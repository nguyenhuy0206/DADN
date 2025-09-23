// split_and_combine.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

// Ground-truth bounding boxes
const bboxMap = {
    "algal_1.jpg": { x: 90, y: 178, width: 366, height: 157, label: "Algal lesion" },
    "blight_2.jpg": { x: 190, y: 120, width: 200, height: 180, label: "Blight 2" },
    "blight_4.jpg": { x: 150, y: 90, width: 220, height: 260, label: "Blight 4" }
};

const scaleX = 512 / 740;
const scaleY = 512 / 485;

const PATCH_SIZE = 128;
const STRIDE = 64;
const NUM_BINS = 64;
const MODEL_INPUT_SIZE = 224;
const alpha = 0.2;

const tileAttentionMap = {};

async function loadImageTensor(imagePath) {
    const buffer = fs.readFileSync(imagePath);
    let imageTensor = tf.node.decodeImage(buffer, 3);
    imageTensor = tf.image.resizeBilinear(imageTensor, [512, 512]);
    imageTensor = imageTensor.div(255.0);
    return imageTensor;
}

async function getAverageHealthyReferenceHistogram(healthyImageNames) {
    const histograms = [];
    for (const fileName of healthyImageNames) {
        const healthyPath = path.join(__dirname, 'img', fileName);
        const healthyTensor = await loadImageTensor(healthyPath);
        const patches = splitIntoPatches(healthyTensor, PATCH_SIZE, STRIDE);

        const centerIdx = Math.floor(patches.length / 2);
        const patchHist = computeColorHistogram(patches[centerIdx].patch);
        histograms.push(patchHist);

        tf.dispose(healthyTensor);
    }
    const stacked = tf.stack(histograms);
    const averageHist = stacked.mean(0);
    tf.dispose(stacked);
    histograms.forEach(h => tf.dispose(h));
    return averageHist;
}

async function loadModel() {
    try {
        const modelPath = `file://${path.join(__dirname, 'model', 'model.json')}`;
        const model = await tf.loadLayersModel(modelPath);
        console.log('Model loaded successfully.');
        return model;
    } catch (err) {
        console.error('Error loading model:', err.message);
        process.exit(1);
    }
}

function computeColorHistogram(patch, bins = NUM_BINS) {
    const [r, g, b] = tf.split(patch, 3, 2);

    const computeChannelHist = (channelTensor) => {
        const flat = channelTensor.flatten();
        const binSize = 1.0 / bins;
        let histogram = new Array(bins).fill(0);
        const values = flat.arraySync();
        for (let val of values) {
            let bin = Math.floor(val / binSize);
            if (bin >= bins) bin = bins - 1;
            histogram[bin]++;
        }
        const total = values.length;
        return tf.tensor1d(histogram.map(v => v / total));
    };

    const rHist = computeChannelHist(r.squeeze());
    const gHist = computeChannelHist(g.squeeze());
    const bHist = computeChannelHist(b.squeeze());

    return tf.concat([rHist, gHist, bHist]);
}

function jensenShannonDivergence(p, q) {
    const epsilon = 1e-8;
    const pSafe = p.add(epsilon);
    const qSafe = q.add(epsilon);
    const m = pSafe.add(qSafe).div(2);

    const kl = (a, b) => tf.tidy(() => a.mul(a.div(b).log()).sum());
    const jsd = kl(pSafe, m).add(kl(qSafe, m)).div(2);
    return jsd.dataSync()[0];
}

function splitIntoPatches(imageTensor, patchSize, stride) {
    const [h, w, c] = imageTensor.shape;
    const patches = [];
    for (let y = 0; y <= h - patchSize; y += stride) {
        for (let x = 0; x <= w - patchSize; x += stride) {
            const patch = imageTensor.slice([y, x, 0], [patchSize, patchSize, c]);
            patches.push({ x, y, patch });
        }
    }
    return patches;
}

async function predictPatch(model, patch) {
    try {
        const resized = tf.image.resizeBilinear(patch, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]).expandDims(0);
        const prediction = model.predict(resized);
        const probs = await prediction.data();
        tf.dispose([resized, prediction]);
        return Array.from(probs);
    } catch (err) {
        console.error('Error predicting patch:', err.message);
        return null;
    }
}

async function visualizeOutput(imagePath, importantTiles, epsilon, lambdaSet, scaledBBox) {
    const canvas = createCanvas(512, 512);
    const ctx = canvas.getContext('2d');

    try {
        const img = await loadImage(imagePath);
        ctx.drawImage(img, 0, 0, 512, 512);
    } catch (err) {
        console.error('Error loading image for visualization:', err.message);
        return;
    }

    // Draw GT Bounding Box (blue)
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.strokeRect(scaledBBox.x, scaledBBox.y, scaledBBox.width, scaledBBox.height);
    ctx.font = '14px Arial';
    ctx.fillStyle = 'blue';
    ctx.fillText('GT Box', scaledBBox.x + 5, scaledBBox.y + 15);

    // Red boxes for top attention tiles
    const topTiles = importantTiles.slice(0, 5);
    for (const tile of topTiles) {
        const { x, y } = tile;
        const att = (typeof tile.attentionEMA !== 'undefined') ? tile.attentionEMA : tile.combinedScore || 0;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
        ctx.fillRect(x, y, PATCH_SIZE, PATCH_SIZE);
        ctx.font = '14px Arial';
        ctx.fillStyle = 'black';
        ctx.fillText(`A=${att.toFixed(2)}`, x + 5, y + 20);
    }

    const buffer = canvas.toBuffer('image/png');
    const lambdaStr = lambdaSet.map(l => l.toFixed(1)).join('-');
    const baseName = path.basename(imagePath, path.extname(imagePath));
    const outputPath = path.join(__dirname, 'output', `${baseName}_lam${lambdaStr}_eps${epsilon}.png`);

    fs.writeFileSync(outputPath, buffer);
    console.log(`Visualization saved as ${outputPath}`);
}

async function processSingleImage(imagePath, model, referenceHist, lambda, epsilon, mode = "full", scaledBBox) {
    const imageTensor = await loadImageTensor(imagePath);
    const patches = splitIntoPatches(imageTensor, PATCH_SIZE, STRIDE);

    const allTiles = [];
    for (let idx = 0; idx < patches.length; idx++) {
        const { x, y, patch } = patches[idx];
        const tileKey = `${path.basename(imagePath)}_${x}_${y}`;
        const previousV = tileAttentionMap[tileKey] || 0;

        const histTensor = computeColorHistogram(patch);
        const jsd = jensenShannonDivergence(referenceHist, histTensor);
        const prediction = await predictPatch(model, patch);

        let unhealthyProb = 0;
        if (prediction.length === 2) {
            unhealthyProb = prediction[1];
        } else {
            unhealthyProb = prediction.reduce((sum, prob, i) => (i === 0 ? sum : sum + prob), 0);
        }

        const entropy = -prediction.reduce((s, p) => s + (p * Math.log(p + 1e-8)), 0);
        const combinedScore = computeAttentionScore(jsd, unhealthyProb, entropy, lambda, mode);

        const newV = alpha * combinedScore + (1 - alpha) * previousV;
        tileAttentionMap[tileKey] = newV;

        const overlapRatio = computeIoU({ x, y, w: PATCH_SIZE, h: PATCH_SIZE }, scaledBBox);
        const tileLabel = overlapRatio >= 0.5 ? 1 : 0;

        allTiles.push({
            image: path.basename(imagePath),
            x, y, jsd, unhealthyProb, entropy,
            combinedScore, attentionEMA: newV, overlapRatio, tileLabel
        });

        tf.dispose(histTensor);
        tf.dispose(patch);
    }

    tf.dispose(imageTensor);
    const selectedTiles = selectTilesByEpsilon(allTiles, epsilon, 5);
    return { allTiles, selectedTiles };
}

function selectTilesByEpsilon(tiles, epsilon, N = 5) {
    const selected = [];
    const usedIndices = new Set();

    for (let i = 0; i < N; i++) {
        const r = Math.random();
        if (r < epsilon) {
            let idx;
            do {
                idx = Math.floor(Math.random() * tiles.length);
            } while (usedIndices.has(idx));
            usedIndices.add(idx);
            selected.push(tiles[idx]);
        } else {
            const sorted = tiles
                .map((tile, i) => ({ ...tile, idx: i }))
                .filter(tile => !usedIndices.has(tile.idx))
                .sort((a, b) => b.attentionEMA - a.attentionEMA);
            if (sorted.length > 0) {
                const chosen = sorted[0];
                usedIndices.add(chosen.idx);
                selected.push(chosen);
            }
        }
    }
    return selected;
}

function computeAttentionScore(jsd, unhealthyProb, entropy, lambda, mode = "full") {
    switch (mode) {
        case "cnn_only":
            return unhealthyProb;
        case "cnn_jsd":
            return (lambda[0] * jsd) + (lambda[1] * unhealthyProb);
        case "cnn_entropy":
            return (lambda[1] * unhealthyProb) + (lambda[2] * entropy);
        case "full":
        default:
            return (lambda[0] * jsd) + (lambda[1] * unhealthyProb) + (lambda[2] * entropy);
    }
}

function computeIoU(tile, bbox) {
    const tileRight = tile.x + tile.w;
    const tileBottom = tile.y + tile.h;
    const bboxRight = bbox.x + bbox.width;
    const bboxBottom = bbox.y + bbox.height;

    const overlapX = Math.max(0, Math.min(tileRight, bboxRight) - Math.max(tile.x, bbox.x));
    const overlapY = Math.max(0, Math.min(tileBottom, bboxBottom) - Math.max(tile.y, bbox.y));

    return (overlapX * overlapY) / (tile.w * tile.h);
}

async function main() {
    const model = await loadModel();
    const healthyImageNames = ['healthy_1.jpg', 'healthy_2.jpg', 'healthy_3.jpg'];
    const referenceHist = await getAverageHealthyReferenceHistogram(healthyImageNames);

    const testImage = 'algal_1.jpg'; // change to 'blight_2.jpg' or 'blight_4.jpg'
    const testImagePath = path.join(__dirname, 'img', testImage);

    const bbox = bboxMap[testImage];
    const scaledBBox = {
        x: Math.round(bbox.x * scaleX),
        y: Math.round(bbox.y * scaleY),
        width: Math.round(bbox.width * scaleX),
        height: Math.round(bbox.height * scaleY)
    };

    const lambdaSet = [0.3, 0.4, 0.3];
    const epsilon = 0.2;

    const result = await processSingleImage(testImagePath, model, referenceHist, lambdaSet, epsilon, "full", scaledBBox);
    await visualizeOutput(testImagePath, result.selectedTiles, epsilon, lambdaSet, scaledBBox);
}

main();
