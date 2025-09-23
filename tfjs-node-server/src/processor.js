const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const { computeColorHistogram, splitIntoPatches, loadImageTensor, computeIoU, savePatch } = require('./imgUtils');
const { ALPHA, PATCH_SIZE } = require('./config');
const { predictPatch } = require('./model');
const logger = require('./logger');
const OUTPUT_DIR = path.join(__dirname, "public", "tiles");

/**
 * Jensen-Shannon Divergence between two normalized histograms (tf.Tensor1D)
 * returns number
 */
function jensenShannonDivergence(p, q) {
    if (!p || !q) {
        return tf.scalar(0); // hoặc NaN tùy ý
    }
    return tf.tidy(() => {
        const m = p.add(q).div(2);
        return klDiv(p, m).add(klDiv(q, m)).div(2);
    });
}


function computeEntropy(probs) {
    // probs: Array<number>
    return -probs.reduce((s, p) => s + (p * Math.log(p + 1e-8)), 0);
}

function computeAttentionScore(jsd, unhealthyProb, entropy, lambda, mode = 'full') {
    switch (mode) {
        case 'cnn_only':
            return unhealthyProb;
        case 'cnn_jsd':
            return (lambda[0] * jsd) + (lambda[1] * unhealthyProb);
        case 'cnn_entropy':
            return (lambda[1] * unhealthyProb) + (lambda[2] * entropy);
        case 'full':
        default:
            return (lambda[0] * jsd) + (lambda[1] * unhealthyProb) + (lambda[2] * entropy);
    }
}

/**
 * Given an image path, model and referenceHist (tf.Tensor1D),
 * returns { allTiles, selectedTiles }
 */
async function processSingleImage(
    imagePath,
    model,
    referenceHist,
    lambda = [0.3, 0.4, 0.3],
    epsilon = 0.2,
    mode = 'full',
    scaledBBox = { x: 0, y: 0, width: 0, height: 0 }
) {
    const imageTensor = await loadImageTensor(imagePath);
    const patches = splitIntoPatches(imageTensor);

    const allTiles = [];
    const selectedTiles = [];

    for (let idx = 0; idx < patches.length; idx++) {
        const { x, y, patch } = patches[idx];

        // chạy predict
        const probs = await predictPatch(model, patch);
        const unhealthyProb =
            probs.length === 2 ? probs[1] : probs.slice(1).reduce((s, p) => s + p, 0);

        // lưu toàn bộ patch info
        allTiles.push({ x, y, probs, unhealthyProb });

        // ⚡️ lưu tile nếu unhealthyProb > 0.5 (tuỳ chỉnh threshold)
        if (unhealthyProb > 0.5) {
            const filename = `tile_${Date.now()}_${x}_${y}.png`;
            const outPath = path.join(OUTPUT_DIR, filename);

            // lưu patch ra file
            savePatch(patch, outPath);

            // lưu vào danh sách selectedTiles để trả về API
            selectedTiles.push({
                x,
                y,
                probs,
                unhealthyProb,
                file: `/tiles/${filename}` // URL cho frontend load
            });
        }

        patch.dispose();
    }

    imageTensor.dispose();

    return { allTiles, selectedTiles };
}

function selectTilesByEpsilon(tiles, epsilon = 0.2, N = 5) {
    const selected = [];
    const used = new Set();

    for (let i = 0; i < N; i++) {
        const r = Math.random();
        if (r < epsilon) {
            let idx;
            do { idx = Math.floor(Math.random() * tiles.length); } while (used.has(idx));
            used.add(idx);
            selected.push(tiles[idx]);
        } else {
            const sorted = tiles
                .map((t, i) => ({ ...t, idx: i }))
                .filter(t => !used.has(t.idx))
                .sort((a, b) => b.attentionEMA - a.attentionEMA);
            if (sorted.length === 0) break;
            const chosen = sorted[0];
            used.add(chosen.idx);
            selected.push(chosen);
        }
    }
    return selected;
}

/**
 * Compute average histogram from array of healthy image paths (filenames)
 * returns tf.Tensor1D
 */
async function getAverageHealthyReferenceHistogram(healthyImageNames) {
    const histograms = [];
    for (const fn of healthyImageNames) {
        const t = await loadImageTensor(fn);
        const patches = splitIntoPatches(t);
        const center = Math.floor(patches.length / 2);
        const hist = computeColorHistogram(patches[center].patch);
        histograms.push(hist);
        t.dispose();
    }
    const stacked = tf.stack(histograms);
    const avg = stacked.mean(0);
    stacked.dispose();
    histograms.forEach(h => h.dispose());
    return avg;
}

module.exports = {
    jensenShannonDivergence,
    computeAttentionScore,
    processSingleImage,
    selectTilesByEpsilon,
    getAverageHealthyReferenceHistogram
};
