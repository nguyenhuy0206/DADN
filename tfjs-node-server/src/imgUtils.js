const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const { IMG_DIR, UPLOAD_DIR, PATCH_SIZE, STRIDE, NUM_BINS } = require('./config');
const logger = require('./logger');
const { MODEL_INPUT_SIZE } = require('./config');

/**
 * Load image file as normalized tensor [H,W,3], resized to 512x512
 */
async function loadImageTensor(imagePath) {
    console.log("👉 loadImageTensor input:", imagePath);

    let absPath;
    if (path.isAbsolute(imagePath)) {
        absPath = imagePath;
    } else if (imagePath.startsWith('uploads')) {
        // nếu là từ multer
        absPath = path.join(UPLOAD_DIR, path.basename(imagePath));
    } else {
        // dataset (chỉ tên file)
        absPath = path.join(IMG_DIR, imagePath);
    }

    console.log("👉 Resolved path:", absPath);

    if (!fs.existsSync(absPath)) {
        throw new Error(`File not found: ${absPath}`);
    }

    const buffer = fs.readFileSync(absPath);
    let imageTensor = tf.node.decodeImage(buffer, 3);
    imageTensor = tf.image.resizeBilinear(imageTensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    imageTensor = imageTensor.div(255.0);
    return imageTensor;
}



function splitIntoPatches(imageTensor, patchSize = PATCH_SIZE, stride = STRIDE) {
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

/**
 * Compute normalized color histogram per channel and concat (R,G,B).
 * Returns tf.Tensor1D length = 3*bins
 */
function computeColorHistogram(patch, bins = NUM_BINS) {
    return tf.tidy(() => {
        // patch assumed in [0,1]
        const channels = tf.split(patch, 3, 2); // list of [H,W,1]
        const channelHists = channels.map(ch => {
            const flat = ch.flatten();
            // scale 0..1 to bins 0..bins-1
            const indices = flat.mul(bins).floor().cast('int32');
            // clamp
            const clipped = tf.clipByValue(indices, 0, bins - 1);
            // oneHot and sum
            const oneHot = tf.oneHot(clipped, bins);
            const hist = oneHot.sum(0);
            const norm = hist.div(hist.sum().add(1e-8));
            // dispose temps inside tidy automatically
            return norm;
        });
        return tf.concat(channelHists);
    });
}

/**
 * utility to write visualization buffer to file (used by visualize)
 */
async function saveCanvasBuffer(canvas, outPath) {
    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(outPath, buffer);
    logger.info('Saved', outPath);
}

/**
 * compute IoU between tile and bbox (tile: {x,y,w,h}, bbox: {x,y,width,height})
 */
function computeIoU(tile, bbox) {
    const tileRight = tile.x + tile.w;
    const tileBottom = tile.y + tile.h;
    const bboxRight = bbox.x + bbox.width;
    const bboxBottom = bbox.y + bbox.height;

    const overlapX = Math.max(0, Math.min(tileRight, bboxRight) - Math.max(tile.x, bbox.x));
    const overlapY = Math.max(0, Math.min(tileBottom, bboxBottom) - Math.max(tile.y, bbox.y));

    return (overlapX * overlapY) / (tile.w * tile.h);
}
function savePatch(patchTensor, outPath) {
    const [h, w, c] = patchTensor.shape;
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');

    // lấy data (normalize [0,1] -> [0,255])
    const data = patchTensor.mul(255).cast('int32').dataSync();

    const imageData = ctx.createImageData(w, h);
    for (let i = 0; i < data.length; i++) {
        // data là [R,G,B,R,G,B,...]
        imageData.data[i] = data[i];
    }

    // bổ sung alpha channel
    for (let i = 3; i < data.length; i += 4) {
        imageData.data[i] = 255;
    }

    ctx.putImageData(imageData, 0, 0);

    fs.writeFileSync(outPath, canvas.toBuffer('image/png'));
}
module.exports = {
    loadImageTensor,
    splitIntoPatches,
    computeColorHistogram,
    saveCanvasBuffer,
    computeIoU,
    savePatch
};
