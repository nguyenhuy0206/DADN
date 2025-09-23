const tf = require('@tensorflow/tfjs-node');
const { computeColorHistogram, splitIntoPatches, loadImageTensor, computeIoU } = require('./imageUtils');
const { ALPHA, PATCH_SIZE } = require('./config');
const { predictPatch } = require('./model');
const logger = require('./logger');

/**
 * Jensen-Shannon Divergence between two normalized histograms (tf.Tensor1D)
 * returns number
 */
function jensenShannonDivergence(p, q) {
  return tf.tidy(() => {
    const eps = 1e-8;
    const pSafe = p.add(eps);
    const qSafe = q.add(eps);
    const m = pSafe.add(qSafe).div(2);
    const kl = (a, b) => a.mul(a.div(b).log()).sum();
    const js = kl(pSafe, m).add(kl(qSafe, m)).div(2);
    const val = js.arraySync();
    return typeof val === 'number' ? val : val[0];
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
async function processSingleImage(imagePath, model, referenceHist, lambda = [0.3,0.4,0.3], epsilon = 0.2, mode = 'full', scaledBBox = {x:0,y:0,width:0,height:0}) {
  const imageTensor = await loadImageTensor(imagePath);
  const patches = splitIntoPatches(imageTensor);

  const allTiles = [];
  // tileAttentionMap kept per process run (can be persisted externally if desired)
  const tileAttentionMap = {};

  for (let idx = 0; idx < patches.length; idx++) {
    const { x, y, patch } = patches[idx];
    const key = `${x}_${y}`;

    const hist = computeColorHistogram(patch);
    const jsd = jensenShannonDivergence(referenceHist, hist);
    const probs = await predictPatch(model, patch); // Array
    const unhealthyProb = (probs.length === 2) ? probs[1] : probs.slice(1).reduce((s,p) => s+p, 0);
    const entropy = computeEntropy(probs);

    const combinedScore = computeAttentionScore(jsd, unhealthyProb, entropy, lambda, mode);

    const prev = tileAttentionMap[key] || 0;
    const ema = ALPHA * combinedScore + (1 - ALPHA) * prev;
    tileAttentionMap[key] = ema;

    const overlapRatio = computeIoU({ x, y, w: PATCH_SIZE, h: PATCH_SIZE }, scaledBBox);
    const tileLabel = overlapRatio >= 0.5 ? 1 : 0;

    allTiles.push({
      x, y, jsd, unhealthyProb, entropy, combinedScore, attentionEMA: ema, overlapRatio, tileLabel, probs
    });

    // dispose tensors of this iteration
    hist.dispose();
    patch.dispose();
  }

  imageTensor.dispose();

  const selected = selectTilesByEpsilon(allTiles, epsilon, 5);

  return { allTiles, selectedTiles: selected };
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
        .sort((a,b) => b.attentionEMA - a.attentionEMA);
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
