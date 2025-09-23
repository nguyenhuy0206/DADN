const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const {MODEL_DIR} = require('./config');
const logger = require('./logger')


async function loadModel(modelFolder = MODEL_DIR) {
  try {
    const modelPath = `file://${path.join(modelFolder, 'model.json')}`;
    const model = await tf.loadLayersModel(modelPath);
    logger.info('Model loaded from', modelPath);
    return model;
  } catch (err) {
    logger.error('Failed to load model:', err.message);
    throw err;
  }
}

/**
 * Predicts a single patch tensor.
 * Returns probabilities as Array<number>
 */
async function predictPatch(model, patchTensor, modelInputSize = 224) {
  // patchTensor: tf.Tensor of shape [H,W,3] (values in [0,1])
  return tf.tidy(() => {
    const resized = tf.image.resizeBilinear(patchTensor, [modelInputSize, modelInputSize]);
    const batched = resized.expandDims(0);
    const out = model.predict(batched);
    // convert to array (synchronous style via dataSync)
    const probs = Array.from(out.dataSync());
    // dispose temporary tensors
    resized.dispose();
    batched.dispose();
    if (out.dispose) out.dispose();
    return probs;
  });
}

module.exports = { loadModel, predictPatch };