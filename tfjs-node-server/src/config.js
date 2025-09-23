const path = require('path')

module.exports = {
    IMG_DIR: path.join(__dirname, '..', 'img'),
    OUTPUT_DIR: path.join(__dirname, '..', 'output'),
    MODEL_DIR: path.join(__dirname, '..', 'model'),
    PATCH_SIZE: 128,
    STRIDE: 64,
    NUM_BINS: 64,
    MODEL_INPUT_SIZE: 224,
    ALPHA: 0.2,
    DEFAULT_LAMBDA: [0.3,0.4,0.3],
    DEFAULT_EPSILON: 0.2,
    SCALE_X: 512 / 740,
    SCALE_Y: 512/485
};