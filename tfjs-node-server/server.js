const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { loadImageTensor } = require('./src/imgUtils');
const { processSingleImage } = require('./src/processor');

// ⚡️ Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

const app = express();
const PORT = 3000;

// static: phục vụ index.html + tiles
app.use(express.static(path.join(__dirname, 'public')));
app.use('/tiles', express.static(path.join(__dirname, 'public', 'tiles')));

let model;

// load model khi server start
(async () => {
  try {
    const modelPath = path.join(__dirname, 'model', 'model.json');
    console.log(`Loading model from: ${modelPath}`);
    model = await tf.loadLayersModel(tf.io.fileSystem(modelPath));
    console.log('Model loaded successfully.');
  } catch (error) {
    console.error('Error loading model:', error);
  }
})();

app.post('/predict', upload.array('images'), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).send('No images uploaded.');
  }

  const results = [];

  for (const file of req.files) {
    try {
      const imageTensor = await loadImageTensor(file.path);

      // chạy processSingleImage → sẽ lưu tile và trả ra URL
      const tileResult = await processSingleImage(file.path, model, null);

      // predict toàn ảnh (resize 224x224 trước khi predict)
      const resized = tf.image.resizeBilinear(imageTensor, [224, 224]).expandDims(0);
      const predictionTensor = model.predict(resized);
      const predictions = await predictionTensor.data();
      const predictionsPercent = Array.from(predictions).map(p => +(p * 100).toFixed(2));

      results.push({
        file: file.originalname,
        savedAs: file.filename,
        predictions: predictionsPercent,
        tiles: tileResult.selectedTiles   // ⚡️ danh sách tile với URL
      });

      tf.dispose([imageTensor, resized, predictionTensor]);

    } catch (err) {
      console.error(`Error predicting ${file.originalname}:`, err);
      results.push({
        file: file.originalname,
        error: 'Prediction failed'
      });
    }
  }

  res.json({ results });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
