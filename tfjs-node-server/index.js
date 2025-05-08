const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

let model;

// Load model khi server khởi động
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

// Root route
app.get('/', (req, res) => {
  res.send('Server is running. Use POST /predict with an image file.');
});

// Predict route
app.post('/predict', upload.single('image'), async (req, res) => {
  console.log('POST /predict accessed');
  if (!req.file) {
    return res.status(400).send('No image uploaded.');
  }

  try {

    const imageBuffer = fs.readFileSync(req.file.path);
    
    const imageTensor = tf.node
        .decodeImage(imageBuffer, 3)
        .resizeBilinear([224, 224]) // Resize to 224x224 instead of 200x200
        .expandDims(0)               // Add batch dimension
        .div(255.0);               // Normalize pixel values

    // Dự đoán
    const predictionTensor = model.predict(imageTensor);
    const predictions = await predictionTensor.data();

    console.log('Predictions:', predictions);

    // Trả kết quả về client
    const predictionsArray = Array.from(predictions);

// Chuyển thành phần trăm và làm tròn
    const predictionsPercent = predictionsArray.map(p => +(p * 100).toFixed(2));

    res.json({ predictions: predictionsPercent });

  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).send('Prediction failed.');
  } finally {
    // Xóa file upload sau khi xử lý
    fs.unlink(req.file.path, (err) => {
      if (err) console.error('⚠️ Error deleting file:', err);
      else console.log('🧹 Uploaded file deleted.');
    });
  }
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
