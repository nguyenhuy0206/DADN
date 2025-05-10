###version of node to run the program
Version of node : v20.19.1

###Testing
on server site: node index.js
on client site: curl -X POST -F "[image_path]" http://localhost:3000/predict

When run successfully, the terminal should be as example:
Loading model from: /Users/admin/Desktop/University/DADN/tfjs-node-server/model/model.json
Server running on http://localhost:3000
Model loaded successfully.
POST /predict accessed
Predictions: Float32Array(6) [
  1.2050749731429278e-8,
  3.317679642123039e-8,
  0.00047478932538069785,
  0.000004734285539598204,
  0.2472582757472992,
  0.7522621750831604
]
🧹 Uploaded file deleted.
POST /predict accessed
Predictions: Float32Array(6) [
  1.2050749731429278e-8,
  3.317679642123039e-8,
  0.00047478932538069785,
  0.000004734285539598204,
  0.2472582757472992,
  0.7522621750831604
]
🧹 Uploaded file deleted.

On the client site will be the class and the predictions for each class(trained with tensor flow):
curl -X POST -F "image=@/Users/admin/Desktop/University/DADN/tfjs-node-server/img/download.jpeg" http://localhost:3000/predict

{"predictions":[0,0,0.05,0,24.73,75.23]}%  



