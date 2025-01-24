const faceapi = require('@vladmandic/face-api');
const fs = require('fs');
const canvas = require('canvas');
const path = require('path');

// Required for `face-api` to work with Node.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function run() {
  // Load models
  const modelPath = path.resolve('./weights'); // Path to downloaded models
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath); // Face detection model
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath); // Face landmarks model
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath); // Face recognition model
  await faceapi.nets.ageGenderNet.loadFromDisk(modelPath); // Age and gender model

  // Load and process an image
  const imgBuffer = fs.readFileSync('./abhishek.jpeg'); // Replace with your image path
  const img = await canvas.loadImage(imgBuffer);

  // Detect faces with landmarks, descriptors, age, and gender
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors()
    .withAgeAndGender(); 

  console.log('Detections:', detections);

  // Draw results on the image
  const out = faceapi.createCanvasFromMedia(img);
  faceapi.draw.drawDetections(out, detections);
  faceapi.draw.drawFaceLandmarks(out, detections);

  // Draw age and gender on the image
  detections.forEach((detection) => {
    const { age, gender, genderProbability } = detection;
    const box = detection.detection.box;
    const label = `${age.toFixed(1)} years, ${gender} (${(genderProbability * 100).toFixed(1)}%)`;
    const drawBox = new faceapi.draw.DrawBox(box, { label });
    drawBox.draw(out);
  });

  // Save output image
  const outStream = fs.createWriteStream('./output-image.jpg');
  const outBuffer = out.toBuffer('image/jpeg');
  outStream.write(outBuffer);
  outStream.end();
  console.log('Output image saved as output-image.jpg');
}

run().catch(console.error);
