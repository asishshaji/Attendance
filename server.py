from flask import Flask, request, jsonify
import cv2
from keras.models import load_model
from utils.faceNetKeras import FaceNet
from utils.faceDetector import FaceDetector
import numpy as np
import json

faceDetector = FaceDetector()
faceNet = FaceNet()

model = load_model('feedNet.h5')

app = Flask(__name__)

mapping = {0: "Arjun MS", 1: "Asish", 2: "Roshan", 3: "Modi", 4: "Obama"}


@app.route('/detect', method=['POST', 'GET'])
def detect():
    if request.method == "POST" and len(request.files) > 0:
        image = request.files['file']
        results = []
        croppedImages, _ = faceDetector.detectFaces(image)
        for i in range(len(croppedImages)):
            image = croppedImages[i]
            image = cv2.resize(image, (160, 160))
            image = image.astype('float') / 255.0
            image = np.expand_dims(image, axis=0)
            embedVal = faceNet.predictEmbedding(image)
            embedVal = np.expand_dims(embedVal, axis=0)
            result = model.predict(embedVal)[0]
            results.append(result)
        students = []

        for res in results:
            id = np.argmax(res)
            students.append(mapping.get(id))

        return jsonify(present=json.dump(mapping))


if __name__ == "__main__":
    app.run(debug=True)
