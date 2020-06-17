import cv2
from keras.models import load_model
from utils.faceNetKeras import FaceNet
from utils.faceDetector import FaceDetector
import numpy as np

np.set_printoptions(suppress=True)

# ms 0, asish 1, roshan 2, modi 3, obama 4

user_dict = {0: 'gayathri', 1: 'gal', 2: 'krishnendhu', 3: 'modi', 4: 'obama', 5: 'chrispine'}

faceDetector = FaceDetector()
faceNet = FaceNet()

model = load_model('feedNet.h5')

path = "/home/asish/PycharmProjects/Attendance/gal.jpg"

croppedImages, _ = faceDetector.detectFaces(path)

for i in range(len(croppedImages)):
    image = croppedImages[i]
    image = cv2.resize(image, (160, 160))
    image = image.astype('float') / 255.0
    image = np.expand_dims(image, axis=0)
    embedVal = faceNet.predictEmbedding(image)
    embedVal = np.expand_dims(embedVal, axis=0)
    result = model.predict(embedVal)[0]
    print(user_dict.get(int(np.argmax(result))), np.max(result))
