import os
from utils.faceDetector import FaceDetector
import cv2
import shutil
from datetime import datetime

faceDetector = FaceDetector()

name = input("Enter the name")
if os.path.exists(f'./dataset/{name}'):
    shutil.rmtree(f'./dataset/{name}')
os.mkdir(f'./dataset/{name}')

croppedImages = []

for file in os.listdir('../datas/fullData'):
    croppedImages, _ = faceDetector.detectFaces(f'/home/asish/PycharmProjects/Attendance/helpers/fullData/{file}')

for i in range(len(croppedImages)):
    cv2.imwrite(f'./dataset/{name}/{datetime.now().microsecond}.jpg', croppedImages[i])
