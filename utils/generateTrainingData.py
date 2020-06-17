import os
from utils.faceDetector import FaceDetector
import cv2
import shutil
from datetime import datetime

faceDetector = FaceDetector()

name = input("Enter the name")
if os.path.exists(f'../datas/dataset/{name}'):
    shutil.rmtree(f'../datas/dataset/{name}')
os.mkdir(f'../datas/dataset/{name}')

croppedImageList = []
for file in os.listdir('../datas/fullData'):
    croppedImages, _ = faceDetector.detectFaces(f'../datas/fullData/{file}')
    for image in croppedImages:
        croppedImageList.append(image)

for image in croppedImageList:
    cv2.imwrite(f'../datas/dataset/{name}/{datetime.now().microsecond}.jpg', image)
