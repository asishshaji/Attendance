from architecture import NN
from utils.faceDetector import FaceDetector
from utils.faceNetKeras import FaceNet
from sklearn.model_selection import train_test_split
import os
import cv2
from keras.utils import to_categorical
import numpy as np
from keras.optimizers import Adam

noOfClasses = 5

faceNet = FaceNet()  # loads the facenet model
faceDetector = FaceDetector()  # loads opencv facedetector

feedForwardNetwork = NN(noOfClasses)

learningRate = 0.01
epochs = 50
batch_size = 32

xImageData = []
y = []

counter = 0

for students in os.listdir('datas/dataset'):
    for photo in os.listdir(f'datas/dataset/{students}'):
        image = cv2.imread(f'datas/dataset/{students}/{photo}')
        image = cv2.resize(image, (160, 160))
        image = image.astype('float') / 255.0
        image = np.expand_dims(image, axis=0)
        embeddingFaceNet = faceNet.predictEmbedding(image)
        xImageData.append(embeddingFaceNet)
        y.append(counter)
        print(students,counter)
    counter = counter + 1

x = np.array(xImageData, dtype='float')
y = np.array(y)
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

y_train = to_categorical(y_train, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)

model = feedForwardNetwork.architecture()

model.compile(optimizer=Adam(learning_rate=learningRate, decay=learningRate / epochs),
                           loss="categorical_crossentropy")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle='true',
                       validation_data=(X_test, y_test))
model.save('feedNet.h5')
