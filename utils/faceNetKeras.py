from keras.models import load_model


class FaceNet:
    def __init__(self):
        self.faceNetModel = load_model('/home/asish/PycharmProjects/Attendance/static/models/facenet_keras.h5')

    def predictEmbedding(self, image):
        return self.faceNetModel.predict(image)[0]
