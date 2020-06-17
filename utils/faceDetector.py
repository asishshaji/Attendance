import cv2


class FaceDetector:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier(
            '/home/asish/PycharmProjects/Attendance/utils/haarcascade_frontalface_default.xml')

    def detectFaces(self, image):
        '''

        :param image: takes image for face detection
        :return:
            croppedImages: Images of faces are cropped
            cord : Coordinates of cropped image
        '''
        img = cv2.imread(image)
        grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cord = []
        croppedImages = []
        faces = self.faceCascade.detectMultiScale(grayScale, 1.3, 5)
        for (x, y, w, h) in faces:
            cord.append([x, y, w, h])
            croppedImages.append(img[y:y + h, x:x + w])

        return croppedImages, cord
