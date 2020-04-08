import cv2
import sys
import os
import numpy as np

class FaceCropper(object):
    CASCADE_PATH = "./haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return []

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return []

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        croppedFaces = []
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (224, 224))
            i += 1
            croppedFaces.append(lastimg)
        
        return np.array(croppedFaces).reshape((len(croppedFaces), 224, 224, 3))


if __name__ == '__main__':
    detecter = FaceCropper()
    detecter.generate('./images/fake.png', False)