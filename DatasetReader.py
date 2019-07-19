import os
import cv2
import numpy as np
from math import isnan, sqrt

class DatasetReaderTUM:
    def __init__(self, datasetPath, scaling=1.0):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self._datasetPath = os.path.join(__location__, datasetPath)
        self._imagesPath = os.path.join(self._datasetPath, "images")
        self._numFrames = len([x for x in os.listdir(self._imagesPath) if x.endswith(".jpg")])
        self._scaling = scaling

        if self._numFrames < 2:
            raise Exception("Not enough images ({}) found, aborting.".format(frameReader.getFramesCount()))
        else:
            print("Found {} images in {}".format(self._numFrames, self._imagesPath))

    def readFrame(self, index=0):
        if index >= self._numFrames:
            raise Exception("Cannot read frame number {} from {}".format(index, self._imagesPath))

        img = cv2.imread(os.path.join(self._imagesPath, "{:05d}.jpg".format(index)), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(img.shape[1] * self._scaling), int(img.shape[0] * self._scaling)))
        return img

    def readCameraMatrix(self):
        cameraFile = os.path.join(self._datasetPath, "camera.txt")
        with open(cameraFile) as f:
            lines = f.readlines()
            fx, fy, cx, cy, _ = list(map(float, lines[0].rstrip().split("\t")))
            in_width, in_height = list(map(int, lines[1].rstrip().split(" ")))
            K = np.zeros((3,3))
            K[0, 0] = fx * in_width
            K[0, 2] = cx * in_width - 0.5
            K[1, 1] = fy * in_height
            K[1, 2] = cy * in_height - 0.5
            K[2, 2] = 1
            print("Constructed camera matrix {}:\n{}".format(K.shape, K))
            return K


    def readGroundtuthPosition(self, frameId):
        groundtruthFile = os.path.join(self._datasetPath, "groundtruthSync.txt")
        with open(groundtruthFile) as f:
            lines = f.readlines()
            _, tx, ty, tz, _, _, _, _ = list(map(float, lines[frameId].rstrip().split(" ")))
            
            if frameId == 0:
                tx_prev, ty_prev, tz_prev = 0, 0, 0
            else:
                _, tx_prev, ty_prev, tz_prev, _, _, _, _ = list(map(float, lines[frameId-1].rstrip().split(" ")))

            if isnan(tx) or isnan(ty) or isnan(tz) or isnan(tx_prev) or isnan(ty_prev) or isnan(tz_prev):
                return float('nan')

            scale = sqrt((tx-tx_prev) * (tx-tx_prev) + (ty-ty_prev) * (ty-ty_prev) + (tz-tz_prev) * (tz-tz_prev))
            position = (tx, ty, tz)
            return position, scale

    def readVignette(self):
        img = cv2.imread(os.path.join(self._datasetPath, "vignette.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(img.shape[1] * self._scaling), int(img.shape[0] * self._scaling)))
        return img

    def getFramesCount(self):
        return self._numFrames

    def getDatasetPath(self):
        return self._datasetPath
