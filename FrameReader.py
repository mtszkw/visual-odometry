import os
import cv2

class FrameReader:
    def __init__(self, datasetPath):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self._datasetPath = os.path.join(__location__, datasetPath)
        self._numFrames = len([x for x in os.listdir(datasetPath) if x.endswith('.jpg')])

        if self._numFrames < 2:
            raise Exception("Not enough images ({}) found, aborting.".format(frameReader.getFramesCount()))
        else:
            print("Found {} images in {}".format(self._numFrames, self._datasetPath))

    def readFrame(self, index=0):
        if index >= self._numFrames:
            raise Exception("Cannot read frame number {} from {}".format(index, self._datasetPath))

        img = cv2.imread(os.path.join(self._datasetPath, "{:05d}.jpg".format(index)), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(img.shape[1] * 3 / 4), int(img.shape[0] * 3 / 4)))
        return img

    def getFramesCount(self):
        return self._numFrames

    def getDatasetPath(self):
        return self._datasetPath
