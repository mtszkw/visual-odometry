import cv2
import os
import time

class FrameReader:
    def __init__(self, datasetPath):
        self._datasetPath = datasetPath
        self._numFrames = len([x for x in os.listdir(datasetPath) if x.endswith('.jpg')])
        print("Initializing FrameReader. Found {} images in dataset {}.".format(self._numFrames, self._datasetPath))

    def readFrame(self, index=0):
        if index >= self._numFrames:
            raise Exception("Cannot read frame number {} from {}".format(index, self._datasetPath))

        return cv2.imread(os.path.join(self._datasetPath, "{:05d}.jpg".format(index)))

    def getFramesCount(self):
        return self._numFrames

    def getDatasetPath(self):
        return self._datasetPath


if __name__ == "__main__":
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    frameReader = FrameReader(os.path.join(__location__, "videos/sequence_11/images/"))

    # If there are less than two frames in directory, throw an exception and quit.
    if frameReader.getFramesCount() < 2:
        raise Exception("Not enough images ({}) found, aborting.".format(frameReader.getFramesCount()))

    # Read first two frames
    prevFrame = frameReader.readFrame(0)
    currFrame = frameReader.readFrame(1)
    
    # Initialize FAST feature detector with non max. suppresion and threshold,
    # Then extract features from 1st and 2nd frame.
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    prevKeypts = fast.detect(prevFrame, None)
    currKeypts = fast.detect(currFrame, None)

    # Find the essential matrix (focal and pp were taken from camera.txt file)
    focalLength = 1.0
    principalPoint = (0.493140000000000	0.499021000000000)
    E = findEssentialMat(prevKeypts, currKeypts, focalLength, principalPoint, RANSAC, 0.99, 1.0) #, mask?)
    # recoverPose(E, currPoints, prevPoints, R, t, focal, pp, mask?)

    # for frameIdx in range(frameReader.getFramesCount()):
    #     frame = frameReader.readFrame(frameIdx)

    #     keypts = fast.detect(frame, None)
    #     frameKeypts = cv2.drawKeypoints(frame, keypts, None, color=(255,0,0))
    #     cv2.imshow("Frame with keypoints", frameKeypts)

    #     if cv2.waitKey(1) == ord('q'):
    #         break

    # cv2.destroyAllWindows()