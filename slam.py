import cv2
import os
import time
import numpy as np


class FrameReader:
    def __init__(self, datasetPath):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self._datasetPath = os.path.join(__location__, datasetPath)
        self._numFrames = len([x for x in os.listdir(datasetPath) if x.endswith('.jpg')])
        print("Found {} images in {}".format(self._numFrames, self._datasetPath))

    def readFrame(self, index=0):
        if index >= self._numFrames:
            raise Exception("Cannot read frame number {} from {}".format(index, self._datasetPath))

        return cv2.imread(os.path.join(self._datasetPath, "{:05d}.jpg".format(index)))

    def getFramesCount(self):
        return self._numFrames

    def getDatasetPath(self):
        return self._datasetPath


def calcWrongFeatureIndices(features, frame, status):
    status_ = status.copy()
    for idx, pt in enumerate(features):
        if pt[0] < 0 or pt[1] < 0 or pt[0] > frame.shape[1] or pt[1] > frame.shape[0]:
            status_[idx] = 0
    wrongIndices = np.where(status_ == 0)[0]
    return wrongIndices
    

if __name__ == "__main__":
    frameReader = FrameReader("videos/sequence_11/images/")

    # If there are less than two frames in directory, throw an exception and quit.
    if frameReader.getFramesCount() < 2:
        raise Exception("Not enough images ({}) found, aborting.".format(frameReader.getFramesCount()))

    # Read first two frames
    prevFrame = cv2.cvtColor(frameReader.readFrame(0), cv2.COLOR_RGB2GRAY)
    currFrame = cv2.cvtColor(frameReader.readFrame(1), cv2.COLOR_RGB2GRAY)

    # Feature detection on the 1st frame
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    prevPts = cv2.KeyPoint_convert(fast.detect(prevFrame))
    print("Detected {} features in the first frame".format(len(prevPts)))

    # Feature tracking on the 2nd frame
    currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevPts, None)

    # Filter out features that were not tracked (status=0) or are outside the image
    wrongIndices = calcWrongFeatureIndices(currPts, currFrame, status)
    prevPts = np.delete(prevPts, wrongIndices, axis=0)
    currPts = np.delete(currPts, wrongIndices, axis=0)
    print(prevPts.shape, currPts.shape)

    # Find the essential matrix (focal and p.p. were taken from camera.txt file)
    fx = 0.349153000000000
    fy = 0.436593000000000
    cx = 0.493140000000000
    cy = 0.499021000000000
    focal = (fx + fy) / 2
    pp = (cx, cy)
    K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
    
    E, mask = cv2.findEssentialMat(prevPts, currPts, focal, pp, cv2.RANSAC, 0.99, 1.0)
    print("Constructed camera matrix {}:\n{}".format(K.shape, K))
    print("Essential matrix {} calculated:\n{}".format(E.shape, E))

    retval, R, t, mask = cv2.recoverPose(E, currPts, prevPts, K)
    print("Computed rotation matrix {}:\n{}".format(R.shape, R))
    print("Computed translation matrix {}:\n{}".format(t.shape, t))
    
    prevFrame = currFrame
    prevPts = currPts

    # Process next frames
    for frameIdx in range(2, frameReader.getFramesCount()):
        # Read next frame and track features
        currFrame = cv2.cvtColor(frameReader.readFrame(frameIdx), cv2.COLOR_RGB2GRAY)
        currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevPts, None)

        # Filter out features that were not tracked (status=0) or are outside the image
        wrongIndices = calcWrongFeatureIndices(currPts, currFrame, status)
        prevPts = np.delete(prevPts, wrongIndices, axis=0)
        currPts = np.delete(currPts, wrongIndices, axis=0)
        print("Tracked {} features in frame #{} after filtering".format(len(currPts), frameIdx))

        # Retrack if too many features were filtered out (less than 50 points left)
        if len(currPts) < 50:
            prevPts = cv2.KeyPoint_convert(fast.detect(currFrame))
            currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevPts, None)
            
            wrongIndices = calcWrongFeatureIndices(currPts, currFrame, status)
            prevPts = np.delete(prevPts, wrongIndices, axis=0)
            currPts = np.delete(currPts, wrongIndices, axis=0)
            print("Too few features in frame #{}, {} features detected after retracking".format(frameIdx, len(currPts)))

        # Display current frame with motion vectors and some information (frame id, # of features)
        currFrameRGB = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2RGB)
        for i in range(len(currPts)-1):
            cv2.circle(currFrameRGB, tuple(currPts[i]), radius=3, color=(255, 0, 0))
            cv2.line(currFrameRGB, tuple(prevPts[i]), tuple(currPts[i]), color=(255, 0, 0))
        cv2.putText(currFrameRGB, "Frame: {}".format(frameIdx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
        cv2.putText(currFrameRGB, "Features: {}".format(len(currPts)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
        cv2.imshow("Frame with keypoints", currFrameRGB)

        if cv2.waitKey(1) == ord('q'):
            break

        # Swap frames before next iteration
        prevFrame = currFrame
        prevPts = currPts

    cv2.destroyAllWindows()