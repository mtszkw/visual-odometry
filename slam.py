import cv2
import numpy as np
from math import isnan

from DatasetReader import DatasetReaderTUM
from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker

# Draws detected and tracked features on a frame (motion vector is drawn as a line).
# @param frame Frame to be used for drawing (will be converted to RGB).
# @param prevPts Previous frame keypoints.
# @param currPts Next frame keypoints.
def drawFrameFeatures(frame, prevPts, currPts):
    currFrameRGB = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2RGB)
    for i in range(len(currPts)-1):
        cv2.circle(currFrameRGB, tuple(currPts[i]), radius=3, color=(200, 100, 0))
        cv2.line(currFrameRGB, tuple(prevPts[i]), tuple(currPts[i]), color=(200, 100, 0))
        cv2.putText(currFrameRGB, "Frame: {}".format(frameIdx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
        cv2.putText(currFrameRGB, "Features: {}".format(len(currPts)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))    
    cv2.imshow("Frame with keypoints", currFrameRGB)
    

def vignetteFiltering(prevPts, currPts, vignette):
    wrongIndices = []
    ret, thVign = cv2.threshold(vignette, 127, 255, cv2.THRESH_BINARY)
    for i in range(len(prevPts)-1):
        if thVign[int(prevPts[i, 1]), int(prevPts[i, 0])] == 0 or thVign[int(currPts[i, 1]), int(currPts[i, 0])] == 0:
            wrongIndices.append(i)
    prevPts = np.delete(prevPts, wrongIndices, axis=0)
    currPts = np.delete(currPts, wrongIndices, axis=0)
    return prevPts, currPts


if __name__ == "__main__":
    # datasetReader = DatasetReaderTUM("videos/sequence_11/", scaling=0.75)
    datasetReader = DatasetReaderKITTI("videos/KITTI/data_odometry_gray/dataset/sequences/00/")

    detector = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    tracker = FeatureTracker()

    currentRot = np.eye(3)
    currentPos = np.zeros((3,1))
    K = datasetReader.readCameraMatrix()
    # vignette = datasetReader.readVignette()
    trajectoryImage = np.zeros((500, 500, 3), np.uint8)
    
    prevPts = np.empty(0)
    prevFrame = datasetReader.readFrame(0)
 
    # Process next frames
    for frameIdx in range(1, datasetReader.getFramesCount()):
        if len(prevPts) < 25:
            prevPts = cv2.KeyPoint_convert(detector.detect(prevFrame))
        
        currFrame = datasetReader.readFrame(frameIdx)
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts, removeOutliers=True)
        # prevPts, currPts = vignetteFiltering(prevPts, currPts, vignette)

        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        # groundTruth = datasetReader.readGroundtuthPosition(frameIdx)
        # if isinstance(groundTruth, tuple):
            # groundPos, groundScale = groundTruth
            # print("Scale for frame {} is {}".format(frameIdx, groundScale))
            # ground_x = int(groundPos[0] + (trajectoryImage.shape[1] / 2)) 
            # ground_z = int(groundPos[2] + (trajectoryImage.shape[0] / 2))
            # cv2.circle(trajectoryImage, (ground_x, ground_z), radius=15, color=(0, 150, 0))
        # else:
            # groundScale = 0.1

        # currentPos = currentPos + groundScale * currentRot.dot(T)
        # currentRot = R.dot(currentRot)

        # x = int(currentPos[0] + (trajectoryImage.shape[1] / 2)) 
        # z = int(currentPos[2] + (trajectoryImage.shape[0] / 2))
        # cv2.circle(trajectoryImage, (x, z), radius=15, color=(200, 200, 200))
        # cv2.imshow("Trajectory", trajectoryImage)

        drawFrameFeatures(currFrame, prevPts, currPts)
        if cv2.waitKey(1) == ord('q'):
            break

        prevFrame = currFrame
        prevPts = currPts

    cv2.destroyAllWindows()