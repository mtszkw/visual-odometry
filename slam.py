import cv2
import numpy as np
from math import isnan
import matplotlib.pyplot as plt

from utils import drawFrameFeatures
from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker

if __name__ == "__main__":   
    datasetReader = DatasetReaderKITTI("videos/KITTI/data_odometry_gray/dataset/sequences/00/")

    currR = np.eye(3)
    currT = np.zeros((3,1))
    K = datasetReader.readCameraMatrix()

    # Initialize feature extraction objects
    prevPts = np.empty(0)
    prevFrame = datasetReader.readFrame(0)
    tracker = FeatureTracker()
    detector = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    
    # Prepare image for drawing trajectory
    voTruthPoints = []
    voTrackedPoints = []
    trajectoryImage = np.zeros((600, 600, 3), np.uint8)
    trajOffsetX, trajOffsetY = (trajectoryImage.shape[1] / 2), (trajectoryImage.shape[0] / 2)
    
    # Process next frames
    for frameIdx in range(1, datasetReader.getFramesCount()-1):
        if len(prevPts) < 100:
            prevPts = cv2.KeyPoint_convert(detector.detect(prevFrame))
        
        currFrame = datasetReader.readFrame(frameIdx)
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts, removeOutliers=True)

        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        truthT, truthScale = datasetReader.readGroundtuthPosition(frameIdx)
        if truthScale > 0.1:
            # truthX = int(0.5 * truthT[0] + trajOffsetX) 
            # truthY = int(trajOffsetY - truthT[2] * 0.5)
            # cv2.circle(trajectoryImage, (truthX, truthY), radius=1, color=(0, 150, 0))

            currT = currT + truthScale * currR.dot(T)
            currR = R.dot(currR)

            # x = int(0.5 * currT[0] + trajOffsetX) 
            # y = int(trajOffsetY - currT[2] * 0.5)
            # cv2.circle(trajectoryImage, (x, y), radius=1, color=(200, 200, 200))
        
            voTruthPoints.append([truthT[0], truthT[2]])
            voTrackedPoints.append([currT[0], currT[2]])
        # cv2.imshow("Trajectory", trajectoryImage)
            # plt.scatter(x, y, c='blue')
            # plt.scatter(truthX, truthY, c='green')
        # plt.imshow(trajectoryImage)

        drawFrameFeatures(currFrame, prevPts, currPts, frameIdx)
        if cv2.waitKey(1) == ord('q'):
            break

        prevFrame = currFrame
        prevPts = currPts

    cv2.destroyAllWindows()

    voTruthPoints = np.array(voTruthPoints)
    voTrackedPoints = np.array(voTrackedPoints)
    plt.title("Trajectory")
    plt.scatter(voTruthPoints[:,0], voTruthPoints[:,1], c='green', label="Estimation")
    plt.scatter(voTrackedPoints[:,0], voTrackedPoints[:,1], c='blue', label="Ground truth")
    plt.legend()
    plt.show()