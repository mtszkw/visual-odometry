import cv2
import numpy as np
from math import isnan, sqrt
import matplotlib.pyplot as plt

from utils import drawFrameFeatures
from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker


if __name__ == "__main__":   
    datasetReader = DatasetReaderKITTI("videos/KITTI/data_odometry_gray/dataset/sequences/00/")

    K = datasetReader.readCameraMatrix()
    currR, currT = np.eye(3), np.zeros((3,1))

    # Initialize feature extraction objects
    prevPts = np.empty(0)
    prevFrame = datasetReader.readFrame(0)
    
    tracker = FeatureTracker()
    detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    
    # Prepare image for drawing trajectory
    voTruthPoints, voTrackPoints = [], []
    plt.show()

    # Process next frames
    for frameIdx in range(1, datasetReader.getFramesCount()-1):
        # if len(prevPts) < 50:
        prevPts = detector.detect(prevFrame)
        prevPts = sorted(prevPts, key = lambda p: p.response, reverse=True)[:1000]
        prevPts = cv2.KeyPoint_convert(prevPts)
        
        currFrame = datasetReader.readFrame(frameIdx)
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts, removeOutliers=True)

        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        truthT, truthScale = datasetReader.readGroundtuthPosition(frameIdx)
        if truthScale > 0.1:
            currT = currT + truthScale * currR.dot(T)
            currR = R.dot(currR)

            voTruthPoints.append([truthT[0], truthT[2]])
            voTrackPoints.append([currT[0], currT[2]])

            plt.cla()
            plt.plot(np.array(voTrackPoints)[:,0], np.array(voTrackPoints)[:,1], c='blue', label="Estimation")
            plt.plot(np.array(voTruthPoints)[:,0], np.array(voTruthPoints)[:,1], c='green', label="Ground truth")
            plt.title("Trajectory")
            plt.legend()
            plt.draw()
            plt.pause(0.01)

        drawFrameFeatures(currFrame, prevPts, currPts, frameIdx)
        if cv2.waitKey(1) == ord('q'):
            break

        prevFrame = currFrame
        prevPts = currPts

    plt.savefig('trajectory.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
