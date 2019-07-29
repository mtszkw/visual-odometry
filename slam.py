import cv2
import numpy as np
import pandas as pd
from math import isnan, sqrt
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

from utils import drawFrameFeatures
from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker

def createPointCloud(points, colors, filename):
    cloud = PyntCloud(pd.DataFrame(
        data=np.hstack((points, colors)),
        columns=["x", "y", "z", "red", "green", "blue"])
    )
    cloud.to_file(filename)


def updateTrajectoryDrawing(trackedPoints, groundtruthPoints):
    plt.cla()
    plt.plot(trackedPoints[:,0], trackedPoints[:,1], c='blue', label="Tracking")
    plt.plot(groundtruthPoints[:,0], groundtruthPoints[:,1], c='green', label="Ground truth")
    plt.title("Trajectory")
    plt.legend()
    plt.draw()
    plt.pause(0.01)


if __name__ == "__main__":   
    tracker = FeatureTracker()
    detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    datasetReader = DatasetReaderKITTI("videos/KITTI/data_odometry_gray/dataset/sequences/00/")

    K = datasetReader.readCameraMatrix()
    prevFrameBGR = datasetReader.readFrame(0)
    
    prevPts = np.empty(0)
    voTruthPoints, voTrackPoints = [], []
    currR, currT = np.eye(3), np.zeros((3,1))
    
    plt.show()

    # Process next frames
    for frameIdx in range(1, datasetReader.getFramesCount()-1):
        prevFrame = cv2.cvtColor(prevFrameBGR, cv2.COLOR_BGR2GRAY)

        # Read current frame (and convert to grayscale)
        currFrameBGR = datasetReader.readFrame(frameIdx)
        currFrame = cv2.cvtColor(currFrameBGR, cv2.COLOR_BGR2GRAY)

        # Detect keypoints in the frame, save only 1000 with best responses
        prevPts = detector.detect(prevFrame)
        prevPts = sorted(prevPts, key = lambda p: p.response, reverse=True)[:1000]
        prevPts = cv2.KeyPoint_convert(prevPts)
        
        # Track features between frames using optical flow
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts, removeOutliers=True)

        # Find the essential matrix using RANSAC and then R matrix and T vector between frames
        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        # Read groundtruth translation T and absolute scale for computing trajectory
        truthT, truthScale = datasetReader.readGroundtuthPosition(frameIdx)
        if truthScale <= 0.1:
            continue

        # Update the pose
        currT = currT + truthScale * currR.dot(T)
        currR = R.dot(currR)

        # Update vectors of tracked and ground truth positions, draw trajectory
        voTrackPoints.append([currT[0], currT[2]])
        voTruthPoints.append([truthT[0], truthT[2]])
        updateTrajectoryDrawing(np.array(voTrackPoints), np.array(voTruthPoints))
        drawFrameFeatures(currFrame, prevPts, currPts, frameIdx)

        if cv2.waitKey(1) == ord('q'):
            break
        
        winSize, minDisp, maxDisp = 5, -1, 63
        stereo = cv2.StereoSGBM_create(minDisparity=minDisp, numDisparities=(maxDisp - minDisp), blockSize=5,
        uniquenessRatio=5, speckleWindowSize=5, speckleRange=5, disp12MaxDiff=1, P1=8*3*winSize**2, P2=32*3*winSize**2)
        disparityMap = stereo.compute(prevFrame, currFrame)

        focal_length = K[0, 0]
        Q2 = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, focal_length*0.05, 0], [0, 0, 0, 1]])

        points_3D = cv2.reprojectImageTo3D(disparityMap, Q2)
        colors = cv2.cvtColor(currFrameBGR, cv2.COLOR_BGR2RGB)
        mask_map = disparityMap > disparityMap.min()
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]
        createPointCloud(output_points, output_colors, "slam.ply")
        
        # Consider current frame as previous for the next step
        prevFrameBGR = currFrameBGR
        prevPts = currPts

    # plt.savefig('trajectory.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
