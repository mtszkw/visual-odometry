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


def normalizePoints(points, focal, pp):
    points = [ [(p[0] - pp[0]) / focal, (p[1] - pp[1]) / focal] for p in points]
    return points


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

    K, focal, pp = datasetReader.readCameraMatrix()
    prevFrameBGR = datasetReader.readFrame(0)
    
    prevPts = np.empty(0)
    voTruthPoints, voTrackPoints = [], []
    rotation, position = np.eye(3), np.zeros((3,1))
    
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
        truthPos, truthScale = datasetReader.readGroundtuthPosition(frameIdx)
        if truthScale <= 0.1:
            continue

        # Update the pose
        position = position + truthScale * rotation.dot(T)
        rotation = R.dot(rotation)

        # Reconstruct 3D points
        if frameIdx == 1:
            P = np.hstack((R, T))
            triangPoints = cv2.triangulatePoints(np.eye(3, 4), P,
                np.transpose(normalizePoints(prevPts, focal=focal, pp=pp)),
                np.transpose(normalizePoints(currPts, focal=focal, pp=pp))
            )

            triangPoints = np.transpose(triangPoints)
            triangPoints = np.array([[x/w, y/w, z/w] for [x, y, z, w] in triangPoints])

            colors = np.array([currFrameBGR[int(pt[1]),int(pt[0])] for pt in prevPts])
            print(colors)
            createPointCloud(triangPoints, colors, "slam_cloud.ply")

        # Update vectors of tracked and ground truth positions, draw trajectory
        voTrackPoints.append([position[0], position[2]])
        voTruthPoints.append([truthPos[0], truthPos[2]])
        updateTrajectoryDrawing(np.array(voTrackPoints), np.array(voTruthPoints))
        drawFrameFeatures(currFrame, prevPts, currPts, frameIdx)

        if cv2.waitKey(1) == ord('q'):
            break
               
        # Consider current frame as previous for the next step
        prevPts, prevFrameBGR = currPts, currFrameBGR
    
    # plt.savefig('trajectory.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
