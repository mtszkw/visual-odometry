import cv2
import numpy as np
import matplotlib.pyplot as plt

from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker
from Reconstruction3D import Reconstruction3D
from utils import drawFrameFeatures, updateTrajectoryDrawing


def normalizePoints(points, focal, pp):
    points = [ [(p[0] - pp[0]) / focal, (p[1] - pp[1]) / focal] for p in points]
    return points


if __name__ == "__main__":   
    tracker = FeatureTracker()
    detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    datasetReader = DatasetReaderKITTI("videos/KITTI/data_odometry_gray/dataset/sequences/00/")

    K, focal, pp = datasetReader.readCameraMatrix()
    prevFrameBGR = datasetReader.readFrame(0)
    
    prevPts = np.empty(0)
    voTruthPoints, voTrackPoints = [], []
    rotation, position = np.eye(3), np.zeros((3,1))
    
    cloudPoints, cloudColors = [], []

    plt.show()

    # Process next frames
    for frameIdx in range(1, 500):
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
        triangPoints = Reconstruction3D().triangulate(prevPts, currPts, P0=np.eye(3,4), P1=np.hstack((rotation, position)))
        triangPoints = [[x/w, y/w, z/w] for [x, y, z, w] in triangPoints]

        colors = [currFrameBGR[int(pt[1]),int(pt[0])] for pt in prevPts]
        cloudColors += colors
        cloudPoints += triangPoints

        # Update vectors of tracked and ground truth positions, draw trajectory
        voTrackPoints.append([position[0], position[2]])
        voTruthPoints.append([truthPos[0], truthPos[2]])
        drawFrameFeatures(currFrame, prevPts, currPts, frameIdx)
        updateTrajectoryDrawing(np.array(voTrackPoints), np.array(voTruthPoints))

        if cv2.waitKey(1) == ord('q'):
            break
               
        # Consider current frame as previous for the next step
        prevPts, prevFrameBGR = currPts, currFrameBGR
    
    Reconstruction3D().createPointCloud(cloudPoints, cloudColors, filename="slam_cloud.ply")

    # plt.savefig('trajectory.png')
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    