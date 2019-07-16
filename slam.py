import cv2
import numpy as np

from FrameReader import FrameReader
from FeatureTracker import FeatureTracker

# Calculates camera matrix given focal and principal point information.
# @returns Camera matrix, focal, principal point.
def calculateCameraMatrix():
    # fx, fy = 0.4, 0.53
    # cx, cy = 0.5, 0.5
    fx, fy = 0.349153000000000, 0.436593000000000
    cx, cy = 0.493140000000000, 0.499021000000000
    in_width, in_height = 1280, 1024
    
    K = np.zeros((3,3))
    K[0, 0] = fx * in_width
    K[0, 2] = cx * in_width - 0.5
    K[1, 1] = fy * in_height
    K[1, 2] = cy * in_height - 0.5
    K[2, 2] = 1
    print("Constructed camera matrix {}:\n{}".format(K.shape, K))
    return K


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
    

if __name__ == "__main__":
    frameReader = FrameReader("videos/sequence_11/images/")
    detector = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    tracker = FeatureTracker()

    K = calculateCameraMatrix()
    
    currentRot = np.eye(3)
    currentPos = np.zeros((3,1))
    trajectoryImage = np.zeros((300, 300, 3), np.uint8)

    prevPts = np.empty(0)
    prevFrame = frameReader.readFrame(0)

    # Process next frames
    for frameIdx in range(1, frameReader.getFramesCount()):
        if len(prevPts) < 25:
            prevPts = cv2.KeyPoint_convert(detector.detect(prevFrame))
        
        currFrame = frameReader.readFrame(frameIdx)
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts)

        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        scale = 1.0 # TODO: not used now
        currentPos = currentPos + scale * currentRot.dot(T)
        currentRot = R.dot(currentRot)

        print(currentPos.transpose())

        x = int(currentPos[0] + (trajectoryImage.shape[1] / 2)) 
        y = int(currentPos[2] + (trajectoryImage.shape[0] / 2))
        cv2.circle(trajectoryImage, (x, y), radius=1, color=(100, 200, 0))
        cv2.imshow("Trajectory", trajectoryImage)

        drawFrameFeatures(currFrame, prevPts, currPts)
        if cv2.waitKey(1) == ord('q'):
            break

        prevFrame = currFrame
        prevPts = currPts

    cv2.destroyAllWindows()