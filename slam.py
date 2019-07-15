import cv2
import numpy as np

from FrameReader import FrameReader
from FeatureTracker import FeatureTracker

# Calculates camera matrix given focal and principal point information.
# @returns Camera matrix, focal, principal point.
def calculateCameraMatrix():
    fx = 0.349153000000000
    fy = 0.436593000000000
    cx = 0.493140000000000
    cy = 0.499021000000000
    
    focal = (fx + fy) / 2
    pp = (cx, cy)
    K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
    return K, focal, pp


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

    # Feature detection on the 1st frame
    prevFrame = frameReader.readFrame(0)
    prevPts = cv2.KeyPoint_convert(detector.detect(prevFrame))
    print("Detected {} features in the first frame".format(len(prevPts)))

    # Feature tracking on the 2nd frame
    currFrame = frameReader.readFrame(1)
    prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts)

    # Calculate camera parameters and find E (focal and p.p. were taken from camera.txt file)
    K, focal, pp = calculateCameraMatrix()
    
    E, mask = cv2.findEssentialMat(prevPts, currPts, focal, pp, cv2.RANSAC, 0.99, 1.0)
    retval, finalR, finalT, mask = cv2.recoverPose(E, currPts, prevPts, K)
    print("Constructed camera matrix {}:\n{}".format(K.shape, K))
    print("Essential matrix {} calculated:\n{}".format(E.shape, E))
    print("Computed rotation matrix {}:\n{}".format(finalR.shape, finalR))
    print("Computed translation matrix {}:\n{}".format(finalT.shape, finalT))
    
    prevFrame = currFrame
    prevPts = currPts

    # trajectoryImage = np.zeros((300, 300, 3), np.uint8)

    # Process next frames
    for frameIdx in range(2, frameReader.getFramesCount()):
        # Retrack if too many features were filtered out (less than ... points left)
        if len(prevPts) < 25:
            prevPts = cv2.KeyPoint_convert(detector.detect(currFrame))
        
        currFrame = frameReader.readFrame(frameIdx)
        prevPts, currPts = tracker.trackFeatures(prevFrame, currFrame, prevPts)

        E, mask = cv2.findEssentialMat(prevPts, currPts, focal, pp, cv2.RANSAC, 0.99, 1.0)
        retval, R, T, mask = cv2.recoverPose(E, currPts, prevPts, K)

        # scale = 1.0 # TODO: not used now
        # finalT = T
        # finalR = finalR * R
        # x = int(finalT[0] + (trajectoryImage.shape[1] / 2)) 
        # y = int(finalT[1] + (trajectoryImage.shape[0] / 2))
        # cv2.circle(trajectoryImage, (x, y), radius=3, color=(100, 200, 0))

        drawFrameFeatures(currFrame, prevPts, currPts)
        if cv2.waitKey(1) == ord('q'):
            break

        prevFrame = currFrame
        prevPts = currPts

    cv2.destroyAllWindows()