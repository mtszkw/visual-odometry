import cv2
import numpy as np

class FeatureTracker:        
    # Finds indices of bad features (with status=0 or position outside the frame).
    # @param features Detected (or tracked) features.
    # @param frame Image matrix.
    # @param status Vector of status flags returned by optical flow.
    # @returns Vector containing indices that should be filtered out.
    def calcWrongFeatureIndices(self, features, frame, status):
        status_ = status.copy()
        for idx, pt in enumerate(features):
            if pt[0] < 0 or pt[1] < 0 or pt[0] > frame.shape[1] or pt[1] > frame.shape[0]:
                status_[idx] = 0
        wrongIndices = np.where(status_ == 0)[0]
        return wrongIndices

    # Tracks features using Lucas-Kanade optical flow and filters out bad features.
    # @param prevFrame Previous image.
    # @param currFrame Current (next) image.
    # @param prevPts Features detected on previous frame.
    # @param removeOutliers Set to true if you want to remove bad features after tracking.
    # @returns Features from previous and current frame (tracked), both filtered.
    def trackFeatures(self, prevFrame, currFrame, prevPts, removeOutliers=False):
        # Feature tracking on the 2nd frame
        currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevPts, None)

        if removeOutliers:
            # Filter out features that were not tracked (status=0) or are outside the image
            wrongIndices = self.calcWrongFeatureIndices(currPts, currFrame, status)
            prevPts = np.delete(prevPts, wrongIndices, axis=0)
            currPts = np.delete(currPts, wrongIndices, axis=0)
            
        return prevPts, currPts
