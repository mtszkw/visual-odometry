import cv2
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

class Reconstruction3D:
    def __init__(self):
        pass

    def triangulate(self, points0, points1, P0, P1):
        # triangPoints = cv2.triangulatePoints(np.eye(3, 4), P,
            # np.transpose(normalizePoints(prevPts, focal=focal, pp=pp)),
            # np.transpose(normalizePoints(currPts, focal=focal, pp=pp))
        # )
        triangPoints = cv2.triangulatePoints(P0, P1, np.transpose(points0), np.transpose(points1))
        return np.transpose(triangPoints)

    def createPointCloud(self, points, colors, filename):
        cloud = PyntCloud(pd.DataFrame(
            data=np.hstack((np.array(points), np.array(colors))),
            columns=["x", "y", "z", "red", "green", "blue"])
        )
        cloud.to_file(filename)