import cv2
import matplotlib.pyplot as plt

# Draws detected and tracked features on a frame (motion vector is drawn as a line).
# @param frame Frame to be used for drawing (will be converted to RGB).
# @param prevPts Previous frame keypoints.
# @param currPts Next frame keypoints.
def drawFrameFeatures(frame, prevPts, currPts, frameIdx):
    currFrameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for i in range(len(currPts)-1):
        cv2.circle(currFrameRGB, tuple(currPts[i]), radius=3, color=(200, 100, 0))
        cv2.line(currFrameRGB, tuple(prevPts[i]), tuple(currPts[i]), color=(200, 100, 0))
        cv2.putText(currFrameRGB, "Frame: {}".format(frameIdx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
        cv2.putText(currFrameRGB, "Features: {}".format(len(currPts)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))    
    cv2.imshow("Frame with keypoints", currFrameRGB)

#
# @param trackedPoints
# @param groundtruthPoints
def updateTrajectoryDrawing(trackedPoints, groundtruthPoints):
    plt.cla()
    plt.plot(trackedPoints[:,0], trackedPoints[:,2], c='blue', label="Tracking")
    plt.plot(groundtruthPoints[:,0], groundtruthPoints[:,2], c='green', label="Ground truth")
    plt.title("Trajectory")
    plt.legend()
    plt.draw()
    plt.pause(0.01)

def savePly(points, colors, output_file):
    vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)]
    vertexes = [ v for v in vertexes if v[2] >= 0 ] # Discard negative z
    dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    array = np.array(vertexes, dtype=dtypes)
    element = plyfile.PlyElement.describe(array, "vertex")
    plyfile.PlyData([element]).write(output_file)