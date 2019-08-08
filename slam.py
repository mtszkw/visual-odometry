import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt

from DatasetReaderKITTI import DatasetReaderKITTI
from FeatureTracker import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing

def savePly(points, colors, output_file):
    vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)]
    vertexes = [ v for v in vertexes if v[2] >= 0 ] # Discard negative z
    dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    array = np.array(vertexes, dtype=dtypes)
    element = plyfile.PlyElement.describe(array, "vertex")
    plyfile.PlyData([element]).write(output_file)


if __name__ == "__main__":
    tracker = FeatureTracker()
    detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    dataset_reader = DatasetReaderKITTI("videos/KITTI/sequences/00/")

    K = dataset_reader.readCameraMatrix()

    prev_points = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot, camera_pos = np.eye(3), np.zeros((3,1))

    plt.show()

    # Process next frames
    for frame_no in range(1, 50):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        prev_points = detector.detect(prev_frame)
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key = lambda p: p.response, reverse=True)[:1000])
    
        # Feature tracking (optical flow)
        prev_points, curr_points = tracker.trackFeatures(prev_frame, curr_frame, prev_points, removeOutliers=True)

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        
        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos = camera_pos + kitti_scale * camera_rot.dot(T)
        camera_rot = R.dot(camera_rot)

        track_positions.append([camera_pos[0], camera_pos[2]])
        kitti_positions.append([kitti_pos[0], kitti_pos[2]])
        drawFrameFeatures(curr_frame, prev_points, curr_points, frame_no)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR

    cv2.destroyAllWindows()

