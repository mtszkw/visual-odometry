import cv2
import os
import time

if __name__ == "__main__":
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    datasetPath = os.path.join(__location__, "videos/sequence_11/images/")
    numFrames = len([x for x in os.listdir(datasetPath) if x.endswith('.jpg')])
    print("Found {} images in dataset {}".format(numFrames, datasetPath))

    if numFrames < 2:
        raise Exception("Not enough images ({}) found in {}, aborting.".format(numFrames, datasetPath))
    
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)

    for frameIdx in range(numFrames):
        frame = cv2.imread(os.path.join(datasetPath, "{:05d}.jpg".format(frameIdx)))
        # cv2.imshow("Frame", frame)

        keypts = fast.detect(frame, None)
        frameKeypts = cv2.drawKeypoints(frame, keypts, None, color=(255,0,0))
        cv2.imshow("Frame with keypoints", frameKeypts)

        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()