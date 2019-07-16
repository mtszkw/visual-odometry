#### Datasets
- [KITTI Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [Monocular Visual Odometry Dataset (TUM Department of Informatics)](https://vision.in.tum.de/data/datasets/mono-dataset)

#### Camera file format
To be described.

#### TUM RGB-D dataset trajectory format
As explained [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats), groundtruthSync.txt file contains consecutive poses of a camera i.e.
```
* Each line in the text file contains a single pose.
* The format of each line is 'timestamp tx ty tz qx qy qz qw'
* timestamp (float) gives the number of seconds since the Unix epoch.
* tx ty tz (3 floats) give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system.
* qx qy qz qw (4 floats) give the orientation of the optical center of the color camera in form of a unit quaternion with respect to the world origin as defined by the motion capture system.
```