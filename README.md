#### Datasets
- [KITTI Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [Monocular Visual Odometry Dataset (TUM Department of Informatics)](https://vision.in.tum.de/data/datasets/mono-dataset)

#### Understanding the camera.txt format
As written [here](https://github.com/JakobEngel/dso#31-dataset-format), values in camera.txt file, in case of FOV camera model, stand for:
```
FOV fx fy cx cy omega
in_width in_height
"crop" / "full" / "fx fy cx cy 0"
out_width out_height
```