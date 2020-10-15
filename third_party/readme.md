This folder contains the third party code:
- A differentiable, 3D mesh renderer using TensorFlow, which we refered to as [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer). The version we use in this repository is 
https://github.com/google/tf_mesh_renderer/tree/a6403fbb36a71443ecb822e435e5724550d2b52b. Please visit ../install.sh for compiling deteils.
- A tensorflow version of VGG face model, which is modified from https://github.com/ZZUTK/Tensorflow-VGG-face.git. The pretrained weights are also downloaded from the same repo.
- Our python implementation of a non-rigid variant of the iterative closest point algorithm. The original MATLAB implementation is from https://github.com/charlienash/nricp.

If you use these code in your research, please cite the following follow papers:

```
@InProceedings{Genova_2018_CVPR,
  author = {Genova, Kyle and Cole, Forrester and Maschinot, Aaron and Sarna, Aaron and Vlasic, Daniel and Freeman, William T.},
  title = {Unsupervised Training for 3D Morphable Model Regression},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```

```
@InProceedings{Parkhi15,
  author = {Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman},
  title = {Deep Face Recognition},
  booktitle = {British Machine Vision Conference},
  year = {2015},
}
```

```
@inproceedings{amberg2007optimal,
  title = {Optimal step nonrigid icp algorithms for surface registration},
  author = {Amberg, Brian and Romdhani, Sami and Vetter, Thomas},
  booktitle = {Computer Vision and Pattern Recognition, 2007. CVPR'07. IEEE Conference on},
  pages = {1--8},
  year = {2007},
  organization = {IEEE}
}
```