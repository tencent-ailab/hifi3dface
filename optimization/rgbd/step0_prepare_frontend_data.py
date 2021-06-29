# -*- coding:utf8 -*-
"""
This file is part of the repo: https://github.com/tencent-ailab/hifi3dface

If you find the code useful, please cite our paper: 

"High-Fidelity 3D Digital Human Head Creation from RGB-D Selfies."
ACM Transactions on Graphics 2021
Code: https://github.com/tencent-ailab/hifi3dface

Copyright (c) [2020-2021] [Tencent AI Lab]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from absl import app, flags
import cv2
import numpy as np
import sys

sys.path.append("../..")
from RGBD_utils.preProcess import preProcess


def load_depth_png_data(capture_dir, prepare_dir):
    rgb_name_all = []
    depth_name_all = []
    name_all = open(os.path.join(prepare_dir, "img_names.txt"))
    line = name_all.readline()
    while 1:
        if line:
            one = line.split(",")
            rgb_name_all.append(one[0])
            dd = one[1][0 : one[1].find(".") + 4]
            depth_name_all.append(dd)
        else:
            break
        line = name_all.readline()
    name_all.close()

    # load img and depth , change to 640 * 480
    imgs_all = []
    depths_all = []
    for i in range(len(rgb_name_all)):
        img = cv2.imread(os.path.join(capture_dir, rgb_name_all[i]))
        imgs_all.append(img)
        depth = preProcess.load_one_PNG_depth(
            os.path.join(capture_dir, depth_name_all[i])
        )
        depths_all.append(depth)

    depths_all = preProcess.depth_bilateral_filter(depths_all)

    # load landmarks , change to 640 * 480
    lmk3d_all = preProcess.load_landmark_rgbd(prepare_dir, 86)
    lmk3d_all = [np.transpose(element) for element in lmk3d_all]

    return imgs_all, depths_all, lmk3d_all, rgb_name_all


def run(capture_dir, prepare_dir, output_dir):
    print("---- step0 start -----")

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    print("capture_dir:", capture_dir)
    print("prepare_dir:", prepare_dir)
    ######################### camera parameters  #############################
    K = {"fx": 592.7247, "fy": 593.7484, "cx": 239.7348, "cy": 319.4659}

    K_depth = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])

    np.savez(os.path.join(output_dir, "camera_matrix.npz"), K_depth=K_depth)

    ####################### get image, depth, lanmdarks ######################
    # depth->png
    img_all, depth_all, lmk3d_all, rgb_name_all = load_depth_png_data(
        capture_dir, prepare_dir
    )

    np.savez(
        os.path.join(output_dir, "formatted_data.npz"),
        lmk3d_all=lmk3d_all,
        img_all=img_all,
        depth_all=depth_all,
        rgb_name_all=rgb_name_all,
    )

    ########################### get 3d keypoints #############################
    pt3d_all = []
    for i in range(len(lmk3d_all)):
        pt2d = lmk3d_all[i]  # 2 * 86
        depth = depth_all[i]

        one_3d = np.zeros((3, len(pt2d[0])))
        for k in range(pt2d.shape[1]):
            dd = depth[int(round(pt2d[1][k])), int(round(pt2d[0][k]))]
            one_3d[0][k] = (pt2d[0][k] - K["cx"]) * dd / K["fx"]
            one_3d[1][k] = (pt2d[1][k] - K["cy"]) * dd / K["fy"]
            one_3d[2][k] = dd
        pt3d_all.append(one_3d)
    np.savez(os.path.join(output_dir, "pt3ds.npz"), pt3d_all=pt3d_all)
    print("---- step0 succeed -----")


def main(_):
    capture_dir = FLAGS.capture_dir
    prepare_dir = FLAGS.prepare_dir
    output_dir = FLAGS.prepare_dir
    run(capture_dir, prepare_dir, output_dir)


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    flags.DEFINE_string("capture_dir", "ori/", "input data directory")
    flags.DEFINE_string("prepare_dir", "prepare/", "output data directory")

    app.run(main)
