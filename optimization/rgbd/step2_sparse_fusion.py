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
from absl import app, flags, logging
import scipy.io as sio
import numpy as np
import sys

sys.path.append("../..")
from RGBD_utils.SparseFusion import (
    find_3d_keypoints_from_landmark_and_depth_86,
    get_trans_base_to_camera,
)
from RGBD_utils.CropMask import crop_depth_for_fusion
from RGBD_utils.PoseTools import PoseTools
from third_party.ply import write_ply, write_obj
import cv2


def load_from_npz(base_dir):
    data = np.load(os.path.join(base_dir, "step1_data_for_fusion.npz"))
    K = data["K"]
    lmk3d_select = data["lmk3d_select"]
    img_select = data["img_select"]
    depth_select = data["depth_select"]
    pt3d_select = data["pt3d_select"]
    first_trans_select = data["first_trans_select"]

    return img_select, depth_select, lmk3d_select, pt3d_select, first_trans_select, K


def run(prefit_dir):
    print("---- step2 start -----")
    print("running base:", prefit_dir)
    (
        img_select,
        depth_select,
        lmk3d_select,
        pt3d_select,
        first_trans_select,
        K,
    ) = load_from_npz(prefit_dir)

    # 1. get mask depth
    _, _, depth_ori_select = crop_depth_for_fusion(
        img_select, depth_select, lmk3d_select
    )

    # 2. fusion 3d points
    pt3d_remove_outliers_camera = find_3d_keypoints_from_landmark_and_depth_86(
        first_trans_select, pt3d_select, lmk3d_select, depth_ori_select, img_select, K
    )

    #  get trans
    trans_base_2_camera = get_trans_base_to_camera(
        pt3d_remove_outliers_camera.transpose(), FLAGS.is_bfm
    )

    if FLAGS.is_bfm:
        thr1 = 20.0
        thr2 = 20.0
    else:
        thr1 = 2.0
        thr2 = 2.0

    pt3d_remove_outliers = PoseTools.backtrans(
        pt3d_remove_outliers_camera, trans_base_2_camera
    )
    bad_idx = pt3d_remove_outliers_camera[2, :] < 1
    pt3d_remove_outliers[:, bad_idx] = -10000

    # # ues y and z coord
    s1 = pt3d_remove_outliers[1:, 0:8]
    s2 = pt3d_remove_outliers[1:, np.array(range(16, 8, -1))]
    loss = np.sum(np.array(abs(abs(s1) - abs(s2))), axis=0)
    mid = np.median(loss[loss < thr1])
    error = np.array(loss > min(mid + 0.5, thr2)).astype(np.int32)
    error_countour = np.concatenate(
        (np.concatenate((error, [0])), error[np.array(range(7, -1, -1))])
    )
    bad_idx = np.where(error_countour[:] > 0)
    pt3d_remove_outliers[:, bad_idx] = -10000

    # save
    np.savez(
        os.path.join(prefit_dir, "step2_fusion.npz"),
        pt3d_remove_outliers=pt3d_remove_outliers,
        trans_base_2_camera=trans_base_2_camera,
        first_trans_select=first_trans_select,
        depth_ori_select=depth_ori_select,
    )

    temp = pt3d_remove_outliers[:, pt3d_remove_outliers[2, :] > -100]  # 3 * n
    write_ply(
        os.path.join(prefit_dir, "vis_pt3d_remove_outliers.ply"),
        temp.transpose(),
        None,
        None,
        True,
    )

    print("---- step2 succeed -----")


def main(_):
    prefit_dir = FLAGS.prefit
    run(prefit_dir)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("prefit", "prefit_bfm/", "output data directory")
    flags.DEFINE_boolean("is_bfm", False, "default: False")
    app.run(main)
