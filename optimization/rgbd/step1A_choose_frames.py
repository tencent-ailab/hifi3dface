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

import sys

sys.path.append("../..")
from absl import app, flags
import numpy as np
import cv2
from RGBD_utils.chooseFrame import chooseFrame
import os


def load_from_npz(base_dir):
    data = np.load(os.path.join(base_dir, "camera_matrix.npz"))
    K = data["K_depth"]

    data = np.load(os.path.join(base_dir, "formatted_data.npz"))
    lmk3d_all = data["lmk3d_all"]
    img_all = data["img_all"]
    depth_all = data["depth_all"]
    rgb_name_all = data["rgb_name_all"]

    temp = np.load(os.path.join(base_dir, "pt3ds.npz"))
    pt3d_all = np.array(temp["pt3d_all"])

    return img_all, depth_all, lmk3d_all, pt3d_all, rgb_name_all, K


def run(prepare_dir, prefit_dir):
    print("---- step1A start -----")
    print("prepare_dir:", prepare_dir)

    # load ori datas
    img_all, depth_all, lmk3d_all, pt3d_all, rgb_name_all, K = load_from_npz(
        prepare_dir
    )

    # check outliners
    list_len = []
    for i in range(len(lmk3d_all)):
        one = lmk3d_all[i]
        boxsize = np.max(one.transpose(), axis=0) - np.min(one.transpose(), axis=0)
        len1 = boxsize[0] * boxsize[1]
        list_len.append(len1)

    baseline = list_len[0]
    inliners = np.where((list_len < baseline * 2) & (list_len > baseline * 0.5))
    img_all = np.array(img_all)[inliners]
    lmk3d_all = np.array(lmk3d_all)[inliners]
    depth_all = np.array(depth_all)[inliners]
    rgb_name_all = np.array(rgb_name_all)[inliners]
    pt3d_all = np.array(pt3d_all)[inliners]

    # first frame
    ref_id = 0
    trans_ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    ref_ok = chooseFrame.check_eye_close(lmk3d_all[ref_id].T)

    if ref_ok <= 0:
        print("!!!! error in check ref eye close !!!!")
        sys.exit()

    # 1.first pose calculation by pnp
    up_down_all, left_right_all = chooseFrame.get_abs_angle_by_orth_pnp(lmk3d_all)

    # 2.cal infor of chin, id, pose, sequences
    id_chin, trans_chin, part_chin, flag_find_chin = chooseFrame.find_chin(
        pt3d_all, up_down_all, left_right_all, ref_id, img_all
    )
    if flag_find_chin <= 0:
        print("!!!! error no chin !!!!")
        sys.exit()

    # 3.find left and right sequences, cal pose
    (
        id_left,
        trans_left,
        id_right,
        trans_right,
        flag_ok,
        score_left,
        score_right,
    ) = chooseFrame.find_left_right_part(
        left_right_all, img_all, pt3d_all, lmk3d_all, ref_id, part_chin
    )
    if flag_ok <= 0:
        print("!!!! error no left or right !!!!")
        sys.exit()

    # 4. choose best 4 images by blur score
    len_left = np.ceil(len(score_left) / 3)
    left_max = np.argmax(np.array(score_left)[0 : int(len_left) + 1])

    len_right = np.ceil(len(score_right) / 3)
    right_max = np.argmax(np.array(score_right)[0 : int(len_right) + 1])

    select_frame_ind_all = [ref_id, id_left[left_max], id_right[right_max], id_chin]
    first_trans_select = [
        trans_ref,
        trans_left[left_max],
        trans_right[right_max],
        trans_chin,
    ]
    img_select = img_all[select_frame_ind_all]
    lmk3d_select = lmk3d_all[select_frame_ind_all]
    pt3d_select = pt3d_all[select_frame_ind_all]
    depth_select = depth_all[select_frame_ind_all]
    new_list = select_frame_ind_all
    rgb_name_select = rgb_name_all[select_frame_ind_all]

    if os.path.exists(prefit_dir) is False:
        os.makedirs(prefit_dir)

    np.savez(
        os.path.join(prefit_dir, "step1_data_for_fusion.npz"),
        first_trans_select=first_trans_select,
        img_select=img_select,
        new_list=new_list,
        depth_select=depth_select,
        rgb_name_select=rgb_name_select,
        K=K,
        lmk3d_select=lmk3d_select,
        pt3d_select=pt3d_select,
        inliners=inliners,
    )

    for i in range(img_select.shape[0]):
        cv2.imwrite(prefit_dir + "/" + str(i) + ".png", img_select[i])
    print("---- step1A succeed -----")


def main(_):
    prepare_dir = FLAGS.prepare_dir
    prefit_dir = FLAGS.prefit
    run(prepare_dir, prefit_dir)


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    flags.DEFINE_string("prepare_dir", "prepare/", "output data directory")
    flags.DEFINE_string("prefit", "prefit/", "output data directory")

    app.run(main)
