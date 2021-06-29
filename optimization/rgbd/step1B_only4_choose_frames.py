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

from absl import app, flags
import scipy.io as sio
import cv2
import numpy as np
import sys

sys.path.append("..")
import scipy.ndimage
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
    print("---- step1B start -----")
    print("running base:", prepare_dir)

    img_all, depth_all, lmk3d_all, pt3d_all, rgb_name_all, K = load_from_npz(
        prepare_dir
    )

    #####################################################  sort to mid - left - right - up ##############################
    # input 4*2*86 (4 is the number of pics,86 is the number of landmarks)
    up_down_all, left_right_all = chooseFrame.get_abs_angle_by_orth_pnp(
        lmk3d_all
    )  # check
    idright = np.argmin(left_right_all)
    idleft = np.argmax(left_right_all)
    tt = list(set(range(4)) - set(list([idright, idleft])))
    maxx = 0
    maxx_id = -1
    minn = 10000
    minn_id = -1
    for i in tt:
        if up_down_all[i] > maxx:
            maxx = up_down_all[i]
            maxx_id = i
        if up_down_all[i] < minn:
            minn = up_down_all[i]
            minn_id = i
    idmid = maxx_id
    idup = minn_id
    new_list = [idmid, idleft, idright, idup]

    lmk3d_select = lmk3d_all[new_list, :, :]
    img_select = img_all[new_list, :, :]
    pt3d_select = pt3d_all[new_list, :, :]
    depth_select = depth_all[new_list, :, :]
    rgb_name_select = rgb_name_all[new_list]

    up_down_select = up_down_all[new_list]
    left_right_select = left_right_all[new_list]

    ##################################################### check close eyes or not ##############################
    for iter in range(4):
        pt2d = lmk3d_select[iter].transpose()  # get n*2 matrix
        if iter == 0:
            ref_eye_close_ok = chooseFrame.check_eye_close(pt2d, 0)
        else:
            ref_eye_close_ok = chooseFrame.check_eye_close(pt2d, 1)
        if ref_eye_close_ok <= 0:
            print("!!!! error in check eye close !!!!")
            sys.exit()

    #################################################### check angles of face ##################################
    flag_angle = 1
    if left_right_select[0] > 5 or left_right_select[0] < -5:
        flag_angle = -1
        print(flag_angle)
    if left_right_select[2] > -15 or left_right_select[2] < -60:
        flag_angle = -2
        print(flag_angle)
    if left_right_select[1] > 60 or left_right_select[1] < 15:
        flag_angle = -3
        print(flag_angle)
    # if up_down_select[3] >  up_down_select[0]-5 or  up_down_select[3] <  up_down_select[0] - 20:
    if (
        up_down_select[3] > up_down_select[0] - 5
        or up_down_select[3] < up_down_select[0] - 30
    ):
        flag_angle = -4
        print(flag_angle)
    if flag_angle <= 0:
        print("!!!! error in check face rotation angles !!!!")
        sys.exit()

    ############################################### calculate pose ################################################
    left_lmk_index = [
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        55,
        57,
        59,
        61,
        62,
        66,
        67,
        72,
        75,
        76,
        78,
        81,
        84,
    ]
    right_lmk_index = [
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        56,
        58,
        60,
        64,
        65,
        69,
        70,
        73,
        74,
        77,
        80,
        83,
        85,
    ]
    up_lmk_index = [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        17,
        22,
        23,
        24,
        25,
        30,
        31,
        32,
        33,
        34,
        66,
        70,
        71,
        74,
        75,
        84,
        85,
    ]
    pt3d_ref = pt3d_select[0].transpose()
    pt3d_left = pt3d_select[1].transpose()
    pt3d_right = pt3d_select[2].transpose()
    pt3d_up = pt3d_select[3].transpose()
    flag_ok2, trans_left = chooseFrame.call_one_pose_by_DLT(
        left_lmk_index, pt3d_ref, pt3d_left
    )
    flag_ok3, trans_right = chooseFrame.call_one_pose_by_DLT(
        right_lmk_index, pt3d_ref, pt3d_right
    )
    flag_ok4, trans_chin = chooseFrame.call_one_pose_by_DLT(
        up_lmk_index, pt3d_ref, pt3d_up
    )

    if flag_ok2 < 0 or flag_ok3 < 0 or flag_ok4 < 0:
        print("!!!! error in call pose! not enough 3d inliners !!!!")
        return -1

    trans_ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    first_trans_select = [trans_ref, trans_left, trans_right, trans_chin]

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
        inliners=[[0, 1, 2, 3]],
    )

    for i in range(img_select.shape[0]):
        cv2.imwrite(prefit_dir + "/" + str(i) + ".png", img_select[i])

    print("---- step1B succeed -----")


def main(_):
    prepare_dir = FLAGS.prepare_dir
    prefit_dir = FLAGS.prefit
    run(prepare_dir, prefit_dir)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("prepare_dir", "prepare/", "output data directory")
    flags.DEFINE_string("prefit", "prefit_bfm/", "output data directory")

    app.run(main)
