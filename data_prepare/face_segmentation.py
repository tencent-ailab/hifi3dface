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

"""
The tensorflow pb file used here is transfered from the pretrained pytorch model: https://github.com/switchablenorms/CelebAMask-HQ
"""


from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import cv2
import os
from numpy.linalg import inv


def crop_image_and_process_landmark_for_seg(
    img, landmark, size=300, orig=False, a=0.4, b=0.8, c=0.6
):
    # use landmarks to crop face
    min_x, min_y = np.amin(landmark, 0)
    max_x, max_y = np.amax(landmark, 0)

    c_x = min_x + 0.5 * (max_x - min_x)
    c_y = min_y + a * (max_y - min_y)  # 0.4 b0.3
    half_x = (max_x - min_x) * b  # 0.8 b0.85
    half_y = (max_y - min_y) * c  # 0.6 b0.7
    half_width = max(half_x, half_y)

    st_x = c_x - half_width
    en_x = c_x + half_width
    st_y = c_y - half_width
    en_y = c_y + half_width

    pad_t = int(max(0 - st_y, 0) + 0.5)
    pad_b = int(max(en_y - img.shape[0], 0) + 0.5)
    pad_l = int(max(0 - st_x, 0) + 0.5)
    pad_r = int(max(en_x - img.shape[1], 0) + 0.5)

    st_x_ = int(st_x + 0.5)
    en_x_ = int(en_x + 0.5)
    st_y_ = int(st_y + 0.5)
    en_y_ = int(en_y + 0.5)

    # pad top and bottom
    pad_img = np.concatenate(
        [np.zeros((pad_t, img.shape[1], 3)), img, np.zeros((pad_b, img.shape[1], 3))],
        axis=0,
    )
    # pad left and right
    pad_img = np.concatenate(
        [
            np.zeros((pad_img.shape[0], pad_l, 3)),
            pad_img,
            np.zeros((pad_img.shape[0], pad_r, 3)),
        ],
        axis=1,
    )
    crop_img = pad_img[
        (st_y_ + pad_t) : (en_y_ + pad_t), (st_x_ + pad_l) : (en_x_ + pad_l), :
    ]

    four_points = [st_x_, en_x_, st_y_, en_y_]

    lmk_x = landmark[:, 0]
    lmk_x = (lmk_x - st_x) / crop_img.shape[1] * size
    lmk_y = landmark[:, 1]
    lmk_y = (lmk_y - st_y) / crop_img.shape[0] * size
    landmark = np.stack([lmk_x + 0.5, lmk_y + 0.5], axis=1)

    if crop_img.shape[0] != crop_img.shape[1]:
        crop_img = cv2.resize(crop_img, (crop_img.shape[0], crop_img.shape[0]))

    old_crop_img = crop_img.copy()
    cur_h, cur_w, _ = crop_img.shape
    while cur_h // 4 > size:
        crop_img = cv2.GaussianBlur(crop_img, (3, 3), 1)
        crop_img = crop_img[::2, :, :]
        crop_img = crop_img[:, ::2, :]
        cur_h, cur_w, _ = crop_img.shape
    crop_img = cv2.resize(crop_img, (size, size))

    ## crop reference image
    if orig:
        return crop_img, old_crop_img, landmark, four_points
    else:
        return crop_img, landmark, four_points


color_list = [
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [255, 255, 255],
    [0, 0, 125],
    [0, 125, 0],
    [125, 0, 0],
    [0, 125, 125],
    [125, 125, 0],
    [125, 0, 125],
    [125, 125, 125],
    [0, 0, 60],
    [0, 60, 0],
    [60, 0, 0],
    [60, 60, 0],
]


def run_face_seg(CROP_LMK3D_batch, CROP_IMG_batch, sess):

    inputs = sess.graph.get_tensor_by_name("input:0")
    outputs = sess.graph.get_tensor_by_name("output:0")

    seg_batch = []
    seg_color_batch = []
    count = 0
    for i in range(CROP_IMG_batch.shape[0]):
        img = CROP_IMG_batch[i, :, :, ::-1].astype(np.float32) / 255.0  # RGB
        (
            test_img,
            ori_img,
            prediction3D,
            four_pointsSeg,
        ) = crop_image_and_process_landmark_for_seg(
            img, CROP_LMK3D_batch[i], size=512, orig=True, a=0.3, b=0.85, c=0.7
        )

        test_img = (test_img - 0.5) / 0.5  # -1 ~ 1
        test_img = cv2.resize(test_img, (512, 512))  # 512 x 512 x 3
        test_img = np.transpose(test_img, [2, 0, 1])  # 3 x 512 x 512
        test_img = test_img[np.newaxis, ...]  # 1 x 3 x 512 x 512

        seg_result = sess.run(
            outputs, feed_dict={inputs: test_img}
        )  # # 1 x 19 x 512 x 512
        seg_result = np.transpose(seg_result[0], [1, 2, 0])  # 512 x 512 x 19

        ###########################################################################
        # back_transform
        (
            test_img,
            ori_img,
            prediction3D,
            four_pointsCrop,
        ) = crop_image_and_process_landmark_for_seg(
            img, CROP_LMK3D_batch[i], size=300, orig=True
        )

        st_x_ = int(four_pointsSeg[0])
        en_x_ = int(four_pointsSeg[1])
        st_y_ = int(four_pointsSeg[2])
        en_y_ = int(four_pointsSeg[3])
        first_point_to_sub1 = np.array([(four_pointsSeg[0], four_pointsSeg[2])] * 4)
        # print(first_point_to_sub1)
        four_points1 = (
            np.array([(st_x_, st_y_), (en_x_, st_y_), (en_x_, en_y_), (st_x_, en_y_)])
            - first_point_to_sub1
        )
        # print(four_pointsSeg[0])
        st_x_ = int(four_pointsCrop[0])
        en_x_ = int(four_pointsCrop[1])
        st_y_ = int(four_pointsCrop[2])
        en_y_ = int(four_pointsCrop[3])
        first_point_to_sub2 = np.array([(four_pointsCrop[0], four_pointsCrop[2])] * 4)
        four_points2 = (
            np.array([(st_x_, st_y_), (en_x_, st_y_), (en_x_, en_y_), (st_x_, en_y_)])
            - first_point_to_sub2
        )
        # print(four_pointsCrop[0])
        st_x_ = max(four_pointsSeg[0], four_pointsCrop[0])
        en_x_ = min(four_pointsSeg[1], four_pointsCrop[1])
        st_y_ = max(four_pointsSeg[2], four_pointsCrop[2])
        en_y_ = min(four_pointsSeg[3], four_pointsCrop[3])
        res_point = np.array(
            [(st_x_, st_y_), (en_x_, st_y_), (en_x_, en_y_), (st_x_, en_y_)]
        )

        common_four_points1 = res_point - first_point_to_sub1
        common_four_points2 = res_point - first_point_to_sub2

        st_x_ = 0
        en_x_ = 512
        st_y_ = 0
        en_y_ = 512
        fix_points1 = np.array(
            [(st_x_, st_y_), (en_x_, st_y_), (en_x_, en_y_), (st_x_, en_y_)]
        )
        st_x_ = 0
        en_x_ = 300
        st_y_ = 0
        en_y_ = 300
        fix_points2 = np.array(
            [(st_x_, st_y_), (en_x_, st_y_), (en_x_, en_y_), (st_x_, en_y_)]
        )
        # print(fix_points)

        H1 = cv2.getPerspectiveTransform(
            np.float32(four_points1), np.float32(fix_points1)
        )
        H2 = cv2.getPerspectiveTransform(
            np.float32(four_points2), np.float32(fix_points2)
        )
        H3 = cv2.getPerspectiveTransform(
            np.float32(common_four_points1), np.float32(common_four_points2)
        )
        H4 = inv(H1)
        H4 = H3.dot(H4)
        H4 = H2.dot(H4)
        # print(H4)
        seg_result = cv2.warpPerspective(seg_result, H4, (300, 300))

        seg_result = np.argmax(seg_result, axis=2)

        seg_result = seg_result[..., np.newaxis]  # 300 x 300 x 1
        seg_result_color = np.zeros((300, 300, 3))
        for m in range(0, 300):
            for n in range(0, 300):
                seg_result_color[m, n] = color_list[seg_result[m, n, 0]]

        seg_batch.append(seg_result)
        seg_color_batch.append(seg_result_color)

        count = count + 1
        # print('has run face_seg: '+ str(count)+' / '+str(CROP_IMG_batch.shape[0]))

    return [np.array(seg_batch), np.array(seg_color_batch)]
