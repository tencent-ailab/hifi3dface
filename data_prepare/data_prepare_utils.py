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

import numpy as np
import os
import cv2


def load_landmark(path, num_lms):
    print("load lm:", path)
    f = open(path)
    arr = f.readlines()
    landmarks = []
    images = []
    for one in arr:
        # print(one)
        splits = one.strip().split(" ")
        imgname = splits[0]  # .split('/')[-1]
        # imgnumber = splits[0].split('/')[-1].split('.')[0]

        lm = [float(n) for n in splits[1 : 1 + num_lms * 2]]
        lm = np.array(lm)
        if lm.shape[0] < num_lms:
            continue
        lm = np.reshape(lm, (num_lms, 2))
        landmarks.append(lm)
        images.append(imgname)
    return landmarks, images


def write_lmk(out_img_name, lmk, fopen):
    text = out_img_name + " "
    for j in range(0, lmk.shape[0]):
        text = text + "%.6f" % lmk[j][0] + " "
        text = text + "%.6f" % lmk[j][1] + " "
    text = text + "\n"
    fopen.write(text)


def write_lmk_no_name(lmk, fopen):
    text = ""
    for j in range(0, lmk.shape[0]):
        text = text + "%.6f" % lmk[j][0] + " "
        text = text + "%.6f" % lmk[j][1] + " "
    text = text + "\n"
    fopen.write(text)


def crop_image_and_process_landmark(img, landmark, landmark2D, size=300, orig=False):
    # use landmarks to crop face
    min_x, min_y = np.amin(landmark, 0)
    max_x, max_y = np.amax(landmark, 0)

    c_x = min_x + 0.5 * (max_x - min_x)
    c_y = min_y + 0.4 * (max_y - min_y)
    half_x = (max_x - min_x) * 0.8
    half_y = (max_y - min_y) * 0.6
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

    lmk_x = landmark[:, 0]
    lmk_x = (lmk_x - st_x) / crop_img.shape[1] * size
    lmk_y = landmark[:, 1]
    lmk_y = (lmk_y - st_y) / crop_img.shape[0] * size
    landmark = np.stack([lmk_x + 0.5, lmk_y + 0.5], axis=1)

    lmk_x2D = landmark2D[:, 0]
    lmk_x2D = (lmk_x2D - st_x) / crop_img.shape[1] * size
    lmk_y2D = landmark2D[:, 1]
    lmk_y2D = (lmk_y2D - st_y) / crop_img.shape[0] * size
    landmark2D = np.stack([lmk_x2D + 0.5, lmk_y2D + 0.5], axis=1)

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
        return crop_img, landmark, landmark2D, old_crop_img
    else:
        return crop_img, landmark, landmark2D, None
