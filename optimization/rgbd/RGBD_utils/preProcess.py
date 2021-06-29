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

import numpy as np
import copy
import cv2
import struct
import os


class preProcess(object):
    def __init__(self):
        pass

    @staticmethod
    def load_one_PNG_depth(one_name):
        # depthImage = cv2.imread(one_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
        depthImage = cv2.imread(one_name, cv2.IMREAD_ANYCOLOR)
        # db = depthImage[:,:,0]
        dg = depthImage[:, :, 1]
        dr = depthImage[:, :, 2]
        depth = dr.astype(float) * 255 + dg.astype(float)
        depth = depth / 10.0

        # clip depth  : 100 - 1000 mm
        depth[(depth > 1000) + (depth < 100)] = 0

        return depth

    @staticmethod
    def load_landmark_rgbd(base_path, num_lms):
        lmkpath = os.path.join(base_path, "lmk_3D_86pts_ori.txt")
        print("load lm:", lmkpath)
        f = open(lmkpath)
        arr_lmk = f.readlines()
        landmarks = []
        for idx in range(len(arr_lmk)):
            one = arr_lmk[idx]
            splits = one.strip().split(" ")
            lm = [float(n) for n in splits[1 : 1 + num_lms * 2]]
            lm = np.array(lm)
            if lm.shape[0] < num_lms:
                continue
            lm = np.reshape(lm, (num_lms, 2))
            landmarks.append(lm)
        return landmarks

    @staticmethod
    def fix_hole_by_valid_mean(depthimg1, mask_valid, fliter_size):
        def use_for_loop(depthimg1, mask_valid, fliter_size):
            fix_hols_depth = copy.copy(depthimg1)
            [w, h] = depthimg1.shape
            for ii in range(w):
                for jj in range(h):
                    if (
                        ii <= fliter_size
                        or ii >= w - fliter_size
                        or jj <= fliter_size
                        or jj >= h - fliter_size
                    ):
                        fix_hols_depth[ii, jj] = 0
                        continue
                    if not mask_valid[ii, jj]:  # empty point
                        sum = 0
                        count = 0
                        for ki in range((-1 * fliter_size), fliter_size):
                            for kj in range((-1 * fliter_size), fliter_size):
                                if mask_valid[ii + ki, jj + kj]:
                                    sum = sum + depthimg1[ii + ki, jj + kj]
                                    count = count + 1
                        if count == 0:
                            fix_hols_depth[ii, jj] = 0
                        else:
                            fix_hols_depth[ii, jj] = float(sum) / float(count)
            return fix_hols_depth

        def use_filters(depthimg1, mask_valid, fliter_size):
            [w, h] = depthimg1.shape
            binary_depth = np.zeros((w, h), np.float)
            binary_depth[mask_valid] = 1
            H = np.ones((fliter_size, fliter_size), np.float) / (
                fliter_size * fliter_size
            )
            num_depth = cv2.filter2D(binary_depth, -1, H) * fliter_size * fliter_size
            sum_depth = cv2.filter2D(depthimg1, -1, H) * fliter_size * fliter_size
            mean_dpeth = sum_depth / (num_depth + 1e-4)
            mean_dpeth[np.isnan(mean_dpeth)] = 0
            return mean_dpeth

        # for-loop method is too slow ,so we change to a faster fliter implement
        return use_filters(depthimg1, mask_valid, fliter_size)

    @staticmethod
    def depth_bilateral_filter(depths):
        depths_filtered_all = []
        for ind in range(len(depths)):
            depth = copy.copy(depths[ind])  # 100-1000 mm, 640 * 480
            mask_valid = (depth < 1000) * (depth > 100)
            cc = depth[mask_valid]
            maxd = cc.max() + 20
            mind = cc.min() - 20
            depth[depth < mind] = mind
            depth[depth > maxd] = maxd
            depthimg1 = (depth - mind) / (maxd - mind)  # normalize to 0-1 float

            #  fill holes
            fliter_size = 6
            base_img = preProcess.fix_hole_by_valid_mean(
                depthimg1, mask_valid, fliter_size
            )
            base_img = base_img.astype(np.float32)

            # use bilateralFilter
            depthimg2 = cv2.bilateralFilter(
                src=base_img, d=9, sigmaColor=100, sigmaSpace=100
            )
            depthimg2 = depthimg2 * (maxd - mind) + mind

            # remove outliner region
            mask_change = abs(depthimg2 - depths[ind]) < 5
            mask_used = mask_change + mask_valid
            depthimg3 = np.zeros(depthimg2.shape)
            depthimg3[mask_used] = depthimg2[mask_used]

            depths_filtered_all.append(depthimg3)

        return np.stack(depths_filtered_all)
