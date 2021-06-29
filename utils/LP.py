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

import cv2
import numpy as np


class LaplacianPyramid(object):
    @staticmethod
    def downSamplePyramids(image, n_level, sigma=1):
        # test OK
        pyramids = [image]
        for i in range(1, n_level):
            temp = pyramids[i - 1].copy()
            rows, cols, _ = temp.shape
            temp = temp[range(0, rows, 2), :, :]
            temp = temp[:, range(0, cols, 2), :]
            temp = cv2.GaussianBlur(temp, (5, 5), sigma)
            pyramids.append(temp)
        return pyramids

    @staticmethod
    def upSample(image):
        rows, cols, channels = image.shape
        up_image = np.zeros((rows * 2, cols, channels))
        up_image[range(0, rows * 2, 2), :, :] = image
        up_image[range(1, rows * 2, 2), :, :] = image
        up_image2 = np.zeros((rows * 2, cols * 2, channels))
        up_image2[:, range(0, cols * 2, 2), :] = up_image
        up_image2[:, range(1, cols * 2, 2), :] = up_image
        return up_image2

    @staticmethod
    def buildLaplacianPyramids(image, n_level):
        h_filter = np.reshape(np.array([1, 4, 6, 4, 1]) / 16.0, [5, 1])
        h_filter = np.matmul(h_filter, h_filter.transpose())
        g_filter = h_filter

        pyramids = []
        cur_image = image
        for i in range(n_level - 1):
            temp = cv2.filter2D(cur_image, -1, h_filter)
            rows, cols, _ = temp.shape
            temp = temp[range(0, rows, 2), :, :]
            temp = temp[:, range(0, cols, 2), :]
            dn_temp = temp.copy()
            temp = LaplacianPyramid.upSample(temp)
            temp = temp[:rows, :cols, :]
            temp = cv2.filter2D(temp, -1, g_filter)
            pyramids.append(cur_image - temp)
            cur_image = dn_temp
        pyramids.append(cur_image)
        return pyramids

    @staticmethod
    def reconstruct(pyramids):
        h_filter = np.reshape(np.array([1, 4, 6, 4, 1]) / 16.0, [5, 1])
        h_filter = np.matmul(h_filter, h_filter.transpose())
        g_filter = h_filter

        for i in range(len(pyramids) - 1, 0, -1):
            temp = pyramids[i]
            temp = LaplacianPyramid.upSample(temp)
            rows, cols, _ = pyramids[i - 1].shape
            temp = temp[:rows, :cols, :]
            temp = cv2.filter2D(temp, -1, g_filter)
            pyramids[i - 1] = pyramids[i - 1] + temp
        return pyramids[0]
