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

import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image, ImageOps
import numpy as np
import cv2
import os


def bilinear_interpolation(x, y, points):
    """Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

    >>> bilinear_interpolation(12, 5.5,
    ...                        [(10, 4, 100),
    ...                         (20, 4, 200),
    ...                         (10, 6, 150),
    ...                         (20, 6, 300)])
    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    points = sorted(points)  # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        print(x1, _x1, x2, _x2)
        print(y1, _y1, y2, _y2)
        raise ValueError("points do not form a rectangle")
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        print(x1, x2, x, y1, y2, y)
        raise ValueError("(x, y) not within the rectangle")

    EPS = 1e-16

    if x2 == x1 and y1 == y1:
        return q11
    elif x2 == x1:
        return (q11 * (y - y1) + q12 * (y2 - y)) / (y2 - y1)
    elif y1 == y2:
        return (q11 * (x - x1) + q21 * (x2 - x)) / (x2 - x1)
    return (
        q11 * (x2 - x) * (y2 - y)
        + q21 * (x - x1) * (y2 - y)
        + q12 * (x2 - x) * (y - y1)
        + q22 * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1))


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    """
    a numpy version of the above torch code.
    The purpose for this snippet is to test the validity of the converted pb file.
    -- by cyj
    """
    # heatmap = heatmap.cpu().numpy()
    assert len(heatmap.shape) == 3, "The dimension of the heatmap is wrong : {}".format(
        heatmap.shape
    )
    assert radius > 0, "The radius is not ok : {}".format(radius)
    num_pts, H, W = heatmap.shape
    assert W > 1 and H > 1, "To avoid the normalization function divide zero"

    # find the approximate location:
    v_heatmap = np.reshape(heatmap, (num_pts, -1))
    index = np.argmax(v_heatmap, axis=1)
    score = []
    for i, ind in enumerate(index):
        score.append(v_heatmap[i, ind])
    score = np.array(score)
    index_w = np.mod(index, W, dtype=np.float32)
    index_h = (index / W).astype(np.float32)

    def normalize(x, L):
        return -1.0 + 2.0 * x / (L - 1)

    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)

    affine_parameter = np.zeros((num_pts, 2, 3), dtype=np.float32)
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2

    def get_feat(heatt, warp_mat, grid_size):
        H, W = heatt.shape[0], heatt.shape[1]
        new_heat = np.zeros(grid_size, dtype=np.float32)
        heat = heatt.copy()
        heat = np.row_stack((heat, np.zeros((1, heat.shape[1]))))
        heat = np.column_stack((heat, np.zeros((heat.shape[0], 1))))
        grad_init_y = np.linspace(-1, 1, grid_size[0])  # *(grid_size[0]-1)/grid_size[0]
        grad_init_x = np.linspace(-1, 1, grid_size[1])  # *(grid_size[1]-1)/grid_size[1]
        for yy in range(grid_size[0]):
            for xx in range(grid_size[1]):
                # find the location of (x, y) in original space
                y = grad_init_y[yy]
                x = grad_init_x[xx]

                loc = np.reshape(np.array([x, y, 1], dtype=np.float64), [3, 1])

                src_loc = np.matmul(warp_mat, loc)  # [-1, 1]

                if (
                    src_loc[0] < -1
                    or src_loc[0] > 1
                    or src_loc[1] < -1
                    or src_loc[1] > 1
                ):
                    continue
                src_loc[0] = min((src_loc[0] + 1) * (W - 1) / 2.0, W)
                src_loc[1] = min((src_loc[1] + 1) * (H - 1) / 2.0, H)

                p1 = [int(src_loc[1]), int(src_loc[0])]
                p2 = [int(src_loc[1]), int(src_loc[0] + 1)]
                p3 = [int(src_loc[1] + 1), int(src_loc[0])]
                p4 = [int(src_loc[1] + 1), int(src_loc[0] + 1)]

                p1.append(heat[p1[0], p1[1]])
                p2.append(heat[p2[0], p2[1]])
                p3.append(heat[p3[0], p3[1]])
                p4.append(heat[p4[0], p4[1]])

                try:
                    value = bilinear_interpolation(
                        src_loc[1], src_loc[0], (p1, p2, p3, p4)
                    )
                except Exception as e:
                    print(e)
                    value = heat[int(src_loc[1]), int(src_loc[0])]

                new_heat[yy, xx] = value
        # feat = cv2.resize(new_heat, grid_size)
        return new_heat

    # extract the sub-region heatmap
    grid_size = (radius * 2 + 1, radius * 2 + 1)
    sub_features = []
    for i in range(len(affine_parameter)):
        warp_mat = affine_parameter[i]
        feat = get_feat(heatmap[i], warp_mat, grid_size)
        feat[feat <= threshold] = np.finfo(float).eps
        sub_features.append(feat)
    sub_features = np.array(sub_features)

    X = np.arange(-radius, radius + 1).reshape(1, -1)
    Y = np.arange(-radius, radius + 1).reshape(-1, 1)

    sum_region = np.sum(sub_features.reshape(num_pts, -1), 1)
    x = np.sum((sub_features * X).reshape(num_pts, -1), 1) / sum_region + index_w
    y = np.sum((sub_features * Y).reshape(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    loc = np.stack([x, y], 1)
    # print(loc)
    # loc = torch.from_numpy(loc)
    # score = torch.from_numpy(score)
    return loc, score  # loc.cuda(), score.cuda()


def detect_2Dlmk68(LMK3D_batch, IMG_batch, sess):
    inputs = sess.graph.get_tensor_by_name("input_image:0")
    outputs = sess.graph.get_tensor_by_name("cpm_stage3:0")

    # print(IMG_batch.shape[0])
    return_lms = []
    count = 0
    for i in range(0, IMG_batch.shape[0]):
        lmk = LMK3D_batch[i]
        img = IMG_batch[i]

        img = img[:, :, ::-1]  # RGB

        x_min = max(np.min(lmk[:, 0]), 0)
        y_min = max(np.min(lmk[:, 1]), 0)
        x_max = np.max(lmk[:, 0])
        y_max = np.max(lmk[:, 1])

        bbox_x1 = int(max(x_min - (x_max - x_min) * 0.2, 0))
        bbox_y1 = int(max(y_min - (y_max - y_min) * 0.2, 0))
        bbox_x2 = int(min(x_max + (x_max - x_min) * 0.2, img.shape[1]))
        bbox_y2 = int(min(y_max + (y_max - y_min) * 0.2, img.shape[0]))

        crop_img = img[bbox_y1:bbox_y2, bbox_x1:bbox_x2, :]
        H, W, _ = crop_img.shape
        ori_crop_img = crop_img.copy()
        crop_img = cv2.resize(crop_img, (128, 128))

        test_img = np.transpose(np.asarray(crop_img), [2, 0, 1]) / 255.0
        test_img = (test_img - 0.5) / 0.5
        test_img = np.expand_dims(test_img, axis=0)

        heatmap = sess.run(outputs, {inputs: test_img})

        heatmap = heatmap[0]
        locations, scores = find_tensor_peak_batch(heatmap, 3, 8)
        locations = locations[
            :68,
        ]
        locations[:, 0] = locations[:, 0] * W / 128.0 + bbox_x1
        locations[:, 1] = locations[:, 1] * H / 128.0 + bbox_y1
        return_lms.append(locations)

        # count = count + 1
        # print('has run 68pt lmk: '+ str(count)+' / '+str(IMG_batch.shape[0]))

    return np.array(return_lms)  # N x 68 x 2
