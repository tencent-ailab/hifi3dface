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

import cv2
from scipy.spatial import ConvexHull

import numpy as np
import math
from skimage.measure import label, regionprops
from skimage import draw


def Shrink_mask(mask, kernel_size):
    kernel = 4 * kernel_size + 1
    mask_ex = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask_ex[mask_ex > 0.9] = 1
    mask_ex[mask_ex != 1] = 0
    return mask_ex


def expand_mask(mask, kernel_size):
    kernel = 4 * kernel_size + 1
    mask_ex = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask_ex[mask_ex > 0.1] = 1
    mask_ex[mask_ex != 1] = 0
    return mask_ex


def threshold(mask):
    mask_out = mask
    mask_out[mask > 0.5] = 1
    mask_out[mask <= 0.5] = 0
    return mask_out


def crop_depth_for_fusion(img_all, depth_all, pt2d_all):
    ##    input: 4 * 2 * n
    use_depth_ori_all = []
    use_depth_crop_half_all = []
    use_depth_crop_eye_mouth_all = []

    for iter in range(4):
        img = img_all[iter]
        pt2d = pt2d_all[iter].transpose()  # n * 2
        depth = depth_all[iter]

        # 1.crop masks by landmark
        base_masks = find_base_mask_86(pt2d, img)

        # 2.crop mask by depth
        (
            valid_mask_filter,
            fusion_mask,
            crop_eye_mouth_mask,
            ori_mask,
        ) = crop_fusion_mask_86(iter, depth, base_masks, pt2d)

        # 3.get final useful depth
        depth_crop_half = np.zeros(depth.shape)
        depth_crop_half[fusion_mask > 0.1] = depth[fusion_mask > 0.1]

        depth_crop_eye_mouth = np.zeros(depth.shape)
        depth_crop_eye_mouth[crop_eye_mouth_mask > 0.1] = depth[
            crop_eye_mouth_mask > 0.1
        ]

        depth_ori = np.zeros(depth.shape)
        depth_ori[ori_mask > 0.1] = depth[ori_mask > 0.1]

        use_depth_ori_all.append(depth_ori)
        use_depth_crop_half_all.append(depth_crop_half)
        use_depth_crop_eye_mouth_all.append(depth_crop_eye_mouth)

    return use_depth_crop_half_all, use_depth_crop_eye_mouth_all, use_depth_ori_all


def get_max_area_of_depth(crop_depth):
    imLabel = label(crop_depth)
    imLabel[np.where(imLabel != 1)] = 0
    final_croped_depth = imLabel

    return final_croped_depth


def crop_fusion_mask_86(iter, depth, base_masks, pt2d):

    up_face_index = np.array([52, 53, 56, 57]) - 1
    mid_mouth_index = np.array([68, 70, 85, 86]) - 1
    left_mouth_index = np.array([8]) - 1
    right_mouth_index = np.array([10]) - 1
    up_face = np.mean(pt2d[up_face_index, :], 0)
    mid_mouth = np.mean(pt2d[mid_mouth_index, :], 0)
    left_mouth = np.mean(pt2d[left_mouth_index, :], 0)
    right_mouth = np.mean(pt2d[right_mouth_index, :], 0)

    inside_mask = expand_mask(base_masks[5], 5)
    mask1 = base_masks[0]
    mask2 = base_masks[1]
    mask3 = base_masks[2]
    mask4 = base_masks[3]
    mask5 = base_masks[4]

    # choose depth between +-10cm of the mean
    valid_mask = np.array((depth > 100) * (depth < 1000)).astype(np.float32)
    mask2s = Shrink_mask(mask2, 5)
    vv = np.where(np.array((mask2s > 0.5) * valid_mask).astype(np.float32) > 0.5)
    mean_d = np.mean(depth[vv[0], vv[1]])
    ori_mask = np.array(
        (depth > mean_d - 100) * (depth < mean_d + 100) * valid_mask
    ).astype(np.float32)
    valid_mask_intersect_mask3 = ori_mask * mask3

    valid_mask_filter = Shrink_mask(
        valid_mask_intersect_mask3, 5
    )  # remove noise at the edge
    valid_mask_filter = threshold(valid_mask_filter + mask1)  # Fill the inside face

    if iter == 0:
        # mid
        def merge_two_mask_by_row(cc, mask_left, mask_right):
            cc = int(round(cc))
            mask = np.zeros(mask_left.shape)
            mask[0:cc, :] = mask_left[0:cc, :]
            mask[cc : mask_left.shape[0], :] = mask_right[cc : mask_left.shape[0], :]
            return mask

        temp_mask = Shrink_mask(mask2, 5)
        fusion_mask = (
            merge_two_mask_by_row(up_face[1], valid_mask_filter, temp_mask)
            * valid_mask_intersect_mask3
        )
        crop_eye_mouth_mask = np.multiply(fusion_mask, valid_mask_intersect_mask3)

    if iter == 1:
        # left
        def zeros_mask_down(cc, mask):
            cc = round(cc)
            cc = int(cc)
            result = np.zeros(mask.shape)
            result[:, 0 : cc - 1] = mask[:, 0 : cc - 1]
            return result

        temp_mask = expand_mask(mask5, 5)
        temp_mask = np.multiply(valid_mask_filter, mask4) - temp_mask
        fusion_mask = np.multiply(
            zeros_mask_down(left_mouth[1], temp_mask), valid_mask_intersect_mask3
        )
        crop_eye_mouth_mask = np.multiply(
            np.multiply(np.multiply(mask4, valid_mask_filter), (1 - inside_mask)),
            valid_mask_intersect_mask3,
        )
    if iter == 2:
        # right
        def zeros_mask_up(cc, mask):
            cc = int(round(cc))
            result = np.zeros(mask.shape)
            result[:, cc - 1 : mask.shape[1]] = mask[:, cc - 1 : mask.shape[1]]
            return result

        temp_mask = expand_mask(mask5, 5)
        temp_mask = np.multiply(valid_mask_filter, mask4) - temp_mask
        fusion_mask = np.multiply(
            zeros_mask_up(right_mouth[0], temp_mask), valid_mask_intersect_mask3
        )
        crop_eye_mouth_mask = np.multiply(
            np.multiply(np.multiply(mask4, valid_mask_filter), (1 - inside_mask)),
            valid_mask_intersect_mask3,
        )
    if iter == 3:
        # up
        def zeros_mask_left(cc, mask):
            cc = int(round(cc))
            result = np.zeros(mask.shape)
            result[cc - 1 : mask.shape[0], :] = mask[cc - 1 : mask.shape[0], :]
            return result

        temp_mask = np.multiply(valid_mask_filter, mask3)
        temp_mask = temp_mask - expand_mask(mask1, 3)

        fusion_mask = np.multiply(
            zeros_mask_left(mid_mouth[1], temp_mask), valid_mask_intersect_mask3
        )
        crop_eye_mouth_mask = np.multiply(
            np.multiply(np.multiply(mask4, valid_mask_filter), (1 - inside_mask)),
            valid_mask_intersect_mask3,
        )
    fusion_mask = get_max_area_of_depth(fusion_mask)
    crop_eye_mouth_mask = get_max_area_of_depth(crop_eye_mouth_mask)

    return valid_mask_filter, fusion_mask, crop_eye_mouth_mask, ori_mask


def find_base_mask_86(pt2d, img):
    # in: pt2d: n * 2
    def find_convhull(pt2d, img):
        shape = img.shape
        base_mask = np.zeros([shape[0], shape[1]])
        hull = ConvexHull(pt2d)
        area = []
        for ind in hull.vertices:
            area.append([int(round(pt2d[ind, 0])), int(round(pt2d[ind, 1]))])
        area = np.array(area)
        base_mask = cv2.fillConvexPoly(base_mask, area, (255))

        return base_mask

    def poly2mask(vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(
            vertex_row_coords, vertex_col_coords, shape
        )
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

    shape = img.shape[0:2]
    mask1 = find_convhull(pt2d[18:, :], img)
    mask2 = find_convhull(pt2d, img)

    props = regionprops(label(mask2))
    meancentre = props[0].centroid
    pt2d_centre = np.zeros((len(pt2d), 2))
    pt2d_centre[:, 1] = np.tile(meancentre, (len(pt2d), 1))[:, 0]
    pt2d_centre[:, 0] = np.tile(meancentre, (len(pt2d), 1))[:, 1]
    dist = pt2d - pt2d_centre
    dist = np.linalg.norm(dist, axis=1)
    maxdist = max(dist) * 1.5
    t = np.linspace(0, 2 * math.pi, 50)
    mask3 = poly2mask(
        maxdist * np.cos(t) + meancentre[0], maxdist * np.sin(t) + meancentre[1], shape
    )
    mask3 = mask3.astype(np.float32)

    maxdist_11 = max(dist) * 1.1
    t = np.linspace(0, 2 * math.pi, 50)
    mask4 = poly2mask(
        maxdist_11 * np.cos(t) + meancentre[0],
        maxdist_11 * np.sin(t) + meancentre[1],
        shape,
    )
    mask4 = mask4.astype(np.float32)

    idx = [
        7,
        8,
        9,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
    ]
    mask5 = find_convhull(pt2d[idx, :], img)

    mask_eye_l = find_convhull(pt2d[35:43, :], img)
    mask_eye_r = find_convhull(pt2d[43:51, :], img)
    mask_mouth = find_convhull(pt2d[66:, :], img)

    inside_mask = mask_eye_l + mask_eye_r + mask_mouth
    inside_mask = expand_mask(inside_mask, 5)

    base_masks = []
    base_masks.append(threshold(mask1))
    base_masks.append(threshold(mask2))
    base_masks.append(threshold(mask3))
    base_masks.append(threshold(mask4))
    base_masks.append(threshold(mask5))
    base_masks.append(threshold(inside_mask))
    return base_masks
