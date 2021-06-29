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

import scipy.io as sio
import cv2
import struct
import numpy as np
from .PoseTools import PoseTools
import random


def extract_3d_vex(cam_coord, depthimg):
    im_h = depthimg.shape[0]
    im_w = depthimg.shape[1]
    depth = np.reshape(depthimg, (1, im_h * im_w))
    cam_pts = np.zeros((3, len(cam_coord[0])))
    cam_pts[0, :] = cam_coord[0, :]
    cam_pts[1, :] = cam_coord[1, :]

    cam_pts[0, :] = cam_pts[0, :] * depth[0, :]
    cam_pts[1, :] = cam_pts[1, :] * depth[0, :]
    cam_pts[2, :] = depth[0, :]
    return cam_pts


def extract_tex(pixel_coord, colorimg):
    im_h = colorimg.shape[0]
    im_w = colorimg.shape[1]
    pt3d_color = np.zeros((3, im_h * im_w))
    for i in range(pixel_coord.shape[1]):
        x = pixel_coord[0, i]
        y = pixel_coord[1, i]

        cc = colorimg[int(y) - 1, int(x) - 1, :]
        pt3d_color[:, i] = cc[:]
    return pt3d_color


def make_one_point(pixel_x, pixel_y, im_w):
    pp = (pixel_y - 1) * im_w + pixel_x
    return pp


def find_overlap(points_model_in_current, depthimg, K):
    pt2d_model = points_model_in_current
    pt2d_model[0, :] = (
        K["fx"] * points_model_in_current[0, :] / points_model_in_current[2, :]
        + K["cx"]
    )
    pt2d_model[1, :] = (
        K["fy"] * points_model_in_current[1, :] / points_model_in_current[2, :]
        + K["cy"]
    )
    im_h = depthimg.shape[0]
    im_w = depthimg.shape[1]
    invalid_around = np.zeros((im_h * im_w))
    mask_around = np.zeros(depthimg.shape)
    idx_in_neutral = []
    d = []
    for i in range(pt2d_model.shape[1]):
        pixel_x = int(round(pt2d_model[0, i]))
        pixel_y = int(round(pt2d_model[1, i]))
        if pixel_x > im_w or pixel_y > im_h:
            continue
        d_img = depthimg[pixel_y - 1][pixel_x - 1]
        if d_img < 100 or d_img > 1000:
            continue
        d.append(d_img)

        idx_in_neutral.append(i)
        kernel = 1
        mask_around[
            pixel_y - kernel : pixel_y + kernel + 1,
            pixel_x - kernel : pixel_x + kernel + 1,
        ] = 1
        for row in range(-1 * kernel, kernel + 1):
            for col in range(-1 * kernel, kernel + 1):
                point = make_one_point(pixel_x + row, pixel_y + col, im_w)
                invalid_around[point - 1] = i
    pair_in_depthimg = np.where(invalid_around != 0)
    pair_in_neutral = invalid_around[pair_in_depthimg]
    idx_in_mask_all_in_neutral = idx_in_neutral
    return mask_around, pair_in_depthimg, pair_in_neutral, idx_in_mask_all_in_neutral


def fusion_two(points1, points2, K, centre):
    p1 = points1[0:2, :].copy()
    p1[0, :] = K["fx"] * points1[0, :] / points1[2, :] + K["cx"]
    p1[1, :] = K["fy"] * points1[1, :] / points1[2, :] + K["cy"]
    dist1 = p1 - np.tile(centre.transpose(), (p1.shape[1], 1)).transpose()
    dist1 = dist1[0, :] * dist1[0, :] + dist1[1, :] * dist1[1, :]
    dist1 = np.sqrt(dist1)
    dist1 = np.power(dist1, 3)
    m = max(dist1) - min(dist1)
    w1 = (dist1 - min(dist1)) / m
    w1 = np.tile(w1, (3, 1))
    out_vertex = out_vertex = w1 * points1 + (1 - w1) * points2

    return out_vertex


def fusion_surfel(
    use_img_all,
    depth_ori_all,
    depth_crop_eye_mouth_all,
    depth_crop_half_all,
    pose_all,
    K,
    use_pt2d_all,
):
    im_h = depth_crop_half_all[1].shape[0]
    im_w = depth_crop_half_all[1].shape[1]
    nx, ny = (im_h, im_w)
    x = np.linspace(1, im_w, im_w)
    y = np.linspace(1, im_h, im_h)
    xv, yv = np.meshgrid(x, y)

    XX = xv.reshape((1, -1))

    xv, yv = np.meshgrid(y, x)
    YY = xv.reshape((1, -1), order="F")

    pixel_coord = np.vstack((np.vstack((XX, YY)), np.ones((1, len(XX[0])))))
    cam_coord = np.zeros((3, len(XX[0])))
    cam_coord[0, :] = (pixel_coord[0, :] - K["cx"]) / K["fx"]
    cam_coord[1, :] = (pixel_coord[1, :] - K["cy"]) / K["fy"]
    cam_coord[2, :] = pixel_coord[2, :]

    ex_pt3ds = []
    ex_colors = []

    for i in range(len(depth_crop_half_all)):
        depthimg = depth_crop_half_all[i]
        depthimg_full = depth_crop_eye_mouth_all[i]
        colorimg = use_img_all[i] / 255
        pose = pose_all[i]
        pt2d = use_pt2d_all[i]

        pt3d = extract_3d_vex(cam_coord, depthimg)
        pt3d = np.squeeze(pt3d)
        pt3d_color = extract_tex(pixel_coord, colorimg)
        pt3d_color = np.squeeze(pt3d_color)

        valid_index = np.where(pt3d[2, :] > 0)
        if i == 0:
            valid_pt3d = pt3d[:, valid_index]
            neutral_pt3ds = valid_pt3d
            neutral_colors = pt3d_color[:, valid_index]
        else:
            neutral_pt3ds = np.squeeze(neutral_pt3ds)
            pw = PoseTools.backtrans(neutral_pt3ds, pose)
            (
                mask_around,
                pair_in_depthimg,
                pair_in_neutral,
                idx_in_mask_all_in_neutral,
            ) = find_overlap(pw, depthimg, K)
            idx_current_ex = np.sort(
                list(set(valid_index[0]) - set(pair_in_depthimg[0]))
            )
            pt3d_current_ex0 = pt3d[:, idx_current_ex]
            pt3d_current_else = pose.dot(
                np.vstack((pt3d_current_ex0, np.ones((1, pt3d_current_ex0.shape[1]))))
            )
            new_vertex_ex = pt3d_current_else
            new_colors_ex = pt3d_color[:, idx_current_ex]
            pt3d_current_overlap = pt3d[:, pair_in_depthimg]
            pt3d_current_overlap = np.squeeze(pt3d_current_overlap)
            inliner = np.where((pt3d_current_overlap[2, :] != 0))
            pt3d_current_overlap = pose.dot(
                np.vstack(
                    (pt3d_current_overlap, np.ones((1, pt3d_current_overlap.shape[1])))
                )
            )
            pt_depth = pt3d_current_overlap[:, inliner[0]]
            pair_in_neutral = pair_in_neutral[inliner[0]]
            pair_in_neutral = [int(x) for x in pair_in_neutral]
            pt_neutral = neutral_pt3ds[:, pair_in_neutral]
            neutral_colors = np.squeeze(neutral_colors)
            color_overlap = neutral_colors[:, pair_in_neutral]
            vertex_overlap = fusion_two(
                pt_depth, pt_neutral, K, pt2d[:, pt2d.shape[1] - 4]
            )

            vertex_left = neutral_pt3ds[
                :,
                np.sort(
                    list(
                        set(range(neutral_pt3ds.shape[1]))
                        - set(idx_in_mask_all_in_neutral)
                    )
                ),
            ]
            color_left = neutral_colors[
                :,
                np.sort(
                    list(
                        set(range(neutral_pt3ds.shape[1]))
                        - set(idx_in_mask_all_in_neutral)
                    )
                ),
            ]
            neutral_pt3ds = np.hstack((vertex_left, vertex_overlap))
            neutral_colors = np.hstack((color_left, color_overlap))

            if i == 1:
                ex_pt3ds = new_vertex_ex
                ex_colors = new_colors_ex
            else:
                ex_pt3ds = np.hstack((ex_pt3ds, new_vertex_ex))
                ex_colors = np.hstack((ex_colors, new_colors_ex))

    mesh_netural = {"pt3ds": neutral_pt3ds, "colors": neutral_colors}
    mesh_ex = {"pt3ds": ex_pt3ds, "colors": ex_colors}
    return mesh_netural, mesh_ex


def fast_crop(mesh_in, with_error_3d_points, id_nose):
    # in: 3 *n
    vertex = mesh_in["pt3ds"]
    num_vertex = vertex.shape[1]
    vertex_nose = with_error_3d_points[:, id_nose - 1]
    dist3 = vertex - np.tile(vertex_nose, (num_vertex, 1)).transpose()
    dist = np.sqrt(
        dist3[0, :] * dist3[0, :]
        + dist3[1, :] * dist3[1, :]
        + dist3[2, :] * dist3[2, :]
    )
    inliner_vertex = np.where(dist < 150)
    mes_out_pt3ds = mesh_in["pt3ds"][:, inliner_vertex]
    mes_out_colors = mesh_in["colors"][:, inliner_vertex]
    mes_out_pt3ds = np.squeeze(mes_out_pt3ds)
    mes_out_colors = np.squeeze(mes_out_colors)
    mesh_out = {"pt3ds": mes_out_pt3ds, "colors": mes_out_colors}
    return mesh_out


def downsampling_voxel(isRandonm, mesh_in, rate):
    # in 3*n
    voxelSize = np.array([rate, rate, rate])
    vertex = mesh_in["pt3ds"]
    colors = mesh_in["colors"]
    vertex_sample = []
    color_sample = []
    idx_sample = []
    minp = np.min(vertex, axis=1)
    maxp = np.max(vertex, axis=1)
    leafSize = (maxp - minp) / voxelSize

    inv_leafSize = 1 / leafSize
    minb = np.floor(minp * inv_leafSize)
    maxb = np.floor(maxp * inv_leafSize)
    divb = maxb - minb + 1
    divb_mul = [1, divb[0], divb[0] * divb[1]]
    ijk1 = np.floor(vertex[0, :] * inv_leafSize[0]) - minb[0]
    ijk2 = np.floor(vertex[1, :] * inv_leafSize[1]) - minb[1]
    ijk3 = np.floor(vertex[2, :] * inv_leafSize[2]) - minb[2]
    idx = ijk1 * divb_mul[0] + ijk2 * divb_mul[1] + ijk3 * divb_mul[2]

    ori_idx = np.argsort(idx)
    """
    if 0:
        data = sio.loadmat(r"D:/rand.mat")
        ori_idx = data['ori_idx'][0]
        rand1 = data['rand'][0]
        test_idx = 0
    """
    voxel_idx = np.sort(idx)
    id_begin = 1
    while id_begin <= vertex.shape[1]:
        choose_vertex = vertex[:, int(ori_idx[id_begin - 1])]
        choose_color = colors[:, int(ori_idx[id_begin - 1])]
        choose_idx = ori_idx[id_begin - 1]
        id_now = id_begin + 1
        if isRandonm == 1:
            while (
                id_now < vertex.shape[1]
                and voxel_idx[id_begin - 1] == voxel_idx[id_now - 1]
            ):
                id_now = id_now + 1
            rand = random.randint(1, id_now - id_begin)
            """
            if 0:
                rand = rand1[test_idx]
                test_idx = test_idx +1
            """
            temp = id_begin + rand
            if temp > vertex.shape[1]:
                break
            choose_vertex = vertex[:, int(ori_idx[temp - 1]) - 1]
            choose_color = colors[:, int(ori_idx[temp - 1]) - 1]
            choose_idx = ori_idx[temp - 1] - 1
        else:
            while (
                id_now < vertex.shape[1]
                and voxel_idx[id_begin - 1] == voxel_idx[id_now - 1]
            ):
                choose_vertex = choose_vertex + vertex[:, int(ori_idx[id_now - 1])]
                # centroid_color = centroid_color + colors(:,indices_idx(1,i));
                id_now = id_now + 1
            choose_vertex = choose_vertex / (id_now - id_begin)
        if id_begin == 1:
            vertex_sample = choose_vertex
            color_sample = choose_color
            idx_sample = choose_idx
        else:
            vertex_sample = np.vstack((vertex_sample, choose_vertex))
            color_sample = np.vstack((color_sample, choose_color))
            idx_sample = np.vstack((idx_sample, choose_idx))

        id_begin = id_now
    mesh_out_pt3ds = vertex_sample.T
    mesh_out_colors = color_sample.T
    mesh_out = {"pt3ds": mesh_out_pt3ds, "colors": mesh_out_colors}
    return mesh_out
