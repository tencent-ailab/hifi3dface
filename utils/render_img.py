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
import numpy as np
import cv2
import sys
import os

sys.path.append("..")

from .basis import load_3dmm_basis, get_geometry, get_region_uv_texture
from .project import Projector, Shader


def project_and_render(
    image,
    ver_xyz,
    trans_Mat,
    K,
    imageH,
    imageW,
    tri,
    tri_vt,
    vt_list,
    tex,
    mask,
    para_illum,
    project_type,
    is_bfm,
    name,
):

    # Project
    (
        norm_image,
        ver_norm,
        alphas,
        clip_xyzw,
        proj_xy,
        proj_z,
        render_depth,
        ver_contour_mask,
        ver_contour_mask_image,
    ) = Projector.generate_proj_information(
        ver_xyz, trans_Mat, K, imageH, imageW, tri, project_type, name
    )

    # Render
    if is_bfm is False:
        render_image_batch, attrs_image, diffuse = Projector.project_uv_render(
            image,
            norm_image,
            clip_xyzw,
            tri,
            tri_vt,
            vt_list,
            imageH,
            imageW,
            tex,
            mask,
            para_illum,
            name,
        )
    else:
        render_image_batch, attrs_image, diffuse = Projector.project_vertex_render(
            image,
            norm_image,
            clip_xyzw,
            tri,
            imageH,
            imageW,
            tex,
            mask,
            para_illum,
            name,
        )

    return (
        norm_image,
        ver_norm,
        proj_xy,
        proj_z,
        render_depth,
        ver_contour_mask,
        render_image_batch,
        attrs_image,
        diffuse,
        ver_contour_mask_image,
    )


def render_img_in_different_pose(
    var_list,
    basis3dmm,
    project_type,
    imageH,
    imageW,
    opt_type="RGB",
    is_bfm=False,
    scale=1.0,
):
    #
    image_batch = var_list["image_batch"]
    batch_size = image_batch.shape[0]

    #
    trans_Mat = Projector.tf_rotationVector_2_trans(
        var_list["pose6"], project_type, scale
    )

    # fake pose parameter -> used for multi view ID loss
    pose_front_batch, pose_left_batch, pose_right_batch = Projector.gen_fix_multi_pose(
        batch_size, project_type
    )

    trans_Mat_FakeM = Projector.tf_rotationVector_2_trans(
        pose_front_batch, project_type, scale=1.0
    )
    trans_Mat_FakeL = Projector.tf_rotationVector_2_trans(
        pose_left_batch, project_type, scale=1.0
    )
    trans_Mat_FakeR = Projector.tf_rotationVector_2_trans(
        pose_right_batch, project_type, scale=1.0
    )

    # fake light parameter -> used for multi view ID loss
    (
        illum_front_batch,
        illum_left_batch,
        illum_right_batch,
    ) = Projector.gen_fix_multi_light(batch_size)

    #### get base_information like vertex, uv map, tri
    # tex: uv_rgb or ver_rgb
    # mask: uv_mask or ver_mask
    if is_bfm is False:
        tri, tri_vt, vt_list, tex, mask, ver_xyz = Projector.generate_base_information(
            basis3dmm, var_list["para_shape"], var_list["para_tex"]
        )
    else:
        tri, tex, mask, ver_xyz = Projector.generate_base_information_BFM(
            basis3dmm, var_list["para_shape"], var_list["para_tex"]
        )
        if opt_type == "RGB":
            ver_xyz = ver_xyz / 10.0
        tri_vt = None
        vt_list = None

    #### real, render in ori view
    (
        norm_image,
        ver_norm,
        proj_xy,
        proj_z,
        render_depth,
        ver_contour_mask,
        render_image_batch,
        attrs_image,
        diffuse,
        _,
    ) = project_and_render(
        image_batch,
        ver_xyz,
        trans_Mat,
        var_list["K"],
        imageH,
        imageW,
        tri,
        tri_vt,
        vt_list,
        tex,
        mask,
        var_list["para_illum"],
        project_type,
        is_bfm,
        "real",
    )

    #### fake, render in different view
    black_image_batch = tf.zeros(
        [batch_size, 300, 300, 3], dtype=tf.float32, name="black_image"
    )

    if opt_type == "RGB":
        fake_view_K = var_list["K"]
    else:
        if is_bfm:
            ver_xyz = ver_xyz / 10.0
        fake_view_K = tf.constant(
            np.array([[[-500.0, 0, 150.0], [0, -500.0, 150.0], [0, 0, 1]]]),
            dtype=tf.float32,
        )
        fake_view_K = tf.tile(fake_view_K, [batch_size, 1, 1])

    _, _, proj_xy_fake_M, _, _, _, render_image_fake_M, _, _, _ = project_and_render(
        black_image_batch,
        ver_xyz,
        trans_Mat_FakeM,
        fake_view_K,
        imageH,
        imageW,
        tri,
        tri_vt,
        vt_list,
        tex,
        mask,
        illum_front_batch,
        project_type,
        is_bfm,
        "fake1",
    )

    _, _, proj_xy_fake_L, _, _, _, render_image_fake_L, _, _, _ = project_and_render(
        black_image_batch,
        ver_xyz,
        trans_Mat_FakeL,
        fake_view_K,
        imageH,
        imageW,
        tri,
        tri_vt,
        vt_list,
        tex,
        mask,
        illum_left_batch,
        project_type,
        is_bfm,
        "fake2",
    )

    _, _, proj_xy_fake_R, _, _, _, render_image_fake_R, _, _, _ = project_and_render(
        black_image_batch,
        ver_xyz,
        trans_Mat_FakeR,
        fake_view_K,
        imageH,
        imageW,
        tri,
        tri_vt,
        vt_list,
        tex,
        mask,
        illum_right_batch,
        project_type,
        is_bfm,
        "fake3",
    )

    render_img_in_ori_pose = {
        "norm_image": norm_image,
        "ver_norm": ver_norm,
        "proj_xy": proj_xy,
        "proj_z": proj_z,
        "ver_contour_mask": ver_contour_mask,
        "render_image": render_image_batch,
        "attrs_image": attrs_image,
        "diffuse": diffuse,
        "ver_xyz": ver_xyz,
        "tex": tex,
        "render_depth": render_depth,
    }
    render_img_in_fake_pose_M = {
        "proj_xy": proj_xy_fake_M,
        "render_image": render_image_fake_M,
    }
    render_img_in_fake_pose_L = {
        "proj_xy": proj_xy_fake_L,
        "render_image": render_image_fake_L,
    }
    render_img_in_fake_pose_R = {
        "proj_xy": proj_xy_fake_R,
        "render_image": render_image_fake_R,
    }
    return (
        render_img_in_ori_pose,
        render_img_in_fake_pose_M,
        render_img_in_fake_pose_L,
        render_img_in_fake_pose_R,
    )
