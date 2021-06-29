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
import sys
import os
import glob
import scipy.io
import skimage.io
from PIL import Image

sys.path.append("..")
import third_party.rasterize_triangles as rasterize
from utils.basis import load_3dmm_basis, get_geometry
from utils.const import *
from utils.misc import tf_blend_uv


def warp_img_to_uv(img_attrs, ver_proj_xy, tri_v, tri_vt, vt_list, uv_size):
    """ unwrap input images to UV space """
    c = img_attrs.get_shape().as_list()[-1]
    uv_position = warp_ver_to_uv(ver_proj_xy, tri_v, tri_vt, vt_list, uv_size)

    # build batch uv position
    batch_size = img_attrs.get_shape().as_list()[0]
    batch_ind = tf.reshape(
        tf.range(batch_size),
        [
            batch_size,
            1,
            1,
            1,
        ],
    )
    batch_ind = tf.tile(batch_ind, [1, uv_size, uv_size, 1])
    uv_position = tf.cast(uv_position, tf.int32)
    batch_uv_pos = tf.concat([batch_ind, uv_position], axis=-1)
    batch_uv_pos = tf.reshape(batch_uv_pos, [batch_size, -1, 3])

    uv_map = tf.gather_nd(img_attrs, batch_uv_pos)
    uv_map = tf.reshape(uv_map, [batch_size, uv_size, uv_size, c])
    return uv_map


def warp_ver_to_uv(
    v_attrs,  # tensor [batch_size, N, 3]
    tri_v,  # tensor
    tri_vt,  # tensor
    vt_list,  # tensor
    uv_size,
):  # int

    if len(v_attrs.shape) != 3:
        raise ValueError("v_attrs must have shape [batch_size, vertex_count, ?].")
    if len(tri_v.shape) != 2:
        raise ValueError("tri_v must have shape [triangles, 3].")
    if len(tri_vt.shape) != 2:
        raise ValueError("tri_vt must have shape [triangles, 3].")
    if len(vt_list.shape) != 2:
        raise ValueError("vt_list must have shape [vertex_texture_count, 2].")
    if tri_vt.dtype != tf.int32:
        raise ValueError("tri_vt must be of type int32.")
    if tri_v.dtype != tf.int32:
        raise ValueError("tri_v must be of type int32.")
    if v_attrs.dtype != tf.float32:
        raise ValueError("v_attrs must be of type float32.")
    if vt_list.dtype != tf.float32:
        raise ValueError("vt_list must be of type float32.")

    # add sample indices to tri_v and tri_vt
    batch_size, v_cnt, n_channels = v_attrs.shape
    n_tri = tri_v.shape[0]
    sample_indices = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, n_tri * 3]),
        [-1],
        name="sample_indices",
    )
    tri_v_list = tf.concat(
        [tf.reshape(tri_v, [-1])] * batch_size, axis=0, name="tri_v_list"
    )
    tri_vt_list = tf.concat(
        [tf.reshape(tri_vt, [-1])] * batch_size, axis=0, name="tri_vt_list"
    )

    tri_v_list = tf.stack(
        [sample_indices, tri_v_list], axis=1, name="sample_tri_v_list"
    )
    tri_vt_list = tf.stack(
        [sample_indices, tri_vt_list], axis=1, name="sample_tri_vt_list"
    )

    # gather vertex attributes
    v_attrs_list = tf.gather_nd(v_attrs, tri_v_list)
    v_attrs_count = tf.ones(dtype=tf.float32, shape=[batch_size * n_tri * 3, 1])

    assert (
        len(v_attrs_list.shape) == 2
        and v_attrs_list.shape[0].value == tri_v_list.shape[0].value
    )

    # add sample indices to vt_list
    n_vt = vt_list.shape[0].value
    vt_attrs_list = tf.scatter_nd(
        tri_vt_list,
        v_attrs_list,
        shape=[batch_size, n_vt, v_attrs.shape[2].value],
        name="vt_attrs_list",
    )
    vt_attrs_count = tf.scatter_nd(
        tri_vt_list, v_attrs_count, shape=[batch_size, n_vt, 1], name="vt_attrs_count"
    )
    vt_attrs_list = tf.div(vt_attrs_list, vt_attrs_count)

    assert len(vt_list.shape) == 2 and vt_list.shape[1].value == 2
    u, v = tf.split(vt_list, 2, axis=1)
    z = tf.random_normal(shape=[n_vt, 1], stddev=0.000001)
    vt_list = tf.concat([(u * 2 - 1), ((1 - v) * 2 - 1), z], axis=1, name="full_vt")
    vt_list = tf.stack([vt_list] * batch_size, axis=0)
    # scatter vertex texture attributes

    renders, _ = rasterize.rasterize_clip_space(
        vt_list,
        vt_attrs_list,
        tri_vt,
        uv_size,
        uv_size,
        [-1] * vt_attrs_list.shape[2].value,
    )
    renders.set_shape((batch_size, uv_size, uv_size, n_channels))
    # renders = tf.clip_by_value(renders, 0, 1)
    # renders = renders[0]
    return renders


def get_mask_from_seg(seg_batch):
    seg_list = tf.split(seg_batch, 19, axis=-1)
    seg_mask = (
        seg_list[SEG_SKIN]
        + seg_list[SEG_NOSE]
        + seg_list[SEG_LEYE]
        + seg_list[SEG_REYE]
        + seg_list[SEG_LBROW]
        + seg_list[SEG_RBROW]
        + seg_list[SEG_ULIP]
        + seg_list[SEG_LLIP]
    )
    return seg_mask


def get_visible_mask(ver_normals):
    _, _, normal_z = tf.split(ver_normals, 3, axis=2)
    ver_mask = tf.cast(tf.less(normal_z, -0.0), tf.float32)
    return ver_mask


# para_pose: should be process according to image size
def unwrap_img_into_uv(
    img_attrs,  # in a batch [batch, H, W , ?](should be tensor)
    proj_xyz,  # tensor
    ver_normals,
    basis3dmm,
    uv_size,
):
    proj_x, proj_y, proj_z = tf.split(proj_xyz, 3, axis=-1)
    proj_yx = tf.concat([proj_y, proj_x], axis=2)

    # get all useful configs
    basis_shape = basis3dmm["basis_shape"]
    # mu_shape = np.transpose(basis3dmm['mu_shape'])
    tri = basis3dmm["tri"]
    tri_vt = basis3dmm["tri_vt"]
    vt_list = basis3dmm["vt_list"]

    # unwrap vertex colors and mask into uv space
    uv_attrs = warp_img_to_uv(
        img_attrs,
        proj_yx,
        tf.constant(tri),
        tf.constant(tri_vt),
        tf.constant(vt_list),
        uv_size,
    )

    # warp_ver_to_uv
    ver_vis_masks = get_visible_mask(ver_normals)
    ver_vis_masks = ver_vis_masks * np.reshape(basis3dmm["mask_face"], [1, -1, 1])
    uv_mask = warp_ver_to_uv(
        ver_vis_masks,
        tf.constant(tri),
        tf.constant(tri_vt),
        tf.constant(vt_list),
        uv_size,
    )

    uv_mask = tf.clip_by_value(uv_mask, 0, 1)
    return uv_attrs, uv_mask
