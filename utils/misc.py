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

""" utility functions used in multiple scripts """
import cv2
import numpy as np
import skimage.io
import tensorflow as tf
import os
import random
from absl import logging
from PIL import Image
from utils.LP import LaplacianPyramid as LP
from utils.tf_LP import TF_LaplacianPyramid as tf_LP
from utils.const import *


class Utils(object):
    """ use for training only. """

    @staticmethod
    def create_photo_loss_mask_from_seg(seg, glassframe):
        # create photo loss mask from face segmentation
        # use skin, nose, eyeglass, lrow, rbrow, ulip, llip
        mask = (
            seg[:, :, :, SEG_SKIN]
            + seg[:, :, :, SEG_NOSE] * 1.0
            + seg[:, :, :, SEG_EYEG] * (1 - glassframe) * 0.5
            + seg[:, :, :, SEG_LBROW] * 1.0
            + seg[:, :, :, SEG_RBROW] * 1.0
            + seg[:, :, :, SEG_ULIP] * 1.0
            + seg[:, :, :, SEG_LLIP] * 1.0
        )
        mask = tf.expand_dims(mask, axis=-1)
        return mask


def tf_detect_glassframe(rgb_img, seg_img):
    # input range [0,1]
    dX = seg_img[:, :, 1:] - seg_img[:, :, :-1]
    dX = tf.pad(dX, ((0, 0), (0, 0), (0, 1)), "constant")
    dY = seg_img[:, 1:, :] - seg_img[:, :-1, :]
    dY = tf.pad(dY, ((0, 0), (0, 1), (0, 0)), "constant")
    G = tf.sqrt(tf.square(dX) + tf.square(dY))
    G = tf.where(tf.greater(G, 0.1), tf.ones_like(G), tf.zeros_like(G))
    G = tf.expand_dims(G, axis=3)

    k = 10
    kernel = np.ones((k, k), np.float32) / (k * k)
    kernel = tf.reshape(kernel, [k, k, 1, 1])

    # from tf_LP import TF_LaplacianPyramid as tf_LP

    # mask = tf_LP.conv_depthwise(G, kernel, strides=[1, 1, 1, 1], padding="SAME")
    # mask = tf.where(tf.greater(mask, 0.01), tf.ones_like(mask), tf.zeros_like(mask))
    # mask = tf.squeeze(mask)

    # convert rgb to hsv
    hsv_img = tf_rgb_to_hsv(rgb_img)
    v_img = hsv_img[:, :, :, 2]
    # v_mask = tf.where(tf.less(v_img, 0.6), tf.ones_like(v_img), tf.zeros_like(v_img))
    # glassframe = v_mask * mask * seg_img
    glassframe = seg_img
    return glassframe


def tf_rgb_to_hsv(rgb_image):
    rgb_norm_image = rgb_image / 255.0
    batch_size, image_height, image_width, n_channel = rgb_image.get_shape().as_list()
    r_img = tf.strided_slice(
        rgb_norm_image, [0, 0, 0, 0], [batch_size, image_height, image_width, 1]
    )
    g_img = tf.strided_slice(
        rgb_norm_image, [0, 0, 0, 1], [batch_size, image_height, image_width, 2]
    )
    b_img = tf.strided_slice(
        rgb_norm_image, [0, 0, 0, 2], [batch_size, image_height, image_width, 3]
    )
    # r_img, g_img, b_img = tf.split(rgb_norm_image, 3, axis=3)
    c_max = tf.reduce_max(rgb_norm_image, axis=3, keepdims=True)
    c_min = tf.reduce_min(rgb_norm_image, axis=3, keepdims=True)
    delta = c_max - c_min
    EPS = 1e-8

    h_branch1 = tf.zeros_like(c_max)
    h_branch2 = 60 * tf.div(g_img - b_img, c_max - c_min + EPS)
    h_branch3 = 60 * tf.div(g_img - b_img, c_max - c_min + EPS) + 360
    h_branch4 = 60 * tf.div(b_img - r_img, c_max - c_min + EPS) + 120
    h_branch5 = 60 * tf.div(r_img - g_img, c_max - c_min + EPS) + 240

    h2 = tf.where(
        tf.logical_and(
            tf.equal(c_max, r_img),
            tf.logical_or(tf.equal(g_img, b_img), tf.greater(g_img, b_img)),
        ),
        h_branch2,
        tf.zeros_like(h_branch2),
    )
    h3 = tf.where(
        tf.logical_and(tf.equal(c_max, r_img), tf.less(g_img, b_img)),
        h_branch3,
        tf.zeros_like(h_branch3),
    )
    h4 = tf.where(tf.equal(c_max, g_img), h_branch4, tf.zeros_like(h_branch4))
    h5 = tf.where(tf.equal(c_max, b_img), h_branch5, tf.zeros_like(h_branch5))
    h = h2 + h3 + h4 + h5

    s_branch2 = 1 - tf.div(c_min, c_max + EPS)
    s = tf.where(tf.equal(c_max, 0), s_branch2, tf.zeros_like(s_branch2))

    v = c_max

    hsv_img = tf.concat([h, s, v], axis=3)
    return hsv_img


def tf_rgb_to_yuv(rgb_image):
    batch_size, image_height, image_width, n_channel = rgb_image.get_shape().as_list()
    R = tf.strided_slice(
        rgb_image, [0, 0, 0, 0], [batch_size, image_height, image_width, 1]
    )
    G = tf.strided_slice(
        rgb_image, [0, 0, 0, 1], [batch_size, image_height, image_width, 2]
    )
    B = tf.strided_slice(
        rgb_image, [0, 0, 0, 2], [batch_size, image_height, image_width, 3]
    )
    # R, G, B = tf.split(rgb_image, 3, axis=3)
    Y = 0.3 * R + 0.59 * G + 0.11 * B
    U = (B - Y) * 0.493
    V = (R - Y) * 0.877
    yuv_image = tf.concat([Y, U, V], axis=3)
    return yuv_image


def tf_yuv_to_rgb(yuv_image):
    batch_size, image_height, image_width, n_channel = yuv_image.get_shape().as_list()
    Y = tf.strided_slice(
        yuv_image, [0, 0, 0, 0], [batch_size, image_height, image_width, 1]
    )
    U = tf.strided_slice(
        yuv_image, [0, 0, 0, 1], [batch_size, image_height, image_width, 2]
    )
    V = tf.strided_slice(
        yuv_image, [0, 0, 0, 2], [batch_size, image_height, image_width, 3]
    )
    # Y, U, V = tf.split(yuv_image, 3, axis=3)
    R = Y + 1.14 * V
    G = Y - 0.39 * U - 0.58 * V
    B = Y + 2.03 * U
    rgb_image = tf.concat([R, G, B], axis=3)
    return rgb_image


def tf_blend_uv(
    base_uv,
    face_uv,
    face_mask,
    match_color=False,
    times=5,
):
    # when match_color=True, use color tone
    assert len(base_uv.get_shape().as_list()) == 4
    assert len(face_uv.get_shape().as_list()) == 4

    uv_size = base_uv.get_shape().as_list()[1]
    k_blur = int(31 * uv_size / 2048)
    kernel_blur = np.ones((k_blur, 1), dtype=np.float32) / k_blur
    kernel_blur_row = np.reshape(kernel_blur, (k_blur, 1, 1, 1))
    kernel_blur_row = tf.constant(kernel_blur_row, name="kernel_blur_row")

    kernel_blur_col = np.reshape(kernel_blur, (1, k_blur, 1, 1))
    kernel_blur_col = tf.constant(kernel_blur_col, name="kernel_blur_col")

    face_mask_blur = face_mask
    face_mask_blur = tf.nn.conv2d(
        face_mask_blur, kernel_blur_row, strides=[1, 1, 1, 1], padding="SAME"
    )
    face_mask_blur = tf.nn.conv2d(
        face_mask_blur, kernel_blur_col, strides=[1, 1, 1, 1], padding="SAME"
    )
    face_mask = face_mask_blur * face_mask

    if match_color:
        # select color from patch
        base_uv_yuv = tf_rgb_to_yuv(base_uv)
        face_uv_yuv = tf_rgb_to_yuv(face_uv)

        sum_base = tf.reduce_sum(base_uv_yuv * face_mask, axis=(1, 2), keepdims=True)
        sum_face = tf.reduce_sum(face_uv_yuv * face_mask, axis=(1, 2), keepdims=True)
        sum_cnt = tf.reduce_sum(face_mask, axis=(1, 2), keepdims=True)
        mu_base = sum_base / sum_cnt
        mu_face = sum_face / sum_cnt

        sum_base_sq = tf.reduce_sum(
            tf.square(base_uv_yuv) * face_mask, axis=(1, 2), keepdims=True
        )
        sum_face_sq = tf.reduce_sum(
            tf.square(face_uv_yuv) * face_mask, axis=(1, 2), keepdims=True
        )
        mu_base_sq = sum_base_sq / sum_cnt
        mu_face_sq = sum_face_sq / sum_cnt

        std_base = tf.sqrt((mu_base_sq - tf.square(mu_base)))
        std_face = tf.sqrt((mu_face_sq - tf.square(mu_face)))

        base_uv_yuv = (base_uv_yuv - mu_base) / std_base * std_face + mu_face
        base_uv = tf_yuv_to_rgb(base_uv_yuv)

    face_uv = face_uv * face_mask + base_uv * (1 - face_mask)

    pyramids1 = tf_LP.buildLaplacianPyramids(base_uv, times)
    pyramids2 = tf_LP.buildLaplacianPyramids(face_uv, times)
    mask_list = tf_LP.downSamplePyramids(face_mask, times)

    blend_pyramids = []
    for j in range(len(pyramids1)):
        mask = tf.clip_by_value(mask_list[j], 0.0, 1.0)
        blend_pyramids.append(pyramids1[j] * (1 - mask) + pyramids2[j] * mask)

    cur_uv = tf_LP.reconstruct(blend_pyramids)
    cur_uv = tf.clip_by_value(cur_uv, 0.0, 1)

    return cur_uv


def blend_uv(
    base_uv,
    face_uv,
    face_mask,
    match_color=False,
    times=5,
):
    # when is_normal=True, do not use color tone
    base_uv = base_uv.astype(np.float32)
    face_uv = face_uv.astype(np.float32)
    face_mask = face_mask.astype(np.float32)

    # contract the face mask a little and blur it
    k_contract = 5
    kernel_contract = np.ones((k_contract, 1), dtype=np.float32) / k_contract
    face_mask_contract = face_mask
    for i in range(int(5 * face_mask.shape[0] / 2048)):
        face_mask_contract = cv2.filter2D(face_mask_contract, -1, kernel_contract)
        face_mask_contract = cv2.filter2D(
            face_mask_contract, -1, np.transpose(kernel_contract)
        )
    face_mask_contract[face_mask_contract < 1] = 0

    # so that the blending edges are smoothy
    k_blur = 41 * base_uv.shape[0] // 2048
    kernel_blur = np.ones((k_blur, 1), dtype=np.float32) / k_blur
    face_mask_blur = face_mask_contract
    for i in range(int(5 * face_mask.shape[0] / 2048)):
        face_mask_blur = cv2.filter2D(face_mask_blur, -1, kernel_blur)  # * face_mask
        face_mask_blur = cv2.filter2D(
            face_mask_blur, -1, np.transpose(kernel_blur)
        )  # * face_mask
    face_mask_blend = face_mask_contract * face_mask_blur

    if match_color:
        # select color from patch

        base_uv_yuv = skimage.color.convert_colorspace(base_uv, "rgb", "yuv")
        face_uv_yuv = skimage.color.convert_colorspace(face_uv, "rgb", "yuv")
        is_valid = face_mask[:, :, 0] > 0.5
        mu_base = np.mean(base_uv_yuv[is_valid], axis=0, keepdims=True)  # part
        mu_face = np.mean(face_uv_yuv[is_valid], axis=0, keepdims=True)  # core
        std_base = np.std(base_uv_yuv[is_valid], axis=0, keepdims=True)
        std_face = np.std(face_uv_yuv[is_valid], axis=0, keepdims=True)

        base_uv_yuv = (base_uv_yuv - mu_base) / std_base * std_face + mu_face
        base_uv_cvt = skimage.color.convert_colorspace(base_uv_yuv, "yuv", "rgb")
        base_uv = np.clip(base_uv_cvt, 0, 1)

    face_uv = face_uv * face_mask + base_uv * (1 - face_mask)

    pyramids1 = LP.buildLaplacianPyramids(base_uv, times)
    pyramids2 = LP.buildLaplacianPyramids(face_uv, times)
    mask_list = LP.downSamplePyramids(face_mask_blend, times)
    blend_pyramids = []
    for j in range(len(pyramids1)):
        mask = np.clip(mask_list[j], 0, 1)
        blend_pyramids.append(pyramids1[j] * (1 - mask) + pyramids2[j] * mask)
    cur_uv = LP.reconstruct(blend_pyramids)
    cur_uv = np.clip(cur_uv, 0, 1)
    return cur_uv
