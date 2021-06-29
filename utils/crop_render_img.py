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


class crop_render_img(object):
    def __init__(self):
        pass

    @staticmethod
    def _tf_get_bbox_by_landmark(landmark3d_batch, a=0.5, b=0.4, c=0.8, d=0.6):
        """compute bbox for cropping faces by landmark.
        :param landmark3d_batch: [batch_size, vertex_num, 3], tf.float32, 86pt 3D landmarks
        :param a: scalar, float32, used in compute bbox center in x dimension
        :param b: scalar, float32, used in compute bbox center in y dimension
        :param c: scalar, float32, used in compute bbox width in x dimension
        :param d: scalar, float32, used in compute bbox height in y dimension

        :return bbox_x1: scalar, tf.float32, x-coordinate of upper left corner in bbox
        :return bbox_y1: scalar, tf.float32, y-coordinate of upper left corner in bbox
        :return bbox_x2: scalar, tf.float32, x-coordinate of lower right corner in bbox
        :return bbox_y2: scalar, tf.float32, y-coordinate of lower right corner in bbox
        """
        min_x = tf.reduce_min(landmark3d_batch[:, :, 0])
        min_y = tf.reduce_min(landmark3d_batch[:, :, 1])
        max_x = tf.reduce_max(landmark3d_batch[:, :, 0])
        max_y = tf.reduce_max(landmark3d_batch[:, :, 1])

        c_x = min_x + a * (max_x - min_x)
        c_y = min_y + b * (max_y - min_y)
        half_x = (max_x - min_x) * c
        half_y = (max_y - min_y) * d
        half_width = tf.maximum(half_x, half_y)

        bbox_x1 = tf.cast(c_x - half_width + 0.5, dtype=tf.int32)
        bbox_x2 = tf.cast(c_x + half_width + 0.5, dtype=tf.int32)
        bbox_y1 = tf.cast(c_y - half_width + 0.5, dtype=tf.int32)
        bbox_y2 = tf.cast(c_y + half_width + 0.5, dtype=tf.int32)

        return bbox_x1, bbox_y1, bbox_x2, bbox_y2

    @staticmethod
    def _tf_crop_resize_face_by_bbox(
        bbox_x1, bbox_y1, bbox_x2, bbox_y2, image, pad_constant=100
    ):
        """Use bboxes to crop and resize faces to 300x300.
        :param bbox_x1: scalar, tf.float32, x-coordinate of upper left corner in bbox
        :param bbox_y1: scalar, tf.float32, y-coordinate of upper left corner in bbox
        :param bbox_x2: scalar, tf.float32, x-coordinate of lower right corner in bbox
        :param bbox_y2: scalar, tf.float32, y-coordinate of lower right corner in bbox
        :param image: [batch_size, imageH, imageW, 3], tf.float32, image to be cropped
        :param pad_constant: scalar, int32, pad size before cropping

        :return image_crop: [batch_size, 300, 300, 3], tf.float32, cropped and resized image
        """
        # apply padding in case the bbox coordinates are out of sight in image
        imageH, imageW = image.get_shape().as_list()[1:3]
        bbox_x1 = bbox_x1 + pad_constant
        bbox_x2 = bbox_x2 + pad_constant
        bbox_y1 = bbox_y1 + pad_constant
        bbox_y2 = bbox_y2 + pad_constant
        image_pad = tf.pad(
            image,
            [
                [0, 0],
                [pad_constant, pad_constant],
                [pad_constant, pad_constant],
                [0, 0],
            ],
            "CONSTANT",
        )
        image_pad = image_pad[:, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :]
        image_crop = tf.image.resize_images(image_pad, (300, 300))
        return image_crop

    @staticmethod
    def tf_crop_by_landmark(image, landmark3d_batch):
        """crop faces by landmarks.
        :param image: [batch_size, imageH, imageW, 3], tf.float32, image to be cropped
        :param landmark3d_batch: [batch_size, vertex_num, 3], tf.float32, 86pt 3D landmarks

        :return image_crop: [batch_size, 300, 300, 3], tf.float32, cropped and resized image
        """
        print(landmark3d_batch.shape)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = crop_render_img._tf_get_bbox_by_landmark(
            landmark3d_batch
        )
        image_crop = crop_render_img._tf_crop_resize_face_by_bbox(
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, image
        )
        return image_crop
