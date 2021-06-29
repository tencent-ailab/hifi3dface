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

"""
Different loss functions:

    - 3d landmark loss
    - photo loss
    - id loss
    - 2d landmark loss
    - reg loss
"""
import sys

sys.path.append("..")

import tensorflow as tf
import numpy as np
from utils.const import (
    SEG_BG,
    SEG_SKIN,
    SEG_NOSE,
    SEG_EYEG,
    SEG_LEYE,
    SEG_REYE,
    SEG_LBROW,
    SEG_RBROW,
    SEG_MOUTH,
    SEG_ULIP,
    SEG_LLIP,
)

EPS = 1e-6
MAX = 1e8


class Losses(object):
    def __init__(self):
        pass

    @staticmethod
    def calc_dist(a, b, opt, mask=None, max_val=MAX):
        if mask is None:
            mask = tf.ones_like(a)
            binary_mask = tf.ones_like(a)
        else:
            mask = tf.where(tf.greater(mask, EPS), mask, tf.zeros_like(mask))
            binary_mask = tf.where(
                tf.greater(mask, EPS), tf.ones_like(mask), tf.zeros_like(mask)
            )

        axes = list(range(len(a.get_shape().as_list())))
        print(axes)
        assert len(axes) > 1

        if opt == "l1":
            error = tf.clip_by_value(tf.abs(a - b), 0, max_val) * mask
            error_sum = tf.reduce_sum(error, axis=axes[1:])
            count = tf.reduce_sum(binary_mask, axis=axes[1:])
            dist = tf.reduce_mean(tf.div(error_sum, count + EPS), name="l1")
        elif opt == "l2":
            error = tf.clip_by_value(tf.square(a - b), 0, max_val) * mask
            error_sum = tf.reduce_sum(error, axis=axes[1:])
            count = tf.reduce_sum(binary_mask, axis=axes[1:])
            dist = tf.reduce_mean(tf.div(error_sum, count + EPS), name="l2")
        elif opt == "l21":
            error = tf.clip_by_value(tf.square(a - b), 0, max_val) * mask
            error_sum = tf.reduce_sum(error, axis=-1, keepdims=True)
            error_sum = tf.reduce_sum(tf.sqrt(error_sum + EPS), axis=axes[1:])
            count = tf.reduce_sum(binary_mask, axis=axes[1:])
            dist = tf.reduce_mean(tf.div(error_sum, count + EPS), name="l21")
        else:
            raise Exception("Invalid loss option")

        return dist

    @staticmethod
    def mult_depth_loss(gt_depth, render_depth, depth_mask):
        front_gt_depth = gt_depth[0]
        front_render_depth = render_depth[0]
        front_depth_mask = depth_mask[0]

        front_err = tf.square(front_gt_depth - front_render_depth)
        front_err = tf.clip_by_value(front_err, 0, 16)
        front_sum_err = tf.reduce_sum(front_err * front_depth_mask)
        front_mean_err = tf.div(front_sum_err, tf.reduce_sum(front_depth_mask))

        left_gt_depth = gt_depth[1]
        left_render_depth = render_depth[1]
        left_depth_mask = depth_mask[1]

        left_err = tf.square(left_gt_depth - left_render_depth)
        left_err = tf.clip_by_value(left_err, 0, 16)
        left_sum_err = tf.reduce_sum(left_err * left_depth_mask)
        left_mean_err = tf.div(left_sum_err, tf.reduce_sum(left_depth_mask))

        right_gt_depth = gt_depth[2]
        right_render_depth = render_depth[2]
        right_depth_mask = depth_mask[2]

        right_err = tf.square(right_gt_depth - right_render_depth)
        right_err = tf.clip_by_value(right_err, 0, 16)
        right_sum_err = tf.reduce_sum(right_err * right_depth_mask)
        right_mean_err = tf.div(right_sum_err, tf.reduce_sum(right_depth_mask))

        loss = front_mean_err + left_mean_err + right_mean_err

        # bottom loss
        bottom_gt_depth = gt_depth[3]
        bottom_render_depth = render_depth[3]
        bottom_depth_mask = depth_mask[3]

        bottom_err = tf.square(bottom_gt_depth - bottom_render_depth)
        bottom_err = tf.clip_by_value(bottom_err, 0, 4)
        bottom_sum_err = tf.reduce_sum(bottom_err * bottom_depth_mask)
        bottom_mean_err = tf.div(bottom_sum_err, tf.reduce_sum(bottom_depth_mask))

        loss = loss + bottom_mean_err

        return loss

    @staticmethod
    def weighted_landmark3d_loss(gt_landmark, pred_landmark):
        gt_landmark = tf.where(
            tf.logical_or(tf.is_nan(gt_landmark), tf.is_nan(pred_landmark)),
            tf.zeros_like(gt_landmark),
            gt_landmark,
        )
        pred_landmark = tf.where(
            tf.logical_or(tf.is_nan(gt_landmark), tf.is_nan(pred_landmark)),
            tf.zeros_like(gt_landmark),
            pred_landmark,
        )
        lmk_weight = np.array([1.0] * 86, dtype=np.float32)

        # contour
        lmk_weight[1] = 2
        lmk_weight[4] = 2
        lmk_weight[8] = 2
        lmk_weight[12] = 2
        lmk_weight[15] = 2

        # eye
        lmk_weight[35] = 5
        lmk_weight[38] = 5
        lmk_weight[41] = 15
        lmk_weight[42] = 15

        lmk_weight[43] = 5
        lmk_weight[46] = 5
        lmk_weight[49] = 15
        lmk_weight[50] = 15

        # nose
        lmk_weight[59] = 5
        lmk_weight[60] = 5
        lmk_weight[63] = 5

        # mouth
        lmk_weight[66] = 5
        lmk_weight[70] = 5

        lmk_weight[68] = 10
        lmk_weight[71] = 10
        lmk_weight[79] = 10
        lmk_weight[82] = 10

        lmk_weight = np.reshape(lmk_weight, [1, 86, 1])
        lmk_loss = tf.reduce_mean(
            tf.square(gt_landmark - pred_landmark) * lmk_weight,
            name="loss/weighted_landmark_loss",
        )
        return lmk_loss

    @staticmethod
    def photo_loss(gt_image, render_image, image_mask):
        with tf.name_scope("loss/photo"):
            loss = Losses.calc_dist(gt_image, render_image, "l1", image_mask, 240)
        return loss

    @staticmethod
    def ws_photo_loss(gt_image, render_image, image_mask):
        gray_gt_image = tf.reduce_mean(gt_image, axis=3, keepdims=True)
        ws_mask = tf.div(1.0, 1 + tf.exp((gray_gt_image * 255.0 - 170) / 20.0))
        loss = Losses.calc_dist(
            gt_image, render_image, "l21", image_mask * ws_mask, 100
        )
        return loss

    @staticmethod
    def reg_loss(para):
        with tf.name_scope("loss/reg"):
            loss = Losses.calc_dist(tf.zeros_like(para), para, "l2")
        return loss

    @staticmethod
    def landmark2d_loss_v2(gt_landmark, ver_xy, ver_mask, ver_normal, keypoint_indices):
        N = gt_landmark.get_shape().as_list()[1]
        pred_landmark = tf.gather(ver_xy, keypoint_indices, axis=1)
        norm_landmark = tf.gather(ver_normal, keypoint_indices, axis=1)
        normal_z = tf.reshape(tf.split(norm_landmark, 3, axis=2)[-1], [-1, N])

        standard_landmark_losses = tf.reduce_mean(
            tf.square(gt_landmark - pred_landmark), axis=2
        )

        invisible_losses = []
        lmk_weight = np.array([1.0] * 18, dtype=np.float32)
        lmk_weight[1] = 5
        lmk_weight[4] = 5
        lmk_weight[8] = 5
        lmk_weight[12] = 5
        lmk_weight[15] = 5

        for i in range(N):  # each ground truth, find the nearest vertex value
            gt_lmk = gt_landmark[:, i : (i + 1), :]

            loss = tf.reduce_sum(
                tf.square(gt_lmk - ver_xy) * ver_mask + MAX * (1 - ver_mask), axis=-1
            )
            loss = tf.reduce_min(loss, axis=1) * lmk_weight[i]
            loss = tf.where(
                tf.greater(loss, float(7.0 / 224 * 300)),
                float(7.0 / 224 * 300) * tf.ones_like(loss),
                loss,
            )
            invisible_losses.append(loss)

        invisible_losses = tf.reshape(tf.stack(invisible_losses, axis=0), [-1, 18])

        losses = tf.where(
            tf.greater(normal_z, 0.0), standard_landmark_losses, invisible_losses
        )
        loss = tf.reduce_mean(losses, name="landmark2d_loss")
        return loss

    @staticmethod
    def vggid_loss(images, N_group, vggpath, layer_name="fc7", data_format="NHWC"):
        from third_party.vggface import VGGFace

        # vgg id loss
        # center crop to (224,224)
        _, H, W, _ = images.get_shape().as_list()
        start = int((float(H) - 224) * 0.5)
        crop_images = images[:, start : (start + 224), start : (start + 224), :]

        vggface = VGGFace(vggpath, trainable=False)
        layers, _, _ = vggface.encoder(
            crop_images, net_name="VGGFaceID", data_format=data_format
        )
        if data_format == "NHWC":
            features = tf.split(
                tf.reduce_mean(layers[layer_name], axis=(1, 2)), N_group, axis=0
            )
        else:
            features = tf.split(
                tf.reduce_mean(layers[layer_name], axis=(2, 3)), N_group, axis=0
            )
        gt_feature = features[0]
        loss = float(0.0)
        weights = [1.0 / (N_group - 1)] * (N_group - 1)
        for i in range(1, N_group):
            pred_feature = features[i]
            cur_loss = Losses.calc_dist(gt_feature, pred_feature, "l2")
            loss += cur_loss * weights[i - 1]
        return loss

    @staticmethod
    def landmark_structure_loss(gt_landmark, pred_landmark):
        # nose structure
        nose_tri_list = [
            [55, 54, 59],
            [55, 59, 61],
            [55, 61, 66],
            [55, 66, 65],
            [55, 65, 64],
            [55, 64, 63],
            [55, 63, 62],
            [55, 62, 60],
            [55, 60, 58],
            [55, 58, 54],
        ]
        nose_tri_list = np.array(nose_tri_list).astype(np.int32) - 1

        # eye structure
        # left
        left_eye_tri_list = [
            [36, 37, 41],
            [37, 42, 41],
            [42, 43, 41],
            [42, 38, 43],
            [38, 40, 43],
            [38, 39, 40],
        ]
        left_eye_tri_list = np.array(left_eye_tri_list).astype(np.int32) - 1
        # right
        right_eye_tri_list = [
            [44, 45, 49],
            [45, 50, 59],
            [50, 51, 49],
            [50, 46, 51],
            [46, 48, 51],
            [46, 47, 48],
        ]
        right_eye_tri_list = np.array(right_eye_tri_list).astype(np.int32) - 1

        # mouth structure
        upper_lip_tri_list = [
            [67, 73, 79],
            [73, 68, 79],
            [68, 69, 79],
            [69, 80, 79],
            [69, 81, 80],
            [69, 70, 81],
            [70, 74, 81],
            [74, 71, 81],
        ]
        upper_lip_tri_list = np.array(upper_lip_tri_list).astype(np.int32) - 1

        lower_lip_tri_list = [
            [77, 82, 76],
            [82, 85, 76],
            [82, 83, 85],
            [83, 72, 85],
            [83, 84, 72],
            [84, 86, 72],
            [84, 75, 86],
            [84, 78, 75],
        ]
        lower_lip_tri_list = np.array(lower_lip_tri_list).astype(np.int32) - 1

        # face structure
        face_tri_list = [
            [36, 39, 55],
            [39, 44, 55],
            [44, 47, 55],
            [36, 55, 67],
            [55, 47, 71],
            [67, 64, 69],
            [64, 71, 69],
        ]
        face_tri_list = np.array(face_tri_list).astype(np.int32) - 1

        # from structure to pairs
        pair_set = set()
        for a, b, c in nose_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))
        for a, b, c in left_eye_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))
        for a, b, c in right_eye_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))
        for a, b, c in upper_lip_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))
        for a, b, c in lower_lip_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))
        for a, b, c in face_tri_list:
            pair_set.add((a, b))
            pair_set.add((b, a))
            pair_set.add((a, c))
            pair_set.add((c, a))
            pair_set.add((b, c))
            pair_set.add((c, b))

        pair_list = np.array(list(pair_set))

        gt_landmark = tf.where(
            tf.is_nan(gt_landmark), tf.zeros_like(gt_landmark), gt_landmark
        )
        pred_landmark = tf.where(
            tf.is_nan(gt_landmark), tf.zeros_like(gt_landmark), pred_landmark
        )

        gt_landmark_a = tf.gather(gt_landmark, pair_list[:, 0], axis=1)
        gt_landmark_b = tf.gather(gt_landmark, pair_list[:, 1], axis=1)
        gt_landmark_dist = tf.sqrt(
            tf.reduce_mean(tf.square(gt_landmark_a - gt_landmark_b), axis=-1)
        )

        pred_landmark_a = tf.gather(pred_landmark, pair_list[:, 0], axis=1)
        pred_landmark_b = tf.gather(pred_landmark, pair_list[:, 1], axis=1)
        pred_landmark_dist = tf.sqrt(
            tf.reduce_mean(tf.square(pred_landmark_a - pred_landmark_b), axis=-1)
        )

        dist = tf.reduce_mean(
            tf.square(gt_landmark_dist - pred_landmark_dist),
            name="loss/landmark_structure_distance",
        )
        return dist

    @staticmethod
    def uv_tv_loss(uv_rgb, uv_mask, uv_weight_mask=None):
        _, height, width, _ = uv_rgb.get_shape().as_list()
        if uv_weight_mask is None:
            uv_weight_mask = tf.ones_like(uv_mask)
        with tf.name_scope("loss/uv_tv_loss"):
            dy_rgb = uv_rgb[:, : (height - 1), :, :] - uv_rgb[:, 1:, :, :]
            dy_rgb = tf.pad(dy_rgb, ((0, 0), (1, 0), (0, 0), (0, 0)))

            dx_rgb = uv_rgb[:, :, : (width - 1), :] - uv_rgb[:, :, 1:, :]
            dx_rgb = tf.pad(dx_rgb, ((0, 0), (0, 0), (1, 0), (0, 0)))

            dxy_rgb = uv_rgb[:, : (height - 1), : (width - 1), :] - uv_rgb[:, 1:, 1:, :]
            dxy_rgb = tf.pad(dxy_rgb, ((0, 0), (1, 0), (1, 0), (0, 0)))

            loss = (
                Losses.calc_dist(
                    dy_rgb, tf.zeros_like(dy_rgb), "l2", uv_mask * uv_weight_mask
                )
                * 1000
                + Losses.calc_dist(
                    dx_rgb, tf.zeros_like(dx_rgb), "l2", uv_mask * uv_weight_mask
                )
                * 1000
                + Losses.calc_dist(
                    dxy_rgb, tf.zeros_like(dxy_rgb), "l2", uv_mask * uv_weight_mask
                )
                * 1000
                + Losses.calc_dist(
                    dy_rgb, tf.zeros_like(dy_rgb), "l2", uv_mask * (1 - uv_weight_mask)
                )
                + Losses.calc_dist(
                    dx_rgb, tf.zeros_like(dx_rgb), "l2", uv_mask * (1 - uv_weight_mask)
                )
                + Losses.calc_dist(
                    dxy_rgb,
                    tf.zeros_like(dxy_rgb),
                    "l2",
                    uv_mask * (1 - uv_weight_mask),
                )
            )
            return loss

    @staticmethod
    def uv_tv_loss2(uv_rgb, uv_mask, uv_weight_mask=None):
        _, height, width, _ = uv_rgb.get_shape().as_list()

        if uv_weight_mask is None:
            uv_weight_mask = np.ones_like(uv_mask)

        with tf.name_scope("loss/uv_tv_loss"):
            dy_rgb = uv_rgb[:, : (height - 1), :, :] - uv_rgb[:, 1:, :, :]
            dy_rgb = tf.pad(dy_rgb, ((0, 0), (1, 0), (0, 0), (0, 0)))

            dx_rgb = uv_rgb[:, :, : (width - 1), :] - uv_rgb[:, :, 1:, :]
            dx_rgb = tf.pad(dx_rgb, ((0, 0), (0, 0), (1, 0), (0, 0)))

            dxy_rgb = uv_rgb[:, : (height - 1), : (width - 1), :] - uv_rgb[:, 1:, 1:, :]
            dxy_rgb = tf.pad(dxy_rgb, ((0, 0), (1, 0), (1, 0), (0, 0)))

            if uv_weight_mask is not None:
                uv_boundary_mask = uv_mask * uv_weight_mask
                uv_non_boundary_mask = uv_mask * (1 - uv_weight_mask)

                axes = list(range(len(uv_rgb.get_shape().as_list())))
                ones_mask = tf.ones_like(uv_boundary_mask)
                zeros_mask = tf.zeros_like(uv_boundary_mask)
                binary_boundary_mask = tf.where(
                    tf.greater(uv_boundary_mask, EPS), ones_mask, zeros_mask
                )
                binary_non_boundary_mask = tf.where(
                    tf.greater(uv_non_boundary_mask, EPS), ones_mask, zeros_mask
                )
            else:
                binary_boundary_mask = tf.ones_like(uv_rgb)
                binary_non_boundary_mask = tf.ones_like(uv_rgb)

            binary_boundary_count = tf.reduce_sum(binary_boundary_mask, axis=axes[1:])
            binary_non_boundary_count = tf.reduce_sum(
                binary_non_boundary_mask, axis=axes[1:]
            )

            stack_delta = tf.concat([dy_rgb, dx_rgb, dxy_rgb], axis=3)
            stack_boundary_mask = tf.concat([binary_boundary_mask] * 3, axis=3)
            stack_non_boundary_mask = tf.concat([binary_non_boundary_mask] * 3, axis=3)
            sum_boundary_delta = tf.reduce_sum(
                tf.square(stack_delta * stack_boundary_mask), axis=axes[1:]
            )
            dist_boundary = (
                tf.reduce_mean(
                    tf.div(sum_boundary_delta, binary_boundary_count + EPS),
                    name="boundary_dist",
                )
                * 200
            )

            sum_non_boundary_delta = tf.reduce_sum(
                tf.square(stack_delta * stack_non_boundary_mask), axis=axes[1:]
            )
            dist_non_boundary = tf.reduce_mean(
                tf.div(sum_non_boundary_delta, binary_non_boundary_count + EPS),
                name="non_boundary_dist",
            )
            loss = dist_boundary + dist_non_boundary
            return loss
