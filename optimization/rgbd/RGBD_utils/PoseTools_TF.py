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

import tensorflow as tf


class PoseTools_TF(object):
    def __init__(self):
        pass

    @staticmethod
    def tf_eulerAngles_2_trans(tf_se3):
        """
        Calculate transformation matrix based on pose parameters and camera option.

        :param:
            para_pose: [ 6,1], pose paramters

        :return:
            T: [ 3, 4], transformation matrix
        """
        pitch, yaw, roll, tx, ty, tz = tf.split(tf_se3, 6, axis=0)
        cos_x = tf.cos(pitch)
        sin_x = tf.sin(pitch)
        cos_y = tf.cos(yaw)
        sin_y = tf.sin(yaw)
        cos_z = tf.cos(roll)
        sin_z = tf.sin(roll)
        zeros = tf.zeros_like(sin_z)
        ones = tf.ones_like(sin_z)

        # compute rotation matrices
        R_x = tf.concat(
            [ones, zeros, zeros, zeros, cos_x, -1 * sin_x, zeros, sin_x, cos_x], axis=1
        )
        R_x = tf.reshape(R_x, [3, 3])
        R_y = tf.concat(
            [cos_y, zeros, sin_y, zeros, ones, zeros, -1 * sin_y, zeros, cos_y], axis=1
        )
        R_y = tf.reshape(R_y, [3, 3])
        R_z = tf.concat(
            [cos_z, -1 * sin_z, zeros, sin_z, cos_z, zeros, zeros, zeros, ones], axis=1
        )
        R_z = tf.reshape(R_z, [3, 3])
        # combine scale and rotation matrix
        rr = tf.matmul(R_z, tf.matmul(R_y, R_x))

        tt = tf.concat([tx, ty, tz], axis=0)

        return rr, tt

    @staticmethod
    def tf_rotationVector_2_trans(pose6):
        """
        :param:
            pose6: [ 6,1], pose paramters

        :return:
            rr :[3,3] , tt:[3,1]
        """
        a, b, c, tx, ty, tz = tf.split(pose6, 6, axis=0)
        theta = tf.sqrt(a * a + b * b + c * c)

        def tf_Rodrigues(a, b, c, theta):
            kx = a / theta
            ky = b / theta
            kz = c / theta
            n = tf.concat([kx, ky, kz], axis=0)  # 3 * 1

            sin_theta = tf.sin(theta)
            cos_theta = tf.cos(theta)
            zeros = tf.zeros_like(sin_theta)
            n_hat = tf.concat(
                [zeros, -1 * kz, ky, kz, zeros, -1 * kx, -1 * ky, kx, zeros], axis=1
            )
            n_hat = tf.reshape(n_hat, [3, 3])
            I = tf.eye(3, 3)

            rr0 = (
                cos_theta * I
                + (1 - cos_theta) * (n * tf.transpose(n))
                + sin_theta * n_hat
            )
            return rr0

        rr = tf.cond(
            tf.squeeze(theta) > 1e-3,
            true_fn=lambda: tf_Rodrigues(a, b, c, theta),
            false_fn=lambda: tf.eye(3, 3),
        )
        tt = tf.concat([tx, ty, tz], axis=0)  # 3 * 1

        return rr, tt

    @staticmethod
    def tf_trans_inverse(rr, tt):
        # in/out : n* 3 * 4
        # rr = TWC[0:3, 0:3]
        # tt = TWC[0:3, 3:]
        rr_1 = tf.transpose(rr, [1, 0])
        tt_1 = -1 * tf.matmul(rr_1, tt)
        return rr_1, tt_1

    @staticmethod
    def tf_apply_trans(in_one, rr, tt):
        # in/out 3*n
        # rr = TCW[0:3, 0:3]
        # tt = TCW[0:3, 3:]
        one = tf.matmul(rr, in_one) + tf.tile(tt, [1, in_one.shape[1]])
        return one

    @staticmethod
    def tf_project_2d(in_proj_xyz, tf_K):
        # in/out : 3 * n
        one3d = tf.matmul(tf_K, in_proj_xyz)
        one3d = tf.stack([one3d[0] / one3d[2], one3d[1] / one3d[2]], 0)
        return one3d
