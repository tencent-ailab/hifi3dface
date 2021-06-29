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

import numpy as np
import math
from numpy import linalg as la
import cv2


class PoseTools(object):
    def __init__(self):
        pass

    @staticmethod
    def so3_2_eulur(R):
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6

        assert isRotationMatrix(R)

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # return np.array([x, y, z])
        return np.array([x, y, z])

    @staticmethod
    def euler_2_so3(theta):
        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(theta[0]), -math.sin(theta[0])],
                [0, math.sin(theta[0]), math.cos(theta[0])],
            ]
        )

        R_y = np.array(
            [
                [math.cos(theta[1]), 0, math.sin(theta[1])],
                [0, 1, 0],
                [-math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )

        R_z = np.array(
            [
                [math.cos(theta[2]), -math.sin(theta[2]), 0],
                [math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    @staticmethod
    def trans_2_eulerAngles(in_trans):
        # in: 3 * 4     # out : 6 * 1
        one_trans = np.copy(in_trans)
        rr = one_trans[0:3, 0:3]
        tt = np.squeeze(one_trans[0:3, 3:])
        so3 = PoseTools.so3_2_eulur(rr)
        se3 = np.array([so3[0], so3[1], so3[2], tt[0], tt[1], tt[2]]).reshape(-1, 1)
        return se3

    @staticmethod
    def eulerAngles_2_trans(se3):
        # in : 6*1 //    #out: 3 * 4
        rr1 = PoseTools.euler_2_so3(se3[0:3])
        tt1 = se3[3:6, :]
        trans1 = np.concatenate((rr1, tt1), axis=1)
        return trans1

    @staticmethod
    def trans_2_rotationVector(in_trans):
        # in: 3 * 4     # out : 6 * 1
        one_trans = np.copy(in_trans)
        rr = one_trans[0:3, 0:3]
        tt = np.squeeze(one_trans[0:3, 3:])
        rv = np.squeeze((cv2.Rodrigues(rr))[0])
        se3 = np.array([rv[0], rv[1], rv[2], tt[0], tt[1], tt[2]]).reshape(
            -1, 1
        )  # 6 * 1
        return se3

    @staticmethod
    def rotationVector_2_trans(in_se3):
        # in : 6*1 //    #out: 3 * 4
        se3 = np.squeeze(np.copy(in_se3)).reshape(-1, 1)  # 6 * 1
        rr = (cv2.Rodrigues(se3[0:3]))[0]  # 3 * 3
        tt = se3[3:6, :].reshape(-1, 1)  # 3 * 1
        trans1 = np.concatenate((rr, tt), axis=1)
        return trans1

        # rv = se3[0:3].reshape(-1, 1) # 3 * 1
        # theta = math.sqrt (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2] )
        # n =  rv / theta # 3 * 1
        # kx = n[0]
        # ky = n[1]
        # kz = n[2]
        # n_hat = np.array([(0,   -1 * kz ,   ky ),
        #                   (kz,     0,       -1 * kx ),
        #                   (-1 * ky,  kx,    0)])
        # I = np.eye(3,3)
        # R = math.cos(theta) * I + (1 - math.cos(theta))* (n*n.T) + math.sin(theta) *n_hat

    @staticmethod
    def trans_inverse(TWC):
        # in/out : 3 * 4
        rr = TWC[0:3, 0:3]
        tt = TWC[0:3, 3:]

        rr_1 = np.transpose(rr, [1, 0])
        tt_1 = -1 * rr_1.dot(tt)
        trans1 = np.concatenate((rr_1, tt_1), axis=1)
        return trans1

    @staticmethod
    def apply_trans(in_one, TCW):
        # in/out 3*n
        rr = TCW[0:3, 0:3]
        tt = TCW[0:3, 3:]
        one = rr.dot(in_one) + np.repeat(tt, in_one.shape[1], axis=1)
        return one

    @staticmethod
    ######################## input,pt3d:3*n,trans:3*4 ####################
    def backtrans(pt3d, trans):
        # in 3 * n / 3* 4
        sR = trans[:, 0:3]
        T = trans[:, 3]
        result_vertex = np.linalg.inv(sR).dot(
            (pt3d - np.tile(T, (len(pt3d[0]), 1)).transpose())
        )
        return result_vertex

    @staticmethod
    def project_2d(in_proj_xyz, K):
        # in/out : 3 * n
        proj_xyz = np.copy(in_proj_xyz)
        proj_xyz = K.dot(proj_xyz)
        # proj_xyz[2, np.where(proj_xyz[2, :]<1)  ]  =sys.float_info.epsilon
        proj_xyz[0, :] = proj_xyz[0, :] / proj_xyz[2, :]
        proj_xyz[1, :] = proj_xyz[1, :] / proj_xyz[2, :]
        return proj_xyz

    @staticmethod
    def pnp_orth(pt2d, PT_3d):
        """
        input :
              pt2d,n*2
              PT_3d,n*3
        output:
              phi,
              gamma,
              theta,
              t3d,
              f

        """
        PT_3d = np.hstack((PT_3d, np.ones((len(PT_3d), 1))))
        inv_p3d = np.linalg.pinv(PT_3d)

        rt_f = np.dot(pt2d.transpose(), inv_p3d.transpose())

        r1t_f = rt_f[0]
        r2t_f = rt_f[1]
        # f1 = math.sqrt(r1t_f[0] * r1t_f[0] + r1t_f[1] * r1t_f[1] + r1t_f[2] * r1t_f[2])
        # f2 = math.sqrt(r2t_f[0] * r2t_f[0] + r2t_f[1] * r2t_f[1] + r2t_f[2] * r2t_f[2])
        f1 = np.linalg.norm(r1t_f[0:3])
        f2 = np.linalg.norm(r2t_f[0:3])
        f = math.sqrt(f1 * f2)
        r1 = r1t_f[0:3] / f1
        r2 = r2t_f[0:3] / f2
        r3 = np.cross(r1, r2)
        R = np.vstack((r1, np.vstack((r2, r3))))

        t1 = r1t_f[3] / f
        t2 = r2t_f[3] / f
        t3 = 0

        # eulur_angles = PoseTools.so3_2_eulur(R)
        theta = math.atan2(r1[0], r1[1])
        gamma = math.asin(-r1[2])
        phi = math.atan2(r2[2], r3[2])

        t3d = f * np.vstack((t1, np.vstack((t2, t3))))
        return phi, gamma, theta, t3d, f

    @staticmethod
    def fit_icp_RT_no_scale(source, target):
        # in: n*3
        npoint = source.shape[0]
        means = np.mean(source, axis=0)
        meant = np.mean(target, axis=0)

        s1 = source - np.tile(means, (npoint, 1))
        t1 = target - np.tile(meant, (npoint, 1))

        W = t1.transpose().dot(s1)
        U, sigma, VT = la.svd(W)
        rotation = U.dot(VT)
        translation_icp = target - rotation.dot(source.transpose()).transpose()
        translation_icp = np.mean(translation_icp, axis=0)

        trans = np.zeros((3, 4))
        trans[:, 0:3] = rotation[:, 0:3]
        trans[:, 3] = translation_icp[:]
        return trans

    @staticmethod
    def fit_icp_scale_RT_next_align_nose(source, target, nose_idx):
        npoint = source.shape[0]
        means = np.mean(source, axis=0)
        meant = np.mean(target, axis=0)

        s1 = source - np.tile(means, (npoint, 1))
        t1 = target - np.tile(meant, (npoint, 1))

        W = t1.transpose().dot(s1)
        U, sigma, VT = la.svd(W)
        rotation = U.dot(VT)

        scale = sum(sum(np.abs(t1))) / sum(sum(abs(rotation.dot(s1.transpose()))))
        translation_nose = target[nose_idx, :] - scale * rotation.dot(
            source[nose_idx, :].transpose()
        )

        translation_icp = target - scale * rotation.dot(source.transpose()).transpose()
        translation_icp = np.mean(translation_icp, axis=0)

        translation = np.zeros(3)
        translation[0:2] = translation_nose[0:2]

        translation[2] = translation_icp[2]
        trans = np.zeros((3, 4))
        trans[:, 0:3] = scale * rotation[:, 0:3]
        trans[:, 3] = translation[:]

        return trans

    @staticmethod
    def fit_icp_RT_with_scale(source, target):
        npoint = source.shape[1]
        means = np.mean(source, 1)
        meant = np.mean(target, 1)
        s1 = source - np.tile(means, (npoint, 1)).transpose()
        t1 = target - np.tile(meant, (npoint, 1)).transpose()
        W = t1.dot(s1.transpose())
        U, sig, V = np.linalg.svd(W)
        rotation = U.dot(V)
        scale = np.sum(np.sum(abs(t1))) / np.sum(np.sum(abs(rotation.dot(s1))))
        translation = target - scale * rotation.dot(source)
        translation = np.mean(translation, 1)

        trans = np.hstack((scale * rotation, translation.reshape(3, 1)))
        return trans
