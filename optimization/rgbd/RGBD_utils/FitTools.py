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
from .PoseTools import PoseTools


def cal_one_loss(
    iter,
    vertex_target,
    mu_key,
    w_key,
    shape,
    trans_base_2_camera,
    pt2d_front,
    K,
    w_pt2d,
    mu_pt2d,
):
    # 3d
    pt3d = mu_key + w_key.dot(shape)
    pt3d = np.reshape(pt3d, (3, -1), order="F")
    err_fit = sum(sum(abs(pt3d - vertex_target)))
    # reg
    err_reg = sum(abs(shape))
    # ed
    pt3d_landmark = mu_pt2d + w_pt2d.dot(shape)
    pt3d_landmark = np.reshape(pt3d_landmark, (-1, 3)).transpose()  # 3 * n
    render_keypoints = PoseTools.project_2d(
        PoseTools.apply_trans(pt3d_landmark, trans_base_2_camera), K
    )
    err_pt2d = sum(sum(abs(render_keypoints[0:2, :] - pt2d_front)))

    loss = err_fit + err_pt2d
    print(
        "iter ",
        iter,
        "2d landmark error: ",
        str(err_pt2d),
        "fit 3d error: ",
        str(err_fit),
        "reg error: ",
        str(err_reg),
    )
    return loss


def get_landmark_term(fit_pt2d, K, shape0, trans_base_2_camera, img_front):
    # in 3 * n

    pt2d_front = fit_pt2d["target"]  # 2 * n
    mu_pt2d = fit_pt2d["mu"]
    weight_pt2d = fit_pt2d["weight"]
    w_pt2d = fit_pt2d["w"]
    weight = np.tile(weight_pt2d, (2, 1))
    n_basis = shape0.shape[0]
    n_points = pt2d_front.shape[1]

    sR_next_to_slam = trans_base_2_camera[0:3, 0:3]
    pt3d = mu_pt2d + w_pt2d.dot(shape0)
    pt3d = np.reshape(pt3d, (3, -1), order="F")

    pt2d_liner = pt2d_front
    pt3d_liner = pt3d

    pt3d_liner = PoseTools.apply_trans(pt3d_liner, trans_base_2_camera)

    def cal_Jacbian_K(P, K):
        x = P[0]
        y = P[1]
        z = P[2]
        z2 = z * z
        JK = np.zeros((2, 3))
        JK[0, 0] = K[0, 0] / z
        JK[1, 1] = K[1, 1] / z
        JK[0, 2] = -K[0, 0] * x / z2
        JK[1, 2] = -K[1, 1] * y / z2
        return JK

    Jacbian_of_K = []
    for i in range(pt3d_liner.shape[1]):
        one = cal_Jacbian_K(pt3d_liner[:, i], K)
        Jacbian_of_K.append(one)

    J_shape = np.zeros((2, n_points, int(n_basis)))
    W_Beta = np.reshape(w_pt2d, (3, -1, n_basis), order="F")
    for i in range(n_basis):
        R_W_Beta = sR_next_to_slam.dot(W_Beta[:, :, i])
        K_R_W_Beta = np.zeros((2, n_points))
        for j in range(n_points):
            K_R_W_Beta[:, j] = Jacbian_of_K[j].dot(R_W_Beta[:, j])
        J_shape[:, :, i] = weight * K_R_W_Beta

    J_shape = np.reshape(J_shape, (-1, n_basis), order="F")
    f_shape0 = weight * (PoseTools.project_2d(pt3d_liner, K)[0:2, :] - pt2d_liner)
    shape_left = J_shape.transpose().dot(J_shape)
    shape_right = -1 * J_shape.transpose().dot(np.reshape(f_shape0, (-1, 1), order="F"))

    return shape_left, shape_right


def get_vertex_term(fit_kp_3d, shape):
    # in 2 * n
    weight_key = fit_kp_3d["weight"]
    w_key = fit_kp_3d["w"]
    mu_key = fit_kp_3d["mu"]
    vertex_target = fit_kp_3d["target"]

    n_points = vertex_target.shape[1]  # numkpt
    n_basis = shape.shape[0]
    weight = np.tile(weight_key, (3, 1))

    W_Beta = np.reshape(w_key, (3, n_points, n_basis), order="F")
    J_shape = np.zeros((3, n_points, n_basis))
    for i in range(n_basis):
        J_shape[:, :, i] = weight * W_Beta[:, :, i]
    J_shape = np.reshape(J_shape, (-1, n_basis), order="F")

    pt3d_fit = mu_key + w_key.dot(shape)
    pt3d_fit = np.reshape(pt3d_fit, (3, -1), order="F")
    f_shape0 = weight * (pt3d_fit - vertex_target)
    shape_left = J_shape.transpose().dot(J_shape)
    shape_right = -1 * J_shape.transpose().dot(np.reshape(f_shape0, (-1, 1), order="F"))

    return shape_left, shape_right


def get_regu_term(paras, fit_reg):
    lambda_reg_paras = fit_reg["weight"]
    sigma_shape = fit_reg["sigma"]
    sigma = sigma_shape.reshape(-1, 1)
    J_sigma = np.diag(np.squeeze(1 / sigma_shape))  # n * n
    f_sigma0 = (paras) / sigma  # n * 1
    sigma_left = J_sigma.transpose().dot(J_sigma)

    sigma_right = -1 * J_sigma.dot(f_sigma0)
    sigma_left = lambda_reg_paras * sigma_left
    sigma_right = lambda_reg_paras * np.reshape(sigma_right, (-1, 1), order="F")
    return sigma_left, sigma_right


def solve_para_shape_iter(
    iter, shape0, trans_base_2_camera, K, img_front, fit_reg, fit_kp_3d, fit_pt2d
):
    # % 1. landmark
    pt2d_left, pt2d_right = get_landmark_term(
        fit_pt2d, K, shape0, trans_base_2_camera, img_front
    )
    # % 2. shape
    shape_left, shape_right = get_vertex_term(fit_kp_3d, shape0)
    # % 3. reg
    reg_left, reg_right = get_regu_term(shape0, fit_reg)

    equationLeft = pt2d_left + shape_left + reg_left
    equationRight = pt2d_right + shape_right + reg_right
    delta_alpha = np.linalg.lstsq(equationLeft, equationRight, rcond=None)[0]
    shape1 = delta_alpha + shape0

    return shape1


def get_kp3d(keypoints_cur):
    keypoints_cur = np.vstack(
        (keypoints_cur * 3, np.vstack((keypoints_cur * 3 + 1, keypoints_cur * 3 + 2)))
    )
    keypoints_cur = np.reshape(keypoints_cur, (1, -1), order="F")[0]
    keypoints_cur = np.array(list(map(int, keypoints_cur)))
    return keypoints_cur


def fit_para_shape_with_kp3d(
    trans_base_2_camera,
    img_all,
    depth_all,
    pt2d_all,
    K,
    K_f,
    vertex_target_in,
    mu_shape,
    pcev_shape,
    sigma_shape,
    kp_next,
    lambda_reg_paras,
    lambda_3d_key,
    lambda_pt2d,
    lambda_project,
    tri_h_used,
):
    # in  : 3 * n
    kp_next = np.squeeze(kp_next)
    img_front = img_all[0]
    pt2d_front_ori = pt2d_all[0]
    n_shape = pcev_shape.shape[1]
    # % % ------------------------   regs - -----------------------------------
    fit_reg = {"weight": lambda_reg_paras, "sigma": sigma_shape}

    # % % ------------------------   3dkeypoints  - -----------------------------------
    #  all the remaining points are reliable and can be used
    vertex_inliner = np.where(vertex_target_in[2, :] > -100)[0]
    keypoints_cur = kp_next[vertex_inliner] - 1  # n
    keypoints_cur = get_kp3d(keypoints_cur)
    w_3d_key = pcev_shape[keypoints_cur, :]
    mu_3d_key = mu_shape[keypoints_cur].reshape(-1, 1)
    target_3d_key = vertex_target_in[:, vertex_inliner]

    nose_wing_idx = range(51, 55)
    mouth_idx = range(66, 86)
    eye_idx = range(35, 51)
    # Especially important points, left and right eye corners, nose tip, and upper and lower lip peaks,
    # these points are very important and well detected
    important_idx = [
        52 - 16 - 1,
        55 - 16 - 1,
        60 - 16 - 1,
        63 - 16 - 1,
        71 - 16 - 1,
        85 - 16 - 1,
        88 - 16 - 1,
    ]
    # % adjust 3d keypoint weight
    weight_3d_key = 5 * lambda_3d_key * np.ones((len(kp_next)))
    weight_3d_key[nose_wing_idx] = lambda_3d_key * 10
    weight_3d_key[mouth_idx] = lambda_3d_key * 10
    weight_3d_key[eye_idx] = lambda_3d_key * 10
    weight_3d_key[important_idx] = lambda_3d_key * 20
    weight_3d_key = weight_3d_key[vertex_inliner]
    fit_kp_3d = {
        "weight": weight_3d_key,
        "w": w_3d_key,
        "mu": mu_3d_key,
        "target": target_3d_key,
    }

    # ------------------------   2d landmark  ------------------------------------
    keypoints_cur = get_kp3d(kp_next - 1)
    w_pt2d = pcev_shape[
        keypoints_cur,
    ]
    mu_pt2d = mu_shape[
        keypoints_cur,
    ].reshape(-1, 1)
    target_pt2d = pt2d_front_ori
    weight_pt2d = lambda_pt2d * np.ones(len(kp_next))
    fit_pt2d = {
        "weight": weight_pt2d,
        "w": w_pt2d,
        "mu": mu_pt2d,
        "target": target_pt2d,
    }

    maxIteration = 4
    shape_current = np.zeros((n_shape, 1), np.float32)
    cal_one_loss(
        0,
        target_3d_key,
        mu_3d_key,
        w_3d_key,
        shape_current,
        trans_base_2_camera,
        target_pt2d,
        K,
        w_pt2d,
        mu_pt2d,
    )

    for iteration in range(1, maxIteration):
        shape_new = solve_para_shape_iter(
            iteration,
            shape_current,
            trans_base_2_camera,
            K,
            img_front,
            fit_reg,
            fit_kp_3d,
            fit_pt2d,
        )
        cal_one_loss(
            iteration,
            target_3d_key,
            mu_3d_key,
            w_3d_key,
            shape_new,
            trans_base_2_camera,
            target_pt2d,
            K,
            w_pt2d,
            mu_pt2d,
        )
        shape_current = shape_new

    print("----optimization done----")

    return shape_current
