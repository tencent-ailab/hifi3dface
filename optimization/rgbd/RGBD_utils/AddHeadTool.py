# -*- coding: utf-8 -*-
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

import math
import scipy.io as sio
import numpy as np
import sys

sys.path.append("../../..")
from third_party.NRICPTool import NRICPTool


class DoubleFitTool(object):
    def __init__(self):
        pass

    @staticmethod
    def fit_double_vertex(
        back_head_idx,
        contour_idx,
        vertex_target,
        vertex_ref,
        lambda_backhead,
        lambda_countour,
        mu_shape,
        pcev_shape,
        sigma_shape,
        lambda_reg_paras,
    ):
        n_shape = pcev_shape.shape[1]
        shape1 = np.zeros((n_shape, 1))
        keypoints_cur = np.reshape(back_head_idx, (1, -1))
        # keypoints_cur =  [i+1 for i in keypoints_cur]
        keypoints_cur1 = [3 * i - 2 for i in keypoints_cur]
        keypoints_cur2 = [3 * i - 1 for i in keypoints_cur]
        keypoints_cur3 = [3 * i for i in keypoints_cur]
        keypoints_cur = np.vstack(
            (keypoints_cur1, np.vstack((keypoints_cur2, keypoints_cur3)))
        )
        keypoints_cur = np.reshape(keypoints_cur, (-1, 1), order="F")[:, 0]
        keypoints_cur = [i - 1 for i in keypoints_cur]
        w_key_bh = pcev_shape[keypoints_cur, :]
        mu_key_bh = mu_shape[keypoints_cur]
        back_head_idx = [i - 1 for i in back_head_idx]
        vertex_target_bh = vertex_ref[:, back_head_idx]
        weight_bh = lambda_backhead * np.ones((1, len(back_head_idx)))

        keypoints_cur = np.reshape(contour_idx, (1, -1))

        keypoints_cur1 = [3 * i - 2 for i in keypoints_cur]
        keypoints_cur2 = [3 * i - 1 for i in keypoints_cur]
        keypoints_cur3 = [3 * i for i in keypoints_cur]
        keypoints_cur = np.vstack(
            (keypoints_cur1, np.vstack((keypoints_cur2, keypoints_cur3)))
        )
        keypoints_cur = np.reshape(keypoints_cur, (-1, 1), order="F")[:, 0]
        keypoints_cur = [i - 1 for i in keypoints_cur]
        w_key_contour = pcev_shape[keypoints_cur, :]
        mu_key_contour = mu_shape[keypoints_cur]
        contour_idx = [i - 1 for i in contour_idx]
        vertex_target_contour = vertex_target[:, contour_idx]
        weight_coutour = lambda_countour * np.ones((1, len(contour_idx)))
        iteration = 0
        maxiteration = 2
        prev_loss = 10000000
        use_vertex_reg = 0
        while True:
            shape = shape1
            if iteration >= maxiteration:
                break
            iteration = iteration + 1
            shape1 = DoubleFitTool.solov_back_head(
                weight_bh,
                w_key_bh,
                mu_key_bh,
                vertex_target_bh,
                weight_coutour,
                w_key_contour,
                mu_key_contour,
                vertex_target_contour,
                shape,
                sigma_shape,
                lambda_reg_paras,
            )

        return shape

    @staticmethod
    def solov_back_head(
        weight_bh,
        w_key_bh,
        mu_key_bh,
        vertex_target_bh,
        weight_coutour,
        w_key_contour,
        mu_key_contour,
        vertex_target_contour,
        shape0,
        sigma_shape,
        lambda_reg_paras,
    ):
        trans_self = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        backhead_left, backhead_right = DoubleFitTool.get_vertex_term(
            vertex_target_bh, trans_self, mu_key_bh, weight_bh, w_key_bh, shape0
        )
        shape_left, shape_right = DoubleFitTool.get_vertex_term(
            vertex_target_contour,
            trans_self,
            mu_key_contour,
            weight_coutour,
            w_key_contour,
            shape0,
        )
        reg_left, reg_right = DoubleFitTool.get_regu_term(
            shape0, sigma_shape, lambda_reg_paras
        )
        equationLeft = backhead_left + shape_left + reg_left
        equationRight = backhead_right + shape_right + reg_right
        delta_alpha = np.linalg.solve(equationLeft, equationRight)
        shape1 = delta_alpha + shape0
        return shape1

    @staticmethod
    def get_vertex_term(vertex_target, trans, mu_key, weight_key, w_key, shape):
        n_points = vertex_target.shape[1]
        n_basis = shape.shape[0]
        weight = np.tile(weight_key, (3, 1))
        sR = trans[:, 0:3]
        sT = trans[:, 3]
        J_shape = np.zeros((3, n_points, n_basis))
        W_Beta = np.reshape(w_key, (3, n_points, n_basis))
        for i in range(n_basis):
            J_shape[:, :, i] = weight * (sR.dot(W_Beta[:, :, i]))
        J_shape = np.reshape(J_shape, (-1, n_basis))
        pt3d_fit = np.reshape(mu_key, (len(mu_key), 1)) + w_key.dot(shape)
        pt3d_fit = np.reshape(pt3d_fit, (3, -1), order="F")
        pt3d_fit = sR.dot(pt3d_fit) + np.tile(sT, (n_points, 1)).transpose()
        f_shape0 = weight * (pt3d_fit - vertex_target)
        shape_left = J_shape.transpose().dot(J_shape)
        shape_right = -1 * J_shape.transpose().dot(
            np.reshape(f_shape0, (-1, 1), order="F")
        )
        return shape_left, shape_right

    @staticmethod
    def get_regu_term(paras, sigma_shape, lambda_reg_paras):
        sigma = sigma_shape
        J_sigma = np.eye(len(sigma))
        for i in range(len(sigma)):
            J_sigma[i][i] = 1 / sigma[i]
        sigma = sigma.reshape((500, 1))
        f_sigma0 = (paras) / sigma
        sigma_left = J_sigma.transpose().dot(J_sigma)
        sigma_right = -1 * J_sigma.dot(f_sigma0)

        sigma_left = lambda_reg_paras * sigma_left
        sigma_right = lambda_reg_paras * sigma_right
        return sigma_left, sigma_right


class AddHeadTool(object):
    def __init__(self):
        pass

    @staticmethod
    def transfer_PCA_format_for_add_head(rgbd_pca, HeadModel):
        def kp3(idx):
            keypoints_cur = np.reshape(idx, (1, -1))
            keypoints_cur1 = [3 * i for i in keypoints_cur]
            keypoints_cur2 = [3 * i + 1 for i in keypoints_cur]
            keypoints_cur3 = [3 * i + 2 for i in keypoints_cur]
            idx3 = np.vstack(
                (keypoints_cur1, np.vstack((keypoints_cur2, keypoints_cur3)))
            )
            idx3 = np.reshape(idx3, (-1, 1), order="F")[:, 0]
            return idx3

        pca_info_h = {}
        head_h_idx3 = kp3(HeadModel["head_h_idx"])
        mu_shape_h = np.squeeze(rgbd_pca["mu_shape"][:, head_h_idx3])
        pcev_h = rgbd_pca["basis_shape"][:, head_h_idx3].transpose()
        ev_f = np.ones(500)
        pca_info_h["mu_shape_h"] = mu_shape_h
        pca_info_h["pcev_h"] = pcev_h
        pca_info_h["ev_f"] = ev_f
        return pca_info_h

    @staticmethod
    def fix_back_head(vertex_fit_h, HeadModel, pca_info_h, USE_MIRROR, USE_SHOUDER_M):
        head_h_idx = HeadModel["head_h_idx"]

        if USE_MIRROR == 1:
            MirrorInfo = HeadModel["MirrorInfo"]
            _, vertex_sym = AddHeadTool.symmetry_one_next_vertex(
                vertex_fit_h, MirrorInfo, head_h_idx
            )
            vertex_fit_h = vertex_sym

        if USE_SHOUDER_M == 1:
            vertex_ref_wh = HeadModel["vertex_std_male"]
        else:
            vertex_ref_wh = HeadModel["vertex_std_female"]

        # 1: do pose alignment
        vertex_ref_h = vertex_ref_wh[:, head_h_idx]
        vertex_input_h = AddHeadTool.pre_alignment(
            vertex_fit_h, vertex_ref_h, HeadModel
        )

        # 2: fix mouth
        MouthFittingInfo = HeadModel["MouthFittingInfo"]
        vertex_target0 = AddHeadTool.fix_mouth(
            vertex_input_h, pca_info_h, MouthFittingInfo
        )

        # 3: add shoudler and backhead
        NICPShoulderInfo = HeadModel["NICPShoulderInfo"]
        tri_next_wh = HeadModel["tri_wh"]
        vertex_output_coord = AddHeadTool.add_shoudler_and_backhead(
            vertex_ref_wh, vertex_target0, NICPShoulderInfo, tri_next_wh, head_h_idx
        )
        return vertex_output_coord

    @staticmethod
    def flip_one_mesh(vertex_1, vertex_left, vertex_right):
        mid_nose_index = 7277 - 1
        vertex_flip = vertex_1.copy()
        vertex_2 = vertex_1.copy()
        vertex_flip[0, :] = 2 * vertex_1[0, mid_nose_index] - vertex_1[0, :]

        vertex_right = [int(i) - 1 for i in vertex_right]
        vertex_left = [int(i) - 1 for i in vertex_left]
        vertex_2[:, vertex_left] = vertex_1[:, vertex_right]
        vertex_2[:, vertex_right] = vertex_1[:, vertex_left]

        vertex_new_left = vertex_flip.copy()
        vertex_new_right = vertex_2.copy()
        vertex_new_left[0, :] = 2 * vertex_1[0, mid_nose_index] - vertex_new_left[0, :]
        vertex_new_right[0, :] = (
            2 * vertex_1[0, mid_nose_index] - vertex_new_right[0, :]
        )
        return vertex_new_left, vertex_new_right

    @staticmethod
    def flip_3n_next_mesh(vertex_all_in, MirrorInfo):
        mirror_vertex_left = MirrorInfo["left"]
        mirror_vertex_right = MirrorInfo["right"]
        for i in range(vertex_all_in.shape[1]):
            vertex = np.reshape(vertex_all_in[:, i], (3, -1), order="F")
            vertex_new_left, vertex_new_right = AddHeadTool.flip_one_mesh(
                vertex, mirror_vertex_left, mirror_vertex_right
            )
            if i == 0:
                vertex_all_flip = vertex_new_left.reshape((-1, 1), order="F")
                vertex_all_flip = np.hstack(
                    (vertex_all_flip, vertex_new_right.reshape((-1, 1), order="F"))
                )
            else:
                vertex_all_flip = np.hstack(
                    (vertex_all_flip, vertex_new_left.reshape((-1, 1), order="F"))
                )
                vertex_all_flip = np.hstack(
                    (vertex_all_flip, vertex_new_right.reshape((-1, 1), order="F"))
                )
        return vertex_all_flip

    def symmetry_one_next_vertex(input_vertex, MirrorInfo, head_h_idx):
        if input_vertex.shape[1] == 18518:
            a1 = input_vertex.copy()
            one_full = np.zeros((3, 20481))
            one_full[:, head_h_idx] = a1
            vertex_all_flip = AddHeadTool.flip_3n_next_mesh(
                one_full.reshape((-1, 1), order="F"), MirrorInfo
            )
            a2 = np.reshape(vertex_all_flip[:, 1], (3, -1), order="F")
            a2 = a2[:, head_h_idx]
            a3 = (a1 + a2) / 2

        if input_vertex.shape[1] == 20481:
            a1 = input_vertex
            vertex_all_flip = AddHeadTool.flip_3n_next_mesh(
                a1.reshape((-1, 1), order="F"), MirrorInfo
            )
            a2 = np.reshape(vertex_all_flip[:, 1], (3, -1), order="F")
            a3 = (a1 + a2) / 2

        vertex_flip = a2
        vertex_sym = a3
        return vertex_flip, vertex_sym

    @staticmethod
    def pre_alignment(vertex_input_h, vertex_xueer_h, HeadModel):
        vertex_xueer_h = np.squeeze(vertex_xueer_h)
        lr_ear_index_h = [2572 - 1, 209 - 1, 6774 - 1]
        lr_eye_index_wh = [12671 - 1, 12323 - 1, 7530 - 1, 7167 - 1]
        lr_eye_index_h = [11666 - 1, 11318 - 1, 7027 - 1, 6664 - 1]

        def call_scale(vertex_input_h, left_idx, right_idx):
            if len(left_idx) == 1:
                in_left = vertex_input_h[:, left_idx]
            else:
                in_left = np.mean(vertex_input_h[:, left_idx], 1)
            if len(right_idx) == 1:
                in_right = vertex_input_h[:, right_idx]
            else:
                in_right = np.mean(vertex_input_h[:, right_idx], 1)
            mid = (in_left + in_right) / 2
            scale = np.linalg.norm(in_right - in_left)
            return mid, scale

        in_mideye, in_scale = call_scale(
            vertex_input_h, lr_eye_index_h[0:2], lr_eye_index_h[2:4]
        )
        xuuer_mideye, xueer_scale = call_scale(
            vertex_xueer_h, lr_eye_index_h[0:2], lr_eye_index_h[2:4]
        )
        _, in_scale = call_scale(
            vertex_input_h, [lr_ear_index_h[0]], [lr_ear_index_h[1]]
        )
        _, xueer_scale = call_scale(
            vertex_xueer_h, [lr_ear_index_h[0]], [lr_ear_index_h[1]]
        )
        vertex_new = vertex_input_h
        scale = xueer_scale / in_scale
        vertex_new = (
            scale
            * (vertex_new - np.tile(in_mideye, (vertex_new.shape[1], 1)).transpose())
            + np.tile(in_mideye, (vertex_new.shape[1], 1)).transpose()
        )
        temp = HeadModel["symmetry_part_index_only_head"]
        mouth_idx = temp["vertex_mouth_idx_h"][0]
        mouth_idx = [i - 1 for i in mouth_idx]
        xueer_mouth = np.mean(vertex_xueer_h[:, mouth_idx], 1)
        in_mouth = np.mean(vertex_new[:, mouth_idx], 1)
        vector_xueer = xueer_mouth - xuuer_mideye
        vector_in = in_mouth - in_mideye
        angle_xueer = math.atan(vector_xueer[2] / np.abs(vector_xueer[1]))
        angle_in = math.atan(vector_in[2] / np.abs(vector_in[1]))
        change_y_angle = angle_in - angle_xueer
        rr = np.array(
            [
                [1, 0, 0],
                [0, np.cos(change_y_angle), -1 * np.sin(change_y_angle)],
                [0, np.sin(change_y_angle), np.cos(change_y_angle)],
            ]
        )
        vertex_new = (
            rr.dot(
                (vertex_new - np.tile(in_mideye, (vertex_new.shape[1], 1)).transpose())
            )
            + np.tile(in_mideye, (vertex_new.shape[1], 1)).transpose()
        )
        vertex_new = (
            vertex_new
            + np.tile(xuuer_mideye, (vertex_new.shape[1], 1)).transpose()
            - np.tile(in_mideye, (vertex_new.shape[1], 1)).transpose()
        )
        return vertex_new

    @staticmethod
    def fix_mouth(vertex_target, pca_info_h, MouthFittingInfo):
        mu_shape = pca_info_h["mu_shape_h"]
        pcev_shape = pca_info_h["pcev_h"]
        ev_f = pca_info_h["ev_f"]
        sigma_shape = np.sqrt(ev_f / np.sum(ev_f))
        idx_em = MouthFittingInfo["idx_em"][0]
        idx_em_need_update = MouthFittingInfo["idx_em_need_update"][0]
        idx_em_in_face = MouthFittingInfo["idx_em_in_face"][0]
        diff_from_face_to_esle = MouthFittingInfo["diff_from_face_to_esle"]
        idx_in_face_used = MouthFittingInfo["idx_in_face_used"][0]
        vertex_ref = vertex_target
        idx_em_need_update = [i - 1 for i in idx_em_need_update]
        idx_em_in_face = [i - 1 for i in idx_em_in_face]
        vertex_ref[:, idx_em_need_update] = (
            vertex_ref[:, idx_em_in_face] + diff_from_face_to_esle
        )
        back_head_idx = [i + 1 for i in idx_em_need_update]

        contour_idx = idx_in_face_used
        lambda_backhead = 100
        lambda_countour = 1000
        lambda_reg_paras = 100

        shape = DoubleFitTool.fit_double_vertex(
            back_head_idx,
            contour_idx,
            vertex_target,
            vertex_ref,
            lambda_backhead,
            lambda_countour,
            mu_shape,
            pcev_shape,
            sigma_shape,
            lambda_reg_paras,
        )

        mu_shape = np.reshape(mu_shape, (len(mu_shape), 1))
        pt3d_bh = mu_shape + pcev_shape.dot(shape)
        pt3d_bh = np.reshape(pt3d_bh, (3, -1), order="F")
        pt3d_result = vertex_target
        idx_em = [i - 1 for i in idx_em]
        pt3d_result[:, idx_em] = pt3d_bh[:, idx_em]
        return pt3d_result

    @staticmethod
    def add_shoudler_and_backhead(
        vertex_ref_wh, vertex_input_h, NICPShoulderInfo, tri_next_wh, head_h_idx
    ):

        idx_backhead = NICPShoulderInfo["idx_backhead"]
        idx_fix_face_part = NICPShoulderInfo["idx_fix_face_part"]
        idx_contout = NICPShoulderInfo["idx_contout"]
        shoulder_idx = NICPShoulderInfo["shoulder_idx"]
        weight_in = NICPShoulderInfo["weight_in"]

        temp_mesh = vertex_ref_wh.copy()
        temp_mesh[:, head_h_idx] = vertex_input_h
        vertex_input_wh = temp_mesh

        vertex_new = NRICPTool.nricp_shoulder(
            vertex_ref_wh,
            vertex_input_wh,
            tri_next_wh,
            shoulder_idx,
            idx_fix_face_part,
            idx_backhead,
            idx_contout,
        )
        idx_contout = [int(i) - 1 for i in idx_contout]
        vertex_smooth = vertex_new.toarray()
        vertex_smooth[:, idx_contout] = (1 - weight_in) * vertex_smooth[
            :, idx_contout
        ] + weight_in * vertex_input_wh[:, idx_contout]
        vertex_output_coord = vertex_smooth

        return vertex_output_coord
