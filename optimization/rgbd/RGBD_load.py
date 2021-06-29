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

sys.path.append("../..")

from utils.basis import load_3dmm_basis, get_geometry, get_region_uv_texture


class RGBD_load(object):
    @staticmethod
    def load_segs(base_path, indexs):
        segpath = os.path.join(base_path, "img_names.txt")
        print("load seg:", segpath)
        f = open(segpath)
        arr_seg = f.readlines()

        # face seg infro
        def seg_re_organize(Seg):
            seg = np.reshape(Seg, (300, 300))
            seg_list = []
            for i in range(19):
                seg_list.append((seg == i).astype(np.float32))
            seg_list = np.stack(seg_list, axis=2)
            return seg_list  # 300 x 300 x 19

        seg_batch1 = []
        for i in range(4):
            # idx = indexs[i][0] - 1
            idx = indexs[i]
            one = arr_seg[idx]
            splits = one.strip().split(",")
            one_name = splits[0].split(".")[0] + ".npy"
            one_seg = np.load(os.path.join(base_path, one_name))
            one_seg = seg_re_organize(one_seg)
            seg_batch1.append(one_seg)
        seg_batch = np.array(
            [seg_batch1[0], seg_batch1[1], seg_batch1[2], seg_batch1[3]]
        )
        return seg_batch

    @staticmethod
    def load_rgbd_datas_npz(prefit_dir, prepare_dir):

        data = np.load(os.path.join(prefit_dir, "step2_fusion.npz"))
        M_base2camera = data["trans_base_2_camera"]  # 3 * 4

        data = np.load(os.path.join(prefit_dir, "step1_data_for_fusion.npz"))
        img_batch = data["img_select"]
        depth_batch = data["depth_select"]
        landmarks = data["lmk3d_select"]

        new_list = data["new_list"]
        inliners = data["inliners"]
        new_list = inliners[0, new_list]

        height = img_batch[0].shape[0]
        width = img_batch[0].shape[1]
        Dep_height = depth_batch[0].shape[0]
        Dep_width = depth_batch[0].shape[1]
        K = data["K"]
        trans_all = data["first_trans_select"]

        M1_M = trans_all[0]
        M1_L = trans_all[1]
        M1_R = trans_all[2]
        M1_B = trans_all[3]
        K_depth = K

        seg_batch = RGBD_load.load_segs(os.path.join(prepare_dir), new_list)

        para_shape = np.transpose(
            np.load(os.path.join(prefit_dir, "para_shape_init.npy"))
        )  # 1 * n
        para_tex = np.load(os.path.join(prefit_dir, "para_tex_init.npy"))  # 1 * n
        info = {
            "img_batch": img_batch,
            "M_base2camera": M_base2camera,
            "M1_M": M1_M,  # M
            "M1_L": M1_L,  # L
            "M1_R": M1_R,  # R
            "M1_B": M1_B,  # B
            "K": K,
            "para_shape": para_shape,
            "height": height,
            "width": width,
            "Dep_height": Dep_height,
            "Dep_width": Dep_width,
            "K_depth": K_depth,
            "depth_batch": depth_batch,
            "lms3D": landmarks,
            "seg_batch": seg_batch,
            "para_tex": para_tex,
        }
        return info

    @staticmethod
    def crop_300300(raw_info, basis3dmm, size=300):
        prefit_head = basis3dmm["mu_shape"] + raw_info["para_shape"].dot(
            basis3dmm["basis_shape"]
        )
        prefit_head = np.transpose(np.reshape(prefit_head, [-1, 3]), [1, 0])
        vertex_slam = raw_info["M_base2camera"].dot(
            np.concatenate((prefit_head, np.ones([1, prefit_head.shape[1]])), axis=0)
        )

        pose_list = []
        pose_list.append(raw_info["M1_M"])
        pose_list.append(raw_info["M1_L"])
        pose_list.append(raw_info["M1_R"])
        pose_list.append(raw_info["M1_B"])
        K_list = []
        img_list = []
        dep_list = []
        seg_list = []
        lmk_list = []
        pad_list = []
        ori_crop_img = []
        for numiter in range(4):
            landmark = raw_info["lms3D"][numiter].T
            img = raw_info["img_batch"][numiter]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change to channels
            depth = raw_info["depth_batch"][numiter]
            init_K = raw_info["K"]
            segs = raw_info["seg_batch"][numiter]
            one_trans = pose_list[numiter]
            rr = one_trans[0:3, 0:3]
            tt = one_trans[0:3, 3:]
            rr_t = np.transpose(rr, [1, 0])
            proj_xyz0 = rr_t.dot(
                vertex_slam - np.repeat(tt, prefit_head.shape[1], axis=1)
            )

            proj_xyz = init_K.dot(proj_xyz0)
            proj_xyz[0, :] = proj_xyz[0, :] / proj_xyz[2, :]
            proj_xyz[1, :] = proj_xyz[1, :] / proj_xyz[2, :]
            proj_xyz = np.transpose(proj_xyz, [1, 0])  # n * 3

            # use landmarks to crop face
            # landmark : n *2
            min_x, min_y = np.amin(landmark, 0)
            max_x, max_y = np.amax(landmark, 0)

            c_x = min_x + 0.5 * (max_x - min_x)
            c_y = min_y + 0.4 * (max_y - min_y)
            half_x = (max_x - min_x) * 0.8
            half_y = (max_y - min_y) * 0.6
            half_width = max(half_x, half_y)

            st_x = c_x - half_width
            en_x = c_x + half_width
            st_y = c_y - half_width
            en_y = c_y + half_width

            pad_t = int(max(0 - st_y, 0) + 0.5)
            pad_b = int(max(en_y - img.shape[0], 0) + 0.5)
            pad_l = int(max(0 - st_x, 0) + 0.5)
            pad_r = int(max(en_x - img.shape[1], 0) + 0.5)

            st_x_ = int(st_x + 0.5)
            en_x_ = int(en_x + 0.5)
            # crop to 2^n
            rate = int(float(en_x_ - st_x_) / float(size))
            if rate != 0:
                en_x_ = en_x_ - (en_x_ - st_x_) % (2 * rate)
            st_y_ = int(st_y + 0.5)
            # en_y_ = int(en_y + 0.5)
            en_y_ = st_y_ + en_x_ - st_x_

            # pad top and bottom
            pad_img = np.concatenate(
                [
                    np.zeros((pad_t, img.shape[1], 3)),
                    img,
                    np.zeros((pad_b, img.shape[1], 3)),
                ],
                axis=0,
            )
            pad_dep = np.concatenate(
                [
                    np.zeros((pad_t, depth.shape[1])),
                    depth,
                    np.zeros((pad_b, depth.shape[1])),
                ],
                axis=0,
            )
            # pad left and right
            pad_img = np.concatenate(
                [
                    np.zeros((pad_img.shape[0], pad_l, 3)),
                    pad_img,
                    np.zeros((pad_img.shape[0], pad_r, 3)),
                ],
                axis=1,
            )
            pad_dep = np.concatenate(
                [
                    np.zeros((pad_dep.shape[0], pad_l)),
                    pad_dep,
                    np.zeros((pad_dep.shape[0], pad_r)),
                ],
                axis=1,
            )
            # crop
            crop_img = pad_img[
                (st_y_ + pad_t) : (en_y_ + pad_t), (st_x_ + pad_l) : (en_x_ + pad_l), :
            ]
            crop_dep = pad_dep[
                (st_y_ + pad_t) : (en_y_ + pad_t), (st_x_ + pad_l) : (en_x_ + pad_l)
            ]

            cur_h0, _, _ = crop_img.shape
            lmk_x = landmark[:, 0]
            lmk_x = lmk_x - st_x
            lmk_y = landmark[:, 1]
            lmk_y = lmk_y - st_y
            landmark1 = np.stack([lmk_x, lmk_y], axis=1)

            # resize to 300 * 300
            dep_300300 = np.zeros((size, size))
            upscale = float(crop_dep.shape[0]) / float(size)
            downscale = float(size) / float(crop_dep.shape[0])
            for i in range(size):
                for j in range(size):
                    dep_300300[i, j] = crop_dep[int(i * upscale), int(j * upscale)]
            img_300300 = cv2.resize(crop_img, (size, size))
            landmark300 = landmark1 * downscale

            crop_K = np.copy(init_K)
            crop_K[0, 0] = init_K[0, 0] * downscale
            crop_K[1, 1] = init_K[1, 1] * downscale
            crop_K[0, 2] = (init_K[0, 2] - st_x) * downscale
            crop_K[1, 2] = (init_K[1, 2] - st_y) * downscale

            proj_xyz = crop_K.dot(proj_xyz0)
            proj_xyz[0, :] = proj_xyz[0, :] / proj_xyz[2, :]
            proj_xyz[1, :] = proj_xyz[1, :] / proj_xyz[2, :]
            proj_xyz = np.transpose(proj_xyz, [1, 0])  # n * 3

            K_list.append(crop_K)
            img_list.append(img_300300)
            dep_list.append(dep_300300)
            seg_list.append(segs)
            lmk_list.append(landmark300)
            ori_crop_img.append(crop_img)

        info = {
            "K_list": np.array(K_list),
            "img_list": np.array(img_list),
            "ori_img": np.array(ori_crop_img),
            "dep_list": np.array(dep_list),
            "seg_list": np.array(seg_list),
            "lmk_list3D": np.array(lmk_list),
            "pose_list": np.array(pose_list),
            "M_base2camera": raw_info["M_base2camera"],
            "para_shape": raw_info["para_shape"],
            "para_tex": raw_info["para_tex"],
            "height": size,
            "width": size,
            "Dep_height": size,
            "Dep_width": size,
        }
        return info

    @staticmethod
    def cal_trans_Mat(M0, M1):
        # M0:3x4 M1:3x4
        w = np.stack([[0.0], [0.0], [0.0], [1.0]], 1)  # 1 x 4
        M0 = np.concatenate([M0, w], axis=0)  # 3 x 4 + 1 x 4 -> 4 x 4
        M0 = np.transpose(M0, [1, 0])  # 4, 4

        rr = M1[0:3, 0:3]  # 3 x 3
        rr_T = np.transpose(rr, [1, 0])  # 3 x 3
        tt = np.reshape(M1[:, 3], [3, 1])  # 3 x 1
        tt = np.matmul(rr_T, tt)  # 3 x 1
        M1 = np.concatenate([rr_T, tt * (-1)], axis=1)  # 3 x 4
        M1 = np.concatenate([M1, w], axis=0)  # 4 x 4
        M1 = np.transpose(M1, [1, 0])  # 4 x 4

        return np.matmul(M0, M1)

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
    def load_and_preprocess_RGBD_data(prefit_dir, prepare_dir, basis3dmm):

        raw_info = RGBD_load.load_rgbd_datas_npz(prefit_dir, prepare_dir)
        info = RGBD_load.crop_300300(raw_info, basis3dmm)

        info["trans_Mat_M"] = RGBD_load.cal_trans_Mat(
            info["M_base2camera"], info["pose_list"][0]
        )
        info["trans_Mat_L"] = RGBD_load.cal_trans_Mat(
            info["M_base2camera"], info["pose_list"][1]
        )
        info["trans_Mat_R"] = RGBD_load.cal_trans_Mat(
            info["M_base2camera"], info["pose_list"][2]
        )
        info["trans_Mat_B"] = RGBD_load.cal_trans_Mat(
            info["M_base2camera"], info["pose_list"][3]
        )

        scale_Mat = np.matmul(
            info["M_base2camera"][0:3][0:3].T, info["M_base2camera"][0:3][0:3]
        )
        scale = np.sqrt(scale_Mat[0, 0])

        info["se3_M"] = RGBD_load.trans_2_rotationVector(
            np.transpose(info["trans_Mat_M"], [1, 0])[0:3] / scale
        )
        info["se3_L"] = RGBD_load.trans_2_rotationVector(
            np.transpose(info["trans_Mat_L"], [1, 0])[0:3] / scale
        )
        info["se3_R"] = RGBD_load.trans_2_rotationVector(
            np.transpose(info["trans_Mat_R"], [1, 0])[0:3] / scale
        )
        info["se3_B"] = RGBD_load.trans_2_rotationVector(
            np.transpose(info["trans_Mat_B"], [1, 0])[0:3] / scale
        )

        info["input_scale"] = np.array([[scale]])
        info["se3_list"] = np.array(
            [info["se3_M"], info["se3_L"], info["se3_R"], info["se3_B"]]
        )

        return info
