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

import scipy.io as scio
import scipy.ndimage
import numpy as np
import sys

sys.path.append("../..")
from RGBD_utils.FitTools import get_kp3d, fit_para_shape_with_kp3d
from absl import app, flags
from third_party.ply import write_ply
from utils.basis import load_3dmm_basis, load_3dmm_basis_bfm
import os


def load_from_npz(base_dir):
    data = np.load(os.path.join(base_dir, "step1_data_for_fusion.npz"))
    K = data["K"]
    lmk3d_select = data["lmk3d_select"]
    img_select = data["img_select"]
    pt3d_select = data["pt3d_select"]

    data = np.load(os.path.join(base_dir, "step2_fusion.npz"))
    pose_all = data["first_trans_select"]
    pt3d_remove_outliers = data["pt3d_remove_outliers"]
    trans_base_2_camera = data["trans_base_2_camera"]
    depth_ori_select = data["depth_ori_select"]

    return (
        K,
        img_select,
        lmk3d_select,
        pt3d_select,
        pose_all,
        pt3d_remove_outliers,
        trans_base_2_camera,
        depth_ori_select,
    )


def load_pca_models(modle_base, basis3dmm_path):
    import h5py

    datas = h5py.File(os.path.join(modle_base, "next_face_info_only_head.mat"), "r")
    kp_86_h = np.asarray(datas.get("kp_86_h")).reshape(-1, 1)
    tempid = np.squeeze(np.asarray(datas.get("vertex_state")))  # n
    id_h = np.where(tempid > 0.5)
    id_h = np.array(id_h)
    id_h_h3 = get_kp3d(id_h)  # n
    tri_h_used = (np.array(datas.get("tri_h_used")).astype(np.int32) - 1).T  # 3 * n

    basis3dmm = scipy.io.loadmat(basis3dmm_path)
    mu_shape = np.array(basis3dmm["mu_shape"][:, id_h_h3]).transpose()  # n * 1
    pcev_shape = np.array(basis3dmm["basis_shape"][:, id_h_h3]).transpose()  # n * 500

    datas = h5py.File(os.path.join(modle_base, "shape_ev.mat"), "r")
    ev_f = np.asarray(datas.get("ev_f")).reshape(-1, 1)
    sigma_shape = np.sqrt(ev_f / np.sum(ev_f))

    return mu_shape, pcev_shape, sigma_shape, tri_h_used, kp_86_h


def load_pca_models_BFM(basis3dmm_path):
    basis3dmm = scipy.io.loadmat(basis3dmm_path)
    mu_shape = basis3dmm["mu_shape"]  # n * 1
    pcev_shape = (basis3dmm["bases_shape"] * basis3dmm["sigma_shape"])[:, :80]  # n * 80
    sigma_shape = np.ones([80, 1])
    tri_h_used = np.transpose(basis3dmm["tri"]).astype(np.int32)  # 3 * n

    kpts_86 = [
        22502,
        22653,
        22668,
        22815,
        22848,
        44049,
        46010,
        47266,
        47847,
        48436,
        49593,
        51577,
        31491,
        31717,
        32084,
        32197,
        32175,
        38779,
        39392,
        39840,
        40042,
        40208,
        39465,
        39787,
        39993,
        40213,
        40892,
        41087,
        41360,
        41842,
        42497,
        40898,
        41152,
        41431,
        13529,
        1959,
        3888,
        5567,
        6469,
        5450,
        3643,
        4920,
        4547,
        9959,
        10968,
        12643,
        14196,
        12785,
        11367,
        11610,
        12012,
        8269,
        8288,
        8302,
        8192,
        6837,
        9478,
        6499,
        10238,
        6002,
        10631,
        6755,
        7363,
        8323,
        9163,
        9639,
        5652,
        7614,
        8216,
        8935,
        11054,
        8235,
        6153,
        10268,
        9798,
        6674,
        6420,
        10535,
        7148,
        8227,
        9666,
        6906,
        8110,
        9546,
        7517,
        8837,
    ]

    kpts_86 = np.array(kpts_86, np.int32) - 1
    kp_86_h = kpts_86.reshape(-1, 1)  # 86 * 1

    return mu_shape, pcev_shape, sigma_shape, tri_h_used, kp_86_h


def run(prefit_dir, modle_path):
    print("---- step3 start -----")
    print("running base:", prefit_dir)
    # load models
    if FLAGS.is_bfm:
        mu_shape, pcev_shape, sigma_shape, tri_h_used, kp_86_h = load_pca_models_BFM(
            FLAGS.basis3dmm_path
        )
    else:
        mu_shape, pcev_shape, sigma_shape, tri_h_used, kp_86_h = load_pca_models(
            modle_path, FLAGS.basis3dmm_path
        )

    # load datas
    (
        K,
        img_select,
        lmk3d_select,
        pt3d_select,
        pose_all,
        pt3d_remove_outliers,
        trans_base_2_camera,
        depth_ori_select,
    ) = load_from_npz(prefit_dir)

    #  sparse fit
    if FLAGS.is_bfm:
        print("BFM")
        lambda_project = 5000
        lambda_reg_paras = 2000000000
        lambda_3d_key = 1000
        lambda_pt2d = 100
    else:
        lambda_project = 5000
        lambda_reg_paras = 1000
        lambda_3d_key = 1000
        lambda_pt2d = 100
    shape = fit_para_shape_with_kp3d(
        trans_base_2_camera,
        img_select,
        depth_ori_select,
        lmk3d_select,
        K,
        K,
        pt3d_remove_outliers,
        mu_shape,
        pcev_shape,
        sigma_shape,
        kp_86_h,
        lambda_reg_paras,
        lambda_3d_key,
        lambda_pt2d,
        lambda_project,
        tri_h_used,
    )

    # save
    prefit_head = mu_shape + pcev_shape.dot(shape)
    prefit_head = np.reshape(prefit_head, (3, -1), order="F")

    np.save(os.path.join(prefit_dir, "para_shape_init.npy"), shape)

    write_ply(
        os.path.join(prefit_dir, "prefit_head.ply"),
        prefit_head.transpose(),
        tri_h_used.transpose(),
        None,
        True,
    )
    print("---- step3 succeed -----")


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO
    prefit_dir = FLAGS.prefit
    model_dir = FLAGS.modle_path
    run(prefit_dir, model_dir)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("prefit", "prefit_bfm/", "output data directory")
    flags.DEFINE_string("modle_path", "../../resources", "modle_path")
    # flags.DEFINE_string('basis3dmm_path', '../resources/shape_exp_bases_rgbd_20200514.mat', 'basis3dmm_path')
    flags.DEFINE_string(
        "basis3dmm_path", "../../resources/BFM2009_Model.mat", "basis3dmm_path"
    )
    flags.DEFINE_string("GPU_NO", "7", "which GPU")
    flags.DEFINE_boolean("is_bfm", True, "default: False")
    app.run(main)
