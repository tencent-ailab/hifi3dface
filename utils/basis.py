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
import glob
import os
import cv2
import scipy.io
import tensorflow as tf


def load_vertex_3dmm_basis(basis_path, tri_v_path=None):
    """load 3dmm basis and other useful files.

    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param tri_v_path:
        - contains mesh triangle definition in geometry. If it is set, it covers the definition in basis_path.
    :param tri_vt_path:
        - contains mesh triangle definition in UV space.

    """
    # NEXT model
    config_mat = np.load(basis_path, encoding="latin1").tolist()

    # fix shape into normalized xyz
    # config_mat['mu_shape'] = config_mat['mu_shape'].reshape((20481, 3))
    # for large shape
    config_mat["mu_shape"] = config_mat["mu_shape"].reshape((3, 20481)).transpose()
    vtx = config_mat["mu_shape"]
    xx = vtx[:, 0]
    yy = vtx[:, 2]
    zz = -vtx[:, 1]
    vtx[:, 1] = yy
    vtx[:, 2] = zz
    mean_vtx = np.mean(vtx, axis=0)
    vtx -= mean_vtx
    config_mat["mu_shape"] = vtx.reshape((1, 61443)).astype(np.float32)

    for i in range(0, config_mat["bases_shape"].shape[1]):
        x = config_mat["bases_shape"][:, i]
        # x = x.reshape((20481, 3))
        # for large pose
        x = x.reshape((3, 20481)).transpose()
        vtx = x
        xx = vtx[:, 0]
        yy = vtx[:, 2]
        zz = -vtx[:, 1]
        vtx[:, 1] = yy
        vtx[:, 2] = zz
        config_mat["bases_shape"][:, i] = vtx.reshape((61443,))
    config_mat["bases_shape"] = np.transpose(config_mat["bases_shape"])
    config_mat["basis_shape"] = config_mat["bases_shape"][:199, :].astype(np.float32)

    # fix texture
    config_mat["mu_tex"] = config_mat["mu_tex"].reshape((3, 20481)).transpose()
    config_mat["mu_tex"] = config_mat["mu_tex"].reshape((1, 61443)).astype(np.float32)

    for i in range(0, config_mat["bases_tex"].shape[1]):
        x = config_mat["bases_tex"][:, i]
        x = x.reshape((3, 20481)).transpose()
        config_mat["bases_tex"][:, i] = x.reshape((61443,))
    config_mat["basis_tex"] = np.transpose(config_mat["bases_tex"]).astype(np.float32)

    config_mat["tri"] = np.load(tri_v_path)["arr_0"]
    config_mat["tri"] = config_mat["tri"].astype(np.int32)

    config_mat["mask_face"] = config_mat["mask_face"].astype(np.float32)

    config_mat["keypoints"] = config_mat["keypoints"].astype(np.int32)
    config_mat["keypoints"] = np.squeeze(config_mat["keypoints"])

    return config_mat


def load_vertex_uv_3dmm_basis(
    basis_path,
    uv_path=None,
    tri_v_path=None,
    tri_vt_path=None,
    vt_path=None,
    uv_weight_mask_path=None,
    is_train=True,
    is_whole_uv=True,
):
    """load 3dmm basis and other useful files.

    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param tri_v_path:
        - contains mesh triangle definition in geometry. If it is set, it covers the definition in basis_path.
    :param tri_vt_path:
        - contains mesh triangle definition in UV space.

    """
    # NEXT model
    config_mat = np.load(basis_path, encoding="latin1").tolist()

    # fix shape into normalized xyz
    # config_mat['mu_shape'] = config_mat['mu_shape'].reshape((20481, 3))
    # for large shape
    config_mat["mu_shape"] = config_mat["mu_shape"].reshape((3, 20481)).transpose()
    vtx = config_mat["mu_shape"]
    xx = vtx[:, 0]
    yy = vtx[:, 2]
    zz = -vtx[:, 1]
    vtx[:, 1] = yy
    vtx[:, 2] = zz
    mean_vtx = np.mean(vtx, axis=0)
    vtx -= mean_vtx
    config_mat["mu_shape"] = vtx.reshape((1, 61443)).astype(np.float32)

    for i in range(0, config_mat["bases_shape"].shape[1]):
        x = config_mat["bases_shape"][:, i]
        # x = x.reshape((20481, 3))
        # for large pose
        x = x.reshape((3, 20481)).transpose()
        vtx = x
        xx = vtx[:, 0]
        yy = vtx[:, 2]
        zz = -vtx[:, 1]
        vtx[:, 1] = yy
        vtx[:, 2] = zz
        config_mat["bases_shape"][:, i] = vtx.reshape((61443,))
    config_mat["bases_shape"] = np.transpose(config_mat["bases_shape"])
    config_mat["basis_shape"] = config_mat["bases_shape"][:199, :].astype(np.float32)

    # fix texture
    config_mat["mu_tex"] = config_mat["mu_tex"].reshape((3, 20481)).transpose()
    config_mat["mu_tex"] = config_mat["mu_tex"].reshape((1, 61443)).astype(np.float32)

    for i in range(0, config_mat["bases_tex"].shape[1]):
        x = config_mat["bases_tex"][:, i]
        x = x.reshape((3, 20481)).transpose()
        config_mat["bases_tex"][:, i] = x.reshape((61443,))
    config_mat["basis_tex"] = np.transpose(config_mat["bases_tex"]).astype(np.float32)

    config_mat["tri"] = np.load(tri_v_path)["arr_0"]
    config_mat["tri"] = config_mat["tri"].astype(np.int32)

    config_mat["mask_face"] = config_mat["mask_face"].astype(np.float32)

    config_mat["keypoints"] = config_mat["keypoints"].astype(np.int32)
    config_mat["keypoints"] = np.squeeze(config_mat["keypoints"])

    basis3dmm = config_mat

    # load uv basis
    if uv_path is not None and not is_whole_uv:
        uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv512.mat")))
        uv_region_bases = {}
        for region_path in uv_region_paths:
            region_name = region_path.split("/")[-1].split("_uv")[0]
            region_config = scipy.io.loadmat(region_path)
            region_config["basis"] = np.transpose(
                region_config["basis"] * region_config["sigma"]
            )
            region_config["indices"] = region_config["indices"].astype(np.int32)
            del region_config["sigma"]
            assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
            uv_region_bases[region_name] = region_config
        basis3dmm["uv"] = uv_region_bases

        if not is_train:
            uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv.mat")))
            uv_region_bases = {}
            for region_path in uv_region_paths:
                region_name = region_path.split("/")[-1].split("_uv")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                uv_region_bases[region_name] = region_config
            basis3dmm["uv2k"] = uv_region_bases

            normal_region_paths = sorted(
                glob.glob(os.path.join(uv_path, "*_normal.mat"))
            )
            normal_region_bases = {}
            for region_path in normal_region_paths:
                region_name = region_path.split("/")[-1].split("_normal")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                normal_region_bases[region_name] = region_config
            basis3dmm["normal2k"] = normal_region_bases

    if uv_path is not None and is_whole_uv:
        config = scipy.io.loadmat(uv_path)
        config["basis"] = config["basis"] * config["sigma"]
        config["indices"] = config["indices"].astype(np.int32)
        del config["sigma"]
        if config["basis"].shape[0] > config["basis"].shape[1]:
            config["basis"] = np.transpose(config["basis"])
        assert config["basis"].shape[0] < config["basis"].shape[1]
        basis3dmm["uv"] = config

        if not is_train:  # add normal
            normal_path = uv_path.replace("uv512", "norm512")
            config_norm = scipy.io.loadmat(normal_path)
            config_norm["basis"] = np.transpose(
                config_norm["basis"] * config_norm["sigma"]
            )
            config_norm["indices"] = config_norm["indices"].astype(np.int32)
            del config_norm["sigma"]
            assert config_norm["basis"].shape[0] < config_norm["basis"].shape[1]
            basis3dmm["normal"] = config_norm

    if tri_v_path is not None:
        tri_v = np.load(tri_v_path)["arr_0"].astype(np.int32)
        basis3dmm["tri"] = tri_v

    if tri_vt_path is not None:
        tri_vt = np.load(tri_vt_path)["arr_0"].astype(np.int32)
        basis3dmm["tri_vt"] = tri_vt

    if vt_path is not None:
        vt_list = np.load(vt_path)["arr_0"].astype(np.float32)
        basis3dmm["vt_list"] = vt_list

    if uv_weight_mask_path is not None:
        uv_weight_mask = cv2.imread(uv_weight_mask_path).astype(np.float32) / 255.0
        basis3dmm["uv_weight_mask"] = np.expand_dims(uv_weight_mask, 0)
        assert uv_weight_mask.shape[0] == 512

    return basis3dmm


def load_3dmm_basis(
    basis_path,
    uv_path=None,
    tri_v_path=None,
    tri_vt_path=None,
    vt_path=None,
    uv_weight_mask_path=None,
    is_train=True,
    is_whole_uv=True,
    limit_dim=-1,
):
    """load 3dmm basis and other useful files.

    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param tri_v_path:
        - contains mesh triangle definition in geometry. If it is set, it covers the definition in basis_path.
    :param tri_vt_path:
        - contains mesh triangle definition in UV space.

    """

    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm["keypoints"] = np.squeeze(basis3dmm["keypoints"])
    basis3dmm["vt_list"] = basis3dmm["vt_list"].astype(np.float32)

    # load uv basis
    if uv_path is not None and not is_whole_uv:
        uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv512.mat")))
        uv_region_bases = {}
        for region_path in uv_region_paths:
            region_name = region_path.split("/")[-1].split("_uv")[0]
            region_config = scipy.io.loadmat(region_path)
            region_config["basis"] = np.transpose(
                region_config["basis"] * region_config["sigma"]
            )
            region_config["indices"] = region_config["indices"].astype(np.int32)
            del region_config["sigma"]
            assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
            uv_region_bases[region_name] = region_config
        basis3dmm["uv"] = uv_region_bases

        if not is_train:
            uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv.mat")))
            uv_region_bases = {}
            for region_path in uv_region_paths:
                region_name = region_path.split("/")[-1].split("_uv")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                uv_region_bases[region_name] = region_config
            basis3dmm["uv2k"] = uv_region_bases

            normal_region_paths = sorted(
                glob.glob(os.path.join(uv_path, "*_normal.mat"))
            )
            normal_region_bases = {}
            for region_path in normal_region_paths:
                region_name = region_path.split("/")[-1].split("_normal")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                normal_region_bases[region_name] = region_config
            basis3dmm["normal2k"] = normal_region_bases

    if uv_path is not None and is_whole_uv:
        config = scipy.io.loadmat(uv_path)
        config["basis"] = config["basis"] * config["sigma"]
        config["indices"] = config["indices"].astype(np.int32)
        del config["sigma"]
        if config["basis"].shape[0] > config["basis"].shape[1]:
            config["basis"] = np.transpose(config["basis"])
        assert config["basis"].shape[0] < config["basis"].shape[1]
        basis3dmm["uv"] = config

        if not is_train:  # add normal
            normal_path = uv_path.replace("uv512", "norm512")
            config_norm = scipy.io.loadmat(normal_path)
            config_norm["basis"] = np.transpose(
                config_norm["basis"] * config_norm["sigma"]
            )
            config_norm["indices"] = config_norm["indices"].astype(np.int32)
            del config_norm["sigma"]
            assert config_norm["basis"].shape[0] < config_norm["basis"].shape[1]
            basis3dmm["normal"] = config_norm

    if tri_v_path is not None:
        tri_v = np.load(tri_v_path)["arr_0"].astype(np.int32)
        basis3dmm["tri"] = tri_v

    if tri_vt_path is not None:
        tri_vt = np.load(tri_vt_path)["arr_0"].astype(np.int32)
        basis3dmm["tri_vt"] = tri_vt

    if vt_path is not None:
        vt_list = np.load(vt_path)["arr_0"].astype(np.float32)
        basis3dmm["vt_list"] = vt_list

    if uv_weight_mask_path is not None:
        uv_weight_mask = cv2.imread(uv_weight_mask_path).astype(np.float32) / 255.0
        basis3dmm["uv_weight_mask"] = np.expand_dims(uv_weight_mask, 0)
        assert uv_weight_mask.shape[0] == 512

    if limit_dim > 0:
        basis3dmm["basis_shape"] = basis3dmm["basis_shape"][:limit_dim, :]

    return basis3dmm


def load_3dmm_basis_bfm(basis_path):
    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm["basis_shape"] = np.transpose(
        basis3dmm["bases_shape"] * basis3dmm["sigma_shape"]
    ).astype(np.float32)
    basis3dmm["basis_exp"] = np.transpose(
        basis3dmm["bases_exp"] * basis3dmm["sigma_exp"]
    ).astype(np.float32)
    basis3dmm["basis_tex"] = np.transpose(
        basis3dmm["bases_tex"] * basis3dmm["sigma_tex"]
    ).astype(np.float32)
    basis3dmm["mu_shape"] = np.transpose(basis3dmm["mu_shape"]).astype(np.float32)
    basis3dmm["mu_exp"] = np.transpose(basis3dmm["mu_exp"]).astype(np.float32)
    basis3dmm["mu_tex"] = np.transpose(basis3dmm["mu_tex"]).astype(np.float32)
    basis3dmm["tri"] = basis3dmm["tri"].astype(np.int32)

    # crop bases
    basis3dmm["basis_shape"] = basis3dmm["basis_shape"][:80, :]
    basis3dmm["basis_tex"] = basis3dmm["basis_tex"][:80, :]

    # get the neighboring relationship of triangles
    edge_to_triangles = {}
    for idx, tri in enumerate(basis3dmm["tri"]):
        v1 = tri[0]
        v2 = tri[1]
        v3 = tri[2]
        try:
            edge_to_triangles[(v1, v2)].append(idx)
        except Exception:
            edge_to_triangles[(v1, v2)] = [idx]

        try:
            edge_to_triangles[(v2, v1)].append(idx)
        except Exception:
            edge_to_triangles[(v2, v1)] = [idx]

        try:
            edge_to_triangles[(v1, v3)].append(idx)
        except Exception:
            edge_to_triangles[(v1, v3)] = [idx]

        try:
            edge_to_triangles[(v3, v1)].append(idx)
        except Exception:
            edge_to_triangles[(v3, v1)] = [idx]

        try:
            edge_to_triangles[(v2, v3)].append(idx)
        except Exception:
            edge_to_triangles[(v2, v3)] = [idx]

        try:
            edge_to_triangles[(v3, v2)].append(idx)
        except Exception:
            edge_to_triangles[(v3, v2)] = [idx]

    tri_pairs = []
    for key in edge_to_triangles:
        relations = edge_to_triangles[key]
        for item_a in relations:
            for item_b in relations:
                if item_a < item_b:
                    tri_pairs.append((item_a, item_b))
    tri_pairs = set(tri_pairs)
    tri_pairs = np.array(list(tri_pairs), np.int32)
    basis3dmm["tri_pairs"] = tri_pairs

    # keypoints
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
    basis3dmm["keypoints"] = kpts_86

    basis3dmm["test"] = "test"

    return basis3dmm


def get_geometry(basis3dmm, para_shape, para_exp=None):
    """compute the geometry according to the 3DMM parameters.
    para_shape: shape parameter
    para_exp: expression parameter
    """
    shape_inc = tf.matmul(para_shape, basis3dmm["basis_shape"])
    geo = basis3dmm["mu_shape"] + shape_inc

    if para_exp is not None:
        exp_inc = tf.matmul(para_exp, basis3dmm["basis_exp"])
        geo_with_exp = geo + exp_inc

    if para_exp is None:
        return tf.reshape(geo, [-1, basis3dmm["basis_shape"].shape[1] // 3, 3])
    else:
        return (
            tf.reshape(geo, [-1, basis3dmm["basis_shape"].shape[1] // 3, 3]),
            tf.reshape(geo_with_exp, [-1, basis3dmm["basis_shape"].shape[1] // 3, 3]),
        )


def get_texture(basis3dmm, para_tex):
    tex_inc = tf.matmul(para_tex, basis3dmm["basis_tex"])
    tex = basis3dmm["mu_tex"] + tex_inc
    return tf.reshape(tex, [-1, basis3dmm["basis_tex"].shape[1] // 3, 3])


def get_region_uv_texture(uv_basis, para, uv_size=512):
    basis = uv_basis["basis"]
    mu = uv_basis["mu"]
    indices = uv_basis["indices"].astype(np.int32)[:, :2]
    batch_size = para.get_shape().as_list()[0]

    fitted_result = tf.matmul(para, basis) + mu
    fitted_result = tf.reshape(fitted_result, [-1, 3])
    if "weight" in uv_basis:
        weight = uv_basis["weight"]
        weight = np.reshape(np.stack([weight] * batch_size, axis=0), [-1, 3])
        fitted_result = fitted_result * weight

    # scatter update
    batch_orders = tf.reshape(tf.range(batch_size), [-1, 1])
    batch_orders = tf.concat([batch_orders] * indices.shape[0], axis=1)
    batch_orders = tf.reshape(batch_orders, [-1, 1])
    batch_indices = tf.concat([indices] * batch_size, axis=0)
    batch_indices = tf.concat([batch_orders, batch_indices], axis=1)
    fitted_uv = tf.scatter_nd(
        batch_indices, fitted_result, shape=[batch_size, uv_size, uv_size, 3]
    )
    fitted_mask = tf.scatter_nd(
        batch_indices,
        tf.ones_like(fitted_result),
        shape=[batch_size, uv_size, uv_size, 3],
    )

    fitted_uv = tf.clip_by_value(fitted_uv, 0, 255)
    fitted_mask = tf.clip_by_value(fitted_mask, 0, 1)
    return fitted_uv, fitted_mask


def get_uv_texture(uv_bases, para_tex_dict):
    region_names = uv_bases.keys()
    full_uv = None
    full_mask = None
    for region_name in region_names:
        uv_basis = uv_bases[region_name]
        para_tex = para_tex_dict[region_name]
        fitted_uv, fitted_mask = get_region_uv_texture(uv_basis, para_tex)
        if full_uv is None:
            full_uv = fitted_uv
            full_mask = fitted_mask
        else:
            full_uv = full_uv + fitted_uv
            full_mask = full_mask + fitted_mask
    full_mask = tf.where(
        tf.greater(full_mask, 0.5), tf.ones_like(full_mask), tf.zeros_like(full_mask)
    )
    return full_uv, full_mask


def scatter_nd_numpy(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    indices_y, indices_x = np.split(indices, 2, axis=1)
    indices_y = np.squeeze(indices_y).tolist()
    indices_x = np.squeeze(indices_x).tolist()
    tuple_indices = [indices_y, indices_x]
    np.add.at(target, tuple_indices, updates)
    return target


# build output from basis
def construct(config, para, uv_size):
    # TODO: add comments
    mu = config["mu"]
    basis = config["basis"]
    indices = config["indices"]
    result = np.matmul(para, basis) + mu
    result = np.reshape(result, [-1, 3])
    if "weight" in config:
        weight = config["weight"]
        result = result * weight
    uv_map = scatter_nd_numpy(indices, result, (uv_size, uv_size, 3))
    uv_mask = scatter_nd_numpy(indices, np.ones_like(result), (uv_size, uv_size, 3))
    uv_mask = np.clip(uv_mask, 0, 1)
    return uv_map, uv_mask


def construct_mask(config, uv_size):
    v_mask = np.ones_like(config["mu"])
    indices = config["indices"]
    v_mask = np.reshape(v_mask, [-1, 3])
    uv_mask = scatter_nd_numpy(indices, v_mask, (uv_size, uv_size, 3))
    uv_mask = np.clip(uv_mask, 0, 1)
    return uv_mask


def np_get_uv_texture(uv_region_bases, para_tex_dict, uv_size=2048):
    uv_map = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
    uv_mask = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
    for key in para_tex_dict:
        region_basis = uv_region_bases[key]
        para = para_tex_dict[key]
        region_uv, region_mask = construct(region_basis, para, uv_size)
        uv_map = uv_map + region_uv
        uv_mask = uv_mask + region_mask
    uv_mask = np.clip(uv_mask, 0, 1)
    return uv_map, uv_mask


def np_get_region_weight_mask(uv_region_bases, region_weight_dict, uv_size=512):
    uv_mask = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
    for key in region_weight_dict:
        region_basis = uv_region_bases[key]
        weight = region_weight_dict[key]
        region_mask = construct_mask(region_basis, uv_size)
        uv_mask = uv_mask + region_mask * weight
    return uv_mask


def np_get_geometry(basis3dmm, para_shape, para_exp=None):
    """compute the geometry according to the 3DMM parameters.
    para_shape: shape parameter
    para_exp: expression parameter
    """
    shape_inc = np.matmul(para_shape, basis3dmm["basis_shape"])
    geo = basis3dmm["mu_shape"] + shape_inc

    if para_exp is not None:
        exp_inc = np.matmul(para_exp, basis3dmm["basis_exp"])
        geo = geo + exp_inc
    return np.reshape(geo, [-1, basis3dmm["basis_shape"].shape[1] / 3, 3])


def np_get_texture(basis3dmm, para_tex):
    """compute the geometry according to the 3DMM parameters.
    para_tex: ver color parameter
    """
    tex_inc = np.matmul(para_tex, basis3dmm["basis_tex"])
    tex = basis3dmm["mu_tex"] + tex_inc
    tex = np.clip(tex, 0, 255)
    return np.reshape(tex, [-1, basis3dmm["basis_tex"].shape[1] / 3, 3])


if __name__ == "__main__":

    # basis3dmm = load_3dmm_basis('../resources/large_next_model_with_86pts_bfm_20190612.npy',
    #        '/dockerdata/yajingchen/RenderUV/basis/codes/tga_weighted_region_bases',
    #        '/dockerdata/yajingchen/RenderUV/tri_20191224.npz',
    #        '/dockerdata/yajingchen/RenderUV/tri_t_20191224.npz',
    #        '/dockerdata/yajingchen/RenderUV/vt_20191224.npz')

    basis3dmm = load_3dmm_basis(
        "../resources/shape_exp_bases_20200424.mat",
        "../resources/uv_tex_basis_20200430_d100.mat",
        tri_vt_path="/dockerdata/yajingchen/RenderUV/tri_t_20191224.npz",
        vt_path="/dockerdata/yajingchen/RenderUV/vt_20191224.npz",
        is_new=True,
    )

    # para_tex = np.load('../../results/eval_results/lihom/wanglihong_tex.npy')
    para_tex = np.random.normal(size=[1, basis3dmm["uv"]["basis"].shape[0]])
    print(para_tex.shape)
    tex = np.matmul(para_tex, basis3dmm["uv"]["basis"]) + basis3dmm["uv"]["mu"]
    # uv = construct(basis3dmm['uv'], para_tex, 512)
    print(np.amax(tex), np.amin(tex))
