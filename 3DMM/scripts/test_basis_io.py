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
import scipy.io
from PIL import Image

import sys

sys.path.append("../..")

from third_party.ply import write_obj


def load_3dmm_basis(basis_path, uv_path=None, is_whole_uv=True, limit_dim=-1):
    """load 3dmm basis and other useful files.

    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param is_whole_uv:
        - bool. indicate whether we use albedo whole uv bases or regional pyramid bases.
    :param limit_dim:
        - int. the number of dimension is used for the geometry bases. Default: 1, indicating using all dimensions.

    """

    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm["keypoints"] = np.squeeze(basis3dmm["keypoints"])

    # load uv basis
    if uv_path is not None and not is_whole_uv:
        uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv512.mat")))
        uv_region_bases = {}
        for region_path in uv_region_paths:
            print("loading %s" % region_path)
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

        uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv.mat")))
        uv_region_bases = {}
        for region_path in uv_region_paths:
            print("loading %s" % region_path)
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

        normal_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_normal.mat")))
        normal_region_bases = {}
        for region_path in normal_region_paths:
            print("loading %s" % region_path)
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

    if limit_dim > 0 and limit_dim < basis3dmm["basis_shape"].shape[0]:
        basis3dmm["basis_shape"] = basis3dmm["basis_shape"][:limit_dim, :]

    return basis3dmm


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
    uv_map = np.clip(uv_map, 0, 255)
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
    return np.reshape(geo, [-1, basis3dmm["basis_shape"].shape[1] // 3, 3])


def np_get_texture(basis3dmm, para_tex):
    """compute the geometry according to the 3DMM parameters.
    para_tex: ver color parameter
    """
    tex_inc = np.matmul(para_tex, basis3dmm["basis_tex"])
    tex = basis3dmm["mu_tex"] + tex_inc
    tex = np.clip(tex, 0, 255)
    return np.reshape(tex, [-1, basis3dmm["basis_tex"].shape[1] // 3, 3])


if __name__ == "__main__":

    # load basis (albedo is whole face)
    basis3dmm = load_3dmm_basis(
        "../files/AI-NExT-Shape.mat",
        "../files/AI-NExT-Albedo-Global.mat",
        is_whole_uv=True,
        limit_dim=-1,
    )

    # randomly generate results for each basis
    rand_para_shape = np.random.normal(size=[1, basis3dmm["basis_shape"].shape[0]])
    rand_para_uv = np.random.normal(size=[1, basis3dmm["uv"]["basis"].shape[0]])

    ver_shape = np_get_geometry(basis3dmm, rand_para_shape)[0]
    uv_texture, _ = construct(basis3dmm["uv"], rand_para_uv, 512)

    print(basis3dmm["vt_list"].shape, basis3dmm["tri"].shape, basis3dmm["tri_vt"].shape)

    # save all files
    write_obj(
        "rand_shape.obj",
        ver_shape,
        basis3dmm["vt_list"],
        basis3dmm["tri"],
        basis3dmm["tri_vt"],
        "face.mtl",
    )
    Image.fromarray(uv_texture.astype(np.uint8)).save("rand_uv.png")

    # load regional pyramid bases (it takes a long time because files are huge)
    basis3dmm = load_3dmm_basis(
        "../files/AI-NExT-Shape-NoAug.mat",
        "../files/AI-NExT-AlbedoNormal-RPB",
        is_whole_uv=False,
        limit_dim=-1,
    )

    rand_para_uv_dict = {}
    for region_name in basis3dmm["uv"]:
        rand_para = np.random.normal(
            size=[1, basis3dmm["uv"][region_name]["basis"].shape[0]]
        )
        rand_para_uv_dict[region_name] = rand_para

    uv_tex512, _ = np_get_uv_texture(basis3dmm["uv"], rand_para_uv_dict, 512)
    uv_tex2048, _ = np_get_uv_texture(basis3dmm["uv2k"], rand_para_uv_dict, 2048)
    uv_norm2048, _ = np_get_uv_texture(basis3dmm["normal2k"], rand_para_uv_dict, 2048)

    Image.fromarray(uv_tex512.astype(np.uint8)).save("uv_tex512.png")
    Image.fromarray(uv_tex2048.astype(np.uint8)).save("uv_tex2048.png")
    Image.fromarray(uv_norm2048.astype(np.uint8)).save("uv_norm2048.png")
