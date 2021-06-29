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
unwrap the input images into UV maps, regardless of the input image size and the UV map size
"""
from absl import flags
import tensorflow as tf
import numpy as np
import sys
import os
import glob
import scipy.io
import skimage.io
from PIL import Image

sys.path.append("..")
import utils.unwrap_utils as unwrap_utils
from utils.basis import load_3dmm_basis, get_geometry
from utils.const import *
from utils.misc import tf_blend_uv


def main(_):
    # load 3dmm
    basis3dmm = load_3dmm_basis(
        FLAGS.basis3dmm_path,
        FLAGS.uv_path,
        is_whole_uv=True,
    )

    if os.path.exists(FLAGS.output_dir) is False:
        os.makedirs(FLAGS.output_dir)

    """ build graph """
    front_image_batch = tf.placeholder(
        dtype=tf.float32, shape=[1, None, None, 3], name="front_image"
    )
    front_image_batch_resized = tf.image.resize_images(
        front_image_batch, (FLAGS.uv_size, FLAGS.uv_size)
    )
    front_seg_batch = tf.placeholder(
        dtype=tf.float32, shape=[1, None, None, 19], name="front_seg"
    )
    front_proj_xyz_batch = tf.placeholder(
        dtype=tf.float32,
        shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
        name="front_proj_xyz",
    )
    front_ver_norm_batch = tf.placeholder(
        dtype=tf.float32,
        shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
        name="front_ver_norm",
    )

    base_uv_path = "../resources/base_tex.png"
    base_uv = Image.open(base_uv_path).resize((FLAGS.uv_size, FLAGS.uv_size))
    base_uv = np.asarray(base_uv, np.float32) / 255
    base_uv_batch = tf.constant(base_uv[np.newaxis, ...], name="base_uv")

    if FLAGS.is_mult_view:
        left_image_batch = tf.placeholder(
            dtype=tf.float32, shape=[1, None, None, 3], name="left_image"
        )
        left_image_batch_resized = tf.image.resize_images(
            left_image_batch, (FLAGS.uv_size, FLAGS.uv_size)
        )
        left_seg_batch = tf.placeholder(
            dtype=tf.float32, shape=[1, None, None, 19], name="left_seg"
        )
        left_proj_xyz_batch = tf.placeholder(
            dtype=tf.float32,
            shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
            name="left_proj_xyz",
        )
        left_ver_norm_batch = tf.placeholder(
            dtype=tf.float32,
            shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
            name="left_ver_norm",
        )

        right_image_batch = tf.placeholder(
            dtype=tf.float32, shape=[1, None, None, 3], name="right_image"
        )
        right_image_batch_resized = tf.image.resize_images(
            right_image_batch, (FLAGS.uv_size, FLAGS.uv_size)
        )
        right_seg_batch = tf.placeholder(
            dtype=tf.float32, shape=[1, None, None, 19], name="right_seg"
        )
        right_proj_xyz_batch = tf.placeholder(
            dtype=tf.float32,
            shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
            name="right_proj_xyz",
        )
        right_ver_norm_batch = tf.placeholder(
            dtype=tf.float32,
            shape=[1, basis3dmm["basis_shape"].shape[1] // 3, 3],
            name="right_ver_norm",
        )

        # read fixed blending masks for multiview
        front_mask_path = "../resources/mid_blend_mask.png"
        left_mask_path = "../resources/left_blend_mask.png"
        right_mask_path = "../resources/right_blend_mask.png"
        front_mask = (
            np.asarray(
                Image.open(front_mask_path).resize((FLAGS.uv_size, FLAGS.uv_size)),
                np.float32,
            )
            / 255
        )
        left_mask = (
            np.asarray(
                Image.open(left_mask_path).resize((FLAGS.uv_size, FLAGS.uv_size)),
                np.float32,
            )
            / 255
        )
        right_mask = (
            np.asarray(
                Image.open(right_mask_path).resize((FLAGS.uv_size, FLAGS.uv_size)),
                np.float32,
            )
            / 255
        )
        mask_front_batch = tf.constant(
            front_mask[np.newaxis, ..., np.newaxis], tf.float32, name="mask_front"
        )
        mask_left_batch = tf.constant(
            left_mask[np.newaxis, ..., np.newaxis], tf.float32, name="mask_left"
        )
        mask_right_batch = tf.constant(
            right_mask[np.newaxis, ..., np.newaxis], tf.float32, name="mask_right"
        )

    front_uv_batch, front_uv_mask_batch = unwrap_utils.unwrap_img_into_uv(
        front_image_batch_resized / 255.0,
        front_proj_xyz_batch * FLAGS.uv_size / 300,
        front_ver_norm_batch,
        basis3dmm,
        FLAGS.uv_size,
    )

    front_uv_seg_batch, _ = unwrap_utils.unwrap_img_into_uv(
        front_seg_batch,
        front_proj_xyz_batch,
        front_ver_norm_batch,
        basis3dmm,
        FLAGS.uv_size,
    )

    if FLAGS.is_mult_view:

        left_uv_batch, left_uv_mask_batch = unwrap_utils.unwrap_img_into_uv(
            left_image_batch_resized / 255.0,
            left_proj_xyz_batch * FLAGS.uv_size / 300,
            left_ver_norm_batch,
            basis3dmm,
            FLAGS.uv_size,
        )

        left_uv_seg_batch, _ = unwrap_utils.unwrap_img_into_uv(
            left_seg_batch,
            left_proj_xyz_batch,
            left_ver_norm_batch,
            basis3dmm,
            FLAGS.uv_size,
        )

        right_uv_batch, right_uv_mask_batch = unwrap_utils.unwrap_img_into_uv(
            right_image_batch_resized / 255.0,
            right_proj_xyz_batch * FLAGS.uv_size / 300,
            right_ver_norm_batch,
            basis3dmm,
            FLAGS.uv_size,
        )

        right_uv_seg_batch, _ = unwrap_utils.unwrap_img_into_uv(
            right_seg_batch,
            right_proj_xyz_batch,
            right_ver_norm_batch,
            basis3dmm,
            FLAGS.uv_size,
        )

        # blend multiview
        left_uv_seg_mask_batch = unwrap_utils.get_mask_from_seg(left_uv_seg_batch)
        right_uv_seg_mask_batch = unwrap_utils.get_mask_from_seg(right_uv_seg_batch)
        front_uv_seg_mask_batch = unwrap_utils.get_mask_from_seg(front_uv_seg_batch)

        cur_seg = tf_blend_uv(
            left_uv_seg_mask_batch,
            right_uv_seg_mask_batch,
            mask_right_batch,
            match_color=False,
        )
        uv_seg_mask_batch = tf_blend_uv(
            cur_seg, front_uv_seg_mask_batch, mask_front_batch, match_color=False
        )

        mask_batch = tf.clip_by_value(
            mask_front_batch + mask_left_batch + mask_right_batch, 0, 1
        )
        uv_mask_batch = mask_batch * uv_seg_mask_batch
        cur_uv = tf_blend_uv(
            left_uv_batch, right_uv_batch, mask_right_batch, match_color=False
        )
        cur_uv = tf_blend_uv(
            cur_uv, front_uv_batch, mask_front_batch, match_color=False
        )
        uv_batch = tf_blend_uv(base_uv_batch, cur_uv, uv_mask_batch, match_color=True)

    else:
        uv_seg_mask_batch = unwrap_utils.get_mask_from_seg(front_uv_seg_batch)
        uv_mask_batch = front_uv_mask_batch * uv_seg_mask_batch
        uv_batch = tf_blend_uv(
            base_uv_batch, front_uv_batch, uv_mask_batch, match_color=True
        )

    uv_batch = tf.identity(uv_batch, name="uv_tex")
    uv_seg_mask_batch = tf.identity(uv_seg_mask_batch, name="uv_seg")
    uv_mask_batch = tf.identity(uv_mask_batch, name="uv_mask")

    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    if FLAGS.write_graph:
        tf.train.write_graph(sess.graph_def, "", FLAGS.pb_path, as_text=True)
        exit()

    """ load data  """
    # seg: [300,300,19], segmentation
    # diffuse: [300,300,3], diffuse images
    # proj_xyz: [N,3]
    # ver_norm: [N,3]
    info_paths = glob.glob(os.path.join(FLAGS.input_dir, "*texture.mat"))

    for info_path in info_paths:
        info = scipy.io.loadmat(info_path)

        if FLAGS.is_mult_view:
            assert info["proj_xyz"].shape[0] >= 3  # front, left, right
            if FLAGS.is_orig_img:
                front_img = info["ori_img"][0][np.newaxis, ...]
                left_img = info["ori_img"][1][np.newaxis, ...]
                right_img = info["ori_img"][2][np.newaxis, ...]
            else:
                front_img = info["diffuse"][0][np.newaxis, ...]
                left_img = info["diffuse"][1][np.newaxis, ...]
                right_img = info["diffuse"][2][np.newaxis, ...]

            uv_tex_res, uv_mask_res = sess.run(
                [uv_batch, uv_mask_batch],
                {
                    front_image_batch: front_img,
                    front_proj_xyz_batch: info["proj_xyz"][0:1, ...],
                    front_ver_norm_batch: info["ver_norm"][0:1, ...],
                    front_seg_batch: info["seg"][0:1, ...],
                    left_image_batch: left_img,
                    left_proj_xyz_batch: info["proj_xyz"][1:2, ...],
                    left_ver_norm_batch: info["ver_norm"][1:2, ...],
                    left_seg_batch: info["seg"][1:2, ...],
                    right_image_batch: right_img,
                    right_proj_xyz_batch: info["proj_xyz"][2:3, ...],
                    right_ver_norm_batch: info["ver_norm"][2:3, ...],
                    right_seg_batch: info["seg"][2:3, ...],
                },
            )
        else:
            assert info["proj_xyz"].shape[0] >= 1
            if FLAGS.is_orig_img:
                front_img = info["ori_img"][0][np.newaxis, ...]
            else:
                front_img = info["diffuse"][0][np.newaxis, ...]

            uv_tex_res, uv_mask_res = sess.run(
                [uv_batch, uv_mask_batch],
                {
                    front_image_batch: front_img,
                    front_proj_xyz_batch: info["proj_xyz"][0:1, ...],
                    front_ver_norm_batch: info["ver_norm"][0:1, ...],
                    front_seg_batch: info["seg"][0:1, ...],
                },
            )

        uv_tex_res = uv_tex_res[0]
        uv_mask_res = uv_mask_res[0]

        prefix = info_path.split("/")[-1].split(".")[0]
        uv_tex_res = uv_tex_res * 255
        uv_mask_res = uv_mask_res * 255
        Image.fromarray(uv_tex_res.astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + "_tex.png")
        )
        Image.fromarray(np.squeeze(uv_mask_res).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + "_mask.png")
        )
        sess.close()


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("input_dir", "unwrap_data_for_debug", "input data directory")
    flags.DEFINE_string(
        "output_dir", "unwrap_output_for_debug", "output data directory"
    )
    flags.DEFINE_boolean("write_graph", False, "write graph")
    flags.DEFINE_boolean("is_mult_view", True, "is multi view")
    flags.DEFINE_boolean(
        "is_orig_img", False, "is using original cropped image / is using resized image"
    )
    flags.DEFINE_integer("uv_size", 512, "uv size")
    flags.DEFINE_string("pb_path", "path to pb file", "output pb file")

    flags.DEFINE_string(
        "basis3dmm_path",
        "../resources/large_next_model_with_86pts_bfm_20190612.npy",
        "basis3dmm path",
    )
    flags.DEFINE_string("uv_path", "../resources/uv_bases", "basis3dmm path")

    tf.app.run(main)
