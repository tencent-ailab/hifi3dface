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
prepare data for pix2pix inference.

use current fitting approach to fit the unwrapped images

Goal: in inference time, we can get a clean (smoothy and beautiful) and clear (sharp) UV texture and normal image.
"""
import tensorflow as tf
import os
import cv2
import time
import numpy as np
from PIL import Image
import glob
import json
import scipy.io
import skimage
import sys

from absl import app, flags, logging

sys.path.append("..")
from utils.basis import (
    load_3dmm_basis,
    get_uv_texture,
    np_get_uv_texture,
    np_get_region_weight_mask,
)
from utils.losses import Losses
from utils.misc import blend_uv
from utils.tf_LP import TF_LaplacianPyramid as tf_LP


def get_weighted_photo_mask(uv_region_bases):
    weight_mask_dict = {
        "eye": 0.00001,
        "eyebrow": 30.0,
        "cheek": 1.0,
        "nose": 0.1,
        "nosetip": 0.1,
        "mouth": 10.0,
        "jaw": 1.0,
        "contour": 0.1,
    }
    photo_weight_mask = np_get_region_weight_mask(
        uv_region_bases, weight_mask_dict, uv_size=512
    )
    return photo_weight_mask


def save_params():

    if os.path.exists(FLAGS.out_dir) is False:
        os.makedirs(FLAGS.out_dir)

    params = {
        "photo_weight": FLAGS.photo_weight,
        "uv_tv_weight": FLAGS.uv_tv_weight,
        "uv_reg_tex_weight": FLAGS.uv_reg_tex_weight,
        "lr": FLAGS.learning_rate,
    }
    with open(os.path.join(FLAGS.out_dir, "para.json"), "w") as fp:
        json.dump(params, fp)


def main(_):

    # save parameters
    save_params()

    mask_batch = tf.placeholder(
        dtype=tf.float32, shape=[1, 512, 512, 1], name="uv_mask"
    )
    tex_batch = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 3], name="uv_tex")

    var_mask_batch = tf.get_variable(
        shape=[1, 512, 512, 1], dtype=tf.float32, name="var_mask", trainable=False
    )
    var_tex_batch = tf.get_variable(
        shape=[1, 512, 512, 3], dtype=tf.float32, name="var_tex", trainable=False
    )

    assign_op = tf.group(
        [tf.assign(var_mask_batch, mask_batch), tf.assign(var_tex_batch, tex_batch)],
        name="assign_op",
    )

    # arrange images by name (load 3dmm)
    basis3dmm = load_3dmm_basis(
        FLAGS.basis3dmm_path,
        FLAGS.uv_path,
        uv_weight_mask_path=FLAGS.uv_weight_mask_path,
        is_train=False,
        is_whole_uv=False,
    )

    # build fitting graph
    uv_region_bases = basis3dmm["uv"]
    para_uv_dict = {}
    for region_name in uv_region_bases:
        region_basis = uv_region_bases[region_name]
        para = tf.get_variable(
            shape=[1, region_basis["basis"].shape[0]],
            initializer=tf.zeros_initializer(),
            name="para_" + region_name,
        )
        para_uv_dict[region_name] = para

    uv_rgb, uv_mask = get_uv_texture(uv_region_bases, para_uv_dict)
    photo_weight_mask = get_weighted_photo_mask(uv_region_bases)

    # build fitting loss
    tot_loss = 0.0
    loss_str = ""

    if FLAGS.photo_weight > 0:
        photo_loss = Losses.ws_photo_loss(
            var_tex_batch, uv_rgb / 255.0, uv_mask * photo_weight_mask * var_mask_batch
        )
        photo_loss = tf.identity(photo_loss, name="photo_loss")
        tot_loss = tot_loss + photo_loss * FLAGS.photo_weight
        loss_str = "photo:{}"

    if FLAGS.uv_tv_weight > 0:
        uv_tv_loss = Losses.uv_tv_loss2(
            uv_rgb / 255, uv_mask, basis3dmm["uv_weight_mask"]
        )
        uv_tv_loss = tf.identity(uv_tv_loss, name="uv_tv_loss")
        tot_loss = tot_loss + uv_tv_loss * FLAGS.uv_tv_weight
        loss_str = loss_str + ";tv:{}"

    if FLAGS.uv_reg_tex_weight > 0:
        uv_reg_tex_loss = 0.0
        for key in para_uv_dict:
            para = para_uv_dict[key]
            reg_region_loss = Losses.reg_loss(para)
            uv_reg_tex_loss = uv_reg_tex_loss + reg_region_loss
        uv_reg_tex_loss = uv_reg_tex_loss / len(para_uv_dict)
        uv_reg_tex_loss = tf.identity(uv_reg_tex_loss, name="uv_reg_tex_loss")
        tot_loss = tot_loss + uv_reg_tex_loss * FLAGS.uv_reg_tex_weight
        loss_str = loss_str + ";reg:{}"

    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optim.minimize(tot_loss, name="train_op")
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        if FLAGS.write_graph:
            tf.train.write_graph(sess.graph_def, "", FLAGS.pb_path, as_text=True)
            exit()

        uv_paths = sorted(glob.glob(os.path.join(FLAGS.data_dir, "*tex.png")))
        mask_paths = sorted(glob.glob(os.path.join(FLAGS.data_dir, "*mask.png")))
        # mask_paths = ['../resources/mask_front_face.png'] * len(uv_paths)
        # uv_paths = sorted(glob.glob(os.path.join(FLAGS.data_dir,'*delight_512.png')))
        # mask_paths = sorted(glob.glob(os.path.join(FLAGS.data_dir, '*mask_512.png')))

        # base uv
        base_uv = Image.open("../resources/base_tex.png")
        base_uv = np.asarray(base_uv, np.float32) / 255.0
        base_normal = Image.open("../resources/base_normal.png")
        base_normal = np.asarray(base_normal, np.float32) / 255.0

        for uv_path, mask_path in zip(uv_paths, mask_paths):
            uv_input = np.asarray(Image.open(uv_path)).astype(np.float32) / 255.0
            mask_input = np.asarray(Image.open(mask_path)).astype(np.float32) / 255.0
            if mask_input.shape[0] != 512:
                mask_input = cv2.resize(mask_input, (512, 512))
            if uv_input.shape[0] != 512:
                uv_input = cv2.resize(uv_input, (512, 512))

            if len(mask_input.shape) != 3:
                mask_input = mask_input[..., np.newaxis]
            mask_input = mask_input[:, :, 0:1]

            uv_input = uv_input[np.newaxis, ...]
            mask_input = mask_input[np.newaxis, ...]

            sess.run(init_op)
            sess.run(assign_op, {tex_batch: uv_input, mask_batch: mask_input})

            for i in range(FLAGS.train_step):
                # l1, l2, l3, l4, l5, _ = sess.run([tot_loss, photo_loss, uv_tv_loss, uv_reg_tex_loss, uv_consistency_loss, train_op])
                l1, l2, l3, l4, _ = sess.run(
                    [tot_loss, photo_loss, uv_tv_loss, uv_reg_tex_loss, train_op]
                )
                if i == 1:
                    start_time = time.time()
                if i % 100 == 0:
                    print(i, loss_str.format(l1, l2, l3, l4))

            para_out_dict = {}
            para_out_dict = sess.run(para_uv_dict)

            face_uv, face_mask = np_get_uv_texture(basis3dmm["uv2k"], para_out_dict)
            face_normal, face_mask = np_get_uv_texture(
                basis3dmm["normal2k"], para_out_dict
            )

            face_uv = np.clip(face_uv, 0, 255)
            face_normal = np.clip(face_normal, 0, 255)
            face_mask = np.clip(face_mask, 0, 1)

            prefix = uv_path.split("/")[-1].split(".")[0]
            print(prefix)

            # Image.fromarray(face_uv.astype(np.uint8)).save(
            #    os.path.join(FLAGS.out_dir, prefix + '_face_uv.png'))

            out_uv = blend_uv(base_uv, face_uv / 255, face_mask, True)
            out_normal = blend_uv(base_normal, face_normal / 255, face_mask, False)
            out_uv = np.clip(out_uv * 255, 0, 255)
            out_normal = np.clip(out_normal * 255, 0, 255)

            out_uv = Image.fromarray(out_uv.astype(np.uint8))
            out_normal = Image.fromarray(out_normal.astype(np.uint8))
            out_uv.save(os.path.join(FLAGS.out_dir, prefix + "_D.png"))
            out_normal.save(os.path.join(FLAGS.out_dir, prefix + "_N.png"))


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "basis3dmm_path",
        "../resources/large_next_model_with_86pts_bfm_20190612.npy",
        "basis3dmm path",
    )
    flags.DEFINE_string("uv_path", "../resources/uv_bases200", "basis3dmm path")

    flags.DEFINE_string(
        "uv_weight_mask_path",
        "../resources/uv_boundary_masks/boundary_512.png",
        "uv weight mask",
    )

    flags.DEFINE_string("data_dir", "unwrap_output_for_debug", "uv data directory")
    flags.DEFINE_string("out_dir", "tmp_out", "fitted output directory")

    flags.DEFINE_float("photo_weight", 1.0, "")
    flags.DEFINE_float("uv_tv_weight", 1.0, "")
    flags.DEFINE_float("uv_reg_tex_weight", 0.001, "")

    flags.DEFINE_integer("train_step", 200, "for each 200epoch save one time")
    flags.DEFINE_float("learning_rate", 0.1, "string : path for 3dmm")

    flags.DEFINE_boolean("write_graph", False, "if true, write graph")
    flags.DEFINE_string("pb_path", "", "path to pb file")

    app.run(main)
