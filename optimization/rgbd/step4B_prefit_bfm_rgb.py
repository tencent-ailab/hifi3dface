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
import scipy.io as scio
import cv2
import tensorflow as tf
import tensorflow.contrib.opt as tf_opt
import os
from absl import app, flags
import sys

sys.path.append("../..")

from RGBD_load import RGBD_load
from utils.render_img import render_img_in_different_pose
from utils.compute_loss import compute_loss
from utils.ply import write_ply, write_obj

from utils.basis import load_3dmm_basis, load_3dmm_basis_bfm


def define_variable(num_of_img, imageH, imageW, para_shape_shape, para_tex_shape, info):

    # variable-trainable=False
    image_batch = tf.get_variable(
        shape=[num_of_img, imageH, imageW, 3],
        dtype=tf.float32,
        name="ori_img",
        trainable=False,
        initializer=tf.constant_initializer(info["img_list"] / 255.0),
    )

    depth_image_batch = tf.get_variable(
        shape=[num_of_img, imageH, imageW],
        dtype=tf.float32,
        name="depth_img",
        trainable=False,
        initializer=tf.constant_initializer(info["dep_list"]),
    )

    input_scale_var = tf.get_variable(
        shape=[1, 1],
        dtype=tf.float32,
        name="input_scale_var",
        trainable=False,
        initializer=tf.constant_initializer(info["input_scale"]),
    )

    segmentation = tf.get_variable(
        shape=[num_of_img, imageH, imageW, 19],
        dtype=tf.float32,
        name="face_segmentation",
        trainable=False,
        initializer=tf.constant_initializer(info["seg_list"]),
    )

    lmk_86_3d_batch = tf.get_variable(
        shape=[num_of_img, 86, 2],
        dtype=tf.float32,
        name="lmk_86_3d_batch",
        trainable=False,
        initializer=tf.constant_initializer(info["lmk_list3D"]),
    )

    K = tf.get_variable(
        shape=[4, 3, 3],
        dtype=tf.float32,
        name="K",
        trainable=False,
        initializer=tf.constant_initializer(info["K_list"]),
    )

    pose6 = tf.get_variable(
        shape=[num_of_img, 6, 1],
        dtype=tf.float32,
        name="para_pose6",
        trainable=False,
        initializer=tf.constant_initializer(info["se3_list"]),
    )

    para_shape = tf.get_variable(
        shape=[1, para_shape_shape],
        dtype=tf.float32,
        name="para_shape",
        trainable=False,
        initializer=tf.constant_initializer(info["para_shape"]),
    )

    # variable-trainable=True
    para_tex = tf.get_variable(
        shape=[1, para_tex_shape],
        dtype=tf.float32,
        name="para_tex",
        trainable=True,
        initializer=tf.zeros_initializer(),
    )

    para_illum = tf.get_variable(
        shape=[num_of_img, 27],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        name="para_illum",
        trainable=True,
    )

    var_list = {
        "image_batch": image_batch,
        "depth_image_batch": tf.reshape(
            depth_image_batch, [num_of_img, imageH, imageW, 1]
        ),
        "para_shape": tf.concat([para_shape] * num_of_img, axis=0),
        "para_tex": tf.concat([para_tex] * num_of_img, axis=0),
        "lmk_86_3d_batch": lmk_86_3d_batch,
        "pose6": pose6,
        "para_illum": para_illum,
        "segmentation": segmentation,
        "K": K,
        "input_scale": input_scale_var,
    }

    return var_list


def build_RGBD_opt_graph(var_list, basis3dmm, imageH, imageW):

    # render ori img to fit pose, light, shape, tex
    # render three fake imgs with fake light and poses for ID loss
    (
        render_img_in_ori_pose,
        render_img_in_fake_pose_M,
        render_img_in_fake_pose_L,
        render_img_in_fake_pose_R,
    ) = render_img_in_different_pose(
        var_list,
        basis3dmm,
        "Pers",
        imageH,
        imageW,
        opt_type="RGBD",
        is_bfm=FLAGS.is_bfm,
        scale=var_list["input_scale"][0, 0],
    )

    # compute loss (include depth loss)
    global_step = tf.Variable(0, name="global_step_train", trainable=False)

    tot_loss, tot_loss_illum = compute_loss(
        FLAGS,
        basis3dmm,
        var_list,
        render_img_in_ori_pose,
        render_img_in_fake_pose_M,
        render_img_in_fake_pose_L,
        render_img_in_fake_pose_R,
        opt_type="RGBD",
        global_step=global_step,
    )

    # optimizer
    learning_rate = tf.maximum(
        tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.lr_decay_step, FLAGS.lr_decay_rate
        ),
        FLAGS.min_learning_rate,
    )
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)

    gvs_illum = optim.compute_gradients(tot_loss_illum)
    gvs = optim.compute_gradients(tot_loss)

    capped_gvs_illum = []
    for grad, var in gvs_illum:
        if grad is not None:
            if var.name.startswith("para_illum"):
                print("optimizing", var.name)
                capped_gvs_illum.append((tf.clip_by_value(grad, -1.0, 1.0), var))

    capped_gvs = []
    for grad, var in gvs:
        if grad is not None:
            if var.name.startswith("para_illum") is False:
                print("optimizing", var.name)
                if var.name.startswith("para_pose"):
                    capped_gvs.append((tf.clip_by_value(grad, -1.0, 1.0) * 0.0001, var))
                else:
                    capped_gvs.append((tf.clip_by_value(grad, -1.0, 1.0), var))

    capped_gvs = capped_gvs + capped_gvs_illum
    train_op = optim.apply_gradients(
        capped_gvs, global_step=global_step, name="train_op"
    )

    out_list = {
        "para_tex": var_list["para_tex"],
        "train_op": train_op,
    }
    return out_list


def RGBD_opt():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO

    # load data
    basis3dmm = load_3dmm_basis_bfm(FLAGS.basis3dmm_path)

    para_tex_init = np.zeros([1, basis3dmm["basis_tex"].shape[0]])  # 1 * 80
    np.save((os.path.join(FLAGS.prefit_dir, "para_tex_init.npy")), para_tex_init)

    # load_RGBD_data, sequence index is: mid --  left -- right -- up
    info = RGBD_load.load_and_preprocess_RGBD_data(
        FLAGS.prefit_dir, FLAGS.prepare_dir, basis3dmm
    )

    imageH = info["height"]
    imageW = info["width"]
    para_shape_shape = info["para_shape"].shape[1]
    para_tex_shape = info["para_tex"].shape[1]

    # build graph
    var_list = define_variable(
        FLAGS.num_of_img, imageH, imageW, para_shape_shape, para_tex_shape, info
    )

    out_list = build_RGBD_opt_graph(var_list, basis3dmm, imageH, imageW)

    # summary_op
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    if os.path.exists(FLAGS.summary_dir) is False:
        os.makedirs(FLAGS.summary_dir)

    # start opt
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction=0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        import time

        starttime = time.time()

        for step in range(FLAGS.train_step):

            if (step % FLAGS.log_step == 0) | (step == FLAGS.train_step - 1):
                out_summary = sess.run(summary_op)
                summary_writer.add_summary(out_summary, step)
                print(step)

                endtime = time.time()
                print("time:" + str(endtime - starttime))
                starttime = time.time()

            sess.run(out_list["train_op"])

        para_tex_out = sess.run(out_list["para_tex"])
        para_tex_out = np.reshape(para_tex_out[0], [1, -1])
        np.save((os.path.join(FLAGS.prefit_dir, "para_tex_init.npy")), para_tex_out)


def main(_):
    RGBD_opt()


if __name__ == "__main__":
    ##FLAGS
    FLAGS = flags.FLAGS

    """ data load parameter """
    flags.DEFINE_boolean("is_bfm", True, "default: True here")
    flags.DEFINE_string(
        "basis3dmm_path", "../../resources/BFM2009_Model.mat", "basis3dmm path"
    )
    flags.DEFINE_string("uv_path", "../../resources/whole_uv512.mat", "uv base")

    flags.DEFINE_string(
        "vggpath", "../../resources/vgg-face.mat", "checkpoint director for vgg"
    )

    flags.DEFINE_string(
        "prefit_dir",
        "../../test_data_debug/prefit_bfm",
        "init poses, shape paras and images .mat file directory",
    )
    flags.DEFINE_string(
        "prepare_dir",
        "../../test_data_debug/prepare",
        "init poses, shape paras and images .mat file directory",
    )

    """ opt parameters """
    flags.DEFINE_string("GPU_NO", "7", "which GPU")

    flags.DEFINE_integer("num_of_img", 4, "")

    flags.DEFINE_float("photo_weight", 10, "")
    flags.DEFINE_float("gray_photo_weight", 0, "")
    flags.DEFINE_float("id_weight", 0, "")
    flags.DEFINE_float("reg_shape_weight", 0, "")
    flags.DEFINE_float("reg_tex_weight", 0, "")
    flags.DEFINE_float("depth_weight", 0, "")
    flags.DEFINE_float("real_86pt_lmk3d_weight", 0, "")
    flags.DEFINE_float("lmk_struct_weight", 0.0, "")

    flags.DEFINE_integer("log_step", 20, "")
    flags.DEFINE_integer("train_step", 100, "")

    flags.DEFINE_float("learning_rate", 0.05, "string : path for 3dmm")
    flags.DEFINE_integer("lr_decay_step", 10, "string : path for 3dmm")
    flags.DEFINE_float("lr_decay_rate", 0.9, "string : path for 3dmm")
    flags.DEFINE_float("min_learning_rate", 0.0000001 * 10, "string : path for 3dmm")

    """ opt out directory """
    flags.DEFINE_string(
        "summary_dir",
        "../../test_data_debug/test_bfm_color_fit2/sum",
        "summary directory",
    )

    app.run(main)
