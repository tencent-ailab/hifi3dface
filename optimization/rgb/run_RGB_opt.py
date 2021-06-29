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

from RGB_load import RGB_load
from utils.render_img import render_img_in_different_pose
from utils.compute_loss import compute_loss
from third_party.ply import write_ply, write_obj

from utils.basis import load_3dmm_basis, load_3dmm_basis_bfm
from PIL import Image


def define_variable(num_of_img, imageH, imageW, para_shape_shape, para_tex_shape, info):

    # variable-trainable=False
    image_batch = tf.get_variable(
        shape=[num_of_img, imageH, imageW, 3],
        dtype=tf.float32,
        name="ori_img",
        trainable=False,
        initializer=tf.constant_initializer(info["img_list"] / 255.0),
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

    lmk_68_2d_batch = tf.get_variable(
        shape=[num_of_img, 68, 2],
        dtype=tf.float32,
        name="lmk_68_2d_batch",
        trainable=False,
        initializer=tf.constant_initializer(info["lmk_list2D"]),
    )

    K = tf.get_variable(
        shape=[1, 3, 3],
        dtype=tf.float32,
        name="K",
        trainable=False,
        initializer=tf.constant_initializer(info["K"]),
    )

    # variable-trainable=True
    pose6 = tf.get_variable(
        shape=[num_of_img, 6, 1],
        dtype=tf.float32,
        name="para_pose6",
        trainable=True,
        initializer=tf.constant_initializer(info["se3_list"]),
    )

    para_shape = tf.get_variable(
        shape=[1, para_shape_shape], dtype=tf.float32, name="para_shape", trainable=True
    )

    para_tex = tf.get_variable(
        shape=[1, para_tex_shape], dtype=tf.float32, name="para_tex", trainable=True
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
        "para_shape": tf.concat([para_shape] * num_of_img, axis=0),
        "para_tex": tf.concat([para_tex] * num_of_img, axis=0),
        "lmk_86_3d_batch": lmk_86_3d_batch,
        "lmk_68_2d_batch": lmk_68_2d_batch,
        "pose6": pose6,
        "para_illum": para_illum,
        "segmentation": segmentation,
        "K": tf.concat([K] * num_of_img, axis=0),
    }

    return var_list


def build_RGB_opt_graph(var_list, basis3dmm, imageH, imageW):

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
        FLAGS.project_type,
        imageH,
        imageW,
        opt_type="RGB",
        is_bfm=FLAGS.is_bfm,
    )

    # compute loss
    tot_loss, tot_loss_illum = compute_loss(
        FLAGS,
        basis3dmm,
        var_list,
        render_img_in_ori_pose,
        render_img_in_fake_pose_M,
        render_img_in_fake_pose_L,
        render_img_in_fake_pose_R,
    )

    # optimizer
    global_step = tf.Variable(0, name="global_step_train", trainable=False)
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
        "para_shape": var_list["para_shape"],
        "para_tex": var_list["para_tex"],
        "ver_xyz": render_img_in_ori_pose["ver_xyz"],
        "tex": render_img_in_ori_pose["tex"],
        "diffuse": render_img_in_ori_pose["diffuse"],
        "proj_xyz": tf.concat(
            [render_img_in_ori_pose["proj_xy"], render_img_in_ori_pose["proj_z"]],
            axis=2,
        ),
        "ver_norm": render_img_in_ori_pose["ver_norm"],
        "train_op": train_op,
    }
    return out_list


def RGB_opt(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO
    # load 3DMM
    if FLAGS.is_bfm is False:
        basis3dmm = load_3dmm_basis(
            FLAGS.basis3dmm_path,
            FLAGS.uv_path,
        )
        para_shape_shape = basis3dmm["basis_shape"].shape[0]
        para_tex_shape = basis3dmm["uv"]["basis"].shape[0]
    else:
        basis3dmm = load_3dmm_basis_bfm(FLAGS.basis3dmm_path)
        para_shape_shape = basis3dmm["basis_shape"].shape[0]
        para_tex_shape = basis3dmm["basis_tex"].shape[0]

    # load RGB data
    info = RGB_load.load_rgb_data(FLAGS.base_dir, FLAGS.project_type, FLAGS.num_of_img)

    imageH = info["img_list"].shape[1]
    imageW = info["img_list"].shape[2]

    # build graph
    var_list = define_variable(
        FLAGS.num_of_img, imageH, imageW, para_shape_shape, para_tex_shape, info
    )

    out_list = build_RGB_opt_graph(var_list, basis3dmm, imageH, imageW)

    # summary_op
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    if os.path.exists(FLAGS.summary_dir) is False:
        os.makedirs(FLAGS.summary_dir)
    if os.path.exists(FLAGS.out_dir) is False:
        os.makedirs(FLAGS.out_dir)

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
                print("step: " + str(step))
                endtime = time.time()
                print("time:" + str(endtime - starttime))
                starttime = time.time()

            if step == FLAGS.train_step - 1 and FLAGS.save_ply:
                print("output_final_result...")
                out_para_shape, out_ver_xyz, out_tex = sess.run(
                    [out_list["para_shape"], out_list["ver_xyz"], out_list["tex"]]
                )
                # output ply
                v_xyz = out_ver_xyz[0]
                if FLAGS.is_bfm is False:
                    uv_map = out_tex[0] * 255.0
                    uv_size = uv_map.shape[0]
                    v_rgb = np.zeros_like(v_xyz) + 200  # N x 3
                    for (v1, v2, v3), (t1, t2, t3) in zip(
                        basis3dmm["tri"], basis3dmm["tri_vt"]
                    ):
                        v_rgb[v1] = uv_map[
                            int((1.0 - basis3dmm["vt_list"][t1][1]) * uv_size),
                            int(basis3dmm["vt_list"][t1][0] * uv_size),
                        ]
                        v_rgb[v2] = uv_map[
                            int((1.0 - basis3dmm["vt_list"][t2][1]) * uv_size),
                            int(basis3dmm["vt_list"][t2][0] * uv_size),
                        ]
                        v_rgb[v3] = uv_map[
                            int((1.0 - basis3dmm["vt_list"][t3][1]) * uv_size),
                            int(basis3dmm["vt_list"][t3][0] * uv_size),
                        ]

                    write_obj(
                        os.path.join(FLAGS.out_dir, "face.obj"),
                        v_xyz,
                        basis3dmm["vt_list"],
                        basis3dmm["tri"].astype(np.int32),
                        basis3dmm["tri_vt"].astype(np.int32),
                    )
                else:
                    v_rgb = out_tex[0] * 255.0

                write_ply(
                    os.path.join(FLAGS.out_dir, "face.ply"),
                    v_xyz,
                    basis3dmm["tri"],
                    v_rgb.astype(np.uint8),
                    True,
                )

                out_diffuse, out_proj_xyz, out_ver_norm = sess.run(
                    [out_list["diffuse"], out_list["proj_xyz"], out_list["ver_norm"]]
                )
                out_diffuse = out_diffuse * 255.0  # RGB 0-255
                scio.savemat(
                    os.path.join(FLAGS.out_dir, "out_for_texture.mat"),
                    {
                        "ori_img": info["img_ori_list"],  # ? x ?
                        "diffuse": out_diffuse,  # 300 x 300
                        "seg": info["seg_list"],  # 300 x 300
                        "proj_xyz": out_proj_xyz,  # in 300 x 300 img
                        "ver_norm": out_ver_norm,
                    },
                )

            sess.run(out_list["train_op"])


if __name__ == "__main__":
    ##FLAGS
    FLAGS = flags.FLAGS
    flags.DEFINE_string("project_type", "Orth", "[Pers, Orth]")

    """ data load parameter """
    flags.DEFINE_string(
        "basis3dmm_path", "../../3DMM/files/AI-NEXT-Shape-NoAug.mat", "basis3dmm path"
    )
    flags.DEFINE_string(
        "uv_path", "../../3DMM/files/AI-NEXT-Albedo-Global.mat", "uv base"
    )
    flags.DEFINE_string(
        "vggpath", "../../resources/vgg-face.mat", "checkpoint director for vgg"
    )
    flags.DEFINE_string(
        "base_dir",
        "../../test_data_debug/prepare_rgb",
        "init poses, shape paras and images .mat file directory",
    )

    """ opt parameters """
    flags.DEFINE_string("GPU_NO", "0", "which GPU")

    flags.DEFINE_integer("num_of_img", 1, "")
    flags.DEFINE_boolean("is_bfm", False, "default: False")

    flags.DEFINE_float("photo_weight", 100.0, "")
    flags.DEFINE_float("gray_photo_weight", 80.0, "")
    flags.DEFINE_float("id_weight", 1.5, "")
    flags.DEFINE_float("reg_shape_weight", 4.0, "")
    flags.DEFINE_float("reg_tex_weight", 8.0, "")
    flags.DEFINE_float("real_86pt_lmk3d_weight", 5.0, "")
    flags.DEFINE_float("real_68pt_lmk2d_weight", 5.0, "")
    flags.DEFINE_float("lmk_struct_weight", 0.0, "")

    flags.DEFINE_integer("log_step", 20, "")
    flags.DEFINE_integer("train_step", 120, "")

    flags.DEFINE_boolean("save_ply", True, "save plys to look or not")

    flags.DEFINE_float("learning_rate", 0.05, "string : path for 3dmm")
    flags.DEFINE_integer("lr_decay_step", 20, "string : path for 3dmm")
    flags.DEFINE_float("lr_decay_rate", 0.9, "string : path for 3dmm")
    flags.DEFINE_float("min_learning_rate", 0.0000001 * 10, "string : path for 3dmm")

    """ opt out directory """
    flags.DEFINE_string("summary_dir", "../../test_data_debug/sum", "summary directory")
    flags.DEFINE_string("out_dir", "../../test_data_debug/out", "output directory")

    app.run(RGB_opt)
