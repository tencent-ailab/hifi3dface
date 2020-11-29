"""
This file is part of the repo: https://github.com/tencent-ailab/hifi3dface

If you find the code useful, please cite our paper: 

"High-Fidelity 3D Digital Human Creation from RGB-D Selfies."
Xiangkai Lin*, Yajing Chen*, Linchao Bao*, Haoxian Zhang, Sheng Wang, Xuefei Zhe, Xinwei Jiang, Jue Wang, Dong Yu, and Zhengyou Zhang. 
arXiv: https://arxiv.org/abs/2010.05562

Copyright (c) [2020] [Tencent AI Lab]

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
#import tensorflow.contrib.opt as tf_opt
import os
from absl import app, flags
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import sys

sys.path.append("../..")

from RGBD_load import RGBD_load
from utils.render_img import render_img_in_different_pose
from utils.compute_loss import compute_loss
from third_party.ply import write_ply, write_obj, read_obj

from utils.basis import load_3dmm_basis, load_3dmm_basis_bfm

from RGBD_utils.AddHeadTool import AddHeadTool


def define_variable(num_of_img, imageH, imageW, para_shape_shape, para_tex_shape, info):

    # variable-trainable=False
    image_batch = tf.compat.v1.get_variable(
        shape=[num_of_img, imageH, imageW, 3],
        dtype=tf.float32,
        name="ori_img",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["img_list"] / 255.0),
    )

    depth_image_batch = tf.compat.v1.get_variable(
        shape=[num_of_img, imageH, imageW],
        dtype=tf.float32,
        name="depth_img",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["dep_list"]),
    )

    input_scale_var = tf.compat.v1.get_variable(
        shape=[1, 1],
        dtype=tf.float32,
        name="input_scale_var",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["input_scale"]),
    )

    segmentation = tf.compat.v1.get_variable(
        shape=[num_of_img, imageH, imageW, 19],
        dtype=tf.float32,
        name="face_segmentation",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["seg_list"]),
    )

    lmk_86_3d_batch = tf.compat.v1.get_variable(
        shape=[num_of_img, 86, 2],
        dtype=tf.float32,
        name="lmk_86_3d_batch",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["lmk_list3D"]),
    )

    K = tf.compat.v1.get_variable(
        shape=[4, 3, 3],
        dtype=tf.float32,
        name="K",
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(info["K_list"]),
    )

    # variable-trainable=True
    if FLAGS.fixed_pose is False:
        pose6 = tf.compat.v1.get_variable(
            shape=[num_of_img, 6, 1],
            dtype=tf.float32,
            name="para_pose6",
            trainable=True,
            initializer=tf.compat.v1.constant_initializer(info["se3_list"]),
        )
    else:
        pose6 = tf.compat.v1.get_variable(
            shape=[num_of_img, 6, 1],
            dtype=tf.float32,
            name="para_pose6",
            trainable=False,
            initializer=tf.compat.v1.constant_initializer(info["se3_list"]),
        )

    para_shape = tf.compat.v1.get_variable(
        shape=[1, para_shape_shape],
        dtype=tf.float32,
        name="para_shape",
        trainable=True,
        initializer=tf.compat.v1.constant_initializer(info["para_shape"]),
    )

    para_tex = tf.compat.v1.get_variable(
        shape=[1, para_tex_shape],
        dtype=tf.float32,
        name="para_tex",
        trainable=True,
        initializer=tf.compat.v1.constant_initializer(info["para_tex"]),
    )

    para_illum = tf.compat.v1.get_variable(
        shape=[num_of_img, 27],
        dtype=tf.float32,
        initializer=tf.compat.v1.zeros_initializer(),
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
        tf.compat.v1.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.lr_decay_step, FLAGS.lr_decay_rate
        ),
        FLAGS.min_learning_rate,
    )
    optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

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


def RGBD_opt(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO

    # load data
    if FLAGS.is_bfm is False:
        basis3dmm = load_3dmm_basis(
            FLAGS.basis3dmm_path,
            FLAGS.uv_path,
        )
    else:
        basis3dmm = load_3dmm_basis_bfm(FLAGS.basis3dmm_path)

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
    summary_op = tf.compat.v1.summary.merge_all()
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.summary_dir)

    if os.path.exists(FLAGS.summary_dir) is False:
        os.makedirs(FLAGS.summary_dir)
    if os.path.exists(FLAGS.out_dir) is False:
        os.makedirs(FLAGS.out_dir)

    # start opt
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction=0.5
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

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

                ## add head
                if FLAGS.is_bfm is False:
                    print("-------------------start add head-------------------")
                    HeadModel = np.load(
                        FLAGS.info_for_add_head, allow_pickle=True
                    ).item()
                    vertex = read_obj(os.path.join(FLAGS.out_dir, "face.obj"))
                    vertex = vertex.transpose()
                    vertex_fit_h = vertex[:, HeadModel["head_h_idx"]]
                    pca_info_h = AddHeadTool.transfer_PCA_format_for_add_head(
                        basis3dmm, HeadModel
                    )
                    vertex_output_coord = AddHeadTool.fix_back_head(
                        vertex_fit_h,
                        HeadModel,
                        pca_info_h,
                        FLAGS.is_add_head_mirrow,
                        FLAGS.is_add_head_male,
                    )
                    write_obj(
                        os.path.join(FLAGS.out_dir, "head.obj"),
                        vertex_output_coord.transpose(),
                        HeadModel["head_vt_list"],
                        HeadModel["head_tri"],
                        HeadModel["head_tri_vt"],
                    )
                    print("-------------------add head successfully-------------------")

                out_diffuse, out_proj_xyz, out_ver_norm = sess.run(
                    [out_list["diffuse"], out_list["proj_xyz"], out_list["ver_norm"]]
                )
                out_diffuse = out_diffuse * 255.0  # RGB 0-255
                scio.savemat(
                    os.path.join(FLAGS.out_dir, "out_for_texture.mat"),
                    {
                        "ori_img": info["ori_img"],  # ? x ?
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

    """ data load parameter """
    flags.DEFINE_string(
        "basis3dmm_path", "../../3DMM/files/AI-NEXT-Shape.mat", "basis3dmm path"
    )
    flags.DEFINE_string(
        "uv_path", "../../3DMM/files/AI-NEXT-Albedo-Global.mat", "uv base"
    )
    flags.DEFINE_string(
        "vggpath", "../../resources/vgg-face.mat", "checkpoint director for vgg"
    )
    flags.DEFINE_string(
        "info_for_add_head", "../resources/info_for_add_head.npy", "info_for_add_head"
    )

    flags.DEFINE_string(
        "prefit_dir",
        "../../test_data_debug/prefit",
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
    flags.DEFINE_boolean("is_bfm", False, "default: False")

    flags.DEFINE_float("photo_weight", 100, "")
    flags.DEFINE_float("gray_photo_weight", 80, "")
    flags.DEFINE_float("id_weight", 1.8, "")
    flags.DEFINE_float("reg_shape_weight", 0.4, "")
    flags.DEFINE_float("reg_tex_weight", 0.0001, "")
    flags.DEFINE_float("depth_weight", 1000, "")
    flags.DEFINE_float("real_86pt_lmk3d_weight", 0.01, "")
    flags.DEFINE_float("lmk_struct_weight", 0.0, "")

    flags.DEFINE_integer("log_step", 10, "")
    flags.DEFINE_integer("train_step", 100, "")

    flags.DEFINE_boolean("save_ply", True, "save plys to look or not")

    flags.DEFINE_boolean("fixed_pose", False, "default: False")

    flags.DEFINE_boolean("is_add_head_mirrow", False, "default: False")
    flags.DEFINE_boolean("is_add_head_male", True, "default: True")

    flags.DEFINE_float("learning_rate", 0.05, "string : path for 3dmm")
    flags.DEFINE_integer("lr_decay_step", 10, "string : path for 3dmm")
    flags.DEFINE_float("lr_decay_rate", 0.9, "string : path for 3dmm")
    flags.DEFINE_float("min_learning_rate", 0.0000001 * 10, "string : path for 3dmm")

    """ opt out directory """
    flags.DEFINE_string("summary_dir", "../../test_data_debug/sum", "summary directory")
    flags.DEFINE_string("out_dir", "../../test_data_debug/out", "output directory")

    app.run(RGBD_opt)
