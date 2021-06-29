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


from .losses import Losses
import tensorflow as tf
from .crop_render_img import crop_render_img


def compute_loss(
    FLAGS,
    basis3dmm,
    var_list,
    render_img_in_ori_pose,
    render_img_in_fake_pose_M,
    render_img_in_fake_pose_L,
    render_img_in_fake_pose_R,
    opt_type="RGB",
    global_step=None,
):
    tot_loss = float(0)
    image_batch = var_list["image_batch"]

    # 3D 86pt lmk loss
    if FLAGS.real_86pt_lmk3d_weight > 0:
        pred_lmk3d_batch = tf.gather(
            render_img_in_ori_pose["proj_xy"], basis3dmm["keypoints"], axis=1
        )
        lmk_86_3d_batch = var_list["lmk_86_3d_batch"]

        lmk3d_86_loss = Losses.weighted_landmark3d_loss(
            pred_lmk3d_batch, lmk_86_3d_batch
        )

        tot_loss = tot_loss + lmk3d_86_loss * FLAGS.real_86pt_lmk3d_weight
        tf.summary.scalar("lmk3d_86_loss", lmk3d_86_loss)

    # 2D 68pt lmk loss
    if opt_type == "RGB":
        if FLAGS.real_68pt_lmk2d_weight > 0:
            lmk_68_2d_batch = var_list["lmk_68_2d_batch"]

            lmk2d_68_loss = Losses.landmark2d_loss_v2(
                lmk_68_2d_batch[:, :18, :],
                render_img_in_ori_pose["proj_xy"],
                render_img_in_ori_pose["ver_contour_mask"],
                render_img_in_ori_pose["ver_norm"],
                basis3dmm["keypoints"][:18],
            )

            tot_loss = tot_loss + lmk2d_68_loss * FLAGS.real_68pt_lmk2d_weight
            tf.summary.scalar("lmk2d_68_loss", lmk2d_68_loss)

    # lmk struct loss
    if (FLAGS.real_86pt_lmk3d_weight > 0) & (FLAGS.lmk_struct_weight > 0):
        lmk_struct_loss = Losses.landmark_structure_loss(
            lmk_86_3d_batch, pred_lmk3d_batch
        )

        tot_loss = tot_loss + lmk_struct_loss * FLAGS.lmk_struct_weight
        tf.summary.scalar("landmark_struct", lmk_struct_loss)

    # add regularization to shape parameter
    if FLAGS.reg_shape_weight > 0:
        reg_shape_loss = Losses.reg_loss(var_list["para_shape"])
        tot_loss = tot_loss + reg_shape_loss * FLAGS.reg_shape_weight
        tf.summary.scalar("reg_shape", reg_shape_loss)

    # add regularization to tex parameter
    if FLAGS.reg_tex_weight > 0:
        reg_tex_loss = Losses.reg_loss(var_list["para_tex"])
        tot_loss = tot_loss + reg_tex_loss * FLAGS.reg_tex_weight
        tf.summary.scalar("reg_tex", reg_tex_loss)

    # get seg mask for photo loss
    from utils.misc import Utils, tf_detect_glassframe

    image_batch_for_seg = image_batch
    glass_seg_image = var_list["segmentation"][:, :, :, 3]
    glass_seg_frame = tf_detect_glassframe(image_batch_for_seg, glass_seg_image)
    seg_mask = Utils.create_photo_loss_mask_from_seg(
        var_list["segmentation"], glass_seg_frame
    )

    # photo loss (300 x 300)
    photo_mask_batch = render_img_in_ori_pose["attrs_image"] * seg_mask
    if FLAGS.photo_weight > 0:
        photo_loss = Losses.photo_loss(
            image_batch, render_img_in_ori_pose["render_image"], photo_mask_batch
        )
        tot_loss = tot_loss + photo_loss * FLAGS.photo_weight
        tf.summary.scalar("photo", photo_loss)

    # gray photo loss (300 x 300)
    gray_image_batch = tf.reduce_mean(image_batch, axis=-1, keepdims=True)
    gray_render_image_batch = tf.reduce_mean(
        render_img_in_ori_pose["render_image"], axis=-1, keepdims=True
    )
    if FLAGS.gray_photo_weight > 0:
        gray_photo_loss = Losses.photo_loss(
            gray_image_batch, gray_render_image_batch, photo_mask_batch
        )
        tot_loss = tot_loss + gray_photo_loss * FLAGS.gray_photo_weight
        tf.summary.scalar("gray_photo", gray_photo_loss)

    # depth loss (300 x 300)
    if opt_type == "RGBD":
        if FLAGS.depth_weight > 0:
            depth_weight = FLAGS.depth_weight - tf.train.exponential_decay(
                FLAGS.depth_weight, global_step, 5, 0.1
            )

            depth_image_batch_GT = var_list["depth_image_batch"]
            depth_image_batch_Pre = render_img_in_ori_pose["render_depth"]

            depth_mask_batch = photo_mask_batch * tf.cast(
                depth_image_batch_GT > 0, tf.float32
            )

            tf.summary.image("depth_mask_batch", depth_mask_batch, max_outputs=6)
            tf.summary.image(
                "depth_image_batch_GT", depth_image_batch_GT, max_outputs=6
            )
            tf.summary.image(
                "depth_image_batch_Pre", depth_image_batch_Pre, max_outputs=6
            )

            depth_loss = Losses.mult_depth_loss(
                depth_image_batch_GT, depth_image_batch_Pre, depth_mask_batch
            )
            tot_loss = tot_loss + depth_loss * depth_weight
            tf.summary.scalar("depth_loss", depth_loss)

    # used for cal grad for light para (id loss not influence light)
    tot_loss_illum = tot_loss

    # ID loss (crop 224 x 224 in vggid_loss)
    pred_lmk3d_front_batch = tf.gather(
        render_img_in_fake_pose_M["proj_xy"], basis3dmm["keypoints"], axis=1
    )
    pred_lmk3d_left_batch = tf.gather(
        render_img_in_fake_pose_L["proj_xy"], basis3dmm["keypoints"], axis=1
    )
    pred_lmk3d_right_batch = tf.gather(
        render_img_in_fake_pose_R["proj_xy"], basis3dmm["keypoints"], axis=1
    )
    render_image_fake_M = crop_render_img.tf_crop_by_landmark(
        render_img_in_fake_pose_M["render_image"], pred_lmk3d_front_batch
    )
    render_image_fake_L = crop_render_img.tf_crop_by_landmark(
        render_img_in_fake_pose_L["render_image"], pred_lmk3d_left_batch
    )
    render_image_fake_R = crop_render_img.tf_crop_by_landmark(
        render_img_in_fake_pose_R["render_image"], pred_lmk3d_right_batch
    )

    if opt_type == "RGB":
        all_images = tf.concat(
            [
                image_batch,
                render_img_in_ori_pose["render_image"],
            ],
            axis=0,
        )
        N_group = 2
    else:  # RGBD
        all_images = tf.concat(
            [
                image_batch,
                render_img_in_ori_pose["render_image"],
                render_image_fake_M,
            ],
            axis=0,
        )
        N_group = 3

    if FLAGS.id_weight > 0:
        id_loss = Losses.vggid_loss(
            all_images * 255.0, N_group, FLAGS.vggpath, "fc7", data_format="NHWC"
        )
        tot_loss = tot_loss + id_loss * FLAGS.id_weight
        tf.summary.scalar("id_loss", id_loss)

    tf.summary.scalar("tot_loss", tot_loss)

    # other summary
    tf.summary.image("image_batch", image_batch, max_outputs=6)
    tf.summary.image(
        "render", render_img_in_ori_pose["render_image"] * 255, max_outputs=6
    )
    tf.summary.image(
        "norm_image", tf.abs(render_img_in_ori_pose["norm_image"]), max_outputs=6
    )
    tf.summary.image("photo_mask", photo_mask_batch * 255, max_outputs=6)
    tf.summary.image("seg_mask", seg_mask, max_outputs=6)
    tf.summary.image("diffuse_batch", render_img_in_ori_pose["diffuse"], max_outputs=6)
    tf.summary.image("render_image_fake_M", render_image_fake_M, max_outputs=6)
    tf.summary.image("render_image_fake_L", render_image_fake_L, max_outputs=6)
    tf.summary.image("render_image_fake_R", render_image_fake_R, max_outputs=6)

    return tot_loss, tot_loss_illum
