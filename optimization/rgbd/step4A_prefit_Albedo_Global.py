# -*- coding:utf8 -*-
'''
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
'''

from absl import flags
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import sys
sys.path.append('../..')

from utils.basis import load_3dmm_basis, get_region_uv_texture, construct
from utils.misc import tf_blend_uv, blend_uv
from utils.losses import Losses
from utils.project import Projector
import utils.unwrap_utils as unwrap_utils

import time
from RGBD_utils.PoseTools import PoseTools as pose


def load_from_npz(base_dir ,basis3dmm):
    shape = np.load(os.path.join(base_dir, "para_shape_init.npy"))
    shape = np.transpose(shape) # 1 * n
    prefit_head = basis3dmm['mu_shape'] + shape.dot(basis3dmm['basis_shape'])
    prefit_head = np.transpose(np.reshape(prefit_head, [-1, 3]), [1, 0])  # 3 * n )

    data = np.load(os.path.join(base_dir, "step2_fusion.npz"))
    trans_base_2_camera = data['trans_base_2_camera'] # 3 * 4

    data = np.load(os.path.join(base_dir, "step1_data_for_fusion.npz"))
    img_select = data['img_select']
    K = data['K']
    trans_all = data['first_trans_select']

    img_list = []
    pro_yx_list = []
    pro_xy_list = []
    prefit_head_camera = pose.apply_trans(prefit_head, trans_base_2_camera)
    
    for i  in range(3):
        one_img = cv2.cvtColor(img_select[i], cv2.COLOR_BGR2RGB)
        one_trans =trans_all[i]
        proj_xyz = pose.apply_trans(prefit_head_camera, pose.trans_inverse(one_trans))
        proj_xyz = pose.project_2d(proj_xyz,K)

        # chaneg to n * 3(for neural render)
        proj_xyz = np.transpose(proj_xyz,[1,0])
        one_yx_left = np.concatenate([proj_xyz[:, 1:2], proj_xyz[:, 0:1]], axis=1)

        img_list.append(one_img)
        pro_yx_list.append(one_yx_left)
        pro_xy_list.append(proj_xyz)

    #load else datas
    base_uv_path = os.path.join(FLAGS.resources_path, 'base_tex.png')
    base_uv = Image.open(base_uv_path).resize((512,512))
    base_uv = np.asarray(base_uv, np.float32)

    # mid-- left  -- right
    preset_masks_list = []
    mid_mask_path = os.path.join(FLAGS.resources_path, 'mid_blend_mask.png')
    left_mask_path = os.path.join(FLAGS.resources_path, 'left_blend_mask.png')
    right_mask_path = os.path.join(FLAGS.resources_path, 'right_blend_mask.png')
    mid_mask = np.asarray(Image.open(mid_mask_path), np.float32) / 255.0
    left_mask = np.asarray(Image.open(left_mask_path), np.float32) / 255.0
    right_mask = np.asarray(Image.open(right_mask_path), np.float32) / 255.0
    preset_masks_list.append(mid_mask)
    preset_masks_list.append(left_mask)
    preset_masks_list.append(right_mask)

    return img_list,pro_yx_list, preset_masks_list,base_uv,pro_xy_list


def main(_):
    print('---- step4 start -----')
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO
    print('running base:',FLAGS.output_dir)

    basis3dmm = load_3dmm_basis(
            FLAGS.basis3dmm_path,
            FLAGS.uv_path,
            is_whole_uv=True)

    # load data (all : 0-255-float32)
    img_list, pro_yx_list, preset_masks_list, base_uv, pro_xyz_list = load_from_npz(FLAGS.output_dir, basis3dmm)

    # set tf datas
    base_uv_batch = tf.constant(base_uv[np.newaxis,...], name='base_uv')
    mask_mid_batch = tf.constant(preset_masks_list[0][np.newaxis, ..., np.newaxis], tf.float32)
    mask_left_batch = tf.constant(preset_masks_list[1][np.newaxis, ..., np.newaxis], tf.float32)
    mask_right_batch = tf.constant(preset_masks_list[2][np.newaxis, ..., np.newaxis], tf.float32)
    mask_batch = tf.clip_by_value(mask_mid_batch + mask_left_batch + mask_right_batch, 0, 1)

    imageH = img_list[0].shape[0]
    imageW = img_list[0].shape[1]
    assert(img_list[0].shape[0] == img_list[1].shape[0] and img_list[0].shape[1] == img_list[2].shape[1])
    image_mid_batch = tf.placeholder(dtype=tf.float32, shape=[1, imageH, imageW, 3], name='image_mid')
    image_left_batch = tf.placeholder(dtype=tf.float32, shape=[1, imageH, imageW, 3], name='image_left')
    image_right_batch = tf.placeholder(dtype=tf.float32, shape=[1, imageH, imageW, 3], name='image_right')

    NV = basis3dmm['basis_shape'].shape[1] // 3
    proj_xyz_mid_batch = tf.placeholder(dtype=tf.float32, shape=[1, NV, 3], name='proj_xyz_mid')
    proj_xyz_left_batch = tf.placeholder(dtype=tf.float32, shape=[1, NV, 3], name='proj_xyz_left')
    proj_xyz_right_batch = tf.placeholder(dtype=tf.float32, shape=[1, NV, 3], name='proj_xyz_right')

    ver_normals_mid_batch, _ = Projector.get_ver_norm(proj_xyz_mid_batch, basis3dmm['tri'], 'normal_mid')
    ver_normals_left_batch, _ = Projector.get_ver_norm(proj_xyz_left_batch, basis3dmm['tri'], 'normal_left')
    ver_normals_right_batch, _ = Projector.get_ver_norm(proj_xyz_right_batch, basis3dmm['tri'], 'normal_right')

    uv_mid_batch, _ = \
            unwrap_utils.unwrap_img_into_uv(
                    image_mid_batch / 255.0,
                    proj_xyz_mid_batch,
                    ver_normals_mid_batch,
                    basis3dmm,
                    512)

    uv_left_batch, _ = \
            unwrap_utils.unwrap_img_into_uv(
                    image_left_batch / 255.0,
                    proj_xyz_left_batch,
                    ver_normals_left_batch,
                    basis3dmm,
                    512)

    uv_right_batch, _ = \
            unwrap_utils.unwrap_img_into_uv(
                    image_right_batch / 255.0,
                    proj_xyz_right_batch,
                    ver_normals_right_batch,
                    basis3dmm,
                    512)

    uv_left_batch.set_shape((1,512,512,3))
    uv_mid_batch.set_shape((1,512,512,3))
    uv_right_batch.set_shape((1,512,512,3))

    # lapulasion pyramid blending

    cur_uv = tf_blend_uv(uv_left_batch, uv_right_batch, mask_right_batch, match_color=False)
    cur_uv = tf_blend_uv(cur_uv, uv_mid_batch, mask_mid_batch, match_color=False)
    uv_batch = tf_blend_uv(base_uv_batch / 255, cur_uv, mask_batch, match_color=True)
    uv_batch = uv_batch * 255

    uv_batch = tf.identity(uv_batch, name='uv_tex')

    print( "uv_batch: ", uv_batch.shape)

    #------------------------------------------------------------------------------------------
    # build fitting graph
    uv_bases = basis3dmm['uv']
    para_tex = tf.get_variable(
            shape=[1,uv_bases['basis'].shape[0]],
            initializer=tf.zeros_initializer(),
            name='para_tex'
            )

    uv_rgb, uv_mask = get_region_uv_texture(uv_bases, para_tex)
    print("uv_rgb: ", uv_rgb.shape )

    # build fitting loss
    input_uv512_batch = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 3], name='gt_uv')
    tot_loss = 0.
    loss_str = 'total:{}'
    if FLAGS.photo_weight > 0:
        photo_loss = Losses.photo_loss(uv_rgb/255.0,input_uv512_batch/255.0, uv_mask)
        tot_loss = tot_loss + photo_loss * FLAGS.photo_weight
        loss_str = '; photo:{}'

    if FLAGS.uv_tv_weight > 0:
        uv_tv_loss = Losses.uv_tv_loss(uv_rgb/255.0, uv_mask)
        tot_loss = tot_loss + uv_tv_loss * FLAGS.uv_tv_weight
        loss_str = loss_str + '; tv:{}'

    if FLAGS.uv_reg_tex_weight > 0:
        uv_reg_tex_loss = Losses.reg_loss(para_tex)
        tot_loss = tot_loss + uv_reg_tex_loss * FLAGS.uv_reg_tex_weight
        loss_str = loss_str + '; reg:{}'
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optim.minimize(tot_loss)


    with tf.Session() as sess:

        if FLAGS.write_graph:
            tf.train.write_graph(sess.graph_def, '', FLAGS.pb_path, as_text=True)
            exit()
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        uv_extract,   o_uv_left_batch,  o_uv_mid_batch, o_uv_right_batch,   o_mask_right_batch= \
                sess.run( [ uv_batch, uv_left_batch,  uv_mid_batch, uv_right_batch ,mask_right_batch],
                 {
                        image_mid_batch: img_list[0][np.newaxis,...],
                        image_left_batch: img_list[1][np.newaxis,...],
                        image_right_batch: img_list[2][np.newaxis,...],

                        proj_xyz_mid_batch: pro_xyz_list[0][np.newaxis,...],
                        proj_xyz_left_batch: pro_xyz_list[1][np.newaxis,...],
                        proj_xyz_right_batch: pro_xyz_list[2][np.newaxis,...] ,
                })
        uv_extract_dump = np.copy(uv_extract)
        uv_extract = uv_extract_dump
        print('  -------------  time wrap and merge:', time.time() - start_time)

        # blur will help
        uv_extract = np.asarray(uv_extract[0], np.float32)
        for _ in range(FLAGS.blur_times):
            kernel = np.reshape(np.array([1.] * FLAGS.blur_kernel * FLAGS.blur_kernel, np.float32), \
                                [FLAGS.blur_kernel, FLAGS.blur_kernel]) / (FLAGS.blur_kernel * FLAGS.blur_kernel)
            uv_extract = cv2.filter2D(uv_extract, -1, kernel)
        if FLAGS.is_downsize:
            uv_extract = cv2.resize(uv_extract, (256, 256))
        uv_extract = cv2.resize(uv_extract, (512, 512))
        uv_extract = uv_extract[np.newaxis, ...]

        # iter fit textrue paras
        start_time = time.time()
        for i in range(FLAGS.train_step):
            l1, l2, l3, l4,  _ = sess.run([tot_loss, photo_loss, uv_tv_loss, uv_reg_tex_loss,   train_op] , {
                    input_uv512_batch: uv_extract
                })
            if i % 50 == 0:
                print(i, loss_str.format(l1, l2, l3, l4) )
        para_tex_out = sess.run(para_tex, {input_uv512_batch: uv_extract})
        print(' ------------- time fit uv paras:',  time.time() -  start_time)

    uv_fit, face_mask = construct(uv_bases, para_tex_out, 512)
    uv_fit = blend_uv(base_uv / 255, uv_fit / 255, face_mask, True, 5)
    uv_fit = np.clip(uv_fit * 255, 0, 255)

    # save
    prefix = "bapi"
    Image.fromarray(np.squeeze(uv_extract_dump).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_tex_merge.png'))
    Image.fromarray(np.squeeze(uv_fit).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_tex_fit.png'))
    Image.fromarray(np.squeeze(o_uv_mid_batch).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_uv_mid_batch.png'))
    Image.fromarray(np.squeeze(o_uv_left_batch).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_uv_left_batch.png'))
    Image.fromarray(np.squeeze(o_uv_right_batch).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_uv_right_batch.png'))
    Image.fromarray(np.squeeze(o_mask_right_batch).astype(np.uint8)).save(
            os.path.join(FLAGS.output_dir, prefix + '_mask_right_batch.png'))

    np.save(os.path.join(FLAGS.output_dir, "para_tex_init.npy"), para_tex_out)
    print('---- step4 succeed -----')


if __name__ == '__main__':

    FLAGS = flags.FLAGS

    flags.DEFINE_string('output_dir',  "prefit",  'output data directory')

    flags.DEFINE_boolean('write_graph', False, 'write graph')
    flags.DEFINE_string('pb_path', 'path to pb file', 'output pb file')

    # for extract and merge
    flags.DEFINE_string('basis3dmm_path', '../resources/shape_exp_bases_rgbd_20200514.mat', 'basis3dmm path')
    flags.DEFINE_string('uv_path', '../resources/whole_uv512.mat', 'basis3dmm path')
    flags.DEFINE_string('resources_path', '../resources/', 'base_uv, mask..')
    # for uv paras fit

    flags.DEFINE_integer('blur_times', 2, 'blur times to preprocess ground truth image')
    flags.DEFINE_integer('blur_kernel', 9, 'blur times to preprocess ground truth image')
    flags.DEFINE_boolean('is_downsize', True, 'if true, down size to 256')

    flags.DEFINE_float('photo_weight', 1.0, '')
    flags.DEFINE_float('uv_tv_weight', 0.00001, '')
    flags.DEFINE_float('uv_reg_tex_weight', 0.00005, '')

    flags.DEFINE_integer('train_step', 100, 'for each 200epoch save one time')
    flags.DEFINE_float('learning_rate', 0.1, 'string : path for 3dmm')
    flags.DEFINE_string('GPU_NO', '7', 'which GPU')

    tf.app.run(main)
