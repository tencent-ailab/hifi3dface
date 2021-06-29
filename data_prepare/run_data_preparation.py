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

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2, os, importlib, math
import os.path as osp
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image, ImageOps

# mtcnn
import detect_face_with_mtcnn

# 86lmk
import detect_3D_landmark

# 68 lmk
import detect_2D_landmark

# crop & seg
from data_prepare_utils import crop_image_and_process_landmark
import face_segmentation

from data_prepare_utils import load_landmark, write_lmk, write_lmk_no_name
from absl import app, flags
import glob
import shutil


def detect_2Dlmk_all_imgs(graph_file, img_dir, lmk3D_txt_path, lmk2D_txt_path):
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        graph_file = graph_file

        with open(graph_file, "rb") as f:
            print("hello")
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            fopen = open(lmk2D_txt_path, "w")

            landmarks3D, images_name = load_landmark(lmk3D_txt_path, 86)
            count = 0
            for i in range(0, len(images_name)):
                img_name = images_name[i]
                img = cv2.imread(os.path.join(img_dir, img_name))
                lmk3D = landmarks3D[i]
                LMK2D_batch = detect_2D_landmark.detect_2Dlmk68(
                    np.array([lmk3D]), np.array([img]), sess
                )
                write_lmk(img_name, np.reshape(LMK2D_batch[0], [68, 2]), fopen)
                count = count + 1
                if (count % 100 == 0) | (count == len(images_name)):
                    print(
                        "has run 68pt lmk: "
                        + str(count)
                        + " / "
                        + str(len(images_name))
                    )

            fopen.close()

    return LMK2D_batch


def crop_img_by_lmk(
    lmk2D_txt_path,
    lmk3D_txt_path,
    lmk2D_crop_txt_path,
    lmk3D_crop_txt_path,
    img_dir,
    out_crop_dir,
    orig=False,
):
    if os.path.exists(out_crop_dir) is False:
        os.makedirs(out_crop_dir)

    fopen2D = open(lmk2D_crop_txt_path, "w")
    fopen3D = open(lmk3D_crop_txt_path, "w")

    landmarks2D, images_name = load_landmark(lmk2D_txt_path, 68)
    landmarks3D, images_name = load_landmark(lmk3D_txt_path, 86)

    count = 0
    for i in range(0, len(images_name)):
        img_name = images_name[i]
        img = cv2.imread(os.path.join(img_dir, img_name))
        (
            crop_img,
            prediction3D,
            prediction2D,
            ori_crop_img,
        ) = crop_image_and_process_landmark(
            img, landmarks3D[i], landmarks2D[i], size=300, orig=orig
        )
        write_lmk(img_name, np.reshape(prediction3D, [86, 2]), fopen3D)
        write_lmk(img_name, np.reshape(prediction2D, [68, 2]), fopen2D)
        cv2.imwrite(os.path.join(out_crop_dir, img_name), crop_img)
        if orig:
            cv2.imwrite(
                os.path.join(out_crop_dir, img_name[:-4] + "_ori" + img_name[-4:]),
                ori_crop_img,
            )

        count = count + 1
        if (count % 100 == 0) | (count == len(images_name)):
            print(
                "has run crop_img_by_lmk: " + str(count) + " / " + str(len(images_name))
            )

    fopen2D.close()
    fopen3D.close()
    return


def face_seg(graph_file, lmk3D_crop_txt_path, out_crop_dir, seg_dir):

    if os.path.exists(seg_dir) is False:
        os.makedirs(seg_dir)

    landmarks3D, images_name = load_landmark(lmk3D_crop_txt_path, 86)

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        graph_file = graph_file

        with open(graph_file, "rb") as f:
            print("hello")
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            count = 0
            for i in range(0, len(images_name)):
                img_name = images_name[i]
                crop_img = cv2.imread(os.path.join(out_crop_dir, img_name))
                lmk3D = landmarks3D[i]
                SEG_batch, SEG_color_batch = face_segmentation.run_face_seg(
                    np.array([lmk3D]), np.array([crop_img]), sess
                )

                np.save(os.path.join(seg_dir, img_name[:-3] + "npy"), SEG_batch[0])
                cv2.imwrite(
                    os.path.join(seg_dir, img_name[:-4] + "_seg.jpg"),
                    SEG_color_batch[0],
                )
                count = count + 1
                if (count % 100 == 0) | (count == len(images_name)):
                    print(
                        "has run face_seg: "
                        + str(count)
                        + " / "
                        + str(len(images_name))
                    )
    return


def prepare_test_data_RGB(img_dir, out_dir):
    mtcnn_dir = os.path.join(out_dir, "mtcnn_result/")
    lmk3D_ori_txt_path = os.path.join(out_dir, "lmk_3D_86pts_ori.txt")
    lmk2D_ori_txt_path = os.path.join(out_dir, "lmk_2D_68pts_ori.txt")
    lmk3D_crop_txt_path = os.path.join(out_dir, "lmk_3D_86pts.txt")
    lmk2D_crop_txt_path = os.path.join(out_dir, "lmk_2D_68pts.txt")
    crop_dir = out_dir
    seg_dir = out_dir
    print("start MTCNN")
    pb_path = os.path.join(FLAGS.pb_path, "mtcnn_model.pb")
    names_list = detect_face_with_mtcnn.detect_with_MTCNN(img_dir, mtcnn_dir, pb_path)

    print("start detect 86pt 3D lmk")
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "lmk3D_86_model.pb")
    detect_3D_landmark.detect_lmk86(
        img_dir, mtcnn_dir, lmk3D_ori_txt_path, names_list, pb_path
    )

    print("start detect 68pt 2D lmk")  # need to transfer RGB in the function
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "lmk2D_68_model.pb")
    detect_2Dlmk_all_imgs(
        pb_path, img_dir, lmk3D_ori_txt_path, lmk2D_ori_txt_path
    )  # N x 68 x 2

    print("start crop by 3D lmk")
    crop_img_by_lmk(
        lmk2D_ori_txt_path,
        lmk3D_ori_txt_path,
        lmk2D_crop_txt_path,
        lmk3D_crop_txt_path,
        img_dir,
        crop_dir,
        orig=True,
    )

    print("start face seg")  # need to transfer RGB in the function
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "faceseg_model.pb")
    face_seg(pb_path, lmk3D_crop_txt_path, crop_dir, seg_dir)

    print("finish RGB data preparation")


def prepare_test_data_RGBD(img_dir, out_dir):
    mtcnn_dir = os.path.join(out_dir, "mtcnn_result/")
    lmk3D_ori_txt_path = os.path.join(out_dir, "lmk_3D_86pts_ori.txt")
    lmk2D_ori_txt_path = os.path.join(out_dir, "lmk_2D_68pts_ori.txt")
    lmk3D_crop_txt_path = os.path.join(out_dir, "lmk_3D_86pts.txt")
    lmk2D_crop_txt_path = os.path.join(out_dir, "lmk_2D_68pts.txt")
    out_img_names_path = os.path.join(out_dir, "img_names.txt")
    crop_dir = out_dir
    seg_dir = out_dir

    print("check RGBD data")
    files = glob.glob(osp.join(img_dir, "*.jpg"))
    files.extend(glob.glob(osp.join(img_dir, "*.JPG")))

    dep_files = glob.glob(osp.join(img_dir, "*.png"))
    dep_files.extend(glob.glob(osp.join(img_dir, "*.PNG")))

    files.sort()
    dep_files.sort()

    if len(files) == len(dep_files) is False:
        raise Exception("The number of depth img is not same with RGB img.")

    for index in range(0, len(files)):
        img = cv2.imread(files[index])
        dep_img = cv2.imread(dep_files[index])

        if img.shape[0] < img.shape[1]:
            img = cv2.transpose(img)
            dep_img = cv2.transpose(dep_img)

            cv2.imwrite(img_dir + files[index].split("/")[-1], img)
            cv2.imwrite(img_dir + dep_files[index].split("/")[-1], dep_img)

    print("start MTCNN")
    pb_path = os.path.join(FLAGS.pb_path, "mtcnn_model.pb")
    names_list, dep_name_list = detect_face_with_mtcnn.detect_with_MTCNN(
        img_dir, mtcnn_dir, pb_path, "depth"
    )

    print("start detect 86pt 3D lmk")
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "lmk3D_86_model.pb")
    detect_3D_landmark.detect_lmk86(
        img_dir, mtcnn_dir, lmk3D_ori_txt_path, names_list, pb_path
    )

    print("start detect 68pt 2D lmk")  # need to transfer RGB in the function
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "lmk2D_68_model.pb")
    detect_2Dlmk_all_imgs(
        pb_path, img_dir, lmk3D_ori_txt_path, lmk2D_ori_txt_path
    )  # N x 68 x 2

    print("start crop by 3D lmk")
    crop_img_by_lmk(
        lmk2D_ori_txt_path,
        lmk3D_ori_txt_path,
        lmk2D_crop_txt_path,
        lmk3D_crop_txt_path,
        img_dir,
        crop_dir,
        orig=False,
    )

    print("start face seg")  # need to transfer RGB in the function
    tf.reset_default_graph()
    pb_path = os.path.join(FLAGS.pb_path, "faceseg_model.pb")
    face_seg(pb_path, lmk3D_crop_txt_path, crop_dir, seg_dir)

    fopen = open(out_img_names_path, "w")
    for i in range(0, len(names_list)):
        text = names_list[i] + "," + dep_name_list[i] + "\n"
        fopen.write(text)
    fopen.close()

    print("finish RGBD data preparation")


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_NO
    if FLAGS.mode == "test_RGB":
        prepare_test_data_RGB(FLAGS.img_dir, FLAGS.out_dir)
    elif FLAGS.mode == "test_RGBD":
        prepare_test_data_RGBD(FLAGS.img_dir, FLAGS.out_dir)
    else:
        raise Exception("mode not in [test_RGB, test_RGBD]!")


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    flags.DEFINE_string("GPU_NO", "1", "which GPU")

    flags.DEFINE_string("pb_path", "../resources/", "path to store pb files")
    flags.DEFINE_string("mode", "test_RGBD", "[test_RGB, test_RGBD]")
    flags.DEFINE_string(
        "img_dir", "./test_data_xk/ori/", ""
    )  # ./test_img_in/  #test_ori/
    flags.DEFINE_string(
        "out_dir", "./test_data_xk/prepare/", ""
    )  # ./test_out/  #test_prepare/

    app.run(main)
