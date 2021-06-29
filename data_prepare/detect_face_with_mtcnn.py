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
# from __future__ import division
# from __future__ import print_function

import cv2, os, importlib, math
import os.path as osp
import numpy as np
import scipy.io as scio
import tensorflow as tf


def create_mtcnn_pb(sess):

    pnet_fun = lambda img: sess.run(
        ("pnet/conv4-2/BiasAdd:0", "pnet/prob1:0"), feed_dict={"pnet/input:0": img}
    )
    rnet_fun = lambda img: sess.run(
        ("rnet/conv5-2/conv5-2:0", "rnet/prob1:0"), feed_dict={"rnet/input:0": img}
    )
    onet_fun = lambda img: sess.run(
        ("onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0", "onet/prob1:0"),
        feed_dict={"onet/input:0": img},
    )

    return pnet_fun, rnet_fun, onet_fun


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generateBoundingBox(
            out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0]
        )

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, "Union")
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, "Union")
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1 : edy[k], dx[k] - 1 : edx[k], :] = img[
                y[k] - 1 : ey[k], x[k] - 1 : ex[k], :
            ]
            if (
                tmp.shape[0] > 0
                and tmp.shape[1] > 0
                or tmp.shape[0] == 0
                and tmp.shape[1] == 0
            ):
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack(
            [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)]
        )
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, "Union")
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1 : edy[k], dx[k] - 1 : edx[k], :] = img[
                y[k] - 1 : ey[k], x[k] - 1 : ex[k], :
            ]
            if (
                tmp.shape[0] > 0
                and tmp.shape[1] > 0
                or tmp.shape[0] == 0
                and tmp.shape[1] == 0
            ):
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1, :]
        points = out1
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack(
            [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)]
        )
        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = (
            np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        )
        points[5:10, :] = (
            np.tile(h, (5, 1)) * points[5:10, :]
            + np.tile(total_boxes[:, 1], (5, 1))
            - 1
        )
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, "Min")
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points


def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


def imresample(img, sz):
    im_data = cv2.resize(
        img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA
    )  # @UndefinedVariable
    return im_data


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def detect_with_MTCNN(origin_images_dir, out_dir, pb_path, mode="no_depth"):
    print("MTCNN detect")

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    minsize = 20  # minimum size of face
    threshold = [0.5, 0.6, 0.6]  # three steps's threshold
    factor = 0.709  # scale factor

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        graph_file = pb_path
        with open(graph_file, "rb") as f:
            print("hello")
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            pnet, rnet, onet = create_mtcnn_pb(sess)

    # find files
    import glob

    if mode == "depth":
        files = glob.glob(osp.join(origin_images_dir, "*.jpg"))
        files.extend(glob.glob(osp.join(origin_images_dir, "*.JPG")))
        dep_files = glob.glob(osp.join(origin_images_dir, "*.png"))
        dep_files.extend(glob.glob(osp.join(origin_images_dir, "*.PNG")))
        files.sort()
        dep_files.sort()
    else:
        files = glob.glob(osp.join(origin_images_dir, "*.jpg"))
        files.extend(glob.glob(osp.join(origin_images_dir, "*.png")))
        files.extend(glob.glob(osp.join(origin_images_dir, "*.JPG")))
        files.extend(glob.glob(osp.join(origin_images_dir, "*.PNG")))
        files.sort()
    print("=========================")
    # print("img:", files)
    # print("Dep:", dep_files)

    # detect face bbox
    count = 0
    names_list = []
    dep_name_list = []
    for index in range(0, len(files)):
        img = cv2.imread(files[index])
        bounding_boxes, points = detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor
        )  # bounding_boxes.shape: (n, 5) points.shape: (10, n)
        if len(bounding_boxes) == 1:
            points = np.transpose(points)
            batch_imgs = img
            batch_bboxes = bounding_boxes
            batch_points = points
            batch_names = files[index].split("/")[-1]
            names_list.append(batch_names)

            scio.savemat(
                os.path.join(out_dir, batch_names[:-4] + ".mat"),
                {
                    "batch_bboxes": batch_bboxes.astype(np.float64),
                    "batch_points": batch_points.astype(np.float64),
                },
            )
            if mode == "depth":
                dep_name_list.append(dep_files[index].split("/")[-1])

        elif len(bounding_boxes) > 1:
            print("too much face to detect by MTCNN, only select first person")
            points = np.transpose(points[:, 0:1])
            batch_imgs = img
            batch_bboxes = bounding_boxes[0:1, :]
            batch_points = points
            batch_names = files[index].split("/")[-1]
            names_list.append(batch_names)

            scio.savemat(
                os.path.join(out_dir, batch_names[:-4] + ".mat"),
                {
                    "batch_bboxes": batch_bboxes.astype(np.float64),
                    "batch_points": batch_points.astype(np.float64),
                },
            )
            if mode == "depth":
                dep_name_list.append(dep_files[index].split("/")[-1])

        else:
            print("no face to detect by MTCNN, please input single person photo")
            pass
            # raise Exception("no face or much face to detect, please input single person photo")

        count = count + 1
        if (count % 100 == 0) | (count == len(files)):
            print("has run MTCNN: " + str(count) + " / " + str(len(files)))
    sess.close()

    if mode == "depth":
        return names_list, dep_name_list
    else:
        return names_list
