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
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
from data_prepare_utils import write_lmk

fixed_pts = (
    np.array(
        [
            73.3451,
            118.4487,
            96.0404,
            77.5112,
            114.8624,
            96.0070,
            95.7575,
            121.6586,
            148.0636,
            147.8569,
        ]
    )
    .reshape((2, -1))
    .transpose()
)
warp_size = [224, 192]  # height, width
pts_mean86 = np.array(
    [
        45.559502,
        104.24780,
        47.094501,
        117.77705,
        49.368011,
        129.97537,
        52.305340,
        141.47940,
        56.249920,
        154.27869,
        63.107460,
        165.01971,
        71.174850,
        172.36023,
        81.929237,
        178.68507,
        97.289093,
        182.29616,
        112.52919,
        177.89139,
        123.33920,
        171.54056,
        131.66940,
        164.26958,
        138.23056,
        153.52193,
        141.85754,
        140.57895,
        144.45071,
        128.84717,
        146.39426,
        116.36816,
        147.67754,
        102.47821,
        55.485870,
        84.340775,
        61.147385,
        78.009048,
        69.581528,
        75.797379,
        78.612648,
        76.860222,
        86.064697,
        79.822960,
        62.489975,
        82.971130,
        69.879005,
        82.226051,
        77.701530,
        82.390945,
        85.335213,
        84.248680,
        105.48699,
        79.453552,
        112.96900,
        76.432724,
        122.02381,
        75.246162,
        130.61064,
        77.285698,
        136.46480,
        83.536705,
        106.27076,
        83.917999,
        113.94787,
        82.011887,
        121.85647,
        81.779221,
        129.34711,
        82.364937,
        63.320316,
        96.792084,
        67.515862,
        94.584686,
        77.845810,
        94.563499,
        81.965393,
        97.318008,
        77.402710,
        98.552208,
        67.509659,
        98.513344,
        72.628456,
        93.677307,
        72.395409,
        99.211624,
        110.02992,
        97.172417,
        114.07248,
        94.319572,
        124.35910,
        94.110191,
        128.54343,
        96.266449,
        124.43443,
        98.040421,
        114.60693,
        98.309441,
        119.23931,
        93.295609,
        119.60595,
        98.848557,
        95.895660,
        93.517433,
        95.888680,
        102.36029,
        95.881584,
        111.20296,
        95.874641,
        120.04578,
        87.517784,
        97.529457,
        104.33669,
        97.407219,
        84.132019,
        116.47855,
        107.81488,
        116.41264,
        80.940468,
        124.97491,
        111.12064,
        124.88945,
        85.455589,
        127.70387,
        90.463188,
        128.69844,
        95.953407,
        129.95752,
        101.45199,
        128.67410,
        106.51112,
        127.66216,
        78.027786,
        147.66968,
        91.463295,
        140.84270,
        96.066689,
        141.89987,
        100.57447,
        140.78816,
        114.46491,
        147.44310,
        96.189842,
        157.38145,
        84.125710,
        143.56898,
        108.07687,
        143.42168,
        109.86893,
        152.31499,
        82.586426,
        152.45120,
        80.742477,
        147.74809,
        111.71300,
        147.55542,
        87.001198,
        146.80209,
        96.081726,
        146.87469,
        105.23645,
        146.70581,
        86.978920,
        148.79839,
        96.164139,
        149.49869,
        105.38802,
        148.72549,
        88.427788,
        156.01730,
        103.95959,
        155.95354,
    ]
)
pts_mean86 = pts_mean86.reshape(86, 2)


def tformfwd(trans, uv):
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv


def findSimilarity(uv, xy, options=None):
    options = {"K": 2}

    #    uv = np.array(uv)
    #    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy.copy()
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def findNonreflectiveSimilarity(uv, xy, options=None):

    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print '--->X.shape: ', X.shape
    # print 'X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print '--->U.shape: ', U.shape
    # print 'U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform:twoUniquePointsReq")

    # print '--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    # print '--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print '--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def detect_lmk86(origin_images_dir, mtcnn_dir, out_dir, names_list, pb_path):

    batch_size = 1

    fopen = open(out_dir, "w")

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        graph_file = pb_path
        with open(graph_file, "rb") as f:
            print("hello")
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            image = sess.graph.get_tensor_by_name("lmk86pt_input:0")
            predict_lanmark = sess.graph.get_tensor_by_name("lmk86pt_output:0")

            count = 0
            for img_name in names_list:

                mtcnn_infor = scio.loadmat(
                    os.path.join(mtcnn_dir, img_name[:-4] + ".mat")
                )
                img = cv2.imread(os.path.join(origin_images_dir, img_name))

                batch_imgs = [img]
                batch_bboxes = [mtcnn_infor["batch_bboxes"].astype(np.float64)]
                batch_points = [mtcnn_infor["batch_points"].astype(np.float64)]

                return_lms = []

                for img_index in range(len(batch_bboxes)):
                    img = batch_imgs[img_index]
                    # concat warped faces
                    batch_faces = []
                    trans_invs = []
                    for face_index in range(len(batch_bboxes[img_index])):
                        # similarity transform
                        mtcnn_landmark = np.transpose(
                            batch_points[img_index][face_index].reshape(2, 5)
                        )
                        trans, trans_inv = get_similarity_transform(
                            mtcnn_landmark, fixed_pts
                        )
                        warp_img = cv2.warpAffine(
                            img, trans[:, 0:2].T, (int(warp_size[1]), int(warp_size[0]))
                        )
                        batch_faces.append(warp_img)
                        trans_invs.append(trans_inv)
                    if len(batch_faces) == 0:
                        return_lms.append(None)
                        continue

                    batch_faces = np.stack(batch_faces, axis=0)

                    # batch mode
                    out_predict_lanmarks = []
                    for i in range(
                        int(math.ceil(len(batch_faces) / float(batch_size)))
                    ):
                        now_batch_faces = batch_faces[
                            i * batch_size : (i + 1) * batch_size
                        ]
                        out_predict_lanmark = sess.run(
                            predict_lanmark, {image: now_batch_faces}
                        )
                        out_predict_lanmarks.append(out_predict_lanmark)
                    out_predict_lanmarks = np.concatenate(out_predict_lanmarks, axis=0)
                    out_predict_lanmarks += pts_mean86

                    # warp back
                    batch_warp_back_lm = []
                    for face_index in range(len(batch_bboxes[img_index])):
                        warp_back_lm = tformfwd(
                            trans_invs[face_index], out_predict_lanmarks[face_index]
                        )
                        # print(warp_back_lm.shape)
                        batch_warp_back_lm.append(np.reshape(warp_back_lm, [-1]))

                    return_lms.extend(batch_warp_back_lm)

                write_lmk(img_name, np.reshape(return_lms[0], [86, 2]), fopen)

                count = count + 1
                if (count % 100 == 0) | (count == len(names_list)):
                    print(
                        "has run 86pt lmk: " + str(count) + " / " + str(len(names_list))
                    )

    fopen.close()
    return
