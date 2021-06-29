# -*- coding:utf8 -*-
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

import scipy.io as sio
import cv2
import struct
import numpy as np
from .PoseTools import PoseTools
import random


def bfs_find_corrospondence(landmarks, height, width, depth):
    num = landmarks.shape[0]
    results = np.zeros((3, num))

    for i in range(num):
        x = int(round(landmarks[i][0])) - 1
        y = int(round(landmarks[i][1])) - 1

        dd = depth[y][x]
        if dd != 0:
            results[0][i] = x
            results[1][i] = y
            results[2][i] = dd
        else:
            visit = np.zeros((height, width))

            visit[y][x] = 1
            queue = [[x + 1, y], [x - 1, y], [x, y - 1], [x, y + 1]]
            while 1:
                if len(queue):
                    x = queue[0][0]
                    y = queue[0][1]
                    queue = queue[1 : len(queue)][:]
                    dd = depth[y][x]
                    if dd != 0:
                        results[0][i] = x
                        results[1][i] = y
                        results[2][i] = dd
                        break
                    if x - 1 > 0 and visit[y][x - 1] == 0:
                        queue.append([x - 1, y])
                        visit[y][x - 1] = 1
                    if y - 1 > 0 and visit[y - 1][x] == 0:
                        queue.append([x, y - 1])
                        visit[y - 1][x] = 1
                    if y + 1 > 0 and visit[y + 1][x] == 0:
                        queue.append([x, y + 1])
                        visit[y + 1][x] = 1
                    if x + 1 > 0 and visit[y][x + 1] == 0:
                        queue.append([x + 1, y])
                        visit[y][x + 1] = 1

                else:
                    break
    return results


def bfs_repair_one(K, pt3d, pt2d, depth, img, part):
    # n * 3
    pt3d_one = np.zeros((len(part), 3))
    ind = 0
    error_list = []
    for i in part:

        pt3d_one[ind] = pt3d[i]
        if pt3d[i][2] == 0:
            error_list.append(i)
        ind = ind + 1

    if len(error_list) > 0:
        shape = img.shape
        error_landmarks = np.zeros((len(error_list), 2))
        ind = 0
        for i in error_list:
            error_landmarks[ind] = pt2d[i]
            ind = ind + 1
        results = bfs_find_corrospondence(error_landmarks, shape[0], shape[1], depth)
        new_depths = results
        # new_depths[0,:] = results[2,:]*(results[0,:]-K['cx'])/K['fx']
        # new_depths[1,:] = results[2,:]*(results[1,:]-K['cy'])/K['fy']
        new_depths[0, :] = results[2, :] * (results[0, :] - K[0, 2]) / K[0, 0]
        new_depths[1, :] = results[2, :] * (results[1, :] - K[1, 2]) / K[1, 1]
        ind = 0
        for i in range(len(error_list)):
            pt3d_one[error_list[i] - 17] = new_depths.transpose()[ind]
            ind = ind + 1
    return pt3d_one


def chect_error_one(K, pt3d, pt2d, depth, img, part):
    # n*3
    pt2d_one = pt2d[part, :]
    pt3d_one = pt3d[part, :]

    # Points with negative depth on pt3d are error points
    error_1 = np.array(pt3d_one[:, 2] < 1).astype(np.int32)

    # Points not in crop depth are also error points
    error_2 = np.zeros(len(part))
    for i in range(len(part)):
        x = int(round(pt2d_one[i][0]))
        y = int(round(pt2d_one[i][1]))
        if depth[y - 1][x - 1] < 1:
            error_2[i] = 1

    error_one = error_1 + error_2
    error_one[error_one > 0] = 1
    return pt3d_one, error_one


def find_3d_keypoints_from_landmark_and_depth_86(
    use_trans_all, use_pt3d_all, use_pt2d_all, use_depth_all, use_img_all, K
):

    pt3d_ref = use_pt3d_all[0].transpose()  # n * 3
    pt2d_ref = use_pt2d_all[0].transpose()  # n * 3
    pt3d_mid = bfs_repair_one(
        K, pt3d_ref, pt2d_ref, use_depth_all[0], use_img_all[0], range(17, 86)
    )

    pt3d_left, error_left = chect_error_one(
        K,
        use_pt3d_all[1].transpose(),
        use_pt2d_all[1].transpose(),
        use_depth_all[1],
        use_img_all[1],
        range(0, 9),
    )
    pt3d_right, error_right = chect_error_one(
        K,
        use_pt3d_all[2].transpose(),
        use_pt2d_all[2].transpose(),
        use_depth_all[2],
        use_img_all[2],
        range(8, 17),
    )
    error_list = np.zeros((86))
    error_list[0:9] = error_left
    error_list[8:17] = error_right

    pose_left = use_trans_all[1]
    pose_right = use_trans_all[2]

    pt3d_left = pose_left.dot(
        np.hstack((pt3d_left, np.ones((pt3d_left.shape[0], 1)))).transpose()
    )  # 3 * 9
    pt3d_right = pose_right.dot(
        np.hstack((pt3d_right, np.ones((pt3d_right.shape[0], 1)))).transpose()
    )  # 3 * 9
    if error_left[8] == 0 and error_right[0] == 0:
        pt9 = (pt3d_left[:, 8] + pt3d_right[:, 0]) / 2
        error_list[8] = 0
    else:
        pt9 = np.array([[-1], [-1], [-1]])
        error_list[8] = 1

    three_3d_points = np.hstack(
        (pt3d_left[:, 0:-1], pt9.reshape(-1, 1), pt3d_right[:, 1:], pt3d_mid.T)
    )
    error_list = np.where(error_list > 0)
    three_3d_points[:, error_list] = -1
    return three_3d_points


def get_trans_base_to_camera(with_error_3d_points_slam, is_bfm):
    ####################### input: n * 3  ###################################
    if is_bfm:
        a1 = [
            -7.23139801e01,
            -7.15076675e01,
            -7.05661774e01,
            -6.81250763e01,
            -6.29898529e01,
            -5.41726456e01,
            -3.89137421e01,
            -1.90310783e01,
            -2.19545674e00,
            1.56179953e01,
            3.57561684e01,
            5.29842300e01,
            6.27171631e01,
            6.75358658e01,
            7.00546646e01,
            7.09011841e01,
            7.15581284e01,
            -5.75549126e01,
            -4.93166542e01,
            -3.64593544e01,
            -2.62986355e01,
            -1.69525738e01,
            -4.71634178e01,
            -3.74854469e01,
            -2.75193386e01,
            -1.54471292e01,
            1.90682487e01,
            2.99945393e01,
            4.11153793e01,
            5.19259186e01,
            5.85614815e01,
            1.75318222e01,
            3.12036495e01,
            4.19002533e01,
            5.02672234e01,
            -4.41655655e01,
            -3.67105942e01,
            -2.30989437e01,
            -1.57808361e01,
            -2.17943840e01,
            -3.67495461e01,
            -2.84216633e01,
            -2.90658684e01,
            1.60581207e01,
            2.42889099e01,
            3.77512665e01,
            4.33805656e01,
            3.68777161e01,
            2.47922916e01,
            2.94666710e01,
            3.06360645e01,
            1.41742384e00,
            1.27253580e00,
            1.51286042e00,
            2.01201681e-02,
            -1.14166441e01,
            1.10063829e01,
            -1.34077501e01,
            1.48980522e01,
            -1.73311424e01,
            1.70757103e01,
            -1.30563917e01,
            -6.53453159e00,
            9.16568279e-01,
            6.86933613e00,
            1.23089561e01,
            -2.46515636e01,
            -4.90839863e00,
            -1.92024767e-01,
            5.43883467e00,
            2.65130939e01,
            -1.99937597e-01,
            -1.53688469e01,
            1.49261389e01,
            1.76253471e01,
            -1.78384418e01,
            -2.03025589e01,
            2.15493469e01,
            -9.91084957e00,
            -2.80126721e-01,
            1.18979940e01,
            -1.16811075e01,
            -1.67604792e00,
            1.10015984e01,
            -9.80548763e00,
            7.99779654e00,
        ]
        a2 = [
            37.36131,
            19.457314,
            4.147109,
            -11.139346,
            -34.128418,
            -51.520515,
            -65.135605,
            -75.5716,
            -75.38477,
            -76.531975,
            -68.16107,
            -53.24735,
            -34.74222,
            -12.196404,
            4.9636893,
            21.914625,
            39.12869,
            51.46235,
            55.340363,
            60.601444,
            58.72639,
            55.12181,
            51.404213,
            52.207603,
            50.137466,
            49.18511,
            56.296253,
            59.2982,
            59.149242,
            53.174667,
            50.42041,
            49.39064,
            51.051556,
            51.795864,
            48.50691,
            33.38993,
            37.49412,
            37.076965,
            31.687813,
            30.684156,
            29.708702,
            38.141052,
            29.32608,
            31.483547,
            37.501278,
            36.779022,
            33.518234,
            30.016329,
            29.942894,
            38.67858,
            29.0896,
            39.121613,
            30.41742,
            17.773756,
            2.2322557,
            31.688046,
            31.562336,
            8.406996,
            6.991427,
            -0.8477457,
            -2.6682057,
            -4.9866652,
            -7.742144,
            -9.772964,
            -7.6465516,
            -5.4153695,
            -33.333046,
            -21.566263,
            -24.960453,
            -22.594265,
            -34.221508,
            -37.97633,
            -26.163889,
            -26.194008,
            -37.478046,
            -37.48826,
            -32.578335,
            -32.358818,
            -31.09593,
            -31.296856,
            -31.188591,
            -31.189629,
            -32.373726,
            -31.116377,
            -39.833355,
            -40.087997,
        ]
        a3 = [
            32.524567,
            25.574074,
            25.45251,
            30.670927,
            38.57534,
            52.16466,
            70.71604,
            89.85496,
            101.13856,
            91.22503,
            70.448586,
            52.94385,
            41.274445,
            31.996029,
            23.64382,
            22.74357,
            32.3493,
            77.98372,
            90.969696,
            100.124725,
            105.6835,
            108.24334,
            94.9855,
            101.22018,
            103.70485,
            106.75156,
            107.61465,
            103.54715,
            96.692535,
            87.71253,
            74.04851,
            105.90956,
            102.46049,
            98.07323,
            91.153854,
            85.90583,
            94.06561,
            94.20951,
            93.472755,
            93.51368,
            91.72802,
            95.336075,
            94.03192,
            92.72418,
            94.18995,
            92.808624,
            86.19275,
            91.256935,
            93.785965,
            95.61713,
            93.65745,
            108.06224,
            113.07293,
            122.35573,
            131.94388,
            97.674995,
            97.4794,
            110.60944,
            107.088455,
            108.09587,
            105.44942,
            113.55085,
            113.3804,
            116.57359,
            112.852615,
            112.694214,
            98.585815,
            115.28394,
            116.20786,
            115.50854,
            97.78003,
            113.53529,
            110.96187,
            110.91211,
            105.91726,
            106.001884,
            100.63327,
            99.94496,
            107.06453,
            109.801384,
            107.37359,
            107.01396,
            111.188995,
            107.86107,
            111.5579,
            111.95899,
        ]
    else:
        a1 = [
            -7.85208222402778,
            -7.54366312674316,
            -7.23474653535443,
            -6.80809671929583,
            -6.09230653082483,
            -4.87651078212643,
            -3.83475234020243,
            -2.11737492922110,
            0.196692177695851,
            2.34190803398287,
            3.91808169999272,
            4.73537027564548,
            5.94108034184592,
            6.59364273461717,
            6.99841927783153,
            7.51040553836296,
            7.86137189208800,
            -6.11444365862811,
            -5.08465131566543,
            -3.66198182674391,
            -2.34233473800948,
            -1.38683716547997,
            -4.85522278875312,
            -3.74606513492472,
            -2.57526465412475,
            -1.31798973321966,
            1.85587028770909,
            2.83916124725813,
            4.08683925990438,
            5.35114638047775,
            6.06811520897587,
            1.79944421267933,
            3.10729110477770,
            4.21531119591557,
            5.17618555081808,
            -4.63965016359563,
            -3.82616992143579,
            -2.52542452413108,
            -1.95012261628636,
            -2.59494439260083,
            -3.97832426823377,
            -2.96117288779393,
            -3.28661513388784,
            2.17940868606990,
            2.80503630428088,
            4.18139769207972,
            4.75941943948944,
            4.21718468647930,
            2.87895781949168,
            3.34828199097913,
            3.63198346073205,
            0.172227865761091,
            0.203203383727210,
            0.237376533455140,
            0.327254178788228,
            -1.15273739422949,
            1.47016515773056,
            -1.26128374003605,
            1.67740927043274,
            -1.74549499350128,
            2.05131517351030,
            -1.14313658159003,
            -0.549086912644775,
            0.269883543859780,
            1.00853876721907,
            1.54351554745658,
            -2.42097077361195,
            -0.359097701137877,
            0.194964792303664,
            0.741509272877570,
            2.74409190227905,
            0.169495015080376,
            -1.34648231972359,
            1.74921856345831,
            1.90604762272808,
            -1.59377731237783,
            -1.96259354615232,
            2.28625235572007,
            -0.897357773102931,
            0.187352317311287,
            1.25053377838451,
            -0.896554576674260,
            0.164446676065311,
            1.22036548778333,
            -0.714083686653618,
            1.06605412891030,
        ]
        a2 = [
            3.00528869326328,
            1.04668676065450,
            -0.889122206388805,
            -2.55095639352405,
            -4.40110946671072,
            -6.09250011054630,
            -6.80311006092336,
            -7.76504378651200,
            -8.02132048724283,
            -7.88507352508978,
            -6.99395549776338,
            -6.22664270284460,
            -4.41106547900483,
            -2.61838706542139,
            -0.798974676934847,
            1.19626594246944,
            3.26406307453894,
            5.34956007644388,
            5.88234362322041,
            6.25539936117420,
            5.92189644117161,
            5.51518530166234,
            5.41936235494157,
            5.49232276656213,
            5.21300145648732,
            4.98458246038797,
            5.51476699947435,
            5.97854932996029,
            6.35800893775269,
            6.01824529600424,
            5.63119854816389,
            4.99878935485879,
            5.28772021256751,
            5.58468060937383,
            5.52380759403738,
            3.51711893108453,
            3.97093962986850,
            3.72379665103676,
            3.16893179461317,
            3.06250270429027,
            3.12598394525833,
            3.91234936445397,
            3.01608979615565,
            3.20700695905134,
            3.85442305445659,
            4.01232958133531,
            3.58685803048397,
            3.30040069521397,
            3.21375342501366,
            4.05645715119317,
            3.15343370157726,
            3.73231308225596,
            2.72831756129082,
            1.43751744494739,
            -0.308762554117067,
            3.16374541998891,
            3.25228883127485,
            0.497205639226081,
            0.563983730067662,
            -0.699581052977720,
            -0.546998599593366,
            -0.933723996508178,
            -1.11943008080303,
            -1.50352213011539,
            -1.15959023129167,
            -1.05748423279695,
            -3.79845336300602,
            -2.94967853722164,
            -3.07951218594857,
            -2.91307019047500,
            -3.72468829824916,
            -4.53680080264178,
            -3.38088050661469,
            -3.34350455842136,
            -4.27814286944599,
            -4.27962023480501,
            -3.80468990632255,
            -3.76477549791434,
            -3.82871978564370,
            -3.83791315639121,
            -3.77954748991920,
            -3.71507929136158,
            -3.71019655230597,
            -3.67866320214353,
            -4.58147416553331,
            -4.60154948641892,
        ]
        a3 = [
            -3.45268775679799,
            -3.43452287063569,
            -3.39702586910112,
            -3.11503255024795,
            -1.83354272671367,
            -0.270888507183879,
            1.68687428234017,
            3.38584872753184,
            4.31442394655653,
            3.20509288854805,
            1.25494456347615,
            -0.560586710039788,
            -1.95890658562151,
            -3.04364572811952,
            -3.40801120848212,
            -3.59179778731285,
            -3.71590908304993,
            2.73612616352353,
            4.41975980255723,
            5.48090442327002,
            5.93428683358328,
            6.04524353810951,
            4.77636652496375,
            5.45201496897447,
            5.66667415132112,
            5.85564197809204,
            5.97514496547339,
            5.81112271327162,
            5.24220917106452,
            4.21076985539817,
            2.83367800367449,
            5.73862211091354,
            5.48545512715199,
            5.20148021574076,
            4.50562848966262,
            3.89260367082094,
            4.88722189513956,
            4.88316139537164,
            4.88248726248746,
            4.80491680840105,
            4.59040123923560,
            5.05274948677072,
            4.85307552118809,
            4.79902665964317,
            4.91906064732271,
            4.71598432433896,
            3.89290781208074,
            4.52768633501309,
            4.77430296217535,
            5.04739331112917,
            4.81188198347252,
            6.00244124275816,
            6.60136741247610,
            7.28652384214304,
            7.92859207328720,
            5.06950285203373,
            5.00127966505672,
            5.84865128412491,
            5.75185260320280,
            5.75017543794407,
            5.52538613137357,
            6.08379721696782,
            5.95677108950612,
            6.20871650186893,
            6.00348069005592,
            6.01346592340405,
            4.60623341955923,
            6.30093749035952,
            6.29854750706875,
            6.33296637708243,
            4.61576413232881,
            6.06648321146848,
            5.83924743900892,
            5.79814962477008,
            5.30575861778606,
            5.25608291235299,
            4.71198286788215,
            4.71612730132530,
            5.44674234351189,
            5.67387976840367,
            5.40407082537025,
            5.22101032807825,
            5.50772088343264,
            5.24190552446463,
            5.84767410591666,
            5.91148803019352,
        ]
    v_kp_ref = np.array([a1, a2, a3]).transpose()
    vertex_inliner = []
    for i in range(len(with_error_3d_points_slam)):
        if with_error_3d_points_slam[i][2] > 10 and i > 16 and (i < 57 or i > 81):
            vertex_inliner.append(i)
            if i == 55:
                nose_idx = len(vertex_inliner) - 1
    vertex_inliner = np.array(vertex_inliner)

    vertex_target = np.zeros((len(vertex_inliner), 3))
    vertex_key = np.zeros((len(vertex_inliner), 3))
    ind = 0
    for i in vertex_inliner:
        vertex_target[ind] = with_error_3d_points_slam[i]
        vertex_key[ind] = v_kp_ref[i]
        ind = ind + 1

    trans_next_2_slam = PoseTools.fit_icp_scale_RT_next_align_nose(
        vertex_key, vertex_target, nose_idx - 1
    )
    return trans_next_2_slam
