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

import numpy as np
import cv2
import math
from .PoseTools import PoseTools


class chooseFrame(object):
    def __init__(self):
        pass

    @staticmethod
    def check_eye_close(pts2d, flag=0):
        # in : n * 2
        left_eye_lmk_index_v = [
            58 - 1,
            59 - 1,
            53 - 1,
            57 - 1,
            54 - 1,
            56 - 1,
        ]  # left eye lmk, vertical index
        left_eye_lmk_index_h = [52 - 1, 55 - 1]  # left eye lmk, horizontal index

        right_eye_lmk_index_v = [
            66 - 1,
            67 - 1,
            61 - 1,
            65 - 1,
            62 - 1,
            64 - 1,
        ]  # right eye lmk, vertical index
        right_eye_lmk_index_h = [60 - 1, 63 - 1]  # right eye lmk, horizontal index
        # flag=0->mid pose, flag=1->left/right pose
        if flag == 0:
            thred = 0.2
        else:
            thred = 0.5
        left_eye_distance = 0
        right_eye_distance = 0
        for ind in range(0, len(left_eye_lmk_index_v), 2):
            x1 = pts2d[left_eye_lmk_index_v[ind]][0]
            y1 = pts2d[left_eye_lmk_index_v[ind]][1]
            x2 = pts2d[left_eye_lmk_index_v[ind + 1]][0]
            y2 = pts2d[left_eye_lmk_index_v[ind + 1]][1]

            left_eye_distance = left_eye_distance + np.linalg.norm(
                np.array([x1, y1]) - np.array([x2, y2])
            )

            x3 = pts2d[right_eye_lmk_index_v[ind]][0]
            y3 = pts2d[right_eye_lmk_index_v[ind]][1]
            x4 = pts2d[right_eye_lmk_index_v[ind + 1]][0]
            y4 = pts2d[right_eye_lmk_index_v[ind + 1]][1]

            right_eye_distance = right_eye_distance + np.linalg.norm(
                np.array([x3, y3]) - np.array([x4, y4])
            )
        x5 = pts2d[left_eye_lmk_index_h[0]][0]
        y5 = pts2d[left_eye_lmk_index_h[0]][1]
        x6 = pts2d[left_eye_lmk_index_h[1]][0]
        y6 = pts2d[left_eye_lmk_index_h[1]][1]
        left_eyes_ratio = left_eye_distance / np.linalg.norm(
            np.array([x5, y5]) - np.array([x6, y6])
        )

        x7 = pts2d[right_eye_lmk_index_h[0]][0]
        y7 = pts2d[right_eye_lmk_index_h[0]][1]
        x8 = pts2d[right_eye_lmk_index_h[1]][0]
        y8 = pts2d[right_eye_lmk_index_h[1]][1]
        right_eyes_ratio = right_eye_distance / np.linalg.norm(
            np.array([x7, y7]) - np.array([x8, y8])
        )
        # print("left_eyes_ratio right_eyes_ratio ",left_eyes_ratio,right_eyes_ratio)
        result_status = 1
        if left_eyes_ratio < thred or right_eyes_ratio < thred:
            result_status = result_status - 2
        return result_status

    @staticmethod
    def get_abs_angle_by_orth_pnp(pt2d_all):
        #################   pt2d: (m*2*n),m is the number of pics ,n is the number of landmarks ############

        preset_PT_3d = np.array(
            [
                [
                    -68.1756973070174,
                    -68.1146470361586,
                    -66.0249619011911,
                    -62.5428837026772,
                    -56.7464771621017,
                    -47.0523705994082,
                    -31.2188583709116,
                    -15.3879495971038,
                    -4.08401359484161,
                    16.5929626641805,
                    33.7166876786039,
                    48.8605803831152,
                    56.8829519914283,
                    62.4308036010388,
                    65.3272669641152,
                    67.3733798672488,
                    67.1870150203456,
                    -54.3237949408625,
                    -45.9681810043115,
                    -35.0726032124679,
                    -24.8322654940756,
                    -12.7729299752417,
                    -45.8693519937360,
                    -36.3826769033241,
                    -25.9141249360281,
                    -14.8096785082511,
                    10.7624649776308,
                    20.5206483307513,
                    33.1056897313524,
                    46.6258909064902,
                    54.0926809905806,
                    12.6266583822557,
                    22.7049645972830,
                    34.4879223577708,
                    44.0535131917368,
                    -41.2673480094191,
                    -35.6089130191105,
                    -21.7764329890458,
                    -16.7468454026988,
                    -25.5757409620726,
                    -37.2345564965293,
                    -28.3139941581866,
                    -31.6768869306443,
                    17.6041471849675,
                    22.6210777240253,
                    34.0961319432436,
                    41.9045520975750,
                    36.7447958534758,
                    25.3302610383278,
                    27.9211579237679,
                    31.9237846382224,
                    -0.583686964484780,
                    0.144336413802457,
                    0.269583123001013,
                    0.229647359245878,
                    -10.8552873508045,
                    8.80169774995196,
                    -14.1943604935433,
                    12.3636086468133,
                    -17.6260973432672,
                    16.1859617393272,
                    -11.8561671740247,
                    -6.14033382730938,
                    0.122603013080397,
                    4.51988955136432,
                    11.2784146784043,
                    -22.3665467791695,
                    -5.12694742849893,
                    -0.261962445311256,
                    5.04729862351575,
                    21.9230047599515,
                    2.19959699419832,
                    -14.3228422073861,
                    14.7897345459004,
                    16.5511881096132,
                    -16.7100684305587,
                    -18.5092145428952,
                    19.6847364244945,
                    -10.4805964476244,
                    0.189005281495954,
                    10.7239229159183,
                    -10.1233814192337,
                    0.733705124663359,
                    10.7879717245791,
                    -7.09363186602643,
                    9.83684660832385,
                ],
                [
                    53.3107940578386,
                    33.6623335532468,
                    13.3328235939212,
                    -6.70656176326401,
                    -26.2279897952572,
                    -39.8772599748791,
                    -50.6031722705214,
                    -55.0682077751379,
                    -54.7205192995213,
                    -54.4383527884069,
                    -45.6074643123763,
                    -36.7972070532683,
                    -26.7844743779424,
                    -8.31142380648254,
                    12.9211104101711,
                    32.3868187022165,
                    52.8705611396547,
                    55.7094497224889,
                    65.4640357632679,
                    68.0262305498897,
                    67.0594026704289,
                    65.1248359510029,
                    58.7888308171010,
                    60.8175914739616,
                    60.2800548576361,
                    57.9635621511934,
                    64.1839275608599,
                    66.9301273508612,
                    68.1410948144991,
                    66.2383559145425,
                    55.8636845216315,
                    57.9597060181801,
                    62.0114248050252,
                    61.3778094800454,
                    59.3384362324800,
                    45.4503766622321,
                    49.4789679497268,
                    49.6233682727380,
                    44.7218523846120,
                    42.5597018661359,
                    43.5828799351429,
                    50.9429154037993,
                    41.9671755981695,
                    44.3637555869468,
                    49.5010235719469,
                    50.2143139274464,
                    46.1853031649069,
                    43.5934880814510,
                    43.1457356201204,
                    50.7859006873030,
                    41.9688807138128,
                    52.2133605010753,
                    40.8606102496554,
                    29.8673390697036,
                    18.5682308483948,
                    47.8126617850450,
                    46.9061150639963,
                    23.1271685381814,
                    23.8289873865312,
                    13.2948429656895,
                    12.6521944496779,
                    10.1998621404392,
                    8.07936564185550,
                    7.02048790726341,
                    7.96680610142852,
                    8.85964829325156,
                    -15.4685231851887,
                    -5.83694476072372,
                    -7.18510079110145,
                    -6.04296059535911,
                    -14.6904905468550,
                    -21.9680454218847,
                    -9.29459523117953,
                    -9.09722248628968,
                    -20.0443614430531,
                    -20.5127768888248,
                    -15.2511924645571,
                    -14.5447326599110,
                    -14.2644213165510,
                    -14.4990315246071,
                    -14.0522679543919,
                    -14.6584850106918,
                    -14.8322370025323,
                    -14.3013495881650,
                    -22.1447253850458,
                    -21.1082651587302,
                ],
                [
                    -23.8053118343091,
                    -27.0445038733228,
                    -25.4519279993035,
                    -21.3747758708710,
                    -10.0427175318756,
                    2.26734761619467,
                    18.4816765410735,
                    34.3221442669071,
                    40.5328508971239,
                    34.8712700765862,
                    22.2864591376731,
                    6.21563701787038,
                    -13.1134717329451,
                    -16.4644090354300,
                    -23.5839491111222,
                    -20.1740307731121,
                    -20.9111590403862,
                    14.2565903252940,
                    24.3935737144901,
                    32.3840414727001,
                    37.1291429990255,
                    39.9932199086069,
                    27.2161675645801,
                    33.7749032608401,
                    36.8089431088900,
                    37.8626154364139,
                    40.2974535831101,
                    38.6138476948733,
                    33.2010273723652,
                    22.5020632801053,
                    13.7408293547814,
                    38.2504652330102,
                    37.6996568238641,
                    34.4917209315425,
                    28.6782251624995,
                    25.2856914494286,
                    31.2533888265002,
                    32.6627868245544,
                    30.0533047871161,
                    32.5050230824477,
                    28.5422698190148,
                    33.5168371456454,
                    31.5280683519493,
                    31.0209067620750,
                    33.3543299414210,
                    32.2212854309802,
                    24.3953248646461,
                    28.9265332073268,
                    33.0077738148317,
                    33.6859149016717,
                    31.0602176320082,
                    40.3393280766325,
                    45.6002020387433,
                    55.0202454254639,
                    62.8978452597018,
                    34.8476961173754,
                    36.5202392654199,
                    43.3278891974560,
                    45.9282464018630,
                    43.4287421024055,
                    47.8949929648990,
                    48.6608554911977,
                    49.5354062065318,
                    53.6401343965816,
                    50.5486291847995,
                    47.9518469656950,
                    40.8297273633096,
                    53.0054425832235,
                    54.0679016180019,
                    53.2652104521736,
                    41.4592065997822,
                    51.9548140335428,
                    48.7692056980377,
                    48.8730272580562,
                    44.6779594870922,
                    44.0230121149454,
                    43.2977936185804,
                    42.8604352591731,
                    48.2764306655049,
                    51.5130563672583,
                    48.6159735298573,
                    47.5215996627314,
                    50.4681746449472,
                    47.7544272187955,
                    50.4640358517105,
                    49.7103891466216,
                ],
            ]
        )

        preset_PT_3d = preset_PT_3d.transpose()  # n * 3

        length = len(pt2d_all)
        phi_all = []
        gamma_all = []
        for i in range(length):

            one_pt2d = pt2d_all[i].transpose()  # n * 2
            pt2d = one_pt2d - np.tile(np.mean(one_pt2d, axis=0), (one_pt2d.shape[0], 1))
            phi, gamma, theta, t3d, f = PoseTools.pnp_orth(pt2d, preset_PT_3d)
            phi1 = phi * 180 / 3.14

            if phi1 > 90:
                phi1 = phi1 - 180
            if phi1 < -90:
                phi1 = phi1 + 180
            phi_all.append(-phi1)
            gamma_all.append(-gamma * 180 / 3.14)

        return np.array(phi_all), np.array(gamma_all)

    @staticmethod
    def ransac_rigid_trans(source, target, n_pairs):
        # in : n * 3
        allowed_error = 6
        nBest = 0
        nPoints = source.shape[0]
        inliers = []

        def computeInliers(trans, source, target, allowedError):
            allowed_error = 6
            nPoints = source.shape[0]

            idx = []
            for i in range(nPoints):
                p1 = source[i, :]
                p2 = target[i, :]
                p1 = np.hstack((p1, 1))
                p1Trans = np.dot(trans, p1.transpose())
                distance = np.linalg.norm(p1Trans - p2)
                if distance < allowed_error:
                    idx.append(i)
            return idx

        for i in range(50):
            samples = np.random.randint(0, nPoints, size=10)

            currentTransformation = PoseTools.fit_icp_RT_no_scale(
                source[samples, :], target[samples, :]
            )
            currentInliers = computeInliers(
                currentTransformation, source, target, allowed_error
            )

            if not currentInliers:
                continue
            n_current_inliers = len(currentInliers)
            # print(n_current_inliers)
            if n_current_inliers > nBest:
                nBest = n_current_inliers
                inliers = currentInliers
        if nBest <= n_pairs:
            trans = []
        trans = PoseTools.fit_icp_RT_no_scale(source[inliers, :], target[inliers, :])
        return trans, inliers, nBest

    @staticmethod
    def call_one_pose_by_DLT(useful_points, in_ref3d, in_pt3d):
        # in : n * 3
        target = in_ref3d[useful_points, :]
        source = in_pt3d[useful_points, :]

        valid_source = []
        valid_target = []
        for i in range(len(target)):
            if target[i][2] > 100 and target[i][2] < 1000:
                valid_target.append(i)
            if source[i][2] > 100 and source[i][2] < 1000:
                valid_source.append(i)
        valid_target = set(valid_target)
        valid_source = set(valid_source)
        valid_ind = list(valid_target & valid_source)

        if len(valid_ind) < 15:
            return -1, None
        source = source[valid_ind, :]
        target = target[valid_ind, :]

        trans, inliers, nInliners = chooseFrame.ransac_rigid_trans(source, target, 10)
        # print(trans, len(inliers), nInliners)
        if nInliners <= 10:
            return -1, None
        return 1, trans

    @staticmethod
    def find_close_angle_index(points, angle):
        index = []
        state = points >= angle
        for i in range(len(state) - 1):
            if state[i] != state[i + 1]:
                if abs(points[i] - angle) < abs(points[i + 1] - angle):
                    index.append(i)
                else:
                    index.append(i + 1)
        return index

    @staticmethod
    def rotationMatrixToEulerAngles(R):
        r1 = math.atan2(R[1][0], R[0][0])
        r2 = math.asin(-R[2][0])
        r3 = math.atan2(R[2][1], R[2][2])
        r = [r1, r2, r3]
        return r

    @staticmethod
    def call_motion_blur(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        num_depth = cv2.filter2D(img_gray, -1, mask)
        # num_depth = scipy.ndimage.correlate(img_gray, mask,mode='nearest')
        (mean, stddv) = cv2.meanStdDev(num_depth)
        score = stddv[0][0]
        return score

    @staticmethod
    def find_chin(pt3d_all, in_up_down_all, left_right_all, ref_id, img_all):
        id_chin = []
        part_chin = []
        flag_ok = 0
        ref_angle = in_up_down_all[ref_id]
        up_down_all = in_up_down_all
        up_down_all[np.where(abs(left_right_all) > 20)] = ref_angle
        min_chin = np.min(np.where(up_down_all == np.min(up_down_all)))
        thres = np.max((7, (ref_angle - up_down_all[min_chin]) / 3 + ref_angle))
        preset_max_chin_angle = ref_angle - thres

        if up_down_all[min_chin] > preset_max_chin_angle:
            flag_ok = -1
            trans_chin = []

        chin_cadidate = chooseFrame.find_close_angle_index(
            up_down_all, preset_max_chin_angle
        )
        if len(chin_cadidate) == 1:
            idx_begin = chin_cadidate[0]
            idx_end = len(up_down_all)
        else:
            idx_begin = chin_cadidate[-2]
            idx_end = chin_cadidate[-1]
        useful_points = [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            17,
            22,
            23,
            24,
            25,
            30,
            31,
            32,
            33,
            34,
            66,
            70,
            71,
            74,
            75,
            84,
            85,
        ]
        ref3d = pt3d_all[ref_id][:, useful_points]
        valid_target = np.where(
            (ref3d[2, :] > 100) & (ref3d[2, :] < 1000)
        )  # Depth prior of face selfie scene
        blur_score_all = []
        angles_all = []
        idx_all = []
        trans_all = []
        count_ok_frame = 0
        for i in range(idx_begin, idx_end):
            pt3d = pt3d_all[i][:, useful_points]
            source = pt3d
            target = ref3d
            valid_source = np.where(
                (source[2, :] > 100) & (source[2, :] < 1000)
            )  # Depth prior of face selfie scene

            valid_source1 = []
            for j in range(len(valid_source[0])):
                valid_source1.append(valid_source[0][j])

            valid_target1 = []
            for j in range(len(valid_target[0])):
                valid_target1.append(valid_target[0][j])
            valid_ind = list(set(valid_source1).intersection(set(valid_target1)))
            if len(valid_ind) < 15:
                print("not enough inliner in pose calulation", i)
            source = source[:, valid_ind]
            target = target[:, valid_ind]
            trans, inliers, nInliners = chooseFrame.ransac_rigid_trans(
                source.transpose(), target.transpose(), 10
            )
            if nInliners <= 10:
                print(
                    "not enough inliner in pose calulation  ",
                    i,
                    " nInliners: ",
                    nInliners,
                )
            angle = chooseFrame.rotationMatrixToEulerAngles(trans[:, 0:3])
            angle1 = ref_angle + -1 * abs(angle[2] * 180 / 3.14)
            if abs(angle1) < 30:
                b = chooseFrame.call_motion_blur(img_all[i])
                blur_score_all.append(b)
                angles_all.append(angle1)
                idx_all.append(i)
                trans_all.append(trans)
                count_ok_frame = count_ok_frame + 1

        if count_ok_frame == 0:
            print("no chin   !!!!")
            flag_ok = -1
            return [], [], [], flag_ok

        threshold = -20
        for i in range(len(angles_all)):
            angles_all[i] = angles_all[i] - threshold
        b = np.argsort((abs(np.array(angles_all))))
        a = np.sort(abs(np.array(angles_all)))
        len1 = int(np.ceil(len(b) / 3))
        c = angles_all.index(max(angles_all[0 : len1 + 1]))
        id1 = b[c]
        id_chin = idx_all[id1]
        trans_chin = trans_all[id1]
        part_chin = chin_cadidate
        flag_ok = 1
        return id_chin, trans_chin, part_chin, flag_ok

    @staticmethod
    def find_two_part_candidate_by_sort(left_right_all, ref_id, part_chin):
        maxx = -1000000
        minn = 1000000
        leftmost_ind = -1
        rightmost_ind = -1
        for i in range(len(left_right_all)):
            if maxx < left_right_all[i]:
                maxx = left_right_all[i]
                leftmost_ind = i
            if minn >= left_right_all[i]:
                minn = left_right_all[i]
                rightmost_ind = i
        Left_min_ANGEL = 10
        Right_min_ANGEL = -10
        Left_MAX_ANGEL = 50
        Right_MAX_ANGEL = -50
        if (left_right_all[ref_id]) > (Left_min_ANGEL) - 5:
            Left_min_ANGEL = left_right_all[ref_id] + 5
        if left_right_all[ref_id] < Right_min_ANGEL + 5:
            Right_min_ANGEL = left_right_all[ref_id] - 5
        if left_right_all[leftmost_ind] < Left_MAX_ANGEL:
            Left_MAX_ANGEL = left_right_all[leftmost_ind] - 1
        if left_right_all[rightmost_ind] > Right_MAX_ANGEL:
            Right_MAX_ANGEL = left_right_all[rightmost_ind] + 1
        print(
            "   min left is : ", Left_min_ANGEL, "    max left  is : ", Left_MAX_ANGEL
        )
        print(
            "   min right is : ",
            Right_min_ANGEL,
            "    max right is : ",
            Right_MAX_ANGEL,
        )

        left_with_chin = np.where(
            (left_right_all > Left_min_ANGEL) & (left_right_all < Left_MAX_ANGEL)
        )
        right_with_chin = np.where(
            (left_right_all < Right_min_ANGEL) & (left_right_all > Right_MAX_ANGEL)
        )
        left1 = np.sort(
            list(set(list(left_with_chin[0])).difference(set(list(part_chin))))
        )
        right1 = np.sort(
            list(set(list(right_with_chin[0])).difference(set(list(part_chin))))
        )

        part_candidate_left = left1
        if len(part_candidate_left) < 1:
            part_candidate_left = left_with_chin
            print(" warning！！！ left face is divided into the chin section")
        part_candidate_right = right1
        if len(part_candidate_right) < 1:
            part_candidate_right = right_with_chin
            print(" warning！！！ right face is divided into the chin section")
        return part_candidate_left, part_candidate_right

    @staticmethod
    def find_best_imgs_in_part(
        ref_lr_angle,
        part_candidate,
        img_all,
        pt3d_all,
        pt2d_all,
        ref_id,
        part_lmk_index,
    ):
        """
        Between the two segments, limit the minimum interval angle to sample, at the same time delete the points with obviously poor quality,
        and recalculate the exact pose relative to the first frame
        Here are the calculations of the rotation and translation relative to the first frame, then use 3d points as icp, plus ransac to filter
        The previous pnp method cannot be used here, because the average 3d point of fwh does not match this person and cannot be used to filter the outliner

        The idea here is almost the same as choosing the chin, the difference is that only one-sided inliner is used when calculating the pose,
        and the logic of selecting the last few pictures
        """
        useful_points = part_lmk_index
        ref3d = pt3d_all[ref_id][:, useful_points]
        valid_target = np.where(
            (ref3d[2, :] > 100) & (ref3d[2, :] < 1000)
        )  # Depth prior of face selfie scene
        angles_all = []
        blur_score_all = []
        good_part_candidate = []
        close_eye_all = []
        trans_all = []
        count_ok_frame = 0
        for ite in range(len(part_candidate)):
            i = part_candidate[ite]
            pt3d = pt3d_all[i][:, useful_points]
            source = pt3d
            target = ref3d
            valid_source = np.where((source[2, :] > 100) & (source[2, :] < 1000))
            valid_ind = list(
                set(list(valid_source[0])).intersection(set(list(valid_target[0])))
            )
            if len(valid_ind) < 15:
                print("not enough inliner in pose calulation", i)
            source = source[:, valid_ind]
            target = target[:, valid_ind]
            trans, inliers, nInliners = chooseFrame.ransac_rigid_trans(
                source.transpose(), target.transpose(), 10
            )
            if nInliners <= 10:
                print(
                    "not enough inliner in pose calulation  ",
                    i,
                    " nInliners: ",
                    nInliners,
                )
            angle = chooseFrame.rotationMatrixToEulerAngles(trans[:, 0:3])
            angle = ref_lr_angle + angle[1] * 180 / 3.14
            if abs(angle) < 40 and abs(angle) > 10:
                angles_all.append(angle)
                good_part_candidate.append(i)
                close_eye_all.append(
                    chooseFrame.check_eye_close(pt2d_all[i].transpose(), 1)
                )
                count_ok_frame = count_ok_frame + 1
                trans_all.append(trans)
                score = chooseFrame.call_motion_blur(img_all[i])
                blur_score_all.append(score)
        if len(angles_all) == 0:
            out_idx = 0
            out_trans_all = []
            return
        angles_all = np.array(angles_all)
        angles_all = abs(angles_all)
        out_idx = []
        num_class = 7
        range_angle = 1 + max(abs(angles_all)) - min(abs(angles_all))
        per_angle = range_angle / (num_class + 1)
        class1 = np.floor((abs(angles_all) - min(abs(angles_all))) / per_angle)
        index_in_all = []
        final_motion_blur_score = []
        for classid in range(int(max(class1)), 0, -1):
            caditate = np.where(class1 == classid)
            ok_cadidate = np.array(caditate[0])[
                np.where(np.array(close_eye_all)[caditate] > 0)
            ]
            if len(ok_cadidate) > 0:
                indx = np.argmax(np.array(blur_score_all)[ok_cadidate])
                final_motion_blur_score.append(blur_score_all[int(ok_cadidate[indx])])
                out_idx.append(good_part_candidate[ok_cadidate[indx]])
                index_in_all.append(ok_cadidate[indx])
        if len(out_idx) == 0:
            out_idx = np.argmax(angles_all)
            out_trans_all = trans_all[out_idx]
            out_score = 1
        else:
            out_trans_all = np.zeros((len(out_idx), 3, 4))
            for i in range(len(out_idx)):
                out_trans_all[i] = trans_all[index_in_all[i]]
            out_score = final_motion_blur_score
        return out_idx, out_trans_all, out_score

    @staticmethod
    def find_left_right_part(
        left_right_all, img_all, pt3d_all, pt2d_all, ref_id, part_chin
    ):
        flag_ok = 1
        (
            part_candidate_left,
            part_candidate_right,
        ) = chooseFrame.find_two_part_candidate_by_sort(
            left_right_all, ref_id, part_chin
        )
        if len(part_candidate_left) == 0:
            print(" error！！！ Missing left face data")
            flag_ok = -1
        if len(part_candidate_right) == 0:
            print("  error！！！ Missing right face data")
            flag_ok = -1
        left_lmk_index = [
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            56,
            58,
            60,
            62,
            63,
            67,
            68,
            73,
            76,
            77,
            79,
            82,
            85,
        ]
        right_lmk_index = [
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            57,
            59,
            61,
            65,
            66,
            70,
            71,
            74,
            75,
            78,
            81,
            84,
            86,
        ]
        for i in range(len(left_lmk_index)):
            left_lmk_index[i] = left_lmk_index[i] - 1
            right_lmk_index[i] = right_lmk_index[i] - 1
        idx_left, trans_left, score_left = chooseFrame.find_best_imgs_in_part(
            left_right_all[ref_id],
            part_candidate_left,
            img_all,
            pt3d_all,
            pt2d_all,
            ref_id,
            left_lmk_index,
        )
        idx_right, trans_right, score_right = chooseFrame.find_best_imgs_in_part(
            left_right_all[ref_id],
            part_candidate_right,
            img_all,
            pt3d_all,
            pt2d_all,
            ref_id,
            right_lmk_index,
        )
        if len(trans_left) == 0:
            print(" error!!! some wrong in left pose calculation")
            flag_ok = -1
        if len(trans_right) == 0:
            print("error!!! some wrong in right pose calculation")
            flag_ok = -1
        return (
            idx_left,
            trans_left,
            idx_right,
            trans_right,
            flag_ok,
            score_left,
            score_right,
        )
