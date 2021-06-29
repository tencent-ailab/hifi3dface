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


flip_vtx_map86 = {
    0: 16,
    1: 15,
    2: 14,
    3: 13,
    4: 12,
    5: 11,
    6: 10,
    7: 9,
    8: 8,
    9: 7,
    10: 6,
    11: 5,
    12: 4,
    13: 3,
    14: 2,
    15: 1,
    16: 0,
    17: 30,
    18: 29,
    19: 28,
    20: 27,
    21: 26,
    22: 34,
    23: 33,
    24: 32,
    25: 31,
    26: 21,
    27: 20,
    28: 19,
    29: 18,
    30: 17,
    31: 25,
    32: 24,
    33: 23,
    34: 22,
    35: 46,
    36: 45,
    37: 44,
    38: 43,
    39: 48,
    40: 47,
    41: 49,
    42: 50,
    43: 38,
    44: 37,
    45: 36,
    46: 35,
    47: 40,
    48: 39,
    49: 41,
    50: 42,
    51: 51,
    52: 52,
    53: 53,
    54: 54,
    55: 56,
    56: 55,
    57: 58,
    58: 57,
    59: 60,
    60: 59,
    61: 65,
    62: 64,
    63: 63,
    64: 62,
    65: 61,
    66: 70,
    67: 69,
    68: 68,
    69: 67,
    70: 66,
    71: 71,
    72: 73,
    73: 72,
    74: 75,
    75: 74,
    76: 77,
    77: 76,
    78: 80,
    79: 79,
    80: 78,
    81: 83,
    82: 82,
    83: 81,
    84: 85,
    85: 84,
}


flip_vtx_map68 = {
    0: 16,
    1: 15,
    2: 14,
    3: 13,
    4: 12,
    5: 11,
    6: 10,
    7: 9,
    8: 8,
    9: 7,
    10: 6,
    11: 5,
    12: 4,
    13: 3,
    14: 2,
    15: 1,
    16: 0,
    17: 26,
    18: 25,
    19: 24,
    20: 23,
    21: 22,
    22: 21,
    23: 20,
    24: 19,
    25: 18,
    26: 17,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 35,
    32: 34,
    33: 33,
    34: 32,
    35: 31,
    36: 45,
    37: 44,
    38: 43,
    39: 42,
    40: 47,
    41: 46,
    42: 39,
    43: 38,
    44: 37,
    45: 36,
    46: 41,
    47: 40,
    48: 54,
    49: 53,
    50: 52,
    51: 51,
    52: 50,
    53: 49,
    54: 48,
    55: 59,
    56: 58,
    57: 57,
    58: 56,
    59: 55,
    60: 64,
    61: 63,
    62: 62,
    63: 61,
    64: 60,
    65: 67,
    66: 66,
    67: 65,
}

# index in matlab, should -1
flip_vtx_map102 = [
    33,
    32,
    31,
    30,
    29,
    28,
    27,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
    19,
    18,
    17,
    16,
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    47,
    46,
    45,
    44,
    43,
    51,
    50,
    49,
    48,
    38,
    37,
    36,
    35,
    34,
    42,
    41,
    40,
    39,
    63,
    62,
    61,
    60,
    65,
    64,
    66,
    67,
    55,
    54,
    53,
    52,
    57,
    56,
    58,
    59,
    68,
    69,
    70,
    71,
    73,
    72,
    75,
    74,
    77,
    76,
    82,
    81,
    80,
    79,
    78,
    87,
    86,
    85,
    84,
    83,
    88,
    90,
    89,
    92,
    91,
    94,
    93,
    97,
    96,
    95,
    100,
    99,
    98,
    102,
    101,
]

PerspCam = {}
PerspCam["fov_y"] = 30
PerspCam["r_scale"] = 15
PerspCam["far_clip"] = 1000
PerspCam["near_clip"] = 10
PerspCam["tz_scale"] = 10
PerspCam["tz_init"] = -700

OrthoCam = {}
OrthoCam["fov_y"] = 3
OrthoCam["r_scale"] = 10
OrthoCam["far_clip"] = 10000
OrthoCam["near_clip"] = 10
OrthoCam["tz_scale"] = 100
OrthoCam["tz_init"] = -20

# define some constant for segmentation region
SEG_BG = 0
SEG_SKIN = 1
SEG_NOSE = 2
SEG_EYEG = 3
SEG_LEYE = 4
SEG_REYE = 5
SEG_LBROW = 6
SEG_RBROW = 7
SEG_MOUTH = 10
SEG_ULIP = 11
SEG_LLIP = 12

# landmark indices for different regions (starting from 1)
lmk86_region = {
    "contour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "left_eyebrow": [18, 19, 20, 21, 22, 26, 25, 24, 23],
    "right_eyebrow": [27, 28, 29, 30, 31, 35, 34, 33, 32],
    "left_eye": [36, 37, 42, 38, 39, 40, 43, 41],
    "right_eye": [44, 45, 50, 46, 47, 48, 51, 49],
    "nose": [52, 53, 54, 55, 56, 58, 60, 62, 63, 64, 65, 66, 61, 59, 57],
    "noseline": [52, 53, 54, 55],
    "mouth": [
        67,
        73,
        68,
        69,
        70,
        74,
        71,
        78,
        81,
        80,
        79,
        77,
        82,
        83,
        84,
        75,
        86,
        72,
        85,
        76,
    ],
}
