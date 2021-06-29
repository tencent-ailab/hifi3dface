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

""" convert texture from output domain to input domain """

import glob
import os
from PIL import Image
import numpy as np
import argparse
import sys

sys.path.append("..")
from utils.misc import blend_uv

parser = argparse.ArgumentParser()
parser.add_argument("--input_fit_dir", help="path to folder containing images")
parser.add_argument("--input_unwrap_dir", help="path to folder containing images")
parser.add_argument("--input_pix2pix_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="path to folder containing output images")
parser.add_argument(
    "--base_tex_path",
    default="../resources/base_tex.png",
    help="path to folder containing output images",
)
a = parser.parse_args()


def main():

    mask_path = "../resources/mask_used.png"
    fit_path = os.path.join(a.input_fit_dir, "out_for_texture_tex_D.png")
    pix2pix_path = os.path.join(a.input_pix2pix_dir, "out_for_texture_tex_D.png")
    unwrap_path = os.path.join(a.input_unwrap_dir, "out_for_texture_tex.png")

    mask_img = (
        np.asarray(Image.open(mask_path).resize((2048, 2048)), np.float32) / 255.0
    )
    mask_img = mask_img[..., np.newaxis]
    fit_img = np.asarray(Image.open(fit_path), np.float32)
    pix2pix_img = np.asarray(Image.open(pix2pix_path), np.float32)
    unwrap_img = np.asarray(Image.open(unwrap_path).resize((2048, 2048)), np.float32)

    fit_mu = np.sum(fit_img * mask_img, axis=(0, 1)) / np.sum(mask_img, axis=(0, 1))
    pix2pix_mu = np.sum(pix2pix_img * mask_img, axis=(0, 1)) / np.sum(
        mask_img, axis=(0, 1)
    )

    pix2pix_img = pix2pix_img - pix2pix_mu + fit_mu
    pix2pix_img = np.clip(pix2pix_img, 0, 255)

    mask_img = np.concatenate([mask_img] * 3, axis=-1)
    pix2pix_img = blend_uv(
        fit_img / 255, pix2pix_img / 255, mask_img, match_color=False, times=7
    )
    pix2pix_img = pix2pix_img * 255
    unwrap_img = blend_uv(
        fit_img / 255, unwrap_img / 255, mask_img, match_color=False, times=7
    )
    unwrap_img = unwrap_img * 255

    if os.path.exists(a.output_dir) is False:
        os.makedirs(a.output_dir)

    Image.fromarray(pix2pix_img.astype(np.uint8)).save(
        os.path.join(a.output_dir, "output_for_texture_tex_D.png")
    )
    Image.fromarray(unwrap_img.astype(np.uint8)).save(
        os.path.join(a.output_dir, "output_for_texture_tex.png")
    )


if __name__ == "__main__":
    main()
