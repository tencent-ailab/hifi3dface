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

"""
modified from https://github.com/affinelayer/pix2pix-tensorflow.

1. data normalization are conducted using statistics of the texture/normal data distribution.
2. Network architecture are modified slightly
3. We add total-variance loss for texture synthesis, and cosine distance loss for normal synthesis.
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from PIL import Image
import cv2

import sys

sys.path.append("..")
from utils.losses import Losses
from utils.misc import blend_uv

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, help="[texture / normal]")
parser.add_argument("--func", default="train", help="[train / freeze / test]")
parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="directory with checkpoint to resume training from or use for testing",
)
parser.add_argument("--pb_path", default=None, help="protobuf file path")

parser.add_argument(
    "--max_steps", type=int, help="number of training steps (0 to disable)"
)
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument(
    "--summary_freq",
    type=int,
    default=100,
    help="update summaries every summary_freq steps",
)
parser.add_argument(
    "--progress_freq",
    type=int,
    default=50,
    help="display progress every progress_freq steps",
)
parser.add_argument(
    "--trace_freq", type=int, default=0, help="trace execution every trace_freq steps"
)
parser.add_argument(
    "--display_freq",
    type=int,
    default=0,
    help="write current training images every display_freq steps",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=5000,
    help="save model every save_freq steps, 0 to disable",
)

parser.add_argument(
    "--separable_conv",
    action="store_true",
    help="use separable convolutions in the generator",
)
parser.add_argument(
    "--aspect_ratio",
    type=float,
    default=1.0,
    help="aspect ratio of output images (width/height)",
)
parser.add_argument(
    "--lab_colorization",
    action="store_true",
    help="split input image into brightness (A) and color (B)",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="number of images in batch"
)
parser.add_argument(
    "--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"]
)
parser.add_argument(
    "--ngf",
    type=int,
    default=64,
    help="number of generator filters in first conv layer",
)
parser.add_argument(
    "--ndf",
    type=int,
    default=64,
    help="number of discriminator filters in first conv layer",
)
parser.add_argument(
    "--scale_size",
    type=int,
    default=1024,
    help="scale images to this size before cropping to 256x256",
)
parser.add_argument(
    "--flip", dest="flip", action="store_true", help="flip images horizontally"
)
parser.add_argument(
    "--no_flip",
    dest="flip",
    action="store_false",
    help="don't flip images horizontally",
)
parser.set_defaults(flip=True)
parser.add_argument(
    "--lr", type=float, default=0.001, help="initial learning rate for adam"
)
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument(
    "--l1_weight",
    type=float,
    default=100.0,
    help="weight on L1 term for generator gradient",
)
parser.add_argument(
    "--tv_weight",
    type=float,
    default=1e-3,
    help="weight on TV term for generator gradient",
)
parser.add_argument(
    "--cos_weight",
    type=float,
    default=1e-3,
    help="weight on cosine distance for generator gradient",
)
parser.add_argument(
    "--gan_weight",
    type=float,
    default=1.0,
    help="weight on GAN term for generator gradient",
)

parser.add_argument(
    "--base_tex_path",
    type=str,
    default="../resources/base_tex.png",
    help="path to base texture file",
)
parser.add_argument(
    "--base_normal_path",
    type=str,
    default="../resources/base_normal.png",
    help="path to base normal file",
)
parser.add_argument(
    "--mu_tex_path",
    type=str,
    default="../resources/mu_tex.npy",
    help="path to mu texture",
)
parser.add_argument(
    "--std_tex_path",
    type=str,
    default="../resources/std_tex.npy",
    help="path to std texture",
)
parser.add_argument(
    "--mu_norm_path",
    type=str,
    default="../resources/mu_norm.npy",
    help="path to mu normal",
)
parser.add_argument(
    "--std_norm_path",
    type=str,
    default="../resources/std_norm.npy",
    help="path to std normal",
)
# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 1000
START_Y = 510
END_Y = 1510
START_X = 548
END_X = 1498

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch"
)

if a.func == "train":
    if a.mode.startswith("tex"):
        Model = collections.namedtuple(
            "Model",
            "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_tv, gen_grads_and_vars, train",
        )
    else:
        Model = collections.namedtuple(
            "Model",
            "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_cos, gen_grads_and_vars, train",
        )
elif a.func == "freeze":
    Model = collections.namedtuple("Model", "outputs")


# mean and std of normal and texture
mu_texture = np.load(a.mu_tex_path)
mu_normal = np.load(a.mu_norm_path)
mu_texture = mu_texture[START_Y:END_Y, START_X:END_X, :]
mu_normal = mu_normal[START_Y:END_Y, START_X:END_X, :]
mu_texture = tf.constant(mu_texture, tf.float32)
mu_normal = tf.constant(mu_normal, tf.float32)

std_texture = np.load(a.std_tex_path)
std_normal = np.load(a.std_norm_path)
std_texture = std_texture[START_Y:END_Y, START_X:END_X, :]
std_normal = std_normal[START_Y:END_Y, START_X:END_X, :]
std_texture = tf.constant(std_texture, tf.float32)
std_normal = tf.constant(std_normal, tf.float32)


def normalize(image, mean=None, std=None):
    if mean is None:
        norm_image = image / 255.0 * 2 - 1
    else:
        norm_image = (image - mean) / std
    return norm_image


def denormalize(image, mean=None, std=None):
    if mean is None:
        denorm_image = (image + 1) / 2 * 255
    else:
        denorm_image = image * std + mean
    return denorm_image


def preprocess(image, mean=None, std=None):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        if mean is None:
            return image * 2 - 1
        elif std is None:
            return image * 255.0 - mean
        else:
            return (image * 255.0 - mean) / std


def deprocess(image, mean=None, std=None):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        if mean is None:
            return (image + 1) / 2
        elif std is None:
            return tf.clip_by_value((image + mean) / 255.0, 0, 1)
        else:
            return tf.clip_by_value((tf.multiply(image, std) + mean) / 255.0, 0, 1)


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(
        batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"
    )
    return tf.layers.conv2d(
        padded_input,
        out_channels,
        kernel_size=4,
        strides=(stride, stride),
        padding="valid",
        kernel_initializer=tf.random_normal_initializer(0, 0.02),
    )


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return tf.layers.separable_conv2d(
            resized_input,
            out_channels,
            kernel_size=4,
            strides=(1, 1),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d_transpose(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    if a.func == "train":
        is_train = True
    else:
        is_train = True
    return tf.layers.batch_normalization(
        inputs,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=is_train,
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
    )


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels"
    )
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                ((srgb_pixels + 0.055) / 1.055) ** 2.4
            ) * exponential_mask
            rgb_to_xyz = tf.constant(
                [
                    #    X        Y          Z
                    [0.412453, 0.212671, 0.019334],  # R
                    [0.357580, 0.715160, 0.119193],  # G
                    [0.180423, 0.072169, 0.950227],  # B
                ]
            )
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(
                xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754]
            )

            epsilon = 6 / 29
            linear_mask = tf.cast(
                xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32
            )
            exponential_mask = tf.cast(
                xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32
            )
            fxfyfz_pixels = (
                xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29
            ) * linear_mask + (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant(
                [
                    #  l       a       b
                    [0.0, 500.0, 0.0],  # fx
                    [116.0, -500.0, 200.0],  # fy
                    [0.0, 0.0, -200.0],  # fz
                ]
            )
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant(
                [-16.0, 0.0, 0.0]
            )

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant(
                [
                    #   fx      fy        fz
                    [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                    [1 / 500.0, 0.0, 0.0],  # a
                    [0.0, 0.0, -1 / 200.0],  # b
                ]
            )
            fxfyfz_pixels = tf.matmul(
                lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz
            )

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                fxfyfz_pixels ** 3
            ) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant(
                [
                    #     r           g          b
                    [3.2404542, -0.9692660, 0.0556434],  # x
                    [-1.5371385, 1.8760108, -0.2040259],  # y
                    [-0.4985314, 0.0415560, 1.0572252],  # z
                ]
            )
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055
            ) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(
            raw_input, dtype=tf.float32
        )  # [0,255] => [0,1]

        assertion = tf.assert_equal(
            tf.shape(raw_input)[2], 3, message="image does not have 3 channels"
        )
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        # raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            # width = raw_input.get_shape().as_list()[1]
            print(width)
            if a.mode.startswith("tex"):
                # a_images = raw_input[:,:width//4,:]
                # b_images = raw_input[:,width//2:3*width//4,:]
                a_images = raw_input[:, : width // 4, :]
                b_images = raw_input[:, width // 2 : 3 * width // 4, :]
                # a_images = raw_input[:,:width//2,:]
                # b_images = raw_input[:,width//2:,:]
                # print('width = %d' % width)
                a_images = preprocess(a_images, mu_texture, std_texture)
                b_images = preprocess(b_images, mu_texture, std_texture)
            else:
                a_rgbimages = preprocess(
                    raw_input[:, : width // 4, :], mu_texture, std_texture
                )
                a_nolimages = preprocess(
                    raw_input[:, width // 4 : width // 2, :], mu_normal, std_normal
                )
                b_rgbimages = preprocess(
                    raw_input[:, width // 2 : 3 * width // 4, :],
                    mu_texture,
                    std_texture,
                )
                b_nolimages = preprocess(
                    raw_input[:, 3 * width // 4 :, :], mu_normal, std_normal
                )
                a_images = tf.concat([a_nolimages, b_rgbimages], axis=2)
                b_images = b_nolimages
                a_images.set_shape([None, None, 6])
                b_images.set_shape([None, None, 3])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image, mask=None, aug=False):
        r = image
        m = mask
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
            if m is not None:
                m = tf.image.random_flip_left_right(m, seed=seed)
        if aug:
            r.set_shape((END_Y - START_Y, END_X - START_X, 3))
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(
            r,
            [a.scale_size, int(round(a.scale_size * a.aspect_ratio))],
            method=tf.image.ResizeMethod.AREA,
        )
        offset = [0, 0]
        offset[0] = tf.cast(
            tf.random_uniform([1], 0, a.scale_size - CROP_SIZE + 1, seed=seed),
            dtype=tf.int32,
        )[0]
        offset[1] = tf.cast(
            tf.random_uniform(
                [1],
                0,
                int(round(a.scale_size * a.aspect_ratio))
                - int(round(CROP_SIZE * a.aspect_ratio))
                + 1,
                seed=seed,
            ),
            dtype=tf.int32,
        )[0]
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(
                r,
                offset[0],
                offset[1],
                CROP_SIZE,
                int(round(CROP_SIZE * a.aspect_ratio)),
            )
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        if a.mode.startswith("tex"):
            mask = Image.open(a.regional_mask_path)
            mask = np.asarray(mask, np.float32) / 255.0
            mask = mask[START_Y:END_Y, START_X:END_X, :]
            input_images = transform(inputs, mask, True)
        else:
            input_images = transform(inputs)

    with tf.name_scope("target_images"):
        if a.mode.startswith("tex"):
            target_images = transform(targets, mask, False)
        else:
            target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch(
        [paths, input_images, target_images], batch_size=a.batch_size
    )
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [
            0,
            tf.cast((x1_shape[1] - x2_shape[1]) // 2, tf.int32),
            tf.cast((x1_shape[2] - x2_shape[2]) // 2, tf.int32),
            0,
        ]
        size = [x1_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]]
        # x1_crop = tf.slice(x1, offsets, size)
        x1_crop = tf.strided_slice(
            x1, offsets, [size[0], offsets[1] + size[1], offsets[2] + size[2], size[3]]
        )
        return tf.concat([x1_crop, x2], 3)


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # print(rectified.get_shape(), '!!!!!!!!!!!!!!!!!!')
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    if a.func == "train":
        layer_specs = [
            (
                a.ngf * 8,
                0.5,
            ),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.5,
            ),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.5,
            ),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.0,
            ),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (
                a.ngf * 4,
                0.0,
            ),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (
                a.ngf * 2,
                0.0,
            ),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (
                a.ngf,
                0.0,
            ),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    else:
        layer_specs = [
            (
                a.ngf * 8,
                0.0,
            ),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.0,
            ),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.0,
            ),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (
                a.ngf * 8,
                0.0,
            ),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (
                a.ngf * 4,
                0.0,
            ),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (
                a.ngf * 2,
                0.0,
            ),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (
                a.ngf,
                0.0,
            ),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = crop_and_concat(
                    layers[-1], layers[skip_layer]
                )  # tf.concat([layers[-1], layers[skip_layer]], 3)  #crop_and_concat(layers[-1], layers[skip_layer])
                sshape = list(layers[skip_layer].get_shape())
                sshape[-1] = sshape[-1] * 2
                input.set_shape(sshape)
            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = crop_and_concat(
            layers[-1], layers[0]
        )  # tf.concat([layers[-1], layers[0]], 3) #crop_and_concat(layers[-1], layers[0])
        sshape = list(layers[0].get_shape())
        sshape[-1] = sshape[-1] * 2
        input.set_shape(sshape)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)

        if a.mode.startswith("tex"):
            output = tf.tanh(output)

        x1_shape = tf.shape(output)
        x2_shape = tf.shape(generator_inputs)
        # offsets for the top left corner of the crop
        offsets = [
            0,
            tf.cast((x1_shape[1] - x2_shape[1]) // 2, tf.int32),
            tf.cast((x1_shape[2] - x2_shape[2]) // 2, tf.int32),
            0,
        ]
        size = [x2_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]]
        output = tf.strided_slice(
            output,
            offsets,
            [size[0], offsets[1] + size[1], offsets[1] + size[2], size[3]],
        )
        sshape = list(generator_inputs.get_shape())
        if a.mode.startswith("tex") is False:
            sshape[-1] = 3
        output.set_shape(sshape)

        layers.append(output)

    return layers[-1]


def create_test_model(inputs):

    with tf.variable_scope("generator"):
        out_channels = 3
        outputs = create_generator(inputs, out_channels)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        outputs=outputs,
    )


def create_train_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 4
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)
            print(output.get_shape().as_list())

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(
            -(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS))
        )

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

        if a.mode.startswith("tex"):
            # NOTE: added by cyj to remove red dots
            mask = Image.open(a.regional_mask_path)
            mask = np.asarray(mask, np.float32) / 255.0
            mask = mask[START_Y:END_Y, START_X:START_X, :]

            _, out_height, out_width, _ = outputs.get_shape().as_list()
            tv_mask = np.expand_dims(1 - mask, 0)
            gen_loss_tv = Losses.uv_tv_loss2(outputs, tv_mask, tv_mask)
            gen_loss = gen_loss + a.tv_weight * gen_loss_tv
        else:
            targetc = tf.nn.l2_normalize(targets, axis=-1)
            outputc = tf.nn.l2_normalize(outputs, axis=-1)
            gen_loss_cos = tf.losses.cosine_distance(targetc, outputc, axis=-1)
            print("gen_loss_cos", gen_loss_cos.get_shape().as_list())

            gen_loss = gen_loss + gen_loss_cos * a.cos_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [
            var
            for var in tf.trainable_variables()
            if var.name.startswith("discriminator")
        ]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(
            discrim_loss, var_list=discrim_tvars
        )
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [
                var
                for var in tf.trainable_variables()
                if var.name.startswith("generator")
            ]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(
                gen_loss, var_list=gen_tvars
            )
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    if a.mode.startswith("tex"):
        update_losses = ema.apply(
            [discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_tv]
        )
    else:
        update_losses = ema.apply(
            [discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_cos]
        )

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    if a.mode.startswith("tex"):
        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_loss_tv=ema.average(gen_loss_tv),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )
    else:
        return Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_loss_cos=ema.average(gen_loss_cos),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def train():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_train_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        if a.mode.startswith("tex"):
            inputs = deprocess(examples.inputs, mu_texture, std_texture)
            targets = deprocess(examples.targets, mu_texture, std_texture)
            outputs = deprocess(model.outputs, mu_texture, std_texture)
        else:
            inputs1 = deprocess(examples.inputs[:, :, :, :3], mu_normal, std_normal)
            inputs2 = deprocess(examples.inputs[:, :, :, 3:], mu_texture, std_texture)
            targets = deprocess(examples.targets, mu_normal, std_normal)
            outputs = deprocess(model.outputs, mu_normal, std_normal)
            inputs = tf.concat([inputs1, inputs2], axis=2)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(
                tf.image.encode_png,
                converted_inputs,
                dtype=tf.string,
                name="input_pngs",
            ),
            "targets": tf.map_fn(
                tf.image.encode_png,
                converted_targets,
                dtype=tf.string,
                name="target_pngs",
            ),
            "outputs": tf.map_fn(
                tf.image.encode_png,
                converted_outputs,
                dtype=tf.string,
                name="output_pngs",
            ),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image(
            "predict_real",
            tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8),
        )

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image(
            "predict_fake",
            tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8),
        )

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    if a.mode.startswith("tex"):
        tf.summary.scalar("generator_loss_tv", model.gen_loss_tv)
    else:
        tf.summary.scalar("generator_loss_cos", model.gen_loss_cos)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()]
        )

    saver = tf.train.Saver(max_to_keep=0)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # training
        start = time.time()

        for step in range(max_steps):

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1
                if a.mode.startswith("tex"):
                    fetches["gen_loss_tv"] = model.gen_loss_tv
                else:
                    fetches["gen_loss_cos"] = model.gen_loss_cos

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(a.summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(
                    results["summary"], results["global_step"]
                )

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(
                    run_metadata, "step_%d" % results["global_step"]
                )

            if should(a.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(
                    results["global_step"] / examples.steps_per_epoch
                )
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print(
                    "progress  epoch %d  step %d  image/sec %0.1f  remaining %dm"
                    % (train_epoch, train_step, rate, remaining / 60)
                )
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                if a.mode.startswith("tex"):
                    print("gen_loss_tv", results["gen_loss_tv"])
                else:
                    print("gen_loss_cos", results["gen_loss_cos"])

            if should(a.save_freq):
                print("saving model")
                saver.save(
                    sess,
                    os.path.join(a.output_dir, "model"),
                    global_step=sv.global_step,
                )

            if sv.should_stop():
                break


def freeze():

    if a.mode.startswith("tex"):
        input_image = tf.placeholder(
            tf.float32, shape=[1, END_Y - START_Y, END_X - START_X, 3], name="inputs"
        )
        input_norm = normalize(input_image)
    else:
        input_tex = tf.placeholder(
            tf.float32,
            shape=[1, END_Y - START_Y, END_X - START_X, 3],
            name="texture_inputs",
        )
        input_normal = tf.placeholder(
            tf.float32,
            shape=[1, END_Y - START_Y, END_X - START_X, 3],
            name="normal_inputs",
        )
        input_tex_norm = normalize(input_tex, mu_texture, std_texture)
        input_normal_norm = normalize(input_normal, mu_normal, std_normal)
        input_norm = tf.concat([input_normal_norm, input_tex_norm], axis=3)

    model = create_test_model(input_norm)

    if a.mode.startswith("tex"):
        output_image = denormalize(model.outputs)
    else:
        output_image = denormalize(model.outputs, mu_normal, std_normal)
        output_image = tf.clip_by_value(output_image, 0, 255)

    output_image = tf.identity(output_image, name="outputs")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            saver.restore(sess, a.checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ["outputs"]
        )
        with tf.gfile.GFile(a.pb_path, "wb") as fp:
            fp.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph" % len(output_graph_def.node))


def test_texture():

    if os.path.exists(a.output_dir) is False:
        os.makedirs(a.output_dir)

    start = time.time()

    base_uv = Image.open(a.base_tex_path)
    base_uv = np.asarray(base_uv, np.float32)

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()

        with open(a.pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        end1 = time.time()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            input_x = sess.graph.get_tensor_by_name("inputs:0")
            output_x = sess.graph.get_tensor_by_name("outputs:0")

            for path in glob.glob(os.path.join(a.input_dir, "*D*.png")):
                print(path)
                img = np.array(Image.open(path)).astype(np.float32)
                crop_img = img[START_Y:END_Y, START_X:END_X]
                crop_img = np.expand_dims(crop_img, 0)

                result = sess.run(output_x, {input_x: crop_img})
                result = np.clip(result, 0, 255)

                start2 = time.time()
                result = sess.run(output_x, {input_x: crop_img})
                end2 = time.time()

                result = result[0]
                mask = np.zeros_like(img)
                mask[START_Y:END_Y, START_X:END_X] = 1.0
                face_tex = np.zeros_like(img)
                face_tex[START_Y:END_Y, START_X:END_X] = result
                img = blend_uv(
                    base_uv / 255, face_tex / 255, mask, match_color=True, times=7
                )
                img = img * 255
                result = Image.fromarray(img.astype(np.uint8))
                result.save(os.path.join(a.output_dir, path.split("/")[-1]))


def test_normal():

    if os.path.exists(a.output_dir) is False:
        os.makedirs(a.output_dir)

    base_normal = Image.open(a.base_normal_path)
    base_normal = np.asarray(base_normal, np.float32)

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()

        with open(a.pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            texture_input_x = sess.graph.get_tensor_by_name("texture_inputs:0")
            normal_input_x = sess.graph.get_tensor_by_name("normal_inputs:0")
            output_x = sess.graph.get_tensor_by_name("outputs:0")

            # refined texture paths, fitted normal paths
            texture_paths = sorted(glob.glob(os.path.join(a.output_dir, "*D.png")))
            normal_paths = sorted(glob.glob(os.path.join(a.input_dir, "*N.png")))

            for tex_path, norm_path in zip(texture_paths, normal_paths):
                print(tex_path, norm_path)
                tex_img = np.array(Image.open(tex_path)).astype(np.float32)
                crop_tex_img = tex_img[START_Y:END_Y, START_X:END_X]
                crop_tex_img = np.expand_dims(crop_tex_img, 0)

                norm_img = np.array(Image.open(norm_path)).astype(np.float32)
                crop_norm_img = norm_img[START_Y:END_Y, START_X:END_X]
                crop_norm_img = np.expand_dims(crop_norm_img, 0)

                result = sess.run(
                    output_x,
                    {texture_input_x: crop_tex_img, normal_input_x: crop_norm_img},
                )
                result = np.clip(result, 0, 255)

                result = result[0]
                mask = np.zeros_like(norm_img)
                mask[START_Y:END_Y, START_X:END_X] = 1.0
                face_norm = np.zeros_like(norm_img)
                face_norm[START_Y:END_Y, START_X:END_X] = result
                norm_img = blend_uv(
                    base_normal / 255, face_norm / 255, mask, match_color=False
                )
                norm_img = norm_img * 255
                result = Image.fromarray(norm_img.astype(np.uint8))
                result.save(os.path.join(a.output_dir, norm_path.split("/")[-1]))


if __name__ == "__main__":
    if a.func == "train":
        train()
    elif a.func == "freeze":
        freeze()
    elif a.func == "test":
        if a.mode.startswith("tex"):
            test_texture()
        else:
            test_normal()
