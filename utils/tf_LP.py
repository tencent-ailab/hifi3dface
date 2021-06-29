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

import tensorflow as tf

# before time optimizing
class TF_LaplacianPyramid(object):
    @staticmethod
    def conv_depthwise(image, kernel, strides, padding):
        channel_list = tf.split(image, image.get_shape().as_list()[-1], axis=-1)
        out_channel_list = []
        for cur_channel in channel_list:
            out_channel = tf.nn.conv2d(
                cur_channel, kernel, strides=strides, padding=padding
            )
            out_channel_list.append(out_channel)
        out_channels = tf.concat(out_channel_list, axis=-1)
        return out_channels

    @staticmethod
    def downSamplePyramids(image, n_level, sigma=1):
        # test OK
        assert len(image.get_shape().as_list()) == 4
        pyramids = [image]
        kernel = tf.reshape(
            tf.constant([1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16], tf.float32),
            [5, 1],
        )
        kernel = tf.reshape(tf.matmul(kernel, kernel, transpose_b=True), [5, 5, 1, 1])

        for i in range(1, n_level):
            temp = pyramids[i - 1]
            _, rows, cols, channels = temp.get_shape().as_list()
            if rows % 2 == 1:
                temp = tf.pad(temp, [(0, 0), (0, 1), (0, 0), (0, 0)])
                rows += 1
            if cols % 2 == 1:
                temp = tf.pad(temp, [(0, 0), (0, 0), (0, 1), (0, 0)])
                cols += 1
            temp = tf.reshape(temp, [-1, rows // 2, 2, cols // 2, 2, channels])
            temp = temp[:, :, 0, :, 0, :]
            temp = tf.pad(temp, [(0, 0), (2, 2), (2, 2), (0, 0)], "REFLECT")
            temp = TF_LaplacianPyramid.conv_depthwise(
                temp, kernel, [1, 1, 1, 1], "VALID"
            )
            pyramids.append(temp)
        return pyramids

    @staticmethod
    def upSample(image):
        _, rows, cols, channels = image.get_shape().as_list()
        image = tf.reshape(image, [-1, rows, cols, 1, channels])
        image = tf.reshape(
            tf.concat([image, image], axis=3), [-1, rows, 1, cols * 2, channels]
        )
        image = tf.reshape(
            tf.concat([image, image], axis=2), [1, rows * 2, cols * 2, channels]
        )
        return image

    @staticmethod
    def buildLaplacianPyramids(image, n_level):
        kernel = tf.reshape(
            tf.constant([1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16], tf.float32),
            [5, 1],
        )
        kernel = tf.reshape(tf.matmul(kernel, kernel, transpose_b=True), [5, 5, 1, 1])

        pyramids = []
        cur_image = image
        for i in range(n_level - 1):
            temp = tf.pad(cur_image, [(0, 0), (2, 2), (2, 2), (0, 0)], "REFLECT")
            temp = TF_LaplacianPyramid.conv_depthwise(
                temp, kernel, [1, 1, 1, 1], "VALID"
            )
            _, rows, cols, channels = temp.get_shape().as_list()
            if rows % 2 == 1:
                temp = tf.pad(temp, [(0, 0), (0, 1), (0, 0), (0, 0)])
                rows += 1
            if cols % 2 == 1:
                temp = tf.pad(temp, [(0, 0), (0, 0), (0, 1), (0, 0)])
                cols += 1
            temp = tf.reshape(temp, [-1, rows // 2, 2, cols // 2, 2, channels])
            temp = temp[:, :, 0, :, 0, :]
            dn_temp = temp
            temp = TF_LaplacianPyramid.upSample(temp)
            temp = temp[:, :rows, :cols, :]
            temp = tf.pad(temp, [(0, 0), (2, 2), (2, 2), (0, 0)], "REFLECT")
            temp = TF_LaplacianPyramid.conv_depthwise(
                temp, kernel, [1, 1, 1, 1], "VALID"
            )
            pyramids.append(cur_image - temp)
            cur_image = dn_temp
        pyramids.append(cur_image)
        return pyramids

    @staticmethod
    def reconstruct(pyramids):
        kernel = tf.reshape(
            tf.constant([1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16], tf.float32),
            [5, 1],
        )
        kernel = tf.reshape(tf.matmul(kernel, kernel, transpose_b=True), [5, 5, 1, 1])

        for i in range(len(pyramids) - 1, 0, -1):
            temp = pyramids[i]
            temp = TF_LaplacianPyramid.upSample(temp)
            _, rows, cols, _ = pyramids[i - 1].get_shape().as_list()
            temp = temp[:, :rows, :cols, :]
            temp = tf.pad(temp, [(0, 0), (2, 2), (2, 2), (0, 0)], "REFLECT")
            temp = TF_LaplacianPyramid.conv_depthwise(
                temp, kernel, [1, 1, 1, 1], "VALID"
            )
            pyramids[i - 1] = pyramids[i - 1] + temp
        return pyramids[0]
