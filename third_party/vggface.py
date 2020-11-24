""" a tensorflow version of VGG face model.
This snippet is modified from https://github.com/ZZUTK/Tensorflow-VGG-face.git
The pretrained weights are also downloaded from the same repo
The "NCHW" data format is added here for running time optimization.
"""
import tensorflow as tf
import numpy as np
import scipy.io
import os


class VGGFace(object):
    def __init__(self, param_path, trainable=True):
        print(os.path.dirname(__file__))
        print(os.getcwd())
        self.data = scipy.io.loadmat(param_path)
        self.trainable = trainable

    def encoder(self, input_maps, net_name="VGGFace", reuse=False, data_format="nhwc"):
        with tf.compat.v1.variable_scope(net_name) as scope:
            if reuse:
                scope.reuse_variables()
            # read meta info
            meta = self.data["meta"]
            classes = meta["classes"]
            class_names = classes[0][0]["description"][0][0]
            normalization = meta["normalization"]
            average_image = np.squeeze(normalization[0][0]["averageImage"][0][0][0][0])
            image_size = np.squeeze(normalization[0][0]["imageSize"][0][0])
            input_maps = tf.image.resize(
                input_maps, (image_size[0], image_size[1])
            )

            # read layer info
            layers = self.data["layers"]
            # current = input_maps
            network = {}
            input_maps = input_maps - average_image

            if data_format == "nchw" or data_format == "NCHW":
                input_maps = tf.transpose(a=input_maps, perm=[0, 3, 1, 2])
            current = input_maps
            network["inputs"] = input_maps  # no much use
            for layer in layers[0]:
                name = layer[0]["name"][0][0]
                layer_type = layer[0]["type"][0][0]
                if layer_type == "conv":
                    if name[:2] == "fc":
                        padding = "VALID"
                    else:
                        padding = "SAME"
                    stride = layer[0]["stride"][0][0]
                    kernel, bias = layer[0]["weights"][0][0]
                    bias = np.squeeze(bias).reshape(-1)
                    kh, kw, kin, kout = kernel.shape

                    if data_format == "nhwc" or data_format == "NHWC":
                        kernel_var = tf.compat.v1.get_variable(
                            name=name + "_weight",
                            dtype=tf.float32,
                            initializer=tf.constant(kernel),
                            trainable=self.trainable,
                        )
                        bias_var = tf.compat.v1.get_variable(
                            name=name + "_bias",
                            dtype=tf.float32,
                            initializer=tf.constant(bias),
                            trainable=self.trainable,
                        )

                        if name[:2] != "fc":
                            conv = tf.nn.conv2d(
                                input=current,
                                filters=kernel_var,
                                strides=(1, stride[0], stride[0], 1),
                                padding=padding,
                            )
                            current = tf.nn.bias_add(conv, bias_var)
                        else:
                            _, cur_h, cur_w, cur_c = current.get_shape().as_list()
                            print(cur_h, cur_w, cur_c, kh, kw, kin)
                            flatten = tf.reshape(
                                current, [-1, cur_h * cur_w * cur_c], name="flatten"
                            )
                            kernel_var = tf.reshape(
                                kernel_var, [kh * kw * kin, kout], name="kernel_flat"
                            )
                            fc = tf.matmul(flatten, kernel_var)
                            current = fc + bias_var
                            current = tf.reshape(current, [-1, 1, 1, kout])
                    else:
                        # kernel = np.transpose(kernel, [2,3,1,0])
                        kernel_var = tf.compat.v1.get_variable(
                            name=name + "_weight",
                            dtype=tf.float32,
                            initializer=tf.constant(kernel),
                            trainable=self.trainable,
                        )
                        bias_var = tf.compat.v1.get_variable(
                            name=name + "_bias",
                            dtype=tf.float32,
                            initializer=tf.constant(bias),
                            trainable=self.trainable,
                        )

                        conv = tf.nn.conv2d(
                            input=current,
                            filters=kernel_var,
                            strides=(1, 1, stride[0], stride[0]),
                            padding=padding,
                            data_format="NCHW",
                        )
                        current = tf.nn.bias_add(conv, bias_var, data_format="NCHW")

                elif layer_type == "relu":
                    current = tf.nn.relu(current)
                elif layer_type == "pool":
                    stride = layer[0]["stride"][0][0]
                    pool = layer[0]["pool"][0][0]
                    if data_format == "nhwc" or data_format == "NHWC":
                        current = tf.nn.max_pool2d(
                            input=current,
                            ksize=(1, pool[0], pool[1], 1),
                            strides=(1, stride[0], stride[0], 1),
                            padding="SAME",
                        )
                    else:
                        current = tf.nn.max_pool2d(
                            input=current,
                            ksize=(1, 1, pool[0], pool[1]),
                            strides=(1, 1, stride[0], stride[0]),
                            padding="SAME",
                            data_format="NCHW",
                        )
                elif layer_type == "softmax":
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

                network[name] = current
                print("network", name, current.get_shape().as_list())

            return network, average_image, class_names


def encoder(modelpath, images, train=True, reuse=False, data_format="NHWC"):
    # apply vgg face encoder
    vggface = VGGFace(modelpath)
    layers, _, _ = vggface.encoder(
        images, net_name="VGGEncoder", reuse=reuse, data_format=data_format
    )
    z = tf.reshape(layers["fc7"], [-1, 4096])
    return z


if __name__ == "__main__":
    vggface = VGGFace("../../resources/vgg-face.mat")
    inputs = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3])
    vggface.encoder(inputs, data_format="NHWC")
