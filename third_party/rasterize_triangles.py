# original source: https://github.com/google/tf_mesh_renderer

# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable triangle rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


dirname = os.path.dirname(__file__)
rasterize_triangles_module = tf.load_op_library(
    os.path.join(dirname, "kernels/rasterize_triangles_kernel.so")
)


# This epsilon should be smaller than any valid barycentric reweighting factor
# (i.e. the per-pixel reweighting factor used to correct for the effects of
# perspective-incorrect barycentric interpolation). It is necessary primarily
# because the reweighting factor will be 0 for factors outside the mesh, and we
# need to ensure the image color and gradient outside the region of the mesh are
# 0.
_MINIMUM_REWEIGHTING_THRESHOLD = 1e-6

# This epsilon is the minimum absolute value of a homogenous coordinate before
# it is clipped. It should be sufficiently large such that the output of
# the perspective divide step with this denominator still has good working
# precision with 32 bit arithmetic, and sufficiently small so that in practice
# vertices are almost never close enough to a clipping plane to be thresholded.
_MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD = 1e-6


def rasterize_clip_space(
    vertices, attributes, triangles, image_width, image_height, background_value
):
    """Rasterizes the input scene and computes interpolated vertex attributes.

    NOTE: the rasterizer does no triangle clipping. Triangles that lie outside the
    viewing frustum (esp. behind the camera) may be drawn incorrectly.

    Args:
      vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
          triplet is an xyz position in model space.
      attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
          attribute_count]. Each vertex attribute is interpolated
          across the triangle using barycentric interpolation.
      triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
          should contain vertex indices describing a triangle such that the
          triangle's normal points toward the viewer if the forward order of the
          triplet defines a clockwise winding of the vertices. Gradients with
          respect to this tensor are not available.
      projection_matrices: 3-D float tensor with shape [batch_size, 4, 4]
          containing model-view-perspective projection matrices.
      image_width: int specifying desired output image width in pixels.
      image_height: int specifying desired output image height in pixels.
      background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
          that lie outside all triangles take this value.

    Returns:
      A 4-D float32 tensor with shape [batch_size, image_height, image_width,
      attribute_count], containing the interpolated vertex attributes at
      each pixel.

    Raises:
      ValueError: An invalid argument to the method is detected.
    """
    if not image_width > 0:
        raise ValueError("Image width must be > 0.")
    if not image_height > 0:
        raise ValueError("Image height must be > 0.")
    if len(vertices.shape) != 3:
        raise ValueError("The vertex buffer must be 3D.")
    batch_size = vertices.shape.as_list()[0]
    vertex_count = vertices.shape.as_list()[1]

    # We map the coordinates to normalized device coordinates before passing
    # the scene to the rendering kernel to keep as many ops in tensorflow as
    # possible.

    homogeneous_coord = tf.ones([batch_size, vertex_count, 1], dtype=tf.float32)
    vertices_homogeneous = tf.concat([vertices, homogeneous_coord], 2)

    # Vertices are given in row-major order, but the transformation pipeline is
    # column major:
    # clip_space_points = tf.matmul(
    #    vertices_homogeneous, projection_matrices, transpose_b=True)
    clip_space_points = vertices_homogeneous

    # Perspective divide, first thresholding the homogeneous coordinate to avoid
    # the possibility of NaNs:
    clip_space_points_w = tf.maximum(
        tf.abs(clip_space_points[:, :, 3:4]), _MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD
    ) * tf.sign(clip_space_points[:, :, 3:4])
    normalized_device_coordinates = clip_space_points[:, :, 0:3] / clip_space_points_w

    per_image_uncorrected_barycentric_coordinates = []
    per_image_vertex_ids = []
    for im in range(vertices.shape[0]):
        (
            barycentric_coords,
            triangle_ids,
            _,
        ) = rasterize_triangles_module.rasterize_triangles(
            normalized_device_coordinates[im, :, :],
            triangles,
            image_width,
            image_height,
        )
        per_image_uncorrected_barycentric_coordinates.append(
            tf.reshape(barycentric_coords, [-1, 3])
        )

        # Gathers the vertex indices now because the indices don't contain a batch
        # identifier, and reindexes the vertex ids to point to a (batch,vertex_id)
        vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
        reindexed_ids = tf.add(vertex_ids, im * vertices.shape.as_list()[1])
        per_image_vertex_ids.append(reindexed_ids)

    uncorrected_barycentric_coordinates = tf.concat(
        per_image_uncorrected_barycentric_coordinates, axis=0
    )
    vertex_ids = tf.concat(per_image_vertex_ids, axis=0)

    # Indexes with each pixel's clip-space triangle's extrema (the pixel's
    # 'corner points') ids to get the relevant properties for deferred shading.
    flattened_vertex_attributes = tf.reshape(
        attributes, [batch_size * vertex_count, -1]
    )
    corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

    # Barycentric interpolation is linear in the reciprocal of the homogeneous
    # W coordinate, so we use these weights to correct for the effects of
    # perspective distortion after rasterization.
    perspective_distortion_weights = tf.math.reciprocal(
        tf.reshape(clip_space_points_w, [-1])
    )
    corner_distortion_weights = tf.gather(perspective_distortion_weights, vertex_ids)

    # Apply perspective correction to the barycentric coordinates. This step is
    # required since the rasterizer receives normalized-device coordinates (i.e.,
    # after perspective division), so it can't apply perspective correction to the
    # interpolated values.
    weighted_barycentric_coordinates = tf.multiply(
        uncorrected_barycentric_coordinates, corner_distortion_weights
    )
    barycentric_reweighting_factor = tf.reduce_sum(
        input_tensor=weighted_barycentric_coordinates, axis=1
    )

    corrected_barycentric_coordinates = tf.divide(
        weighted_barycentric_coordinates,
        tf.expand_dims(
            tf.maximum(barycentric_reweighting_factor, _MINIMUM_REWEIGHTING_THRESHOLD),
            axis=1,
        ),
    )

    # Computes the pixel attributes by interpolating the known attributes at the
    # corner points of the triangle interpolated with the barycentric coordinates.
    weighted_vertex_attributes = tf.multiply(
        corner_attributes, tf.expand_dims(corrected_barycentric_coordinates, axis=2)
    )
    summed_attributes = tf.reduce_sum(input_tensor=weighted_vertex_attributes, axis=1)
    attribute_images = tf.reshape(
        summed_attributes, [batch_size, image_height, image_width, -1]
    )

    # Barycentric coordinates should approximately sum to one where there is
    # rendered geometry, but be exactly zero where there is not.
    alphas = tf.clip_by_value(
        tf.reduce_sum(input_tensor=2.0 * corrected_barycentric_coordinates, axis=1), 0.0, 1.0
    )
    alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

    attributes_with_background = (
        alphas * attribute_images + (1.0 - alphas) * background_value
    )

    print("End of rasterization")

    return attributes_with_background, alphas


@tf.RegisterGradient("RasterizeTriangles")
def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
    # Gradients are only supported for barycentric coordinates. Gradients for the
    # z-buffer are possible as well but not currently implemented.
    del df_dids, df_dz
    return (
        rasterize_triangles_module.rasterize_triangles_grad(
            op.inputs[0],
            op.inputs[1],
            op.outputs[0],
            op.outputs[1],
            df_dbarys,
            op.get_attr("image_width"),
            op.get_attr("image_height"),
        ),
        None,
    )
