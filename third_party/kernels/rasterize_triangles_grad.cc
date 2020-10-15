// original source: https://github.com/google/tf_mesh_renderer

// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace {

// Threshold for a barycentric coordinate triplet's sum, below which the
// coordinates at a pixel are deemed degenerate. Most such degenerate triplets
// in an image will be exactly zero, as this is how pixels outside the mesh
// are rendered.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// If the area of a triangle is very small in screen space, the corner vertices
// are approaching colinearity, and we should drop the gradient to avoid
// numerical instability (in particular, blowup, as the forward pass computation
// already only has 8 bits of precision).
constexpr float kMinimumTriangleArea = 1e-13;

}  // namespace

namespace tf_mesh_renderer {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;

REGISTER_OP("RasterizeTrianglesGrad")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Input("barycentric_coordinates: float32")
    .Input("triangle_ids: int32")
    .Input("df_dbarycentric_coordinates: float32")
    .Attr("image_width: int")
    .Attr("image_height: int")
    .Output("df_dvertices: float32");

class RasterizeTrianglesGradOp : public OpKernel {
 public:
  explicit RasterizeTrianglesGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
    OP_REQUIRES(context, image_width_ > 0,
                InvalidArgument("Image width must be > 0, got ", image_width_));

    OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
    OP_REQUIRES(
        context, image_height_ > 0,
        InvalidArgument("Image height must be > 0, got ", image_height_));
  }

  ~RasterizeTrianglesGradOp() override {}

  void Compute(OpKernelContext* context) override {
    const Tensor& vertices_tensor = context->input(0);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(vertices_tensor.shape()),
        InvalidArgument(
            "RasterizeTrianglesGrad expects vertices to have shape (-1, 3)."));
    auto vertices_flat = vertices_tensor.flat<float>();
    const unsigned int vertex_count = vertices_flat.size() / 3;
    const float* vertices = vertices_flat.data();

    const Tensor& triangles_tensor = context->input(1);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(triangles_tensor.shape()),
        InvalidArgument(
            "RasterizeTrianglesGrad expects triangles to be a matrix."));
    auto triangles_flat = triangles_tensor.flat<int>();
    const int* triangles = triangles_flat.data();

    const Tensor& barycentric_coordinates_tensor = context->input(2);
    OP_REQUIRES(context,
                TensorShape({image_height_, image_width_, 3}) ==
                    barycentric_coordinates_tensor.shape(),
                InvalidArgument(
                    "RasterizeTrianglesGrad expects barycentric_coordinates to "
                    "have shape {image_height, image_width, 3}"));
    auto barycentric_coordinates_flat =
        barycentric_coordinates_tensor.flat<float>();
    const float* barycentric_coordinates = barycentric_coordinates_flat.data();

    const Tensor& triangle_ids_tensor = context->input(3);
    OP_REQUIRES(
        context,
        TensorShape({image_height_, image_width_}) ==
            triangle_ids_tensor.shape(),
        InvalidArgument(
            "RasterizeTrianglesGrad expected triangle_ids to have shape "
            " {image_height, image_width}"));
    auto triangle_ids_flat = triangle_ids_tensor.flat<int>();
    const int* triangle_ids = triangle_ids_flat.data();

    // The naming convention we use for all derivatives is d<y>_d<x> ->
    // the partial of y with respect to x.
    const Tensor& df_dbarycentric_coordinates_tensor = context->input(4);
    OP_REQUIRES(
        context,
        TensorShape({image_height_, image_width_, 3}) ==
            df_dbarycentric_coordinates_tensor.shape(),
        InvalidArgument(
            "RasterizeTrianglesGrad expects df_dbarycentric_coordinates "
            "to have shape {image_height, image_width, 3}"));
    auto df_dbarycentric_coordinates_flat =
        df_dbarycentric_coordinates_tensor.flat<float>();
    const float* df_dbarycentric_coordinates =
        df_dbarycentric_coordinates_flat.data();

    Tensor* df_dvertices_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({vertex_count, 3}),
                                            &df_dvertices_tensor));
    auto df_dvertices_flat = df_dvertices_tensor->flat<float>();
    float* df_dvertices = df_dvertices_flat.data();
    std::fill(df_dvertices, df_dvertices + vertex_count * 3, 0.0f);

    // We first loop over each pixel in the output image, and compute
    // dbarycentric_coordinate[0,1,2]/dvertex[0x, 0y, 1x, 1y, 2x, 2y].
    // Next we compute each value above's contribution to
    // df/dvertices, building up that matrix as the output of this iteration.
    for (unsigned int pixel_id = 0; pixel_id < image_height_ * image_width_;
         ++pixel_id) {
      // b0, b1, and b2 are the three barycentric coordinate values
      // rendered at pixel pixel_id.
      const float b0 = barycentric_coordinates[3 * pixel_id];
      const float b1 = barycentric_coordinates[3 * pixel_id + 1];
      const float b2 = barycentric_coordinates[3 * pixel_id + 2];

      if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff) {
        continue;
      }

      const float df_db0 = df_dbarycentric_coordinates[3 * pixel_id];
      const float df_db1 = df_dbarycentric_coordinates[3 * pixel_id + 1];
      const float df_db2 = df_dbarycentric_coordinates[3 * pixel_id + 2];

      const int triangle_at_current_pixel = triangle_ids[pixel_id];
      const int* vertices_at_current_pixel =
          &triangles[3 * triangle_at_current_pixel];

      // Extract vertex indices for the current triangle.
      const int v0_id = 3 * vertices_at_current_pixel[0];
      const int v1_id = 3 * vertices_at_current_pixel[1];
      const int v2_id = 3 * vertices_at_current_pixel[2];

      // Extract x,y components of the vertices' normalized device coordinates.
      const float v0x = vertices[v0_id];
      const float v0y = vertices[v0_id + 1];
      const float v1x = vertices[v1_id];
      const float v1y = vertices[v1_id + 1];
      const float v2x = vertices[v2_id];
      const float v2y = vertices[v2_id + 1];

      // The derivatives share a common denominator, as the screen space
      // area of the triangle is common to all three vertices.
      // Note this quantity is actually twice the area (i.e. the size of the
      // parallelogram given by the screen space cross product), but we only use
      // the ratio of areas, and we compute all areas this way.
      const float triangle_area =
          (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x);
      // Same calculation applies to clockwise and counter-clockwise triangles.
      if (std::abs(triangle_area) < kMinimumTriangleArea) {
        continue;
      }

      // Derivatives of all three baricentric coordinates with respect to the
      // x-y coordinates of a single vertex share a common factor.
      const float db_dv0 = b0 / triangle_area;
      const float db_dv1 = b1 / triangle_area;
      const float db_dv2 = b2 / triangle_area;

      // Derivatives of barycentric coordinates with respect to x-y coordinates
      // of a single vertex differ by a simple constant.
      const float db0_dvx = v2y - v1y;
      const float db0_dvy = v1x - v2x;
      const float db1_dvx = v0y - v2y;
      const float db1_dvy = v2x - v0x;
      const float db2_dvx = v1y - v0y;
      const float db2_dvy = v0x - v1x;

      // Derivatives of the final function with respect to x coordinate of any
      // vertex share a common factor, as do those with respect to y coordinate.
      const float df_dvx =
          df_db0 * db0_dvx + df_db1 * db1_dvx + df_db2 * db2_dvx;
      const float df_dvy =
          df_db0 * db0_dvy + df_db1 * db1_dvy + df_db2 * db2_dvy;

      df_dvertices[v0_id] += db_dv0 * df_dvx;
      df_dvertices[v0_id + 1] += db_dv0 * df_dvy;
      df_dvertices[v1_id] += db_dv1 * df_dvx;
      df_dvertices[v1_id + 1] += db_dv1 * df_dvy;
      df_dvertices[v2_id] += db_dv2 * df_dvx;
      df_dvertices[v2_id + 1] += db_dv2 * df_dvy;
    }
  }

 private:
  int image_width_;
  int image_height_;
};

REGISTER_KERNEL_BUILDER(Name("RasterizeTrianglesGrad").Device(DEVICE_CPU),
                        RasterizeTrianglesGradOp);

}  // namespace tf_mesh_renderer
