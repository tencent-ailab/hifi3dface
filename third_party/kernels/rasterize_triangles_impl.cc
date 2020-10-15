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
#include <cmath>

#include "rasterize_triangles_impl.h"

namespace tf_mesh_renderer {

namespace {

// Takes the minimum of a, b, and c, rounds down, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMin(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::floor(std::min(std::min(a, b), c))), low),
      high);
}

// Takes the maximum of a, b, and c, rounds up, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMax(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::ceil(std::max(std::max(a, b), c))), low),
      high);
}

// Converts to fixed point with 16 fractional bits and 16 integer bits.
// Overflows for values outside of (-2^15, 2^15).
inline int32 FixedPoint(float f) { return static_cast<int>(f * (1 << 16)); }

// Determines whether the point p lies counter-clockwise (CCW) of a directed
// edge between vertices v0 and v1.
bool IsCCW(int32 v0x, int32 v0y, int32 v1x, int32 v1y, int32 px, int32 py) {
  int32 ex = v1x - v0x;
  int32 ey = v1y - v0y;
  int32 x = px - v0x;
  int32 y = py - v0y;
  // p is CCW of v1 - v0 if det(A) >= 0, where A:
  // | v1x - v0x, px - v0x |
  // | v1y - v0y, py - v0y |
  int64 ex_y = int64{ex} * int64{y};
  int64 ey_x = int64{ey} * int64{x};
  return ex_y >= ey_x;
}

// Determines whether the point p lies inside a triangle.
// Accepts both front-facing and back-facing triangles.
bool PixelIsInsideTriangle(int32 v0x, int32 v0y, int32 v1x, int32 v1y,
                           int32 v2x, int32 v2y, int32 px, int32 py) {
  // Returns true if the point is counter clockwise to all the edges of either
  // the front facing or back facing triangle.
  return (IsCCW(v0x, v0y, v1x, v1y, px, py) &&
          IsCCW(v1x, v1y, v2x, v2y, px, py) &&
          IsCCW(v2x, v2y, v0x, v0y, px, py)) ||
         (IsCCW(v1x, v1y, v0x, v0y, px, py) &&
          IsCCW(v2x, v2y, v1x, v1y, px, py) &&
          IsCCW(v0x, v0y, v2x, v2y, px, py));
}

}  // namespace

void RasterizeTrianglesImpl(const float* vertices, const int32* triangles,
                            int32 triangle_count, int32 image_width,
                            int32 image_height, int32* triangle_ids,
                            float* barycentric_coordinates, float* z_buffer) {
  const float half_image_width = 0.5 * image_width;
  const float half_image_height = 0.5 * image_height;
  for (int32 triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
    const int32 v0_x_id = 3 * triangles[3 * triangle_id];
    const int32 v1_x_id = 3 * triangles[3 * triangle_id + 1];
    const int32 v2_x_id = 3 * triangles[3 * triangle_id + 2];

    // Convert NDC vertex positions to viewport coordinates.
    const float v0x = (vertices[v0_x_id] + 1.0) * half_image_width;
    const float v0y = (vertices[v0_x_id + 1] + 1.0) * half_image_height;
    const float v1x = (vertices[v1_x_id] + 1.0) * half_image_width;
    const float v1y = (vertices[v1_x_id + 1] + 1.0) * half_image_height;
    const float v2x = (vertices[v2_x_id] + 1.0) * half_image_width;
    const float v2y = (vertices[v2_x_id + 1] + 1.0) * half_image_height;

    // Find the triangle bounding box enlarged to the nearest integer and
    // clamped to the image boundaries.
    const int left = ClampedIntegerMin(v0x, v1x, v2x, 0, image_width);
    const int right = ClampedIntegerMax(v0x, v1x, v2x, 0, image_width);
    const int bottom = ClampedIntegerMin(v0y, v1y, v2y, 0, image_height);
    const int top = ClampedIntegerMax(v0y, v1y, v2y, 0, image_height);

    // Convert coordinates to fixed-point to make triangle intersection
    // testing consistent and prevent cracks.
    const int32 fv0x = FixedPoint(v0x);
    const int32 fv0y = FixedPoint(v0y);
    const int32 fv1x = FixedPoint(v1x);
    const int32 fv1y = FixedPoint(v1y);
    const int32 fv2x = FixedPoint(v2x);
    const int32 fv2y = FixedPoint(v2y);

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy) {
      for (int ix = left; ix < right; ++ix) {
        const float px = ix + 0.5;
        const float py = iy + 0.5;

        if (!PixelIsInsideTriangle(fv0x, fv0y, fv1x, fv1y, fv2x, fv2y,
                                   FixedPoint(px), FixedPoint(py))) {
          continue;
        }

        const int pixel_idx = iy * image_width + ix;

        // Compute twice the area of two barycentric triangles, as well as the
        // triangle they sit in; the barycentric is the ratio of the triangle
        // areas, so the factor of two does not change the result.
        const float twice_triangle_area =
            (v2x - v0x) * (v1y - v0y) - (v2y - v0y) * (v1x - v0x);
        const float b0 = ((px - v1x) * (v2y - v1y) - (py - v1y) * (v2x - v1x)) /
                         twice_triangle_area;
        const float b1 = ((px - v2x) * (v0y - v2y) - (py - v2y) * (v0x - v2x)) /
                         twice_triangle_area;
        // The three upper triangles partition the lower triangle, so we can
        // compute the third barycentric coordinate using the other two:
        const float b2 = 1.0f - b0 - b1;

        const float v0z = vertices[v0_x_id + 2];
        const float v1z = vertices[v1_x_id + 2];
        const float v2z = vertices[v2_x_id + 2];
        const float z = b0 * v0z + b1 * v1z + b2 * v2z;

        // Skip the pixel if it is farther than the current z-buffer pixel or
        // beyond the near or far clipping plane.
        if (z < -1.0 || z > 1.0 || z > z_buffer[pixel_idx]) {
          continue;
        }

        triangle_ids[pixel_idx] = triangle_id;
        z_buffer[pixel_idx] = z;
        barycentric_coordinates[3 * pixel_idx + 0] = b0;
        barycentric_coordinates[3 * pixel_idx + 1] = b1;
        barycentric_coordinates[3 * pixel_idx + 2] = b2;
      }
    }
  }
}

}  // namespace tf_mesh_renderer
