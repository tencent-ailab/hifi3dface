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

#include <fstream>

#include "gtest/gtest.h"
#include "rasterize_triangles_impl.h"

#include "third_party/lodepng.h"

namespace tf_mesh_renderer {
namespace {

typedef unsigned char uint8;

const int kImageHeight = 480;
const int kImageWidth = 640;

std::string GetRunfilesRelativePath(const std::string& filename) {
  const std::string srcdir = std::getenv("TEST_SRCDIR");
  const std::string test_data = "/tf_mesh_renderer/mesh_renderer/test_data/";
  return srcdir + test_data + filename;
}

void LoadPng(const std::string& filename, std::vector<uint8>* output) {
  unsigned width, height;
  unsigned error = lodepng::decode(*output, width, height, filename.c_str());
  ASSERT_TRUE(error == 0) << "Decoder error: " << lodepng_error_text(error);
}

void SavePng(const std::string& filename, const std::vector<uint8>& image) {
  unsigned error =
      lodepng::encode(filename.c_str(), image, kImageWidth, kImageHeight);
  ASSERT_TRUE(error == 0) << "Encoder error: " << lodepng_error_text(error);
}

void FloatRGBToUint8RGBA(const std::vector<float>& input,
                         std::vector<uint8>* output) {
  output->resize(kImageHeight * kImageWidth * 4);
  for (int y = 0; y < kImageHeight; ++y) {
    for (int x = 0; x < kImageWidth; ++x) {
      for (int c = 0; c < 3; ++c) {
        (*output)[(y * kImageWidth + x) * 4 + c] =
            input[(y * kImageWidth + x) * 3 + c] * 255;
      }
      (*output)[(y * kImageWidth + x) * 4 + 3] = 255;
    }
  }
}

void ExpectImageFileAndImageAreEqual(const std::string& baseline_file,
                                     const std::vector<float>& result,
                                     const std::string& comparison_name,
                                     const std::string& failure_message) {
  std::vector<uint8> baseline_rgba, result_rgba;
  LoadPng(GetRunfilesRelativePath(baseline_file), &baseline_rgba);
  FloatRGBToUint8RGBA(result, &result_rgba);

  const bool images_match = baseline_rgba == result_rgba;

  if (!images_match) {
    const std::string result_output_path =
        "/tmp/" + comparison_name + "_result.png";
    SavePng(result_output_path, result_rgba);
  }

  EXPECT_TRUE(images_match) << failure_message;
}

class RasterizeTrianglesImplTest : public ::testing::Test {
 protected:
  void CallRasterizeTrianglesImpl(const float* vertices, const int32* triangles,
                                  int32 triangle_count) {
    const int num_pixels = image_height_ * image_width_;
    barycentrics_buffer_.resize(num_pixels * 3);
    triangle_ids_buffer_.resize(num_pixels);

    constexpr float kClearDepth = 1.0;
    z_buffer_.resize(num_pixels, kClearDepth);

    RasterizeTrianglesImpl(vertices, triangles, triangle_count, image_width_,
                           image_height_, triangle_ids_buffer_.data(),
                           barycentrics_buffer_.data(), z_buffer_.data());
  }

  // Expects that a pixel is covered by verifying that its barycentric
  // coordinates sum to one.
  void ExpectIsCovered(int x, int y) const {
    constexpr float kEpsilon = 1e-6f;
    auto it = barycentrics_buffer_.begin() + y * image_width_ * 3 + x * 3;
    EXPECT_NEAR(*it + *(it + 1) + *(it + 2), 1.0, kEpsilon);
  }

  int image_height_ = 480;
  int image_width_ = 640;
  std::vector<float> barycentrics_buffer_;
  std::vector<int32> triangle_ids_buffer_;
  std::vector<float> z_buffer_;
};

TEST_F(RasterizeTrianglesImplTest, CanRasterizeTriangle) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8,  0.0, 0.5,
                                       0.3,  0.5,  -0.5, 0.3};
  const std::vector<int32> triangles = {0, 1, 2};

  CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);
  ExpectImageFileAndImageAreEqual("Simple_Triangle.png", barycentrics_buffer_,
                                  "triangle", "simple triangle does not match");
}

TEST_F(RasterizeTrianglesImplTest, CanRasterizeTetrahedron) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8, 0.0, 0.5, 0.3,
                                       0.5,  -0.5, 0.3, 0.0, 0.0, 0.0};
  const std::vector<int32> triangles = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};

  CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 4);

  ExpectImageFileAndImageAreEqual("Simple_Tetrahedron.png",
                                  barycentrics_buffer_, "tetrahedron",
                                  "simple tetrahedron does not match");
}

TEST_F(RasterizeTrianglesImplTest, WorksWhenPixelIsOnTriangleEdge) {
  // Verifies that a pixel that lies exactly on a triangle edge is considered
  // inside the triangle.
  image_width_ = 641;
  const int x_pixel = image_width_ / 2;
  const float x_ndc = 0.0;
  constexpr int yPixel = 5;

  const std::vector<float> vertices = {x_ndc, -1.0, 0.5,  x_ndc, 1.0,
                                       0.5,   0.5,  -1.0, 0.5};
  {
    const std::vector<int32> triangles = {0, 1, 2};

    CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
  {
    // Test the triangle with the same vertices in reverse order.
    const std::vector<int32> triangles = {2, 1, 0};

    CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 1);

    ExpectIsCovered(x_pixel, yPixel);
  }
}

TEST_F(RasterizeTrianglesImplTest, CoversEdgePixelsOfImage) {
  // Verifies that the pixels along image edges are correct covered.

  const std::vector<float> vertices = {-1.0, -1.0, 0.0, 1.0,  -1.0, 0.0,
                                       1.0,  1.0,  0.0, -1.0, 1.0,  0.0};
  const std::vector<int32> triangles = {0, 1, 2, 0, 2, 3};

  CallRasterizeTrianglesImpl(vertices.data(), triangles.data(), 2);

  ExpectIsCovered(0, 0);
  ExpectIsCovered(image_width_ - 1, 0);
  ExpectIsCovered(image_width_ - 1, image_height_ - 1);
  ExpectIsCovered(0, image_height_ - 1);
}

}  // namespace
}  // namespace tf_mesh_renderer
