#!/bin/bash
echo "compiling rasterizer"
TF_INC=/home/haoxianzhang/anaconda3/envs/py36tf18_tl/lib/python3.6/site-packages/tensorflow/include
TF_LIB=/home/haoxianzhang/anaconda3/envs/py36tf18_tl/lib/python3.6/site-packages/tensorflow
# you might need the following to successfully compile the third-party library
tf_mesh_renderer_path=$(pwd)/third_party/kernels/
g++ -std=c++11 \
    -shared $tf_mesh_renderer_path/rasterize_triangles_grad.cc $tf_mesh_renderer_path/rasterize_triangles_op.cc $tf_mesh_renderer_path/rasterize_triangles_impl.cc $tf_mesh_renderer_path/rasterize_triangles_impl.h \
    -o $tf_mesh_renderer_path/rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I$TF_INC -L$TF_LIB -ltensorflow_framework -O2

if [ "$?" -ne 0 ]; then echo "compile rasterizer failed"; exit 1; fi