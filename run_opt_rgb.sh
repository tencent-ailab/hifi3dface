#!/bin/bash
GPU_NO=0;
is_bfm="False"

# # constants
basic_path=$(pwd)/3DMM/files/;
resources_path=$(pwd)/resources/;

uv_base="$basic_path/AI-NEXT-Albedo-Global.mat"
uv_regional_pyramid_base="$basic_path/AI-NEXT-AlbedoNormal-RPB/"

if [ $is_bfm == "False" ];then
    shape_exp_bases="$basic_path/AI-NEXT-Shape-NoAug.mat"
else
    shape_exp_bases="$resources_path/BFM2009_Model.mat"
fi

vggpath="$resources_path/vgg-face.mat"
pb_path=$resources_path/PB/

# # data directories
ROOT_DIR=$(pwd)/test_data/RGB/test1/single_img/;
img_dir=$ROOT_DIR

########################################################
echo "prepare datas";
cd ./data_prepare

prepare_dir="$ROOT_DIR/prepare_rgb"

python -u run_data_preparation.py \
        --GPU_NO=${GPU_NO}  \
        --mode='test_RGB' \
        --pb_path=${pb_path} \
        --img_dir=${img_dir} \
        --out_dir=${prepare_dir} 

if [ "$?" -ne 0 ]; then echo "data prepare failed"; exit 1; fi

cd ..

########################################################
echo "start RGB opt";

if [ $is_bfm == "False" ];then
    shape_out_dir=${ROOT_DIR}/our_opt_RGB
else
    shape_out_dir=${ROOT_DIR}/bfm_opt_RGB
fi


cd ./optimization/rgb

train_step=150
log_step=20
learning_rate=0.05
lr_decay_step=20
lr_decay_rate=0.9

photo_weight=100.0
gray_photo_weight=80.0
reg_shape_weight=0.5
reg_tex_weight=2.0
id_weight=1.0
real_86pt_lmk3d_weight=5.0
real_68pt_lmk2d_weight=5.0
lmk_struct_weight=0

num_of_img=1
project_type="Pers"

python run_RGB_opt.py \
--GPU_NO=${GPU_NO} \
--is_bfm=${is_bfm} \
--basis3dmm_path=${shape_exp_bases} \
--uv_path=${uv_base} \
--vggpath=${vggpath} \
--base_dir=${prepare_dir} \
--log_step=${log_step} \
--train_step=${train_step} \
--learning_rate=${learning_rate} \
--lr_decay_step=${lr_decay_step} \
--lr_decay_rate=${lr_decay_rate} \
--photo_weight=${photo_weight} \
--gray_photo_weight=${gray_photo_weight} \
--id_weight=${id_weight} \
--reg_shape_weight=${reg_shape_weight} \
--reg_tex_weight=${reg_tex_weight} \
--real_86pt_lmk3d_weight=${real_86pt_lmk3d_weight} \
--real_68pt_lmk2d_weight=${real_68pt_lmk2d_weight} \
--lmk_struct_weight=${lmk_struct_weight} \
--num_of_img=${num_of_img} \
--out_dir=${shape_out_dir} \
--summary_dir=${shape_out_dir}/summary \
--project_type=${project_type} 

if [ "$?" -ne 0 ]; then echo "RGB opt failed"; exit 1; fi

cd ../..
########################################################
if [ $is_bfm == "False" ];then
    echo "start generate HD texture";
    cd ./texture

    echo "step0: start unwrap";
    CUDA_VISIBLE_DEVICES=${GPU_NO} python -u step0_unwrapper.py \
        --basis3dmm_path=${shape_exp_bases} \
        --uv_path=${uv_base} \
        --uv_size=512 \
        --is_mult_view=False \
        --is_orig_img=True \
        --input_dir=${shape_out_dir} \
        --output_dir=${ROOT_DIR}/unwrap

    if [ "$?" -ne 0 ]; then echo "unwrap failed"; exit 1; fi

    echo "step1: start fit AlbedoNormal_RPB";
    CUDA_VISIBLE_DEVICES=${GPU_NO} python -u step1_fit_AlbedoNormal_RPB.py \
        --basis3dmm_path=${shape_exp_bases} \
        --uv_path=${uv_regional_pyramid_base} \
        --write_graph=False \
        --data_dir=${ROOT_DIR}/unwrap \
        --out_dir=${ROOT_DIR}/fit_unwrap \
        --uv_tv_weight=0.1 \
        --uv_reg_tex_weight=0.001 \
        --learning_rate=0.1 \
        --train_step=200

    if [ "$?" -ne 0 ]; then echo "fit UV failed"; exit 1; fi

    echo "step2: generate tex";
    CUDA_VISIBLE_DEVICES=${GPU_NO} python -u step2_pix2pix.py --mode texture --func test --pb_path ${pb_path}/pix2pix_tex.pb \
        --input_dir=${ROOT_DIR}/fit_unwrap \
        --output_dir=${ROOT_DIR}/pix2pix

    if [ "$?" -ne 0 ]; then echo "generate tex failed"; exit 1; fi

    echo "step2: generate norm";
    CUDA_VISIBLE_DEVICES=${GPU_NO} python -u step2_pix2pix.py --mode normal --func test --pb_path ${pb_path}/pix2pix_norm.pb \
        --input_dir=${ROOT_DIR}/fit_unwrap \
        --output_dir=${ROOT_DIR}/pix2pix

    if [ "$?" -ne 0 ]; then echo "generate norm failed"; exit 1; fi

    FIT_DIR=$ROOT_DIR/fit_unwrap
    PIX_DIR=$ROOT_DIR/pix2pix
    UNWRAP_DIR=$ROOT_DIR/unwrap
    OUT_DIR=$ROOT_DIR/pix2pix_convert

    echo "step3: convert_texture_domain";
    python -u step3_convert_texture_domain.py \
        --input_fit_dir=$FIT_DIR \
        --input_pix2pix_dir=$PIX_DIR \
        --input_unwrap_dir=$UNWRAP_DIR \
        --output_dir=$OUT_DIR

    if [ "$?" -ne 0 ]; then echo "convert_texture_domain failed"; exit 1; fi

    cd ..
fi

########################################################
echo "output results";

results_dir=$ROOT_DIR/results
mkdir -p $results_dir

if [ $is_bfm == "False" ];then
    scp $(pwd)/test.mtl $results_dir/
    scp ${shape_out_dir}/face.obj $results_dir/head.obj
    scp $ROOT_DIR/pix2pix_convert/output_for_texture_tex_D.png $results_dir/albedo.png
    scp $ROOT_DIR/pix2pix/out_for_texture_tex_N.png $results_dir/normal.png
else
    scp ${shape_out_dir}/face* $results_dir/
fi

echo "finish";