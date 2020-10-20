#!/bin/bash
GPU_NO=0;
is_bfm="False"

# # constants
basic_path=$(pwd)/3DMM/files/;
resources_path=$(pwd)/resources/;

uv_base="$basic_path/AI-NEXT-Albedo-Global.mat"
uv_regional_pyramid_base="$basic_path/AI-NEXT-AlbedoNormal-RPB/"

if [ $is_bfm == "False" ];then
    shape_exp_bases="$basic_path/AI-NEXT-Shape.mat"
else
    shape_exp_bases="$resources_path/BFM2009_Model.mat"
fi

vggpath="$resources_path/vgg-face.mat"
info_for_add_head="$resources_path/info_for_add_head.npy"
pb_path=$resources_path/PB/

# # data directories
is_only_four_frame="False"
ROOT_DIR=$(pwd)/test_data/RGBD/test1/;
img_dir="$ROOT_DIR"

########################################################
echo "prepare datas";

prefit_dir="$ROOT_DIR/prefit"
prepare_dir="$ROOT_DIR/prepare"

cd ./data_prepare

python -u run_data_preparation.py \
        --GPU_NO=${GPU_NO}  \
        --mode='test_RGBD' \
        --pb_path=${pb_path} \
        --img_dir=${img_dir}/ \
        --out_dir=${prepare_dir}

if [ "$?" -ne 0 ]; then echo "data prepare failed"; exit 1; fi

cd ..

########################################################
cd ./optimization/rgbd

echo "start RGBD opt process";

echo "step 0: load datas";
python -u step0_prepare_frontend_data.py \
        --capture_dir=$img_dir \
        --prepare_dir=${prepare_dir}

if [ "$?" -ne 0 ]; then echo "(step 0) load datas failed"; exit 1; fi

echo "step 1: choose frames : mid-left-right-up";
if [ $is_only_four_frame == "False" ];then
    python -u step1A_choose_frames.py \
            --prepare_dir=${prepare_dir}/    \
            --prefit=${prefit_dir}/
else
    python -u step1B_only4_choose_frames.py \
            --prepare_dir=${prepare_dir}/    \
            -prefit=${prefit_dir}/
fi

if [ "$?" -ne 0 ]; then echo "(step 1) choose frames failed"; exit 1; fi

echo "step 2: sparse fusion ";
python -u step2_sparse_fusion.py \
        --is_bfm=${is_bfm} \
        --prefit=${prefit_dir}/

if [ "$?" -ne 0 ]; then echo "(step 2) sparse fusion failed"; exit 1; fi

echo "step 3: prefit shape ";
python -u step3_prefit_shape.py \
        --GPU_NO=${GPU_NO}  \
        --is_bfm=${is_bfm} \
        --prefit=${prefit_dir}/    \
        --modle_path=${resources_path} \
        --basis3dmm_path=${shape_exp_bases}

if [ "$?" -ne 0 ]; then echo "(step 3) prefit shape failed"; exit 1; fi

echo "step 4: prefit Albedo_Global uv ";
if [ $is_bfm == "False" ];then
    python -u step4A_prefit_Albedo_Global.py \
            --GPU_NO=${GPU_NO}  \
            --is_bfm=${is_bfm} \
            --basis3dmm_path=${shape_exp_bases}  \
            --uv_path=${uv_base}  \
            --resources_path=${resources_path}  \
            --output_dir=${prefit_dir}/
else
    python -u step4B_prefit_bfm_rgb.py \
            --GPU_NO=${GPU_NO}  \
            --num_of_img=4 \
            --is_bfm=${is_bfm} \
            --basis3dmm_path=${shape_exp_bases} \
            --uv_path=${uv_base} \
            --vggpath=${vggpath} \
            --prefit_dir=${prefit_dir} \
            --prepare_dir=${prepare_dir} \
            --summary_dir=${prefit_dir}/sum_prefit_bfm_tex/ 
fi

if [ "$?" -ne 0 ]; then echo "(step 4) prefit uv failed"; exit 1; fi

echo "step 5: start RGBD opt ";

if [ $is_bfm == "False" ];then
    shape_out_dir=${ROOT_DIR}/our_opt_RGBD
else
    shape_out_dir=${ROOT_DIR}/bfm_opt_RGBD
fi

# constants
log_step=10
learning_rate=0.05
lr_decay_step=10
lr_decay_rate=0.9
photo_weight=100
gray_photo_weight=80
reg_shape_weight=0.4
reg_tex_weight=0.0001
depth_weight=1000
id_weight=1.8
real_86pt_lmk3d_weight=0.01
lmk_struct_weight=0
train_step=100
is_fixed_pose="False"
is_add_head_mirrow="False"
is_add_head_male="True"

python step5_run_RGBD_opt.py \
--GPU_NO=${GPU_NO} \
--is_bfm=${is_bfm} \
--basis3dmm_path=${shape_exp_bases} \
--uv_path=${uv_base} \
--vggpath=${vggpath} \
--info_for_add_head=${info_for_add_head} \
--prefit_dir=${prefit_dir} \
--prepare_dir=${prepare_dir} \
--log_step=${log_step} \
--train_step=${train_step} \
--learning_rate=${learning_rate} \
--lr_decay_step=${lr_decay_step} \
--lr_decay_rate=${lr_decay_rate} \
--photo_weight=${photo_weight} \
--gray_photo_weight=${gray_photo_weight} \
--depth_weight=${depth_weight} \
--id_weight=${id_weight} \
--reg_shape_weight=${reg_shape_weight} \
--reg_tex_weight=${reg_tex_weight} \
--real_86pt_lmk3d_weight=${real_86pt_lmk3d_weight} \
--lmk_struct_weight=${lmk_struct_weight} \
--num_of_img=4 \
--fixed_pose=${is_fixed_pose} \
--is_add_head_mirrow=${is_add_head_mirrow} \
--is_add_head_male=${is_add_head_male} \
--out_dir=${shape_out_dir} \
--summary_dir=${shape_out_dir}/summary \

if [ "$?" -ne 0 ]; then echo "(step 5) RGBD opt failed"; exit 1; fi

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
        --is_mult_view=True \
        --is_orig_img=False \
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
    scp ${shape_out_dir}/head* $results_dir/
    scp $ROOT_DIR/pix2pix_convert/output_for_texture_tex_D.png $results_dir/albedo.png
    scp $ROOT_DIR/pix2pix/out_for_texture_tex_N.png $results_dir/normal.png
else
    scp ${shape_out_dir}/face* $results_dir/
fi

echo "finish";