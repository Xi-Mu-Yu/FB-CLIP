#!/bin/bash

echo "=========================================="
echo "Begin Test"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=8


MODEL_PATH="./1030_w015_no_learn_pool——2——16121824_Ares_10505/mvtec_epoch_1_model.pth"

# MODEL_PATH="./1030_w015_visa16121824——Ares_111/visa_epoch_2_model.pth"

# SAVE_PATH="./test_1027_noloss"
# SAVE_PATH="./test_1029_w015——pic——1"
# SAVE_PATH="./test_1101"
SAVE_PATH="./test_inference2"
# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "error: no model: $MODEL_PATH"
    exit 1
fi

# 定义多个数据集
DATASETS=(
    # "./data/visa visa"
    # "./data/mvtec mvtec"
    # "./data/CVC-ClinicDB clinic"
    # "./data/EndoTect EndoTect"
    # "./data/ISBI ISBI"
    # "./data/DTD-Synthetic DTD"
    # "./data/Kvasir kvasir"
    # "./data/btad btad"
    # "./data/MPDD mpdd"
    # "./data/SDD SDD"
    # "./data/dagm DAGM_KaggleUpload"
    # "./data/colon colon"
    "./data/BrainMRI BrainMRI"
    # "./data/Br35H Br35H"
    # "./data/TN3K TN3K"
    # "./data/HeadCT_anomaly_detection headct"
    # "./data/SDD SDD"
)

# 创建保存目录
mkdir -p ${SAVE_PATH}
mkdir -p ${SAVE_PATH}/visualizations

for ENTRY in "${DATASETS[@]}"; do
    TEST_DATA_PATH=$(echo $ENTRY | awk '{print $1}')
    TEST_DATASET=$(echo $ENTRY | awk '{print $2}')

    echo "------------------------------------------"
    echo "Start Test: ${TEST_DATASET}"
    echo "Path: ${TEST_DATA_PATH}"
    echo "------------------------------------------"

    python test_with_trained_model_pic.py \
        --model_path ${MODEL_PATH} \
        --test_data_path ${TEST_DATA_PATH} \
        --test_dataset ${TEST_DATASET} \
        --save_path ${SAVE_PATH}/${TEST_DATASET} \
        --depth 9 \
        --n_ctx 12 \
        --t_n_ctx 4 \
        --feature_layers 1 6 12 18 24\
        --image_size 518 \
        --sigma 4 \
        --visualize_samples 0 \
        --use_gaussian_filter \


    echo "Finish ${TEST_DATASET},result in: ${SAVE_PATH}/${TEST_DATASET}"
    echo ""
done

echo "=========================================="
echo "=========================================="

