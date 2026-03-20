#!/bin/bash

# RealAD数据集逐类别测试脚本（内存优化版本）
# 每次只加载一个类别，测试完后释放内存

export CUDA_VISIBLE_DEVICES=6
# 设置数据路径
REALAD_DATA_PATH="/home/dataset/realiad_512"


MODEL_PATH="./train_on_mvtec/mvtec_epoch_1_model.pth"
SAVE_PATH="./test_realAD_results"

# 创建保存目录
mkdir -p ${SAVE_PATH}

echo "======================================"
echo "RealAD数据集逐类别测试"
echo "======================================"
echo "数据路径: ${REALAD_DATA_PATH}"
echo "模型路径: ${MODEL_PATH}"
echo "结果保存路径: ${SAVE_PATH}"
echo ""
echo "此脚本将逐个类别加载和测试，节省内存"
echo "======================================"
echo ""

# 运行测试
python test_realAD_by_class.py \
    --model_path ${MODEL_PATH} \
    --test_data_path ${REALAD_DATA_PATH} \
    --test_dataset realAD \
    --save_path ${SAVE_PATH} \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --feature_layers 1 6 12 18 24 \
    --image_size 518 \
    --sigma 4 \
    --test_batch_size 8 \
    --visualize_samples 3 \
    --seed 111

echo ""
echo "======================================"
echo "测试完成！"
echo "结果保存在: ${SAVE_PATH}"
echo "======================================"


