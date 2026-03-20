#!/bin/bash


echo "=========================================="
echo "多数据集跨域异常检测脚本"
echo "=========================================="

# 设置GPU
export CUDA_VISIBLE_DEVICES=1

# 基础参数设置
SAVE_PATH="./train_visa_your" 
EPOCHS=3
TRAIN_STEPS_PER_EPOCH=20
BATCH_SIZE=4
LEARNING_RATE=0.00005
IMAGE_SIZE=518
SIGMA=4
SEED=42

# 创建保存目录
mkdir -p ${SAVE_PATH}

echo "开始多数据集跨域测试..."
echo "训练数据集: VisA"

# 运行多数据集测试（一个训练集，多个测试集）
python cross_domain_test.py \
    --train_data_path "./data/visa" \
    --test_data_path "./data/mvtec" \
    --save_path ${SAVE_PATH} \
    --train visa \
    --test mvtec \
    --epoch ${EPOCHS} \
    --train_steps_per_epoch ${TRAIN_STEPS_PER_EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --image_size ${IMAGE_SIZE} \
    --sigma ${SIGMA} \
    --multi_dataset \
    --test_datasets mvtec   \
    --feature_layers 1 6 12 18 24 \
    --seed ${SEED}

echo ""
echo "=========================================="
echo "多数据集跨域测试完成！"
echo "=========================================="
echo "结果保存在: ${SAVE_PATH}"
echo "汇总报告: ${SAVE_PATH}/multi_dataset_summary.txt"
echo ""


echo ""
echo "如需查看完整结果，请查看:"
echo "  ${SAVE_PATH}/multi_dataset_summary.txt"