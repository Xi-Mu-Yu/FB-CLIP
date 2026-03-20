#!/usr/bin/env python3
"""
逐类别测试脚本 - 内存优化版本
每次只加载一个类别的数据，测试完后释放内存
适用于大型数据集如realAD
"""

import FBCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import FBCLIP_PromptLearner
from utils import get_transform, normalize
from dataset import Dataset, generate_class_info
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
import matplotlib.pyplot as plt
import cv2
import gc
import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_trained_model(model_path, args, device):
    """加载训练好的模型"""
    print(f"正在加载训练好的模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    FBCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }
    
    model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=FBCLIP_parameters)
    
    for param in model.parameters():
        param.requires_grad_(False)
    
    prompt_learner = FBCLIP_PromptLearner(model.to("cpu"), FBCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.FB_params(args=args, device=device)
    
    checkpoint = torch.load(model_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    if "model_trainable_params" in checkpoint:
        for name, param_data in checkpoint["model_trainable_params"].items():
            for param_name, param in model.named_parameters():
                if param_name == name:
                    param.data.copy_(param_data)
                    break
    
    print(f"模型加载完成! 训练轮数: {checkpoint.get('epoch', 'Unknown')}")
    
    return model, prompt_learner

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    """将异常分数图叠加到原图上"""
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualize_test_results(img, anomaly_map, gt_mask, save_path, cls_name, sample_idx, dataset_name, is_anomaly):
    """可视化测试结果"""
    cls_save_path = os.path.join(save_path, 'visualizations', cls_name)
    os.makedirs(cls_save_path, exist_ok=True)
    
    # 转换为numpy数组
    img_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = ((img_np * [0.26862954, 0.26130258, 0.27577711] + [0.48145466, 0.4578275, 0.40821073]) * 255).astype(np.uint8)
    
    anomaly_np = anomaly_map.squeeze().cpu().numpy()
    anomaly_np = (anomaly_np - anomaly_np.min()) / (anomaly_np.max() - anomaly_np.min() + 1e-8)
    
    gt_np = gt_mask.squeeze().cpu().numpy()
    
    # 保存图像
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_original.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_gt.png'), (gt_np * 255).astype(np.uint8))
    
    anomaly_colored = cv2.applyColorMap((anomaly_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_anomaly.png'), anomaly_colored)
    
    overlay_img = apply_ad_scoremap(img_np, anomaly_np, alpha=0.5)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_overlay.png'), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

def test_single_class(model, prompt_learner, args, dataset_name, cls_name, device):
    """测试单个类别"""
    print(f"\n{'='*60}")
    print(f"正在测试类别: {cls_name}")
    print(f"{'='*60}")
    
    # 数据预处理
    preprocess, target_transform = get_transform(args)
    
    # 只加载当前类别的数据
    test_data = Dataset(
        root=args.test_data_path, 
        transform=preprocess, 
        target_transform=target_transform, 
        dataset_name=dataset_name,
        cls_name=cls_name  # 只加载这个类别
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.test_batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"类别 {cls_name} 的样本数: {len(test_data)}")
    
    # 初始化结果存储
    results = {
        'gt_sp': [],
        'pr_sp': [],
        'imgs_masks': [],
        'anomaly_maps': [],
        'inference_times': []
    }
    
    model.eval()
    
    import time
    
    # 测试当前类别
    for batch_idx, batch_items in enumerate(tqdm(test_dataloader, desc=f"Testing {cls_name}")):
        batch_image = batch_items['img'].to(device)
        batch_gt_mask = batch_items['img_mask']
        
        # 处理anomaly标签 - 如果已经是tensor就直接使用，否则转换
        if isinstance(batch_items['anomaly'], torch.Tensor):
            batch_anomaly_label = batch_items['anomaly'].to(device)
        else:
            batch_anomaly_label = torch.tensor(batch_items['anomaly'], dtype=torch.long).to(device)
        
        current_batch_size = batch_image.shape[0]
        
        # 处理ground truth mask
        batch_gt_mask[batch_gt_mask > 0.5] = 1
        batch_gt_mask[batch_gt_mask <= 0.5] = 0
        
        # 调整到标准尺寸
        standard_size = (518, 518)
        batch_gt_mask_resized = F.interpolate(batch_gt_mask.float(), size=standard_size, mode='nearest')
        batch_gt_mask_resized = batch_gt_mask_resized.byte()
        
        results['imgs_masks'].append(batch_gt_mask_resized)
        results['gt_sp'].extend(batch_anomaly_label.detach().cpu())
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # 生成prompts
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            
            # 预测
            predict_labels, predict_masks, consistency_loss = model.FB_encode(
                batch_image, args=args, 
                prompts=prompts, 
                tokenized_prompts=tokenized_prompts, 
                compound_prompts_text=compound_prompts_text
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            results['inference_times'].append(inference_time)
            
            # 图像级预测
            text_probs = predict_labels.detach().cpu()
            results['pr_sp'].extend(text_probs)
            
            # 像素级异常图
            anomaly_maps = predict_masks.detach().cpu()
            anomaly_maps = F.interpolate(anomaly_maps, size=standard_size, mode='bilinear', align_corners=False)
            
            # 高斯滤波
            if args.use_gaussian_filter:
                for i in range(anomaly_maps.shape[0]):
                    anomaly_map_np = anomaly_maps[i].squeeze().cpu().numpy()
                    anomaly_map_filtered = gaussian_filter(anomaly_map_np, sigma=args.sigma)
                    anomaly_maps[i] = torch.from_numpy(anomaly_map_filtered).unsqueeze(0)
            
            results['anomaly_maps'].append(anomaly_maps)
            
            # 可视化前几个样本
            if batch_idx == 0:
                for i in range(min(args.visualize_samples, current_batch_size)):
                    visualize_test_results(
                        batch_image[i:i+1], anomaly_maps[i:i+1], batch_gt_mask_resized[i:i+1], 
                        args.save_path, cls_name, i, dataset_name,
                        batch_anomaly_label[i]
                    )
    
    # 计算指标
    print(f"\n计算 {cls_name} 的指标...")
    results['imgs_masks'] = torch.cat(results['imgs_masks'])
    results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()
    
    # 包装成metrics函数需要的格式
    results_dict = {cls_name: results}
    
    image_auroc = image_level_metrics(results_dict, cls_name, "image-auroc")
    image_ap = image_level_metrics(results_dict, cls_name, "image-ap")
    pixel_auroc = pixel_level_metrics(results_dict, cls_name, "pixel-auroc")
    pixel_aupro = pixel_level_metrics(results_dict, cls_name, "pixel-aupro")
    
    # 计算推理速度
    cls_times = np.array(results['inference_times'])
    cls_mean_time = np.mean(cls_times)
    cls_per_image_time = cls_mean_time / args.test_batch_size
    cls_throughput = (args.test_batch_size * 1000) / cls_mean_time
    
    print(f"\n{cls_name} 测试结果:")
    print(f"  图像级 AUROC: {image_auroc:.4f} ({image_auroc*100:.1f}%)")
    print(f"  图像级 AP:    {image_ap:.4f} ({image_ap*100:.1f}%)")
    print(f"  像素级 AUROC: {pixel_auroc:.4f} ({pixel_auroc*100:.1f}%)")
    print(f"  像素级 AUPRO: {pixel_aupro:.4f} ({pixel_aupro*100:.1f}%)")
    print(f"  推理速度: {cls_per_image_time:.2f} ms/图像 ({cls_throughput:.2f} FPS)")
    
    # 清理内存
    del test_data, test_dataloader, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'class': cls_name,
        'image_auroc': image_auroc,
        'image_ap': image_ap,
        'pixel_auroc': pixel_auroc,
        'pixel_aupro': pixel_aupro,
        'per_image_time': cls_per_image_time,
        'throughput': cls_throughput
    }

def main():
    parser = argparse.ArgumentParser("逐类别测试脚本")
    
    # 模型和数据路径
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--test_dataset", type=str, required=True, help="测试数据集名称")
    parser.add_argument("--save_path", type=str, default='./test_results_by_class', help='结果保存路径')
    
    # 模型参数
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[5,11,17,24])
    parser.add_argument("--features_list", type=int, nargs="+", default=[5,11,17,24])
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[1,2,3,11,17,24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=int, default=4)
    
    # 测试参数
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--visualize_samples", type=int, default=3)
    parser.add_argument("--use_gaussian_filter", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--classes", type=str, nargs="+", help="指定要测试的类别列表，不指定则测试所有类别")
    
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载模型
    model, prompt_learner = load_trained_model(args.model_path, args, device)
    model.eval()
    
    # 获取数据集的所有类别
    obj_list, _ = generate_class_info(args.test_dataset)
    
    # 如果指定了类别，只测试指定的类别
    if args.classes:
        test_classes = [cls for cls in args.classes if cls in obj_list]
        if len(test_classes) < len(args.classes):
            missing = set(args.classes) - set(test_classes)
            print(f"警告: 以下类别不存在: {missing}")
    else:
        test_classes = obj_list
    
    print(f"\n将测试以下 {len(test_classes)} 个类别:")
    print(test_classes)
    
    # 逐个类别测试
    all_results = []
    
    for idx, cls_name in enumerate(test_classes):
        print(f"\n进度: {idx+1}/{len(test_classes)}")
        
        try:
            result = test_single_class(model, prompt_learner, args, args.test_dataset, cls_name, device)
            all_results.append(result)
        except Exception as e:
            print(f"错误: 测试类别 {cls_name} 时出错: {e}")
            continue
    
    # 汇总结果
    print("\n" + "="*60)
    print("所有类别测试完成！汇总结果:")
    print("="*60)
    
    table_data = []
    for result in all_results:
        table_data.append([
            result['class'],
            f"{result['pixel_auroc']*100:.1f}",
            f"{result['pixel_aupro']*100:.1f}",
            f"{result['image_auroc']*100:.1f}",
            f"{result['image_ap']*100:.1f}",
            f"{result['per_image_time']:.2f}",
            f"{result['throughput']:.2f}"
        ])
    
    # 计算平均值
    avg_pixel_auroc = np.mean([r['pixel_auroc'] for r in all_results])
    avg_pixel_aupro = np.mean([r['pixel_aupro'] for r in all_results])
    avg_image_auroc = np.mean([r['image_auroc'] for r in all_results])
    avg_image_ap = np.mean([r['image_ap'] for r in all_results])
    avg_per_image_time = np.mean([r['per_image_time'] for r in all_results])
    avg_throughput = np.mean([r['throughput'] for r in all_results])
    
    table_data.append([
        "平均",
        f"{avg_pixel_auroc*100:.1f}",
        f"{avg_pixel_aupro*100:.1f}",
        f"{avg_image_auroc*100:.1f}",
        f"{avg_image_ap*100:.1f}",
        f"{avg_per_image_time:.2f}",
        f"{avg_throughput:.2f}"
    ])
    
    headers = ["类别", "Pixel AUROC", "Pixel AUPRO", "Image AUROC", "Image AP", "ms/图像", "FPS"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存结果到文件
    results_file = os.path.join(args.save_path, f'{args.test_dataset}_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"测试数据集: {args.test_dataset}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"测试类别数: {len(all_results)}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write(f"\n\n平均指标:\n")
        f.write(f"  Pixel AUROC: {avg_pixel_auroc*100:.2f}%\n")
        f.write(f"  Pixel AUPRO: {avg_pixel_aupro*100:.2f}%\n")
        f.write(f"  Image AUROC: {avg_image_auroc*100:.2f}%\n")
        f.write(f"  Image AP: {avg_image_ap*100:.2f}%\n")
        f.write(f"  推理速度: {avg_per_image_time:.2f} ms/图像 ({avg_throughput:.2f} FPS)\n")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 保存JSON格式的详细结果
    json_file = os.path.join(args.save_path, f'{args.test_dataset}_results.json')
    with open(json_file, 'w') as f:
        json.dump({
            'dataset': args.test_dataset,
            'model_path': args.model_path,
            'results': all_results,
            'average': {
                'pixel_auroc': float(avg_pixel_auroc),
                'pixel_aupro': float(avg_pixel_aupro),
                'image_auroc': float(avg_image_auroc),
                'image_ap': float(avg_image_ap),
                'per_image_time': float(avg_per_image_time),
                'throughput': float(avg_throughput)
            }
        }, f, indent=2)
    
    print(f"详细结果已保存到: {json_file}")

if __name__ == "__main__":
    main()
