#!/usr/bin/env python3
"""
跨域异常检测脚本
在MVTec数据集上训练，在VISA数据集上进行零样本测试
"""

import FBCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import FBCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize, get_transform
from dataset import Dataset
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
import sys

class Tee:
    def __init__(self, file_path, stream):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, 'a', buffering=1)
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

def setup_global_logging(save_path):
    """Redirect stdout and stderr to a log file while keeping console output."""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')
    sys.stdout = Tee(log_file, sys.stdout)
    sys.stderr = Tee(log_file, sys.stderr)
    print(f"日志将保存到: {log_file}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    
    if inputs.shape != targets.shape:
        print(f"Warning: inputs shape {inputs.shape} != targets shape {targets.shape}")
        if len(inputs.shape) != len(targets.shape):
            if len(inputs.shape) > len(targets.shape):
                targets = targets.unsqueeze(-1)
            else:
                inputs = inputs.unsqueeze(-1)
    
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def dice_loss(inputs, targets, smooth=1e-6, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()

    if inputs.shape != targets.shape:
        print(f"Warning: inputs shape {inputs.shape} != targets shape {targets.shape}")
        if len(inputs.shape) != len(targets.shape):
            if len(inputs.shape) > len(targets.shape):
                targets = targets.unsqueeze(-1)
            else:
                inputs = inputs.unsqueeze(-1)

    # 展平
    inputs_flat = inputs.view(inputs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # 计算交集和并集
    intersection = (inputs_flat * targets_flat).sum(dim=1)
    union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    loss = 1 - (2 * intersection + smooth) / (union + smooth)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def focal_dice_loss(inputs, targets, alpha=0.25, gamma=2, smooth=1e-6, reduction="mean"):
    """
    Focal Dice Loss: 结合Focal Loss和Dice Loss的优势
    - 解决类别不平衡问题（Focal Loss特性）
    - 关注形状和重叠度（Dice Loss特性）
    - 对难分类样本给予更多关注
    
    Args:
        inputs: 预测结果 [B, C, H, W] 或 [B, H, W]
        targets: 真实标签 [B, C, H, W] 或 [B, H, W]
        alpha: Focal Loss权重参数，控制正负样本权重
        gamma: Focal Loss聚焦参数，控制难易样本权重
        smooth: 平滑参数，避免除零
        reduction: 损失聚合方式
    """
    inputs = inputs.float()
    targets = targets.float()

    if inputs.shape != targets.shape:
        print(f"Warning: inputs shape {inputs.shape} != targets shape {targets.shape}")
        if len(inputs.shape) != len(targets.shape):
            if len(inputs.shape) > len(targets.shape):
                targets = targets.unsqueeze(-1)
            else:
                inputs = inputs.unsqueeze(-1)

    # 展平
    inputs_flat = inputs.view(inputs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # 计算Dice系数
    intersection = (inputs_flat * targets_flat).sum(dim=1)
    union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    
    # 计算Dice Loss
    dice_loss = 1 - dice_coeff
    
    # 计算Focal权重
    # 对于正样本（targets=1），权重为alpha
    # 对于负样本（targets=0），权重为(1-alpha)
    alpha_t = alpha * targets_flat + (1 - alpha) * (1 - targets_flat)
    
    # 计算预测概率（用于Focal权重）
    p_t = inputs_flat * targets_flat + (1 - inputs_flat) * (1 - targets_flat)
    
    # 应用Focal权重到Dice Loss
    focal_weight = alpha_t * ((1 - p_t) ** gamma)
    focal_dice_loss = focal_weight.mean(dim=1) * dice_loss
    
    if reduction == "mean":
        loss = focal_dice_loss.mean()
    elif reduction == "sum":
        loss = focal_dice_loss.sum()
    else:
        loss = focal_dice_loss

    return loss

def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)

def visualize_cross_domain_results(image, anomaly_map, gt_mask, save_path, cls_name, source_domain, target_domain):
    """可视化跨域测试结果"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 异常图
    anomaly_np = anomaly_map.squeeze().cpu().numpy()
    im1 = axes[1].imshow(anomaly_np, cmap='hot')
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Ground Truth
    gt_np = gt_mask.squeeze().cpu().numpy()
    axes[2].imshow(gt_np, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # 叠加显示
    overlay = img_np.copy()
    anomaly_resized = cv2.resize(anomaly_np, (img_np.shape[1], img_np.shape[0]))
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], anomaly_resized)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.suptitle(f'{cls_name} - {source_domain}→{target_domain} Cross-Domain')
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{cls_name}_{source_domain}_to_{target_domain}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def train_on_source_domain(args):
    """在源域（MVTec）上训练模型"""
    print("=" * 50)
    print("在MVTec数据集上训练模型")
    print("=" * 50)
    
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据预处理
    preprocess, target_transform = get_transform(args)
    
    # 模型参数
    FBCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # 加载模型
    model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=FBCLIP_parameters)
    # model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    # MVTec训练数据
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.train)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    

    # 初始化prompt learner
    prompt_learner = FBCLIP_PromptLearner(model.to("cpu"), FBCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.FB_params(args=args, device=device)

    # 优化器 - 分别处理模型参数和prompt learner参数
    model_params = model.get_trainable_parameters()
    prompt_params = list(prompt_learner.parameters())
    
    # 检查参数是否都是叶子张量
    all_params = []
    for param in model_params + prompt_params:
        if param.requires_grad and param.is_leaf:
            all_params.append(param)
        else:
            print(f"跳过非叶子张量: {param.shape}, requires_grad: {param.requires_grad}, is_leaf: {param.is_leaf}")
    
    optimizer = torch.optim.Adam(all_params, lr=args.learning_rate, betas=(0.5, 0.999))

    print("Start train...")
    
    # 训练
    model.eval()
    prompt_learner.train()
    
    # 存储每个epoch的结果
    epoch_results = []
    
    for epoch in range(args.epoch):
        total_loss = []
        j = 0
        for i, items in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            # if i >= args.train_steps_per_epoch:  # 限制每个epoch的训练步数
            #     break
            j = j+1
                
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            labels = label.to(device)
            imgs = image.to(device)
            gts = gt.to(device)
            
            # 训练
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            predict_labels, predict_masks, consistency_loss = model.FB_encode(
                imgs, args=args, 
                prompts=prompts, 
                tokenized_prompts=tokenized_prompts, 
                compound_prompts_text=compound_prompts_text
            )
            
            # 确保gts是正确的维度
            if len(gts.shape) == 1:
                gts = gts.unsqueeze(0).unsqueeze(0)
                h_w = int(gts.shape[-1] ** 0.5)
                gts = gts.reshape(1, 1, h_w, h_w)
            elif len(gts.shape) == 2:
                gts = gts.unsqueeze(0).unsqueeze(0)
            elif len(gts.shape) == 3:
                gts = gts.unsqueeze(1)
            
            # gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
            predict_masks= F.interpolate(predict_masks, size=gts[0].shape[-2:], mode='bilinear')
            gts[gts < 0.5] = 0
            gts[gts > 0.5] = 1
            
            # 计算各个损失组件
            # 图像级损失：predict_labels 是概率，需要使用 focal_loss
            focal_label_loss = focal_loss(predict_labels, labels.float())
            focal_mask_loss = focal_loss(predict_masks, gts)
            
            # 像素级损失：使用 Dice Loss 替代 L1 Loss
            dice_loss_fn = BinaryDiceLoss()
            dice_mask_loss = dice_loss_fn(predict_masks.squeeze(1), gts.squeeze(1).float())*0.2
            
            loss = focal_label_loss + (focal_mask_loss + dice_mask_loss) + consistency_loss
            # loss = focal_label_loss + (focal_mask_loss + dice_mask_loss) 
            # 打印损失组件信息（每10个epoch打印一次）m
            # if j % 10 == 0 and len(total_loss) == 0:
            if j % 10 == 0 :
                print(f"Loss components - Focal_label: {focal_label_loss.item():.6f}, "
                    f"Focal_mask: {focal_mask_loss.item():.6f}, "
                    f"Dice_mask: {dice_mask_loss.item():.6f}")
                print(f"Consistency loss: {consistency_loss.item():.6f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        avg_loss = np.mean(total_loss)
        logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        if epoch <1 :
            continue

        # 每个epoch后保存模型
        model_path = os.path.join(args.save_path, f'{args.train}_epoch_{epoch}_model.pth')
        
        # 获取模型的所有可训练参数
        model_trainable_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_trainable_params[name] = param.data.clone()
        
        torch.save({
            "prompt_learner": prompt_learner.state_dict(),
            "model_trainable_params": model_trainable_params,
            "epoch": epoch,
            "loss": avg_loss,
            "source_domain": args.train
        }, model_path)
        
        print(f"Epoch {epoch} 模型已保存到: {model_path}")
        
        # 每个epoch后进行测试
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 训练完成，开始测试...")
        print(f"{'='*60}")
        
        # 测试当前epoch的模型
        test_results = test_on_target_domain(model, prompt_learner, args, epoch=epoch)
        epoch_results.append({
            'epoch': epoch,
            'loss': avg_loss,
            'test_results': test_results
        })
        
        # 保存当前epoch的测试结果
        save_epoch_results(epoch_results, args.save_path, epoch)

    # 保存最终模型
    final_model_path = os.path.join(args.save_path, f'{args.train}_final_model.pth')
    
    # 获取模型的所有可训练参数
    model_trainable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_trainable_params[name] = param.data.clone()
    
    torch.save({
        "prompt_learner": prompt_learner.state_dict(),
        "model_trainable_params": model_trainable_params,
        "epoch": args.epoch - 1,
        "loss": np.mean(total_loss),
        "source_domain": args.train
    }, final_model_path)
    
    print(f"\n训练完成! 最终模型已保存到: {final_model_path}")
    print(f"所有epoch模型保存在: {args.save_path}")
    
    return model, prompt_learner, epoch_results

def load_trained_model_for_testing(model_path, args, device):
    """加载训练好的模型用于测试"""
    print(f"正在加载训练好的模型: {model_path}")
    
    # 模型参数
    FBCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }
    
    # 加载模型
    model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=FBCLIP_parameters)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)
    
    # 初始化prompt learner
    prompt_learner = FBCLIP_PromptLearner(model.to("cpu"), FBCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.FB_params(args=args, device=device)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载prompt learner权重
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    # 加载模型的可训练参数
    if "model_trainable_params" in checkpoint:
        print("加载模型的可训练参数...")
        for name, param_data in checkpoint["model_trainable_params"].items():
            if name in dict(model.named_parameters()):
                # 找到对应的参数并加载
                for param_name, param in model.named_parameters():
                    if param_name == name:
                        param.data.copy_(param_data)
                        break
        print(f"已加载 {len(checkpoint['model_trainable_params'])} 个模型可训练参数")
    else:
        print("⚠️  警告: 模型文件中未找到可训练参数，使用默认参数")
    
    print(f"模型加载完成!")
    print(f"训练轮数: {checkpoint.get('epoch', 'Unknown')}")
    print(f"最终损失: {checkpoint.get('loss', 'Unknown')}")
    print(f"源域: {checkpoint.get('source_domain', 'Unknown')}")
    
    return model, prompt_learner

def test_on_target_domain(model, prompt_learner, args, epoch=None):
    """在目标域（VISA）上测试模型"""
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    print("=" * 50)
    print(f"在VISA数据集上进行零样本测试{epoch_str}")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据预处理
    preprocess, target_transform = get_transform(args)
    
    # VISA测试数据
    test_data = Dataset(root=args.test_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.test)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    
    # 获取测试数据集的类别列表
    obj_list = test_data.obj_list
    print(f"VISA数据集类别: {obj_list}")
    
    results = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []

    print("开始零样本测试...")
    
    # 按类别分组测试数据 - 优化版本
    class_data = {}
    for batch_items in test_dataloader:
        batch_size = batch_items['img'].shape[0]
        cls_names = batch_items['cls_name']
        
        for i in range(batch_size):
            cls_name = cls_names[i]
            if cls_name not in class_data:
                class_data[cls_name] = []
            
            # 创建单个样本的数据项
            item = {}
            for key in batch_items:
                if isinstance(batch_items[key], torch.Tensor):
                    item[key] = batch_items[key][i:i+1]  # 保持batch维度
                else:
                    item[key] = [batch_items[key][i]]
            
            class_data[cls_name].append(item)
    
    print(f"发现 {len(class_data)} 个类别，每个类别的样本数:")
    for cls_name, data_list in class_data.items():
        print(f"  {cls_name}: {len(data_list)} 个样本")
    
    # 逐个类别进行测试
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    
    for cls_idx, (cls_name, data_list) in enumerate(class_data.items()):


        print(f"\n{'='*60}")
        print(f"正在测试类别 {cls_idx+1}/{len(class_data)}: {cls_name}")
        print(f"样本数量: {len(data_list)}")
        print(f"{'='*60}")
        
        # if "pcb" not in cls_name:
        #     continue
        
        # 测试当前类别的所有样本 - 使用批次处理
        batch_size = args.test_batch_size
        total_samples = len(data_list)
        
        print(f"使用批次大小: {batch_size} 进行测试")
        
        for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Testing {cls_name}"):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_items = data_list[batch_start:batch_end]
            current_batch_size = len(batch_items)
            
            # 准备批次数据
            batch_images = []
            batch_gt_masks = []
            batch_anomaly_labels = []
            
            for item in batch_items:
                batch_images.append(item['img'])
                batch_gt_masks.append(item['img_mask'])
                batch_anomaly_labels.append(item['anomaly'])
            
            # 合并批次数据
            batch_image = torch.cat(batch_images, dim=0).to(device)
            batch_gt_mask = torch.cat(batch_gt_masks, dim=0)
            batch_anomaly_label = torch.cat(batch_anomaly_labels, dim=0)
            
            # 处理ground truth mask
            batch_gt_mask[batch_gt_mask > 0.5] = 1
            batch_gt_mask[batch_gt_mask <= 0.5] = 0
            
            # 将ground truth mask调整到标准尺寸518x518以保持一致性
            standard_size = (518, 518)
            batch_gt_mask_resized = F.interpolate(batch_gt_mask.float(), size=standard_size, mode='nearest')
            batch_gt_mask_resized = batch_gt_mask_resized.byte()
            
            # 存储ground truth数据
            results[cls_name]['imgs_masks'].append(batch_gt_mask_resized)
            results[cls_name]['gt_sp'].extend(batch_anomaly_label.detach().cpu())

            with torch.no_grad():
                # 使用训练好的prompt learner生成参数
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
                
                # 零样本预测
                predict_labels, predict_masks, consistency_loss = model.FB_encode(
                    batch_image, args=args, 
                    prompts=prompts, 
                    tokenized_prompts=tokenized_prompts, 
                    compound_prompts_text=compound_prompts_text
                )
                
                # 获取图像级预测结果
                text_probs = predict_labels.detach().cpu()
                results[cls_name]['pr_sp'].extend(text_probs)
                
                # 获取像素级异常图
                anomaly_maps = predict_masks.detach().cpu()
                
                # 确保异常图与ground truth mask尺寸匹配
                batch_gt_mask_resized = batch_gt_mask.squeeze().cpu()
                if len(batch_gt_mask_resized.shape) == 3:  # [batch, H, W]
                    gt_h, gt_w = batch_gt_mask_resized.shape[1], batch_gt_mask_resized.shape[2]
                    anomaly_maps = F.interpolate(anomaly_maps, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
                
                # 为了确保后续concatenation的一致性，将所有anomaly_maps调整到518x518标准尺寸
                standard_size = (518, 518)
                anomaly_maps = F.interpolate(anomaly_maps, size=standard_size, mode='bilinear', align_corners=False)
                
                # 应用高斯滤波（可选）
                for i in range(anomaly_maps.shape[0]):
                    anomaly_map_np = anomaly_maps[i].squeeze().cpu().numpy()
                    anomaly_map_filtered = gaussian_filter(anomaly_map_np, sigma=args.sigma)
                    anomaly_maps[i] = torch.from_numpy(anomaly_map_filtered).unsqueeze(0)
                
                results[cls_name]['anomaly_maps'].append(anomaly_maps)
                
                # 可视化前几张图像的结果
                for i in range(min(3, current_batch_size)):  # 每个批次只可视化前3张图像
                    item_idx = batch_start + i
                    if item_idx < 3:  # 每个类别只可视化前3张图像
                        visualize_cross_domain_results(
                            batch_image[i:i+1], anomaly_maps[i:i+1], batch_gt_mask_resized[i:i+1], 
                            os.path.join(args.save_path, "visualizations"), 
                            f"{cls_name}_{item_idx}", "mvtec", "visa"
                        )
        
        # 计算当前类别的指标
        print(f"\n计算 {cls_name} 的指标...")
        results[cls_name]['imgs_masks'] = torch.cat(results[cls_name]['imgs_masks'])
        results[cls_name]['anomaly_maps'] = torch.cat(results[cls_name]['anomaly_maps']).detach().cpu().numpy()
        
        # 计算图像级指标
        image_auroc = image_level_metrics(results, cls_name, "image-auroc")
        image_ap = image_level_metrics(results, cls_name, "image-ap")
        pixel_auroc = pixel_level_metrics(results, cls_name, "pixel-auroc")
        pixel_aupro = pixel_level_metrics(results, cls_name, "pixel-aupro")
        
        # 显示当前类别结果
        print(f"\n{cls_name} 测试结果:")
        print(f"  图像级 AUROC: {image_auroc:.4f} ({image_auroc*100:.1f}%)")
        print(f"  图像级 AP:    {image_ap:.4f} ({image_ap*100:.1f}%)")
        print(f"  像素级 AUROC: {pixel_auroc:.4f} ({pixel_auroc*100:.1f}%)")
        print(f"  像素级 AUPRO: {pixel_aupro:.4f} ({pixel_aupro*100:.1f}%)")
        
        # 添加到结果列表
        table = [cls_name, 
                f"{pixel_auroc*100:.1f}", 
                f"{pixel_aupro*100:.1f}", 
                f"{image_auroc*100:.1f}", 
                f"{image_ap*100:.1f}"]
        table_ls.append(table)
        
        image_auroc_list.append(image_auroc)
        image_ap_list.append(image_ap)
        pixel_auroc_list.append(pixel_auroc)
        pixel_aupro_list.append(pixel_aupro)
        

    # 计算最终平均指标
    table_ls.append(['MEAN', 
                    f"{np.mean(pixel_auroc_list)*100:.1f}",
                    f"{np.mean(pixel_aupro_list)*100:.1f}", 
                    f"{np.mean(image_auroc_list)*100:.1f}",
                    f"{np.mean(image_ap_list)*100:.1f}"])
    
    results_table = tabulate(table_ls, headers=['Objects', 'Pixel AUROC', 'Pixel AUPRO', 'Image AUROC', 'Image AP'], tablefmt="pipe")
    
    print("\n" + "=" * 80)
    print("跨域零样本测试最终结果")
    print("=" * 80)
    print(results_table)
    
    # 保存结果
    with open(os.path.join(args.save_path, 'cross_domain_results.txt'), 'w') as f:
        f.write("MVTec → VISA 跨域零样本测试结果\n")
        f.write("=" * 80 + "\n")
        f.write(results_table)
        f.write(f"\n\n详细统计:\n")
        f.write(f"图像级 AUROC: {np.mean(image_auroc_list):.4f} ± {np.std(image_auroc_list):.4f}\n")
        f.write(f"图像级 AP:    {np.mean(image_ap_list):.4f} ± {np.std(image_ap_list):.4f}\n")
        f.write(f"像素级 AUROC: {np.mean(pixel_auroc_list):.4f} ± {np.std(pixel_auroc_list):.4f}\n")
        f.write(f"像素级 AUPRO: {np.mean(pixel_aupro_list):.4f} ± {np.std(pixel_aupro_list):.4f}\n")
    
    # 准备返回的测试结果
    test_results = {
        'image_auroc': np.mean(image_auroc_list),
        'image_ap': np.mean(image_ap_list),
        'pixel_auroc': np.mean(pixel_auroc_list),
        'pixel_aupro': np.mean(pixel_aupro_list),
        'image_auroc_std': np.std(image_auroc_list),
        'image_ap_std': np.std(image_ap_list),
        'pixel_auroc_std': np.std(pixel_auroc_list),
        'pixel_aupro_std': np.std(pixel_aupro_list),
        'detailed_results': table_ls[:-1],  # 排除MEAN行
        'mean_results': table_ls[-1]  # MEAN行
    }
    
    epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
    results_file = os.path.join(args.save_path, f'cross_domain_results{epoch_str}.txt')
    
    print(f"\n结果已保存到: {args.save_path}")
    print(f"可视化结果已保存到: {os.path.join(args.save_path, 'visualizations')}")
    print(f"详细结果文件: {results_file}")
    
    return test_results

def save_epoch_results(epoch_results, save_path, current_epoch):
    """保存每个epoch的测试结果"""
    results_file = os.path.join(save_path, 'epoch_progress.txt')
    
    with open(results_file, 'w') as f:
        f.write("Epoch Progress Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':<8} {'Loss':<12} {'Image AUROC':<15} {'Pixel AUROC':<15} {'Pixel AUPRO':<15}\n")
        f.write("-" * 80 + "\n")
        
        for result in epoch_results:
            epoch = result['epoch']
            loss = result['loss']
            test_results = result['test_results']
            
            f.write(f"{epoch:<8} {loss:<12.6f} {test_results['image_auroc']:<15.4f} "
                   f"{test_results['pixel_auroc']:<15.4f} {test_results['pixel_aupro']:<15.4f}\n")
    
    print(f"Epoch {current_epoch} 结果已保存到: {results_file}")

def cross_domain_test(args):
    """跨域异常检测主函数"""
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "visualizations"), exist_ok=True)
    
    print("开始跨域异常检测实验")
    print(f"源域: {args.train.upper()} (训练)")
    print(f"目标域: {args.test.upper()} (零样本测试)")
    print(f"结果保存路径: {args.save_path}")
    
    # 步骤1: 在源域上训练（每个epoch后自动测试）
    model, prompt_learner, epoch_results = train_on_source_domain(args)
    
    # 打印最终训练总结
    print("\n" + "=" * 80)
    print("训练完成总结")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Loss':<12} {'Image AUROC':<15} {'Pixel AUROC':<15} {'Pixel AUPRO':<15}")
    print("-" * 80)
    
    for result in epoch_results:
        epoch = result['epoch']
        loss = result['loss']
        test_results = result['test_results']
        
        print(f"{epoch:<8} {loss:<12.6f} {test_results['image_auroc']:<15.4f} "
              f"{test_results['pixel_auroc']:<15.4f} {test_results['pixel_aupro']:<15.4f}")
    
   
    # 保存最终总结
    summary_file = os.path.join(args.save_path, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':<8} {'Loss':<12} {'Image AUROC':<15} {'Pixel AUROC':<15} {'Pixel AUPRO':<15}\n")
        f.write("-" * 80 + "\n")
        
        for result in epoch_results:
            epoch = result['epoch']
            loss = result['loss']
            test_results = result['test_results']
            
            f.write(f"{epoch:<8} {loss:<12.6f} {test_results['image_auroc']:<15.4f} "
                   f"{test_results['pixel_auroc']:<15.4f} {test_results['pixel_aupro']:<15.4f}\n")
     
    
    print(f"\n训练总结已保存到: {summary_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Cross-Domain Anomaly Detection", add_help=True)
    
    # 数据路径
    parser.add_argument("--train_data_path", type=str, default="../AF-CLIP/data/mvtec", help="训练数据路径")
    parser.add_argument("--test_data_path", type=str, default="../../AF-CLIP/data/visa", help="测试数据路径")
    parser.add_argument("--save_path", type=str, default='./cross_domain_results', help='结果保存路径')
    
    # 模型参数
    parser.add_argument("--depth", type=int, default=9, help="learnable text embedding depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable text embedding length")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[1,6,12,18,24], help="feature layers")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--train", type=str, default='mvtec', help="训练数据集名称")
    parser.add_argument("--test", type=str, default='visa', help="测试数据集名称")
    
    # 训练参数
    parser.add_argument("--epoch", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    parser.add_argument("--test_batch_size", type=int, default=8, help="测试批次大小")
    parser.add_argument("--train_steps_per_epoch", type=int, default=20, help="每个epoch的训练步数")
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    
    # 多数据集测试选项
    parser.add_argument("--multi_dataset", action="store_true", help="是否进行多数据集测试（一个训练集，多个测试集）")
    parser.add_argument("--test_datasets", type=str, nargs="+", default=["visa", "mvtec", "sdd"], 
                       help="要测试的数据集列表 (visa, mvtec, sdd, cifar10)")
    
    args = parser.parse_args()
    setup_global_logging(args.save_path)
    print("参数设置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    setup_seed(args.seed)
    
    # 根据参数选择测试模式

    cross_domain_test(args)
