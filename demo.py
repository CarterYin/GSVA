#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSVA Demo Script 
用于单张图片的分割推理和可视化
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import transformers
from torch.utils.data import DataLoader
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型相关模块
from model import LisaGSVAForCausalLM, add_task_tokens, init_vision_seg_for_model
import model.llava.conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.mm_utils import tokenizer_image_token


def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


class GSVADemo:
    def __init__(self, precision: str = "fp32", lora_r: int = 0, force_cpu: bool = False):
        """初始化GSVA Demo"""
        self.device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if precision == "bf16" else torch.half if precision == "fp16" else torch.float32
        self.lora_r = lora_r
        
        # 设置设备特定的确定性选项
        if self.device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        
        # 设置模型路径
        self.model_paths = self._setup_model_paths()
        
        # 初始化tokenizer
        print(f"正在加载tokenizer: {self.model_paths['mllm_model_path']}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_paths['mllm_model_path'],
            cache_dir=None,
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
            local_files_only=True
        )
        
        # 添加特殊token
        self.tokenizer, self.args = self._add_task_tokens()
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载预训练权重
        self._load_pretrained_weights()
            
        # 设置对话模板
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
        
        print(f"GSVA Demo 初始化完成，使用设备: {self.device}")
        
    def _setup_model_paths(self):
        """设置模型路径"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        model_paths = {
            'model_path': os.path.join(models_dir, "gsva-7b-ft-gres.bin"),
            'mllm_model_path': os.path.join(models_dir, "llava-llama-2-13b-chat-lightning-preview"),
            'vision_tower': os.path.join(models_dir, "clip-vit-large-patch14"),
            'segmentation_model_path': os.path.join(models_dir, "sam_vit_h_4b8939.pth")
        }
        
        # 检查模型文件是否存在
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在: {name} -> {path}")
            print(f"找到模型文件: {name} -> {path}")
            
        return model_paths
        
    def _add_task_tokens(self):
        """添加任务相关的特殊token"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加[SEG]和[REJ] token
        self.tokenizer.add_tokens("[SEG]")
        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        
        self.tokenizer.add_tokens("[REJ]")
        rej_token_idx = self.tokenizer("[REJ]", add_special_tokens=False).input_ids[0]
        
        # 添加图像token
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        # 创建args对象
        class Args:
            def __init__(self):
                self.seg_token_idx = seg_token_idx
                self.rej_token_idx = rej_token_idx
                self.use_mm_start_end = True
                self.local_rank = 0
                self.lora_alpha = 16
                self.lora_dropout = 0.05
                self.lora_target_modules = "q_proj,v_proj"
                self.eval_only = True
                self.train_mask_decoder = True
                self.out_dim = 256
                self.ce_loss_weight = 1.0
                self.dice_loss_weight = 0.5
                self.bce_loss_weight = 2.0
                
        args = Args()
        args.torch_dtype = self.torch_dtype
        args.local_rank = 0
        args.lora_r = self.lora_r
        
        return self.tokenizer, args
        
    def _create_model(self):
        """创建GSVA模型"""
        # 在模型创建前确保随机种子已设置
        set_seed(42)
        
        model_args = {
            "train_mask_decoder": True,
            "out_dim": 256,
            "ce_loss_weight": 1.0,
            "dice_loss_weight": 0.5,
            "bce_loss_weight": 2.0,
            "seg_token_idx": self.args.seg_token_idx,
            "segmentation_model_path": self.model_paths['segmentation_model_path'],
            "vision_tower": self.model_paths['vision_tower'],
            "use_mm_start_end": True,
            "tokenizer": self.tokenizer,
            "rej_token_idx": self.args.rej_token_idx
        }
        
        model = LisaGSVAForCausalLM.from_pretrained(
            self.model_paths['mllm_model_path'],
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            **model_args
        )
        
        model = init_vision_seg_for_model(model, self.tokenizer, self.args)
        model.resize_token_embeddings(len(self.tokenizer))
        model = model.to(self.device)
        
        # 确保模型处于推理模式
        model.eval()
        
        # 设置模型为确定性模式
        for module in model.modules():
            try:
                if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
                    module.dropout.p = 0.0
                elif hasattr(module, 'p') and hasattr(module, 'training'):
                    module.p = 0.0
            except (AttributeError, TypeError):
                # 跳过无法设置的模块
                continue
        
        return model
        
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        print(f"正在加载预训练权重: {self.model_paths['model_path']}")
        try:
            state_dict = torch.load(self.model_paths['model_path'], map_location="cpu")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"警告: 缺少 {len(missing_keys)} 个键")
            if unexpected_keys:
                print(f"警告: 多余 {len(unexpected_keys)} 个键")
            print("预训练权重加载完成!")
        except Exception as e:
            print(f"警告: 预训练权重加载失败: {e}")
        
    def _preprocess_image(self, image: Image.Image):
        """预处理图像"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        original_size = image.size
        image_np = np.array(image)
        
        # SAM模型预处理 (1024x1024)
        sam_img_size = 1024
        sam_image_resized = cv2.resize(image_np, (sam_img_size, sam_img_size))
        sam_image_tensor = torch.from_numpy(sam_image_resized).permute(2, 0, 1).float()
        
        sam_pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        sam_pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        sam_image_tensor = (sam_image_tensor - sam_pixel_mean) / sam_pixel_std
        
        # CLIP模型预处理 (224x224)
        clip_img_size = 224
        clip_image_resized = cv2.resize(image_np, (clip_img_size, clip_img_size))
        clip_image_tensor = torch.from_numpy(clip_image_resized).permute(2, 0, 1).float()
        
        clip_pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        clip_pixel_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        clip_image_tensor = (clip_image_tensor / 255.0 - clip_pixel_mean) / clip_pixel_std
        
        images = sam_image_tensor.unsqueeze(0)
        images_clip = clip_image_tensor.unsqueeze(0)
        
        return images, images_clip, original_size
        
    def _create_conversation(self, description: str) -> str:
        """创建对话"""
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        
        enhanced_prompt = f"{DEFAULT_IMAGE_TOKEN}\nPlease segment the {description} in this image. Focus on the complete object boundaries and provide a precise segmentation mask."
        
        conv.append_message(conv.roles[0], enhanced_prompt)
        conv.append_message(conv.roles[1], "I will segment the [SEG].")
        
        return conv.get_prompt()
        
    def _prepare_inputs(self, image: Image.Image, description: str):
        """准备模型输入"""
        images, images_clip, original_size = self._preprocess_image(image)
        conversation = self._create_conversation(description)
        
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation = conversation.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
        input_ids = tokenizer_image_token(conversation, self.tokenizer, return_tensors="pt")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        
        offset = torch.tensor([0, 1])
        masks_list = [torch.zeros((1, original_size[1], original_size[0]))]
        label_list = [torch.zeros((original_size[1], original_size[0]))]
        resize_list = [(1024, 1024)]
        do_segs = [True]
        
        return {
            "images": images.to(self.device, dtype=self.torch_dtype),
            "images_clip": images_clip.to(self.device, dtype=self.torch_dtype),
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "attention_masks": attention_mask.to(self.device),
            "offset": offset.to(self.device),
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "do_segs": do_segs,
            "inference": True
        }
        
    def forward(self, model, image: Image.Image, description: str):
        """核心推理函数"""
        # 确保推理过程中的确定性
        torch.set_grad_enabled(False)
        
        with torch.no_grad():
            inputs = self._prepare_inputs(image, description)
            outputs = model(**inputs)
            pred_masks = outputs["pred_masks"]
            
            if len(pred_masks) > 0 and len(pred_masks[0]) > 0:
                pred_mask = pred_masks[0][0].cpu().numpy()
                # 使用sigmoid将logits转换为概率值
                pred_mask_prob = 1 / (1 + np.exp(-pred_mask))
                threshold = max(0.3, pred_mask_prob.max() * 0.6)
                binary_mask = (pred_mask_prob > threshold).astype(np.uint8)
                
                # 形态学操作改善mask质量
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                
                original_size = image.size
                if binary_mask.shape != (original_size[1], original_size[0]):
                    binary_mask = cv2.resize(binary_mask, (original_size[0], original_size[1]))
                    binary_mask = (binary_mask > 0.5).astype(np.uint8)
                
                mask_area = np.sum(binary_mask)
                total_area = binary_mask.shape[0] * binary_mask.shape[1]
                coverage_ratio = mask_area / total_area
                
                if coverage_ratio < 0.001 or coverage_ratio > 0.8:
                    return {
                        "success": False,
                        "mask": None,
                        "confidence": float(pred_mask_prob.max()),
                        "original_size": original_size,
                        "reason": f"Mask coverage ratio ({coverage_ratio:.3f}) out of reasonable range",
                        "debug_info": {
                            "raw_max": float(pred_mask.max()),
                            "prob_max": float(pred_mask_prob.max())
                        }
                    }
                
                return {
                    "success": True,
                    "mask": binary_mask,
                    "confidence": float(pred_mask_prob.max()),
                    "original_size": original_size,
                    "coverage_ratio": coverage_ratio,
                    "debug_info": {
                        "raw_max": float(pred_mask.max()),
                        "prob_max": float(pred_mask_prob.max())
                    }
                }
            else:
                return {
                    "success": False,
                    "mask": None,
                    "confidence": 0.0,
                    "original_size": image.size,
                    "reason": "No mask predicted"
                }
                
    def visualize_result(self, image: Image.Image, result, output_path: str):
        """可视化分割结果"""
        if not result["success"]:
            print(f"分割失败: {result.get('reason', '未知原因')}")
            return
            
        # 设置matplotlib支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=12)
        ax1.axis('off')
        
        # 分割mask
        mask = result["mask"]
        ax2.imshow(mask, cmap='gray')
        title = f"Segmentation Mask (Confidence: {result['confidence']:.3f})"
        if 'coverage_ratio' in result:
            title += f"\nCoverage: {result['coverage_ratio']:.3f}"
        ax2.set_title(title, fontsize=12)
        ax2.axis('off')
        
        # 叠加结果
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
        colored_mask[mask == 1] = [1, 0, 0, 0.5]
        
        image_array = np.array(image)
        ax3.imshow(image_array)
        ax3.imshow(colored_mask)
        ax3.set_title("Segmentation Result", fontsize=12)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到: {output_path}")
        print(f"分割成功 - 置信度: {result['confidence']:.3f}, 覆盖率: {result.get('coverage_ratio', 0):.3f}")
        
        # 添加调试信息
        if 'debug_info' in result:
            print(f"调试信息 - 原始logits最大值: {result['debug_info']['raw_max']:.3f}, 转换后概率最大值: {result['debug_info']['prob_max']:.3f}")
        
    def process_image(self, image_path: str, description: str, output_path: str):
        """处理单张图像"""
        image = Image.open(image_path)
        result = self.forward(self.model, image, description)
        self.visualize_result(image, result, output_path)
        return result


def main():
    parser = argparse.ArgumentParser(description="GSVA Demo - 图像分割推理")
    parser.add_argument("--input_image_path", required=True, type=str, help="输入图像路径")
    parser.add_argument("--prompt", required=True, type=str, help="对象描述")
    parser.add_argument("--output_image_path", required=True, type=str, help="输出可视化结果路径")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "bf16", "fp16"], help="推理精度")
    parser.add_argument("--lora_r", default=0, type=int, help="LoRA rank")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--seed", default=42, type=int, help="随机种子，用于确保结果可重现")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    if not os.path.exists(args.input_image_path):
        print(f"错误: 输入图像文件不存在: {args.input_image_path}")
        return
        
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
    
    print("正在初始化GSVA Demo...")
    demo = GSVADemo(
        precision=args.precision,
        lora_r=args.lora_r,
        force_cpu=args.force_cpu
    )
    
    print(f"正在处理图像: {args.input_image_path}")
    print(f"描述: {args.prompt}")
    
    result = demo.process_image(args.input_image_path, args.prompt, args.output_image_path)
    
    if result["success"]:
        print(f"分割成功! 置信度: {result['confidence']:.3f}")
    else:
        print(f"分割失败: {result.get('reason', '未知原因')}")


if __name__ == "__main__":
    main() 