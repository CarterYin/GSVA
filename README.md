# GSVA Demo 使用说明

## 环境要求

### 依赖包
```bash
conda create -n gsva_demo python=3.10
```

```bash
conda activate gsva_demo
```

```bash
pip install -r requirements.txt
```

## 模型文件

使用前需要准备以下模型文件并放置在models/文件夹下：（注意是models不是model）
- https://cloud.tsinghua.edu.cn/d/1423fb16fdb9445e8155/
- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- https://huggingface.co/liuhaotian/llava-pretrain-llama-2-13b-chat
- https://huggingface.co/openai/clip-vit-large-patch14


## 使用方法

### 基本用法

```bash
python demo.py --input_image_path images/real_sample_image.jpg --prompt "white cloud" --output_image_path results/cloud.png --force_cpu
```
```bash
python demo.py --input_image_path images/person_test.jpg --prompt "man in the image" --output_image_path results/man_fixed.png --force_cpu
```
```bash
python demo.py --input_image_path images/car.jpg --prompt "a blue car" --output_image_path results/car.png --force_cpu
```

#### 可选参数
- `--precision`: 推理精度，可选 `fp32`、`bf16`、`fp16` (默认: fp32)
- `--lora_r`: LoRA rank，用于LoRA微调 (默认: 0)

### 可视化输出
程序会生成一个包含三张子图的图像：
1. **原始图像**: 输入的原始图像
2. **分割Mask**: 二值化的分割掩码，显示置信度
3. **叠加结果**: 分割结果叠加在原图上的效果
