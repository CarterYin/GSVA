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

## Hugging Face

```bash
pip install -U huggingface_hub
```

设置国内镜像站：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

登陆Hugging Face：
```bash
huggingface-cli login
```

## 下载模型
```bash
(gsva_demo) yinchao@ubuntu:~/GSVA$ mkdir -p models
(gsva_demo) yinchao@ubuntu:~/GSVA$ cd models
(gsva_demo) yinchao@ubuntu:~/GSVA/models$ git clone https://hf-mirror.com/liuhaotian/llava-llama-2-13b-chat-lightning-preview
(gsva_demo) yinchao@ubuntu:~/GSVA/models$ git clone https://hf-mirror.com/openai/clip-vit-large-patch14
(gsva_demo) yinchao@ubuntu:~/GSVA/models$ wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

使用前需要准备以下模型文件并放置在models/文件夹下：（注意是models不是model）
- https://cloud.tsinghua.edu.cn/d/1423fb16fdb9445e8155/
- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- https://huggingface.co/liuhaotian/llava-pretrain-llama-2-13b-chat
- https://huggingface.co/openai/clip-vit-large-patch14

### 模型文件下载
- 可能会遇到无法直接下载到服务器，可以下载到本地后推送到远程服务器

```bash
yinchao@yinchaodeMacBook-Air ~ % wget https://cloud.tsinghua.edu.cn/seafhttp/files/997e42b0-170d-4b5e-bd22-b5088fbe1aa8/gsva-7b-ft-gres.bin
```

例如：（打开终端后记得先cd到文件的本地目录）
```bash
scp gsva-7b-ft-gres.bin y****@*****:/home/yinchao/GSVA/models
```



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
