# VideoBooth

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.99999-b31b1b.svg)](https://arxiv.org/abs/2311.99999) -->
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](xxxx)
[![Project Page](https://img.shields.io/badge/VideoBooth-Website-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/VideoBooth-project/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/10DxH1JETzI)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVideoBooth&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


This repository will contain the implementation of the following paper:
> **VideoBooth: Diffusion-based Video Generation with Image Prompts**<br>
> [Yuming Jiang](https://yumingj.github.io/), [Tianxing Wu](https://tianxingwu.github.io/), [Shuai Yang](https://williamyang1991.github.io/), [Chenyang Si](https://chenyangsi.top/), [Dahua Lin](http://dahua.site/), [Yu Qiao](https://scholar.google.com.sg/citations?user=gFtI-8QAAAAJ&hl=en), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), [Ziwei Liu](https://liuziwei7.github.io/)<br>

From [MMLab@NTU](https://www.mmlab-ntu.com/) affliated with S-Lab, Nanyang Technological University and Shanghai AI Laboratory.

## Overview
Our VideoBooth generates videos with the subjects specified in the image prompts.
![overall_structure](./assets/teaser.png)


## Installation

1. Clone the repository.

```shell
git clone https://github.com/Vchitect/VideoBooth.git
cd VideoBooth
```

2. Install the environment.

```shell
conda env create -f environment.yml
conda activate videobooth
```

3. Download pretrained models ([Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [VideoBooth](https://huggingface.co/yumingj/VideoBooth_models/tree/main)), and put them under the folder `./pretrained_models/`.


## Inference

Here, we provide one example to perform the inference.

``` shell
python sample_scripts/sample.py --config sample_scripts/configs/panda.yaml
```

If you want to use your own image, you need to segment the object first. We use [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to segment the subject from images.

## Training

VideoBooth is training in a coarse-to-fine manner.

# Stage 1: Coarse Stage Training

``` shell
srun --mpi=pmi2 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29125 train_stage1.py \
--model TAVU \
--num-frames 16 \
--dataset WebVideoImageStage1  \
--frame-interval 4 \
--ckpt-every 1000 \
--clip-max-norm 0.1 \
--global-batch-size 16 \
--reg-text-weight 0 \
--results-dir ./results \
--pretrained-t2v-model path-to-t2v-model \
--global-mapper-path path-to-elite-global-model
```

# Stage 2: Fine Stage Training

``` shell
srun --mpi=pmi2 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29125 train_stage2.py \
--model TAVU \
--num-frames 16 \
--dataset WebVideoImageStage2  \
--frame-interval 4 \
--ckpt-every 1000 \
--clip-max-norm 0.1 \
--global-batch-size 16 \
--reg-text-weight 0 \
--results-dir ./results \
--pretrained-t2v-model path-to-t2v-model \
--global-mapper-path path-to-stage1-model
```

## Dataset Preparation

You can download our proposed dataset in [HuggingFace](https://huggingface.co/datasets/yumingj/VideoBoothDataset).

```shell
# merge the splited zip files
zip -F webvid_parsing_2M_split.zip --out single-archive.zip

# replace the path-to-webvid-parsing to this path
unzip single-archive.zip

# replace the path-to-videobooth-subset to this path
unzip webvid_parsing_videobooth_subset.zip
```


## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{jiang2023videobooth,
    author = {Jiang, Yuming and Wu, Tianxing and Yang, Shuai and Si, Chenyang and Lin, Dahua and Qiao, Yu and Loy, Chen Change and Liu, Ziwei},
    title = {VideoBooth: Diffusion-based Video Generation with Image Prompts},
    year = {2023}
}
```

