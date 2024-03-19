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


## TODO

- [ ] Release the training code.
- [ ] Release the training dataset.


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

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{jiang2023videobooth,
    author = {Jiang, Yuming and Wu, Tianxing and Yang, Shuai and Si, Chenyang and Lin, Dahua and Qiao, Yu and Loy, Chen Change and Liu, Ziwei},
    title = {VideoBooth: Diffusion-based Video Generation with Image Prompts},
    year = {2023}
}
```

