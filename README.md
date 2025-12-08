# ZeroPlantSeg

<p>
  <a href="https://arxiv.org/abs/2509.09116" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2509.09116-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
  </a>
</p>

This repository contains the official implementation of [Zero-shot Hierarchical Plant Segmentation via Foundation Segmentation Models and Text-to-image Attention](https://arxiv.org/abs/2509.09116)
(Junhao Xing, Ryohei Miyakawa, Yang Yang, Xinpeng Liu, [Risa Shinoda](https://sites.google.com/view/risashinoda/home), [Hiroaki Santo](https://sites.google.com/view/hiroaki-santo/), Yosuke Toda, [Fumio Okura](https://fokura.jp/)).

# Installation
### Local Setup
Clone this repo and install the requirements:
```shell
git clone https://github.com/JunhaoXing/ZeroPlantSeg.git
pip3 install -r requirements.txt
```


### Docker
Build the environment as `docker/README.md`.

# Usage
### Preliminaries
Run `ckpt_download.sh` to acquire necessay checkpoints:
```shell
sh ./ckpt_download.sh
```

### 1. Leaf Mask Collection
To collect leaf mask candidates, run `leaf_mask_collection.py` with
```shell
python3 leaf_mask_collection.py --dataset ${dataset_name} --mode ${val_or_test}
```
### 2. Leaf Instance Segmentation
To get leaf instance segmentation, run `leaf_segmentation.py` with
```shell
python3 leaf_segmentation.py --dataset ${dataset_name} --mode ${val_or_test}
```
### 3. Plant Instance Sementation
To get plant instance segmentation, run `plant_segmentation.py` with
```shell
python3 leaf_segmentation.py --dataset ${dataset_name} --mode ${val_or_test}
```

