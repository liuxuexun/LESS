# LESS: Label-Efficient and Single-Stage Referring 3D Instance Segmentation

This repository is the offical implementation of paper [LESS: Label-Efficient and Single-Stage Referring 3D Instance Segmentation]()

## Abstract

Referring 3D Segmentation is a visual-language task that segments all points of the specified object from a 3D point cloud described by a sentence of query. Previous works perform a two-stage paradigm, first conducting language-agnostic instance segmentation then matching with given text query. However, the semantic concepts from text query and visual cues are separately interacted during the training, and both instance and semantic labels for each object are required, which is time consuming and human-labor intensive. To mitigate these issues, we propose a novel Referring 3D Segmentation pipeline, Label-Efficient and Single-Stage, dubbed LESS, which is only under the supervision of efficient binary mask. Specifically, we design a Point-Word Cross-Modal Alignment module for aligning the fine-grained features of points and textual embedding. Query Mask Predictor module and Query-Sentence Alignment module are introduced for coarse-grained alignment between masks and query. Furthermore, we propose an area regularization loss, which coarsely reduces irrelevant background predictions on a large scale. Besides, a point-to-point contrastive loss is proposed concentrating on distinguishing points with subtly similar features. Through extensive experiments, we achieve state-of-the-art performance on ScanRefer dataset by surpassing the previous methods about 3.7% mIoU using only binary labels.

## Installation

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.x`.

- Create a conda virtual environment

  ```
  conda create -n less python=3.8
  conda activate less
  ```
- Clone the repository

  ```
  git clone https://github.com/mellody11/LESS.git
  ```
- Install the dependencies

  1. Install [Pytorch 1.10](https://pytorch.org/)

     ```
     pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
     ```
  2. Install spconv

     ```
     pip install spconv-cu114
     ```
  3. Install pytorch scatter

     Go to [here](https://pytorch-geometric.com/whl/) and find the one match your pytorch version.

     e.g.  `torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl`, download it and install by

     ```
     pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
     ```
  4. Install other dependency

     ```
     pip install -r requirements.txt
     ```
- Setup, Install pointgroup_ops.

  ```
  cd lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
LESS
├── data
│   ├── scannet
│   │   ├── scans
│   │   ├── scans_test
```

Pre-process ScanNet data

```
cd data/scannet/
python batch_load_scannet_data.py
```

After running the script the scannet dataset structure should look like below.

```
LESS
├── data
│   ├── scannet
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── scannet_data

```

### ScanRefer dataset

Visit their [repository](https://github.com/daveredrum/ScanRefer?tab=readme-ov-file) and download the ScanRefer dataset.

Put the ScanRefer dataset as follow.

```
LESS
├── data
│   ├── scanrefer
│   │   ├── ScanRefer_filtered_train.json
│   │   ├── ScanRefer_filtered_train.txt
│   │   ├── ScanRefer_filtered_val.json
│   │   ├── ScanRefer_filtered_val.txt
│   │   ├── ScanRefer_filtered.json
```

## Training & Inference

We evaluate the method while training.

```
sh scripts/train.sh
```

## Citation

If you find this work useful in your research, please cite:

```
@article{liu2024less,
  title={LESS: Label-Efficient and Single-Stage Referring 3D Segmentation},
  author={Liu, Xuexun and Xu, Xiaoxu and Li, Jinlong and Zhang, Qiudan and Wang, Xu and Sebe, Nicu and Ma, Lin},
  journal={arXiv preprint arXiv:2410.13294},
  year={2024}
}
```

## Ancknowledgement

We Sincerely thanks for [Spformer](https://github.com/sunjiahao1999/SPFormer) and [Scanrefer](https://github.com/daveredrum/ScanRefer) repos. This repo is build upon them.
