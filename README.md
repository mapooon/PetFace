# PetFace (ECCV2024 Oral)
<a href='https://arxiv.org/abs/2407.13555'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://dahlian00.github.io/PetFacePage/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 

The official PyTorch implementation for the following paper:
> [**PetFace: A Large-Scale Dataset and Benchmark for Animal Identification**](https://arxiv.org/abs/2407.13555),  
> [Risa Shionoda](https://sites.google.com/view/risashinoda/home)* and [Kaede Shiohara](https://mapooon.github.io/)* (*Equal contribution),   
> *ECCV 2024 Oral*

## TL;DR: We established a large-scale animal identification dataset with more than 250k IDs across 13 families




![Overview](fig/teaser.png)

# !!!Attention!!!
### Our PetFace dataset, code in this repository, and pretrained models are  **for non-commercial research purpose only**.

# Changelog
[2024/10/07] The code for face alignment on your own images was released.  
[2024/09/02] Installation instruction was updated.  
[2024/08/14] PetFace was selected as an ORAL presentation at ECCV2024 (with 3x Strong Accept!!!)🎉  
[2024/07/27] Pretrained models, training code, and evaluation were released. Also, "split" folder is updated.  
[2024/07/19] This repository was released.  

# Dataset
Fill in a [google form](https://docs.google.com/forms/d/e/1FAIpQLSfRPJaCmU6oQ4X_uB6H-EM5MSeczKczZxbQ5H9FMRS4KNY59w/viewform) for access to the dataset.  
- We manually confirm the submitted forms and therefore it may take some time. Please be patient while the dataset link arrives. 
- If you do not receive the link after 3 days, we may have missed your form or the form was not filled in correctly. We are sorry, but please submit the proper form again. (Please do not start a thread on Issue)
- We will not reply if you enter an e-mail address that is not affiliated with your organization (such as Gmail).
- We will not reply if the form is written in a language other than English.

## Dataset directory
Place the dataset as follows:
```
. (Root of this repository)
└── data
    └── PetFace
        ├── images
        │   └── cat
        │       └── 000000
        │           └── 00.png
        ├── split
        │   └── cat
        │       ├── train.csv
        │       ├── val.txt
        │       ├── test.txt
        │       ├── reidentification.csv 
        │       └── verification.csv
        └── annotations
            └── cat.csv
         
```
train.csv: file names and id labels for training  
val.txt: file names for validation (not used in this codebase)  
test.txt: file names for verification (not used in this codebase)  
verification.csv: pairs of file names to verify and labels indicating whether the pairs have the same ID  
reidentification.csv: file names and id labels for re-identification  

# Setup
## Docker (Recommended)
1) Pull a docker image:
```
docker pull pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
```
2) Replace the path in `exec.sh`.  
3) Execute the dokcer image:
```
bash exec.sh
```
4) Install packages:
```
bash install.sh
```

## pip (Unrecommended)
Install packages:
```
pip install -r requirements.txt
```

# Testing
Pretrained weights are provided on [google drive](https://drive.google.com/drive/folders/1XZHxlvRUZSQeFrztz0GaKgyVUSCh1lT6?usp=sharing). 

## Re-identification
For example, you can run the evaluation of re-identification for cat as follows:
```
CUDA_VISIBLE_DEVICES=0 python3 src/reidentification.py -m arcface -w pretrained/arcface/cat.pt -i data/PetFace/split/cat/reidentification.csv -o results/reidentification/arcface/cat.csv
```
Then, you can compute the top-k (k={1,2,3,4,5}) accuracy:
```
python3 src/compute_topk_acc.py --topk 5 -i results/reidentification/arcface/cat.csv
```

## Verification
For example, you can run the evaluation of re-identification for cat as follows:
```
CUDA_VISIBLE_DEVICES=0 python3 src/verification.py -w pretrained/arcface/cat.pt -i data/PetFace/split/cat/verification.csv -o results/verification/arcface/cat.csv
```
Then, you can compute AUC:
```
python3 src/compute_auc.py -i results/verification/arcface/cat.csv
```

## Face Alignment on Your Own Images
We provide the source key points in `keypoints` folder to align images.  
First, you need to detect 5 keypoints of your own image by [AnyFace](https://github.com/IS2AI/AnyFace) and save them as a `.npy` file.  
Then, you can align the images by:
```sh
python3 src/face_align.py --tgt /path/to/your/keypoints.npy --img /path/to/your/image.jpeg --src /path/to/src/keypoints.npy --out /path/to/output/image.jpg
```


# Training
For example, you can run the training for cat as follows:
```
CUDA_VISIBLE_DEVICES=0 python3 src/train_arcface.py src/configs/cat.py  --output outputs/cat/arcface
```

Also, you can train an arcface model on all the species (families):
```
CUDA_VISIBLE_DEVICES=0 python3 src/train_unified.py src/configs/unified.py  --output outputs/unified
```

# Acknowledgement
We borrow some code from [insightface](https://github.com/deepinsight/insightface), [pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss), and [triplet-loss-with-pytorch](https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch).

# Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{shinoda2025petface,
  title={PetFace: A large-scale dataset and benchmark for animal identification},
  author={Shinoda, Risa and Shiohara, Kaede},
  booktitle={European Conference on Computer Vision},
  pages={19--36},
  year={2025},
  organization={Springer}
}
```

# Our Related Work
### [[GitHub]](https://github.com/dahlian00/OpenAnimalTracks) OpenAnimalTracks: A Dataset for Animal Track Recognition (ICIP2024)