# [FG 2024] Benchmarking Skeleton-based Motion Encoder Models for Clinical Applications


![License](https://img.shields.io/badge/license-MIT-blue) [![arXiv](https://img.shields.io/badge/arXiv-2405.17817-b31b1b.svg)](https://arxiv.org/abs/2405.17817)


## Introduction
This project is created as part of the research for the paper titled "[Benchmarking Skeleton-based Motion Encoder Models for Clinical Applications: Estimating Parkinson’s Disease Severity in Walking Sequences](https://arxiv.org/abs/2405.17817)" accepted at IEEE  international conference on automatic face \& gesture recognition (FG 2024). 

## Installation
```bash
git clone https://github.com/TaatiTeam/MotionEncoders_parkinsonism_benchmark.git
cd MotionEncoders_parkinsonism_benchmark
pip install -r requirements.txt
```

## Data
Dataloaders will be added soon.

## Demo
Demo will be added soon.

## Leaderboard

| Model          | F1 Score | Paper/Source |
| ---------------|----------|--------------|
| MixSTE   | 0.41    | [Link](https://paperswithcode.com/paper/mixste-seq2seq-mixed-spatio-temporal-encoder) |
| MotionAGFormer  | 0.42    | [Link](https://paperswithcode.com/paper/motionagformer-enhancing-3d-human-pose) |
| MotionBERT-LITE    | 0.43    | [Link](https://paperswithcode.com/paper/motionbert-unified-pretraining-for-human) |
| POTR   | 0.46    | [Link](https://paperswithcode.com/paper/pose-transformers-potr-human-motion) |
| MotionBERT    | 0.47    | [Link](https://paperswithcode.com/paper/motionbert-unified-pretraining-for-human) |
| PD STGCN  | 0.48    | [Link](https://paperswithcode.com/paper/abc) |
| PoseFormerV2    | 0.59    | [Link](https://paperswithcode.com/paper/poseformerv2-exploring-frequency-domain-for) |
| PoseFormerV2-Finetuned  | 0.62    | [Link](https://paperswithcode.com/paper/abc) |


For detailed rankings, visit the [Paperswithcode Leaderboard](https://paperswithcode.com/sota/classification-on-full-body-parkinsons).


## Acknowledgement
Special thanks to the creators of the dataset for making their clinical data publicly available:
- [A public data set of walking full-body](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.992585/full)

Our code also refers to the following repositories. We thank the authors for releasing their codes.

- [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [POTR](https://github.com/idiap/potr)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer/tree/master)
- [stgcn_parkinsonism_prediction](https://github.com/TaatiTeam/stgcn_parkinsonism_prediction)




## Citation
Please cite our paper if this library helps your research:
```
@inproceedings{PDmotionBenchmark2024,
  title     =   {Benchmarking Skeleton-based Motion Encoder Models for Clinical Applications: Estimating Parkinson’s Disease Severity in Walking Sequences}, 
  author    =   {Vida Adeli, Soroush Mehraban, Yasamin Zarghami, Irene Ballester, Andrea Sabo, Andrea Iaboni, Babak Taati},
  booktitle =   {2024 18th IEEE international conference on automatic face & gesture recognition (FG 2024)},
  year      =   {2024}
}
```
