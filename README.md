# FDSNet
Official Pytorch Code base for [Food Image Segmentation based on Deep and Shallow Dual-branch Network]
## Using the code:
The code is stable while using Python 3.8, CUDA >=11.3.0
## Datasets
FoodSeg103：[Link](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)

UECFoodPixComplete:[Link](https://mm.cs.uec.ac.jp/uecfoodpix/)

## Data Format
Make sure to put the files as the following structure:
```
├── FDSNet
│     ├── ...
│     ├── ...
├── dataset
│     ├── FoodSeg103/UECFoodPixComplete
│           ├── Images
│               ├── img_dir
│                   ├── allImage
│                       ├── 00000001.jpg
│                       ├── 00000002.jpg
│                       ├── 00000003.jpg
│                       ├── ...
│               ├── ann_dir
│                   ├── allImage
│                       ├── 00000001.png
│                       ├── 00000002.png
│                       ├── 00000003.png
│                       ├── ...
│           ├── ImageSets
│               ├── train.txt
│                   ├── 00000001
│                   ├── 00000002
│                   ├── 00000003
│                   ├── ...
│               ├── test.txt
│                   ├── 00000001
│                   ├── 00000002
│                   ├── 00000003
│                   ├── ...
