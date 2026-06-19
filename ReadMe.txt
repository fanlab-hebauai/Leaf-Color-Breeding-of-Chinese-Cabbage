# UAV-Based Single-Plant Segmentation of Chinese Cabbage

This repository provides the source code used for UAV-based single-plant segmentation of Chinese cabbage under field conditions. The workflow was developed to support high-throughput leaf color phenotyping by extracting individual Chinese cabbage plants from UAV multispectral and RGB orthomosaic images.

The pipeline includes image cropping, dataset preparation, data augmentation, UNet-based semantic segmentation, model evaluation, and single-plant mask generation.

## Overview

UAV multispectral images were acquired using a DJI Phantom 4 Multispectral platform. Orthomosaic RGB and multispectral images were generated after image stitching and radiometric calibration. Individual Chinese cabbage plants were manually annotated as regions of interest and then used to train a semantic segmentation model.

The trained model was used to generate binary masks for individual plants, which were further used for single-plant feature extraction and leaf color phenotyping.

## Workflow

The main workflow includes:

1. UAV image acquisition and orthomosaic generation
2. Image cropping into fixed-size patches
3. Manual annotation and label preparation
4. Dataset splitting into training, validation, and test sets
5. Data augmentation
6. UNet-based semantic segmentation model training
7. Model evaluation using pixel-level metrics
8. Binary mask prediction and single-plant extraction

## Repository Structure
├── Model/                  # UNet and segmentation model scripts
├── augment.py              # Data augmentation
├── caijian.py              # Image cropping
├── compare.py              # Comparison between models or prediction results
├── dataProcess.py          # Data preprocessing
├── hebing.py               # Image or result merging
├── huafen.py               # Dataset splitting
├── huafen shujuji.py       # Dataset partitioning script
├── jiaobing1.py            # Image processing utility
├── jiaobingbi.py           # Image processing utility
├── pinghualoss.py          # Loss function-related script
├── seg_metrics.py          # Segmentation evaluation metrics
├── seg_unet.py             # UNet segmentation model
├── test.py                 # Model testing or inference
├── tichuzhaopian.py        # Single-plant image extraction
├── unet1.py                # UNet model variant
└── zhengkuai.py            # Image tiling/block processing

## Requirements
Python 3.6.13
TensorFlow 2.1.5
NumPy
OpenCV
GDAL
scikit-learn
Pillow
matplotlib
## BrDeepGS for Leaf Color Genomic Prediction

In addition to UAV-based single-plant segmentation, this repository also provides the source code for BrDeepGS, a deep learning model used for genomic prediction of leaf color in Chinese cabbage.

`BrDeepGS.py` was used for model training, cross-validation, independent test set prediction, and performance evaluation. The model uses SNP genotype data as input and leaf color phenotype values as output.

The main dependencies for BrDeepGS include:
Python 3.10.18
pandas
NumPy
scikit-learn
SciPy
PyTorch 2.9.1

## Data
All data required for running the scripts in this repository are provided in the `Test Data` folder.