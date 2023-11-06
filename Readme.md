# Monocular Odometry using Optical flow and Deep Neural Networks

This project focuses on training and testing end to end Monocular visual odometry models using deep learning techniques. Visual odometry is a key component in robotics and autonomous systems, enabling the estimation of a camera or robot's motion and position based on visual input.

## Overview

- The code uses deep convolutional neural networks (CNNs) and GPUs for optical flow estimation.
- The code leverages [PTLFLow]() library to utilize the flow models 
- Several models, such as `Pcnn`, `Pcnn1`, and `Pcnn2`, are available for testing and training.
- Training data is typically sourced from datasets like KITTI odometry and MPI-Sintel.
- The code includes both training and testing modes to help develop and evaluate visual odometry models.

## Requirements

- Python 3.x
- PyTorch
- PIL (Python Imaging Library)
- tqdm
- Matplotlib
- NumPy
- PTLflow library

## Getting Started

1. Clone the repository.

2. Ensure you have the required libraries installed.

   ```bash
   pip install -r requirements.txt

Download KITTI data[here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Download the KITTI images and only  the right camera color images (image_02 folder) are required
the downloaded images will be placed at dataset/sequences/images_2/00/, dataset/images_2/01, ...
the images offered by KITTI is already rectified
Download the ground truth pose from KITTI Visual Odometry
you need to enter your email to request the pose data here
and place the ground truth pose at dataset/poses/

Downloand ptlflow from[here](https://github.com/hmorimitsu/ptlflow)

and export the path so that the flow models cna be accessed directly

    ```bash
    export PYTHON_PACKAGE_PATH="/path/to/ptlflow/ptlflow"


The models can be run directly on Kitti dataset using optical flow models as ptllflow models  

    ```bash
    python your_script.py --model Pcnn --mode train --datapath /path/to/your/dataset --bsize 8 --lr 0.0001 --train_iter 200 --seq 00 --checkpoint_path /path/to/checkpoints --save_res /path/to/results

    python your_script.py --model Pcnn --mode test --datapath /path/to/your/dataset --seq 00 --checkpoint_load /path/to/checkpoint/model.pth --save_res /path/to/results


# BibTex

@misc{morimitsu2021ptlflow,
  author = {Henrique Morimitsu},
  title = {PyTorch Lightning Optical Flow},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hmorimitsu/ptlflow}}
}