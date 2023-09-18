# Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 1. Project Summary

In this repo, we implement U-Net Semantic Segmentation from scratch for Carvana Image Masking Challenge. The main difference between this architecture and the original paper is that we use padded convolutions instead of valid (unpadded) convolutions.

### 1.1. Architecture

The U-Net architecture is a popular CNN model for image segmentation. It was first introduced in 2015, and has since been widely adopted in various field. The U-Net consists of an encoder path to capture features, a decoder path for generating a segmentation map, and skip connections connect the encoder and decoder paths, enabling the model to combine low-level and high-level features. The U-Net effectively captures details and context, making it ideal for segmentation tasks. The U-Net architecture is as follow.

<p align="center">
  <img src="media/u-net-architecture.png" width="350" title="hover text">
</p>

### 1.2. Dataset
The dataset is obtained from Kaggle's Carvana Image Masking Challenge in 2017. The goal is to develop an algorithm that automatically removes the photo studio background from the car.

In this repository, we only use the training set, and split it into train/val/test set. We don't download the test set from Kaggle because it has a very huge size. The data consists of car images and their corresponding masks as  shown below.

<p align="center">
  <img src="media/0cdf5b5d0ce1_04.jpg" width="350" title="hover text">
  <img src="media/0cdf5b5d0ce1_04_mask.gif" width="350" title="hover text">
</p>

### 1.3. Results

After training the data for 10 epoch, we achieved an accuracy of 99.2% on the test set. We consider this result as satisfactory enough considering the short time of development. 

We also convert the model into Onnx runtime to speed up the inference time by 2x. The media below compares the FPS performance of the original torch model and onnx model.

## 2. Usage

### 2.1. Dependencies

The dependencies of this project includes:

- numpy
- torch
- torchvision
- Pillow
- opencv
- albumentations

### 2.2. Training

We use automatic mixed precision to speed up the training process. The model is trained for 10 epochs. To perform training, you could run this command on terminal:
 ```
python train.py
```

### 2.3. Evaluate

To evaluate the data on the test set, run this command
 ```
python test.py
```

### 2.4. Inference an Image

To perform an inference on a single image, run this command:

```
python inference.py 
```

Make sure to change the image_path and output_path in the inference.py script. Figure 2 is the example result of the inference.

<p align="center">
  <img src="media/0ee135a3cccc_04.jpg" width="350" title="hover text">
  <img src="media/masked_car_w.jpg" width="350" title="hover text">
</p>


<!-- ### 2.5. Speed Up Inference 

We also convert the model to Onnx runtime to speed up the inference time. The onnx model is available in this path. To perform inference on onnx runtime run this command

```
python inference_onnx.py 
```

Make sure to change the image_path and output_path in the inference_onnx.py script. -->

## Credit 

This project is mainly for learning purposes and is heavily based on the tutorial from [Aladdin Person's Youtube channel](https://www.youtube.com/watch?v=IHq1t7NxS8k). The original code is available [here](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet).
