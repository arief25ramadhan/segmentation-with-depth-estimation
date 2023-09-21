# Multi Task Learning: Semantic Segmentation and Depth Estimation Using a Single Network

<img src="media/architecture.png" alt="drawing" width="110"/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This project aims to create a joint semantic segmentation and deepth estimation model from scratch. This algorithm can perform segmentation and depth estimation simultaneously within a single network. We willl provide the step-by-step instructions, code, and resources to guide you through the process of implementing this multi task model from the ground up.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training](#training)
   - [Data Preparation](#data-preparation)
   - [Model Configuration](#model-configuration)
   - [Training Process](#training-process)
5. [Inference](#inference)
6. [Performance Tuning](#performance-tuning)

## 1. Introduction

Object detection is a fundamental task in computer vision, and YOLO is one of the most popular and effective approaches. This project aims to build the YOLOv3 from the ground up. By following the steps outlined here, we hope you will gain a deep understanding of the architecture, training process, and inference pipeline of YOLOv3.

### 1.1. A brief explanation of the Model

Yolo v3 uses a single neural network to look at the full image. It divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. YOLOv3 uses a few tricks to improve training and increase performance from its previous iteration, including: multi-scale predictions, and better backbone classifier. The figure below shows the architecture of YOLOv3.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*d4Eg17IVJ0L41e7CTWLLSg.png" alt="drawing" width="800"/>

Visit this [web page](https://pjreddie.com/darknet/yolo/) or [paper](https://arxiv.org/abs/1804.02767) for a more detailed technical explanation of YOLOv3 model.

## 2. Getting Started

### 2.1. Prerequisites
Before you begin, make sure you have the following prerequisites installed:

- Python 3
- Pytorch
- NumPy
- OpenCV
- PIL

### 2.2. Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/arief25ramadhan/segmentation-with-depth-estimation.git
   ```

2. Navigate to the project directory:

   ```bash
   cd segmentation-with-depth-estimations
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### 2.3. Data Preparation

1. Download the PASCAL VOC dataset and annotations from [here](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video). The PASCAL VOC is a dataset of 20 classes of person, vehicles, animals, and indoor objects. The detailed list can be seen in the config.py script.

2. Organize your dataset in the following directory structure:

   ```
   dataset/
   |-- PASCAL_VOC/
       |-- images/          # Contain images
       |-- labels/          # Contain labels
       |-- train.csv        # Define image belongs to train set
       |-- test.csv         # Define image belongs to test set
   ```


## 3. Project Structure

```
|-- dataset/                 # Folder containing data files
|-- model/                   # Folder containing model files
|-- train.py                 # Training script
|-- dataset.py               # Script to load dataset
|-- loss.py                  # Script to load loss function
|-- utils.py                 # Script containing helper functions
|-- config.py                # Script containing hyperparameters
|-- model.py                 # Script to load model
|-- inference.py             # Script to perform inference
```


## 4. Training

Training YOLOv3 from scratch requires model configuration, and the training process itself.

### 4.1. Model Configuration

Configure the YOLOv3 model architecture in the `model.py` file. You can download the pretrained model from [here](https://www.kaggle.com/datasets/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694).

### 4.2. Training Process

Run the training script to start training:

```bash
python train.py
```

## 5. Inference

After training your YOLOv3 model, you can perform inference on images.

```bash
python inference.py --image path/to/your/image.jpg --model path/to/your/model_weights.h5
```

## 6. Performance Results

After training the model for 10 epochs, we look at the model's performance qualiatively and quantitatively. Figure below shows some of the inference results of our Yolo V3 model. We can see that the model does mispredicted. This project is only for learning. So, creating the most accurate model, which requires a lot of tuning and training, is not our priority.

<img src="media/000015_predicted.jpg" alt="drawing" width="400"/>
<img src="media/000004_predicted.jpg" alt="drawing" width="400"/>

We train the Yolo V3 from the checkpoint created by Aladdin Persson. The performance of the model is shown by the table below.

| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 77.64             |
| YOLOv3 Further Trained  | 78.2              |


## 7. Acknowledgement

### 7.1. The Original Paper
The implementation is based on the [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) by Joseph Redmon and Ali Farhadi.

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

### 7.2. The Code 

This project is for learning purposes and made by following the tutorial by Aladdin Persson in his [Youtube channel](https://www.youtube.com/watch?v=Grir6TZbc1M). The original code is also available in [his repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3).
