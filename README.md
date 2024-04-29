# Autism Image Classification.
## Overview
This project aims to develop a deep learning model using Convolutional Neural Networks (CNNs) and try different models to classify Autism images. The dataset used for this task contains facial images of individuals with and without Autism.
## Dataset
The dataset consists of images collected from [Kaggle](https://www.kaggle.com/datasets/cihan063/autism-image-data/data). It contains a total of **2540 images**, with **1270 images** belonging to each class (Autism and Non-Autism) in the **training set**. Additionally, there are **150 images** per class in the **test set**.
## Model Architecture CNN
The CNN architecture used for this classification task is as follows.
* **Three convolutional layers** are added: The first layer has **32 filters of size 3x3**, using **ReLU activation** function followed by **64 and 128** filters.
* Each convolutional layer is followed by a max-pooling layer **(MaxPooling2D)** to downsample the feature maps.Max pooling **reduces the spatial dimensions** of the feature maps by taking the maximum value in each 2x2 window.
* **Flatten Layer**: Flattens the output of the convolutional layers into a **1D array** to prepare for the fully connected layers.
* **Fully Connected Layers**:
  
   - **Dense (512 units, ReLU activation)**:
     - A fully connected layer with 512 neurons, which serves as a hidden layer for learning high-level representations.
   - **Dropout (rate=0.5)**:
     - **Dropout regularization** with a rate of 0.5 is applied to **prevent overfitting** by randomly dropping 50% of the neurons during training.
* **Output Layer**:
   - Dense (1 unit, **Sigmoid activation**):
     - The output layer consists of a **single neuron** with a sigmoid activation function, which outputs the probability of the input image belonging to the positive class (e.g., Autism).
     - For binary classification tasks, sigmoid activation is commonly used to produce a probability value between 0 and 1.
* **Model Compilation**:
  - **Optimizer: Adam**
     - **Adam** optimization algorithm minimizes the binary cross entropy loss function during training.
## Model Architecture with Transfer Learning (VGG16)
This model architecture utilizes **transfer learning** with the **VGG16** convolutional neural network (CNN) model pre-trained on the **ImageNet dataset**. The VGG16 model is loaded without its top fully connected layers, and custom fully connected layers are added on top for fine-tuning the specific classification task.
* **Model Compilation**:
   - **Optimizer**: RMSprop
     - **RMSprop** optimization algorithm is used to minimize the binary cross-entropy loss function during training.
    
## Augmentation techniques applied with Keras ImageDataGenerator class
1. **Rotation**:
   - Randomly rotates images through any degree between **0 and 360 degrees**.
   - During rotation, some pixels may move outside the image, leaving empty areas that need to be filled in. The default **fill mode is "nearest"**, which fills empty areas with the nearest pixel value.

2. **Shift Range**:
   - **`height_shift_range` and `width_shift_range`** parameters indicate the percentage of the image's width or height to shift. These are float values that define the range of random horizontal or vertical shifts applied to the images.

3. **Flipping**:
   - **`horizontal_flip` and `vertical_flip`** parameters enable **flipping images along the horizontal or vertical axis**, respectively. This introduces variations in the orientation of images and helps prevent overfitting.

4. **Zoom Range**:
   - **`zoom_range`** parameter **randomly zooms the images**. It allows for random zooming in or out of the images, providing additional variations in the dataset.
## Training Procedure
The dataset is split into **training, validation, and test sets**. The model is trained for [20] epochs with a batch size of [16].

## Model Performace
In the evaluation of our models, we observed the following results:
1. **CNN Model**:
 - Test Accuracy: 81.25%
 - Test Loss: 0.447287
2. **Transfer Learning (VGG16) Model**:
- Test Accuracy: 80.00%
- Test Loss: 0.427432

## Code Availability :
* The codes for CNN are available in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ChEVSC6OqhFonUGS3tPYnd-9iDQrbMF#scrollTo=Wlz3OUJXO3AO) .
* For Transfer learning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BmeEionBHxqzaPk9MUEMq2yTOrbdLuLO#scrollTo=45jl5mf0j046).

 
