# Image Classifier Project README

This project involves building an image classifier using the CIFAR-10 dataset. The code is organized into different sections to perform various tasks, from data loading and exploration to building and training both an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for image classification.

## Table of Contents

1. [Dataset Loading and Exploration](#dataset-loading-and-exploration)
2. [Data Preprocessing](#data-preprocessing)
3. [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
4. [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
5. [Model Evaluation](#model-evaluation)
6. [Sample Image Predictions](#sample-image-predictions)

## 1. Dataset Loading and Exploration

The CIFAR-10 dataset is loaded using TensorFlow and split into training and testing sets. The shapes of the datasets and labels are checked to ensure proper loading.

## 2. Data Preprocessing

The pixel values of the images are normalized to a range of [0, 1]. Additionally, functions are defined to visualize sample images with their corresponding labels.

## 3. Artificial Neural Network (ANN)

An ANN with a simple architecture is constructed using Keras. The model is compiled with the Stochastic Gradient Descent (SGD) optimizer and trained on the normalized training data.

## 4. Convolutional Neural Network (CNN)

A CNN is built for image classification, consisting of convolutional and pooling layers. The model is compiled using the Adam optimizer and trained on the normalized training data.

## 5. Model Evaluation

The performance of both the ANN and CNN models is evaluated on the test set. Classification reports and confusion matrices are generated for model assessment.

## 6. Sample Image Predictions

The trained models are used to make predictions on sample images from the test set. Predicted classes are compared with true labels, and sample images are visualized with their predicted classes.

Feel free to explore the code sections and adapt them for your own image classification projects!

