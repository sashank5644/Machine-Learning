# Multi-Layer Perceptron

**Overview**

This project implements a simple feedforward neural network using PyTorch to apporximate the function Y = X^2. The model is designed to learn the mapping from input X to output Y through training with synthetic data.

**Project Structure** 

* **Model Definition**: A neural network is defined with two fully connected layers. The first layer maps the input to a hidden layer with 10 neurons, and the second layer maps it to the output.
* **Training Data Generation**: The training data consists of 1000 samples of X values randomly generated from [-1, 1]. The corresponding Y values are calculated as X^2.
* **Training Process**: The model is trained for 10 epochs, minimizing the mean squared error (MSE) between predicted and actual Y values.

**Dependencies**

torch

    pip install torch

**To Run Locally**

Clone Repository

    git clone https://github.com/yourusername/repository-name.git

Run mlp.py