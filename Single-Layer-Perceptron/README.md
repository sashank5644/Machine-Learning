# Single Layer Perceptron for linear regression with PyTorch

**Overview**

This project implements a simple linear regression model using PyTorch to approximate the linear function Y = 3X + 1. The model is designed to learn the relationship between input X and output Y through training on synthetic data.

**Project Structure**

* **Model Definition**: A neural network is defined with a single fully connected layer that maps the input to the output.

* **Training Data Generation**: The training data consists of 100 samples of X values randomly generated from the range [-1, 1]. The corresponding Y values are calculated using the equation Y = 3X + 1.

* **Training Process**: The model is trained for 10 epochs, minimizing the mean squared error (MSE) between predicted and actual Y values.

**Dependencies** 

torch

    pip install torch

**To Run Locally**

Clone Repository

    git clone https://github.com/yourusername/repository-name.git

Run slp.py