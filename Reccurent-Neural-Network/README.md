# Recurrent Neural Network (RNN) for Sequence Prediction

**Overview**

This project implements a Recurrent Neural Network (RNN) using PyTorch to process sequential data. The model is designed to handle input sequences and predict outputs based on learned patterns from the data. This RNN architecture is particularly useful for tasks such as time series forecasting, natural language processing, and other applications involving sequential information.

**Project Structure**

* **Model Definition**: The RNN architecture is implemented through the RNN class, which includes essential components for creating a functional model. In the constructor (__init__), parameters such as input feature size, hidden layer size, number of layers, output classes, and sequence length are initialized. An RNN layer is created with the specified parameters, and a fully connected layer is set up to map the outputs from the RNN to the target classes. The forward method is responsible for initializing the hidden state, processing the input through the RNN, reshaping the output, and passing it through the fully connected layer to generate final predictions.

**Dependencies**

torch
torchvision
numpy

    pip install torch torchvision numpy pandas


**To Run Locally**

Clone Repository

    git clone https://github.com/yourusername/repository-name.git

Run rnn.py