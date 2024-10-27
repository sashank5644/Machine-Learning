# Image Classification

**Overview**

This project accurately classifies pet images into two categories (cats and dogs) through the training and use of a Convolution Neural Network (CNN). Utilizing PyTorch and torchvision libraries, the model processes images, extracts features, and predicts the class of each image based on its learned representations.

**Project Structure**

* Data Preparation: The dataset consists of images organized in folders (./petimages) for each class. The images are resized, normalized, and transformed into tensors for model input.
* Dataset Split: The dataset is divided into training and testing sets, with 80% of the data used for training and 20% for evaluation.
* Model Architecture: The CNN model comprises several convolutional layers followed by fully connected layers to classify the images. The architecture includes:
    * Two convolutional blocks for feature extraction.
    * A classifier that maps the extracted features to the output classes.

**Dependencies**

PyTorch
torchvision
scikit-learn

    pip install torch torchvision scikit-learn


**To Run Locally (GPU recommended)**

Clone Repository

    git clone https://github.com/yourusername/repository-name.git

Navigate to "Image Classification" project folder

    cd "Image Classification"

Run cnn.py

