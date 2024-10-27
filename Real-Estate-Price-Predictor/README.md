# Real Estate Price Prediction with Linear Regression

**Overview**

This project implements a linear regression model to predict real estate prices based on specific features such as house age, distance to the nearest MRT station, and the number of convenience stores. Using the "realestate.csv" dataset, the goal is to determine how these factors influence house prices per unit area.

**Project Structure**

* **Data Preparation**: The project loads the dataset from a CSV file and inspects its structure to understand the column names and data types. Then a check for missing values is performed to identify any data that may require imputation or removal. Subsequently, columns are renamed for easier access, improving clarity for later analysis.

* **Feature Selection**: In this phase, the target variable is identified as the price per unit area, labeled as Price_Area. The features selected for prediction include house age (H_Age), the distance to the nearest MRT station (Distance), and the number of convenience stores (Conv_stores).

* **Model Training**: The dataset is split into training and testing sets using three different random states (0, 50, and 101). A linear regression model is then trained on the training set, allowing it to learn the relationships between the selected features and the target variable without overfitting the model to the training data. 

* **Model Evaluation**: After training, the model's performance is assessed using the testing set. Evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are calculated to determine the model's accuracy and effectiveness in predicting real estate prices.

**Dependencies**
pandas
scikit-learn
NumPy

    pip install pandas scikit-learn numpy

**To Run Locally**

Clone Repository

    git clone https://github.com/yourusername/repository-name.git

Navigate to "Real-Estate-Price-Predictor" project folder

    cd "Real-Estate-Price-Predictor"

Run linearRegression.py
