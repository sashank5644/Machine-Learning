# These are the packages that we are importing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Read the dataset and looking at the head to gauge the column names and whether column names actually exists

lin_reg_df = pd.read_csv('realestate.csv')

lin_reg_df.head()

# Perform the basic Null Check to decide whether imputation or drop is required

lin_reg_df.isnull().sum()

# Rename the column names to make it easy for usage through the rename() function
# Set new column names as: 'SL No', 'Txn_Dt', 'H_Age', 'Distance', 'Conv_stores', 'Lat', 'Long', 'Price_Area'

lin_reg_df.rename(columns={
    'No': 'SL No',
    'X1 transaction date': 'Txn_Dt',
    'X2 house age': 'H_Age',
    'X3 distance to the nearest MRT station': 'Distance',
    'X4 number of convenience stores': 'Conv_stores',
    'X5 latitude': 'Lat',
    'X6 longitude': 'Long',
    'Y house price of unit area': 'Price_Area'
}, inplace=True)

# Split the dataset into target and feature values such that you consider only the following features: House Age, Distance to MRT station and Number of Convenience stores
# While we consider Price per Unit Area as the Target variable
# You can consider other features as well, just make sure to change the X (feature) variable

y = lin_reg_df['Price_Area']

X = lin_reg_df[['H_Age', 'Distance', 'Conv_stores']]

# After that test the model with random_state - 0, 50 and 101 and report the one that gave the best performance based on MSE, MAE and RMSE

random_state_list = [0, 50, 101]

min_MAE, min_MSE, min_RMSE, best_rdm_st = float('inf'), float('inf'), float('inf'), 0

for rdm_st in random_state_list:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rdm_st)# train-test split: 75 - 25

    model_LR = LinearRegression()# init the Linear Regression model

    model_LR.fit(X_train, y_train)# fit the data into the model

    y_pred = model_LR.predict(X_test)# Predict using this model

    # Use sklearn.metrics to get the values of MAE and MSE

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)# RMSE is square root of MSE
    
    if MAE < min_MAE:
        
        min_MAE = MAE
        min_MSE = MSE
        min_RMSE = RMSE
        best_rdm_st = rdm_st
        

    print("For random state = {}, the values are: ".format(rdm_st))
    print("Mean Absolute Error: ", MAE)
    print("Mean Squared Error: ", MSE)
    print("Root Mean Squared Error: ", RMSE)
    print("========================================================")
    print("\n")

# Report the random state that gave the best result and the respective values of MAE, MSE and RMSE

best_st = best_rdm_st
print(best_st) 

best_MAE = min_MAE 
print(best_MAE) 

best_MSE = min_MSE
print(best_MSE) 

best_RMSE = min_RMSE
print(best_RMSE) 


most_sig_wt, idx = 0, 0

for index, wt in enumerate(model_LR.coef_):
    
    if most_sig_wt < abs(wt):
        
        most_sig_wt = wt
        idx = index

most_sig_col = X.columns[idx]

print(most_sig_col) 

intercept_val = round(model_LR.intercept_, 2)# Put the value here 

print(intercept_val)
