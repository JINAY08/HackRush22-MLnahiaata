# Code written by: Team ML nahi aata [HackRush22] - Kush Patel, Haikoo Khandor, Jinay Dagli
# Importing necessary libraries and commands
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
# Importing the given test Dataset 
data = pd.read_csv('../input/competition-dataset/test.csv') 
print(data.head())

# Importing the given training Dataset
data_file_path = "../input/competition-dataset/train.csv"
home_data = pd.read_csv(data_file_path)
y = home_data.Expected
print(home_data.describe)
# Creating X (data)
features = ['Id']
X = home_data[features]
print()
print(X.head())

print()
print(home_data.columns)

#Spliting the dataset into the training and test Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')

# Specifying the number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 35)]
# Specifying the number of features to consider at every split
max_features = ['auto','sqrt']
# Specifying the maximum number of levels in the tree
max_depth = [2,4]
# Specifying the minimum number of samples required to split a node
min_samples_split = [2,5]
# Specifying the minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
# The method of selecting samples for training each tree
bootstrap = [True,False]
# Creating the parameter grid (param grid)
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)
rf_Model = RandomForestClassifier()

rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 10, verbose=2, n_jobs = 4)
rf_Grid.fit(X_train, y_train)
print(rf_Grid.best_params_)
y_predict = rf_Grid.predict(X_test)
print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
rf_val_mse = mean_squared_error(y_predict, y_test)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mse))


  
# Commented code written earlier (not efficient)

# #Create a Random Forest regressor object from Random Forest Regressor class
# RFReg = RandomForestRegressor(n_estimators = 1000, random_state = 1, min_samples_leaf=3, min_samples_split = 2, max_features=1)

  
# #Fit the random forest regressor with training data represented by X_train and y_train
# RFReg.fit(X_train, y_train)
# #Predicted Height from test dataset w.r.t Random Forest Regression
# y_predict_rfr = RFReg.predict(X_test)
# rf_val_mse = mean_squared_error(y_predict_rfr, y_test)
# ''' Visualise the Random Forest Regression by creating range of values from min value of X_train to max value of X_train  
# having a difference of 0.01 between two consecutive values'''
# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mse))

# #Model Evaluation using R-Square for Random Forest Regression
# # Define a random forest model
# rf_model = RandomForestRegressor(random_state=1)
# rf_model.fit(X_train, y_train)
# rf_val_predictions = rf_model.predict(X_test)
# rf_val_mse = mean_squared_error(rf_val_predictions, y_test)
# ''' Visualise the Random Forest Regression by creating range of values from min value of X_train to max value of X_train  
# having a difference of 0.01 between two consecutive values'''
# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mse))


height_pred = rf_Grid.predict(X_test)

#Creating the csv file.
pred = pd.DataFrame(height_pred)
sub_home_data = pd.read_csv("../input/competition-dataset/sample_submission.csv")
datasets=pd.concat([sub_home_data['Id'],pred],axis=1)
datasets.columns=['Id','Predicted']
datasets.to_csv('sample.csv',index=False)
