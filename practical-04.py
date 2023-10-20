# step 1: import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step 2: import the dataset
dataset = pd.read_csv('energy.csv')

#print(dataset)
#print(dataset.shape)

# step 3: extract the IV and DV

x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 4].values
#print(x)
#print(y)


# step 4 : split the data into training and testing

from sklearn.model_selection import train_test_split

x_train ,x_test,y_train,y_test  = train_test_split(x,y , test_size=0.2 , random_state=0)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)

# print(x_test.shape)

# step 5 : implementing the multiple linear regression

from sklearn.linear_model import LinearRegression

obj = LinearRegression()
obj.fit(x_train , y_train)
y_pred =  obj.predict(x_test)


# print(y_pred)


# Step 6 : evaluating the performance of the model

from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

# mean absolute error
mae = mean_absolute_error(y_test , y_pred);
print("Mean Absolute Error:" , mae , "\n")

# mean squared error

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error is :" , mse , "\n")

# R squared

rsquare =  r2_score(y_test, y_pred)

print("R Square :" , rsquare , "\n")

# Step 7: (optional) visualizing the correlation between the independant variables
dataset.corr()

import seaborn as sns

sns.heatmap(dataset.corr() , annot =True)
