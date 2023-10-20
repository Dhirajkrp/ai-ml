# step 1: import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# step 2: import the dataset
dataset = pd.read_csv('Salary_Data.csv')

print("Dataset: \n")
print(dataset)

# step3: extract the independant and dependant variables
x = dataset.iloc[: , :-1].values

y= dataset.iloc[:, 1].values

# print("Independant variable")
print(x)
# print("dependant variable")
print(y)

# # step 4:
x_train ,x_test ,y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=0)

print ("\nX train:\n")
print (x_train)
print ("\nY train:\n")
print (y_train)
print ("\nX test:\n")
print (x_test)
print ("\nY test:\n")
print (y_test)
#######################################################################################

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)


#########################################################################################
# step 5: apply liner regression
y_predict = reg.predict(x_test)

y_predict


#########################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


mean_absolute_error(y_test , y_predict)

mean_squared_error(y_test , y_predict)

r2_score(y_test , y_predict)

#########################################################################################

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_predict,color='blue')
plt.title('YOE vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
