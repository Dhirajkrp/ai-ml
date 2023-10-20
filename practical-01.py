#  step 1: import the libraries
import pandas as pd
import numpy as np

# step 2: import the dataset
dataset= pd.read_csv('data.csv')
# print(dataset)

# step 3: extract the dependatant and independant variable
x = dataset.iloc[:, :-1].values

# print(x)

y = dataset.iloc[:, 3].values

# print(y)

# step 4: data pre-processing

#  returns the number of null values of every column.
# dataset.isnull().sum()


dataset[dataset['Age'].isnull()]
# replaces all the missing values with the constant values
# dataset.fillna(0)

print (dataset)

#  drop the rows having missing values
# dataset.dropna(how="any")

#  drop the rows which is completely blank
# dataset.dropna(how="all")

ffilldata = dataset.fillna(method="ffill")

print(ffilldata)
bfilldata = dataset.fillna(method="bfill")

print(bfilldata)

