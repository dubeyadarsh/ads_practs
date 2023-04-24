import pandas as pd
import numpy as np 

train_df =  pd.read_csv("titanic_train.csv")
test_df= pd.read_csv("titanic_test.csv")

train_df.columns
# Top 5 rows in your dataset 
train_df.head(10)

#Bottom 5 rows in your dataset
train_df.tail()

# Describe  function used to find the total count, mean, standard deviation, minimum value, maximum value,
#25,50,75 percentage of the dataset value
train_df.describe()

#how many nulls
print(train_df.isnull().sum())

# To remove the null values we can use fillna method 
# For example:

train_df.Cabin = train_df.Cabin.fillna("unknown")
print(train_df.isnull().sum())
# It is one of the method of data cleaning!!!!!!!!

#how does the data looks like
print(train_df.shape)# total rows X column  count 
print("\n")
print(train_df.dtypes)# Each column data type

# info gives count and datatype of it
train_df.info()
print('_'*40)
test_df.info()