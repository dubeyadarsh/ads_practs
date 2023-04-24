import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import pearsonr
titanic_df = pd.read_csv('titanic.csv')

titanic_df.columns = ['PassengerID','Survival','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']


print('mean=',titanic_df['Age'].mean())
print('median=',titanic_df['Age'].median())
print('mode=',titanic_df['Age'].mode())
print("Variance=", np.var(titanic_df['Age']))
print("Mean=", np.mean(titanic_df['Age']))
print("Min=", np.min(titanic_df['Age']))
print("Max=", np.max(titanic_df['Age']))
print("Range=", np.max(titanic_df['Age']) - np.min(titanic_df['Age']))
print('Standard Deviation=',titanic_df['Age'].std())
titanic_df['Age'].describe()

plt.figure(figsize=(12,6))
sns.distplot(titanic_df['Fare'], fit=norm, color="r")

print("Skew of data: %f " % titanic_df['Fare'].skew())
print("Kurtosis of data: %f " % kurtosis(titanic_df['Fare'],fisher=False))


list1 = titanic_df['Fare']
list2 = titanic_df['Pclass']
corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)
