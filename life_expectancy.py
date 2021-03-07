import pandas as pd
import numpy as np
pd.set_option('display.max_rows',800)
pd.set_option('display.max_columns',500)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#machinlearing libaries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import random

df = pd.read_csv("Life Expectancy Data.csv")

df.head()
df.info()
df.describe()

#printing catgorical and numerical coloumns
num_col=df.select_dtypes(include=np.number).columns
print('Numerical columns: \n',num_col)

cat_col=df.select_dtypes(exclude=np.number).columns
print('Catgorical columns: \n',cat_col)

#chanage catgrorical to numerical 
from sklearn import preprocessing

#label econder object knows how to understand word lebels.
labelencoder=preprocessing.LabelEncoder()

# detecting and treating missing values
print(df.isna().sum())

#numer of rows x column
print(df.shape)

# will replace the null value by mean by ignoring column country since it is in catgorical
for i in df.columns.drop('Country'):
    df[i].fillna(df[i].mean(),inplace=True)
#check it
#print(df.isna().sum())

#numer of rows x column
#print(df.shape)

# lets check the distribution of y variable (Life expectancy)
plt.figure(figsize=(8,8),dpi=88)
sns.boxplot(df['Life expectancy'])
plt.title('life exoectancy box plot')
plt.show()

plt.figure(figsize=(8,8))
plt.title('Life expectancy  distribution diagram')
sns.distplot(df['Life expectancy'])

#we can see that there are some outliers and linearly correlated

#checking multi corelationamong features
plt.figure(figsize=(15,15))
p=sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',center=0)

#pair plot to know z r/n  b/n d/f features
ax=sns.pairplot(df)

# split input features and out puts ones
x=df.drop(columns=['Life expectancy','Country'])
y=df['Life expectancy']

#split train and test data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1234)

#ecnode column status 
df['Status']=labelencoder.fit_transform(df['Status'])

df.head()
