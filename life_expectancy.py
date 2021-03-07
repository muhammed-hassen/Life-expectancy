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

#ecnode column status 
df['Status']=labelencoder.fit_transform(df['Status'])

df.head()
