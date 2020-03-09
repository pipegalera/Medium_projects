"""
Created on June 2019

@author: Pipe Galera

Ask a home buyer to describe their dream house, and they probably won't begin
with the height of the basement ceiling or the proximity to an east-west railroad.
But this playground competition's dataset proves that much more influences price
negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential
homes in Ames, Iowa, this competition challenges you to predict the final price
of each home.
"""

# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
%matplotlib inline
sns.set()

# Import test and train datasets

df_train = pd.read_csv('/Users/mac/GitHub/Kaggle/house_competition/data/train.csv', index_col=0)
df_test = pd.read_csv('/Users/mac/GitHub/Kaggle/house_competition/data/test.csv', index_col=0)


# Exploratory Data Analisys (EDA)

df_test.shape
df_train.shape

df_train.describe()

df_train.select_dtypes(include=['object']).describe() # 43 categorical columns
df_train.select_dtypes(exclude=['object']).describe() # 37 numerical columns

y = df_train.SalePrice

# Visual Analisys

sns.distplot(y)
plt.title('Distribution of SalePrice');
y.skew()

sns.distplot(np.log(df_train.SalePrice))
plt.title('Distribution of Log-transformed SalePrice')
plt.xlabel('log(SalePrice)');
np.log(y).skew()

# Correlations for feature selection

df_train.corr()['SalePrice'].sort_values(ascending=False).head(20)

# First baseline: House Price = mean. accuracy_score = 0.42

SalePrice.describe()

df_test['SalePrice'] = y.mean()
df_test[['Id', 'SalePrice']].to_csv('/Users/mac/GitHub/Kaggle/house_competition/predictions/baseline_mean.csv', index = False)


# Data Cleaning: Finding NaN values

df_train.select_dtypes(exclude=['object']).isna().sum().sort_values(ascending = False).head()
df_train.select_dtypes(include=['object']).isna().sum().sort_values(ascending = False).head(17)

# Data Cleaning: Finding outliers

fig = plt.figure(figsize=(12, 18))

for i in range(len(df_train.select_dtypes(exclude=['object']).columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=df_train.select_dtypes(exclude=['object']).iloc[:,i])

plt.tight_layout()
plt.show()

# Outliers
"""
- LotofFrontage > 300
- LotArea > 200000
- OverallQual < 2.7
- BsmtFinSF1 > 4000
- BsmtFinSF2 > 1450
- TotalBsmtSF >6000
- 1stFlrSF >4000
"""

# Impute missing variables with Label Encoder
df_train_num_features = df_train.select_dtypes(exclude=['object'])
df_train_nonnum_features = df_train.select_dtypes(include=['object'])

df_train_num_features.isna().sum().sort_values(ascending = False).head()

encoder = LabelEncoder()
features = df_train_num_features.columns
le = LabelEncoder()
for i in features:
    df_train_num_features[i] = le.fit_transform(df_train_num_features[i])

df_train_num_features.isna().sum().sort_values(ascending = False).head()

df_train_nonnum_features.head()
features = df_train_nonnum_features.columns
le = LabelEncoder()
for i in features:
    df_train_nonnum_features[i] = le.fit_transform(df_train_nonnum_features[i].astype(str))


df_train_nonnum_features.isna().sum().sort_values(ascending = False).head()

# Transforming data to reduce skew

df_train_num_features['SalePrice_log'].head()

np.log(df_train.SalePrice).head()

df_train_num_features['SalePrice'] = np.log(df_train_num_features['SalePrice']);
df_train_num_features = df_train_num_features.rename(columns={'SalePrice': 'SalePrice_log'})
df_train_num_features.SalePrice_log.skew()

# First ML Model

SalePrice.describe() # 1460

data_train = data2.iloc[:1460]
data_test = data2.iloc[1460:]


X = data_train.values
test = data_test.values
y = SalePrice.values

clf = tree.DecisionTreeRegressor(random_state=1)
clf.fit(X, y);

SalePrice = clf.predict(test)
SalePrice
df_test['SalePrice']


df_test[['Id', 'SalePrice']].to_csv('/Users/mac/GitHub/Kaggle/house_competition/predictions/1st_dec_tree.csv', index=False)
