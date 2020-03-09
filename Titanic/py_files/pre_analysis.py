"""
@author: Pipe Galera

When the Titanic sank, 1502 of the 2224 passengers and crew were killed.
One of the main reasons for this high level of casualties was the lack of
lifeboats on this self-proclaimed "unsinkable" ship.

Those that have seen the movie know that some individuals were more likely to
survive the sinking (lucky Rose) than others (poor Jack).
"""

# Import modules
%reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import tree, svm
from sklearn.preprocessing import LabelEncoder
import os

# Figures inline and set visualization style
%matplotlib inline
sns.set()

# Load data
os.chdir("C:/Users/fgm.si/Documents/GitHub/side_projects/titanic")
label_data = pd.read_csv('raw_data/train.csv')
unlabel_data = pd.read_csv('raw_data/test.csv')


# Exploratory Data Analisys (EDA)
label_data.shape
unlabel_data.shape
label_data.head(2)
unlabel_data.head(2)

label_data.info()
label_data.describe()

# concat data to cleanning and feature ingenearing
target = label_data.Survived
data = pd.concat([unlabel_data, label_data.drop("Survived", axis = 1)], sort = False)

# Cleanning the datasets
data = data.replace("female", 1)
data = data.replace("male", 0)
data['Age'] = data.Age.fillna(unlabel_data.Age.median())
data['Fare'] = data.Fare.fillna(unlabel_data.Fare.median())

# Definning the submission
submission = pd.DataFrame(unlabel_data["PassengerId"])
submission.shape

# Visual Analysis
sns.countplot(x = 'Survived', data = label_data);
sns.countplot(x = 'Sex', data = label_data);
sns.catplot(x = 'Survived', col = 'Sex', kind = 'count', data = label_data)
label_data.groupby(['Sex']).Survived.sum()

# Save data
submission.to_csv("out_data/submission.csv", index = False)
target.to_csv("out_data/target.csv", index = False, header = False)

#######################
# Feature engineering #
#######################

# Extracting titles from the names
data_eng = data.drop(['PassengerId', 'Ticket'], axis = 1).copy()
data_eng.columns
data_eng['Title'] = data_eng.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x = 'Title', data = data_eng);
plt.xticks(rotation = 45);


data_eng['Title'] = data_eng['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data_eng['Title'] = data_eng['Title'].replace(['Don', 'Dona', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data_eng)

data_eng.head()
data_eng = data_eng.drop(['Name'], axis = 1)

# Making cabinless a feature
-data_eng.Cabin.isnull()
data_eng['Has_Cabin'] = -data_eng.Cabin.isnull()
data_eng.head()
data_eng = data_eng.drop(['Cabin'], axis = 1)

# Missing values imputing
data_eng.info()
data_eng.isna().sum()
data_eng['Embarked'] = data_eng.Embarked.fillna('S')
data_eng.info()

# Binning age
data_eng['CatAge'] = pd.qcut(data_eng.Age, q = 4, labels = False )
data_eng['CatFare']= pd.qcut(data_eng.Fare, q = 4, labels = False)
data_eng.head()

data_eng = data_eng.drop(['Age', 'Fare'], axis = 1)

#  Number of Members in Family Onboard
data_eng['Fam_Size'] = data_eng.Parch + data_eng.SibSp
data_eng.tail()
data_eng = data_eng.drop(['SibSp','Parch'], axis=1)
data_eng.head()

# Transform to numerical values
le = LabelEncoder()
data_eng["Embarked"] = le.fit_transform(data_eng["Embarked"])
data_eng["Title"] = le.fit_transform(data_eng["Title"])
data_eng["Has_Cabin"] = le.fit_transform(data_eng["Has_Cabin"])
data_eng.head()

# Save data
data_eng.to_csv("out_data/data_eng.csv", index = False)
