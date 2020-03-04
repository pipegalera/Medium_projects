"""
Created on Tue May 28 12:30:00 2019

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
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os
os.chdir("C:/Users/fgm.si/Documents/GitHub/kaggle_competitions/titanic_competition")

# Figures inline and set visualization style
%matplotlib inline
sns.set()

# Import test and train datasets

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


# Exploratory Data Analisys (EDA)

df_train.head()
df_test.head()

df_train.info()
df_train.describe()

# Visual Analysis

sns.countplot(x = 'Survived', data = df_train);
sns.countplot(x = 'Sex', data = df_train);
sns.factorplot(x = 'Survived', col = 'Sex', kind = 'count', data = df_train)
df_train.groupby(['Sex']).Survived.sum()

# First baseline: none survived

df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('/Users/mac/GitHub/Kaggle/titanic_competition/predictions/no_survivors.csv', index = False)

# Second baseline: women survived

print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].PassengerId.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].PassengerId.count())


df_test['Survived'] = df_test.Sex == 'female'
df_test.head()
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()

df_test[['PassengerId', 'Survived']].to_csv('/Users/mac/GitHub/Kaggle/titanic_competition/predictions/women_survived.csv', index = False)

# Preprocessing: combine train and test for data manipulation

survived_train = df_train.Survived
data = pd.concat([df_train.drop(['Survived'], axis = 1), df_test], sort = False)
data.info()

# Impute missing numerical variables and create dummies

data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data.info()

data = pd.get_dummies(data, columns = ['Sex'], drop_first = True)
data.head()


# Create first ML model

data2 = data[['Sex_male', 'Fare', 'Age', 'Pclass', 'SibSp']]
data2.info()

survived_train.describe()

data2.iloc[891:]
data_train = data2.iloc[:891]
data_test = data2.iloc[891:]

X = data_train.values
test = data_test.values
y = survived_train.values

clf = tree.DecisionTreeClassifier(max_depth = 3)
clf.fit(X, y)

y_pred = clf.predict(test)
df_test['Survived'] = y_pred


df_test[['PassengerId', 'Survived']].to_csv('/Users/mac/GitHub/Kaggle/titanic_competition/predictions/1st_dec_tree.csv', index=False)


# Selecting hyperparameter to avoid overfitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, stratify = y)

dep = np.arange(1,9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

for i,k in enumerate(dep):
    clf = tree.DecisionTreeClassifier(max_depth=k)
    clf.fit(X_train, y_train)
    train_accuracy[i] = clf.score(X_train, y_train)
    test_accuracy[i] = clf.score(X_test, y_test)

plt.title('clf: Varying depth of tree')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# Feature engineer (extracting titles from the names)

data.Name.head()

data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x = 'Title', data = data);
plt.xticks(rotation = 45);


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);

data.head()

# Feature engineer (cabinless)

data['Has_Cabin'] = -data.Cabin.isnull()
data.head()

data3 = data.copy()
data3.drop(['Survived','Cabin', 'Name', 'PassengerId', 'Ticket'], axis = 1, inplace = True)

data3.head()
data3.info()

# Missing values imputing

data3['Age'] = data3.Age.fillna(data3.Age.median())
data3['Fare'] = data3.Fare.fillna(data3.Fare.median())
data3['Embarked'] = data3.Embarked.fillna('S')
data3.info()
data3.head()


# Binning age

data3['CatAge'] = pd.qcut(data3.Age, q = 4, labels = False )
data3['CatFare']= pd.qcut(data3.Fare, q = 4, labels = False)
data3.head()

data3 = data3.drop(['Age', 'Fare'], axis = 1)
data3.head()

#  Number of Members in Family Onboard

data3['Fam_Size'] = data3.Parch + data3.SibSp

data3 = data3.drop(['SibSp','Parch'], axis=1)
data3.head()


# Transform to numerical values

data3_dum = pd.get_dummies(data3, drop_first = True)
data3_dum.head()


# Create second ML model

data3_train = data3_dum.iloc[:891]
data3_test = data3_dum.iloc[891:]
X = data3_train.values
test = data3_test.values
y = survived_train.values



dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

clf = tree.DecisionTreeClassifier()
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)
clf_cv.fit(X, y);

print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


y_pred = clf_cv.predict(test)
df_test['Survived'] = y_pred
df_test[['PassengerId', 'Survived']].to_csv('/Users/mac/GitHub/Kaggle/titanic_competition/predictions/dec_tree_feat_eng.csv', index=False)
