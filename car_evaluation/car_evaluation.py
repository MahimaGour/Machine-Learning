# -*- coding: utf-8 -*-
"""car_evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MUSC1HpXrnGo4Ova7qt1QbnDSk2gdy9_
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

x = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']

df = pd.read_csv('car.csv', names = x)

df.head()

for i in x:
  print(df[i].unique())

df['doors'].replace('5more', '5', inplace=True)
df['persons'].replace('more', '6', inplace=True)

for i in x:
  print(df[i].unique())

df.doors = pd.to_numeric(df.doors)
df.persons = pd.to_numeric(df.persons)

df.info()

df.buying = df.buying.astype('category')
df.maint = df.maint.astype('category')
df.lug_boot = df.lug_boot.astype('category')
df.safety = df.safety.astype('category')
df.target = df.target.astype('category')

df.info()

for i in x:
  if i!='doors' and i != 'persons':
   df[i] = df[i].cat.codes

df.head()

X = df.iloc[:,0:6]
y = df.iloc[:,6]

df.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=1000)
model1.fit(x_train, y_train)
model1.score(x_test, y_test)

from sklearn.svm import SVC
model2 = SVC()
model2.fit(x_train, y_train)
model2.score(x_test, y_test)

from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
model3.score(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier()
model4.fit(x_train, y_train)
model4.score(x_test, y_test)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
model5 = GaussianNB()
model5.fit(x_train, y_train)
model5.score(x_test, y_test)

model6 = BernoulliNB()
model6.fit(x_train, y_train)
model6.score(x_test, y_test)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
model7 = XGBClassifier()
model7.fit(x_train, y_train)
model7.score(x_test, y_test)

from sklearn.ensemble import GradientBoostingClassifier
model8 = GradientBoostingClassifier()
model8.fit(x_train, y_train)
model8.score(x_test, y_test)

