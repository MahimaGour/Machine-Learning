# -*- coding: utf-8 -*-
"""Bank_predictive_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11vndB8DnTYiTqRJz_LwVpCwp_ZhUkQUc

---

# **Exploring the data**


---
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Bank_data.csv')
df.head()

df.info()

"""---
# **Data Preprocessing**


---
"""

df = df.drop(['nr.employed'], axis=1)

col = df.iloc[:,[1,2,3,4,5,6,7,8,9,14]]
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values='unknown', strategy='most_frequent')
imp_fit = imp.fit_transform(col)
col_miss = pd.DataFrame(imp_fit, columns = ['job',  'marital',  'education',    'default',  'housing',  'loan', 'contact',  'month', 'day_of_week', 'poutcome'])

col_miss.head()

num_col = df.iloc[:,[0,15,16,17,18,19]]
num_col_df = pd.DataFrame( num_col )
all = [ col_miss,num_col_df ]
data = pd.concat( all, axis=1 )

data.head()

print(data.education.unique())
print(data.job.unique())
print(data.contact.unique())

temp = data.drop(['y'], axis=1)
final_df = pd.get_dummies(temp)
target = data['y']

final_df.head()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_df = ss.fit_transform(final_df)

ss_data = pd.DataFrame(ss_df, columns=['age',	'emp',	'cons.price',	'cons.conf',	'euri',	'job_admin',	'job_blue-collar',	'job_entrepreneur',	'job_housemaid',	'job_management',	'job_retired',	'job_self-employed',	'job_services',	'job_student', 'job_technician',	'job_unemployed',	'marital_divorced',	'marital_married',	'marital_single',	'education_basic.4y',	'education_basic.6y',	'education_basic.9y',	'education_high.school',	'education_illiterate',	'education_professional.course',	'education_university.degree',	'default_no',	'default_yes',	'housing_no',	'housing_yes',	'loan_no',	'loan_yes',	'contact_cellular',	'contact_telephone',	'month_apr',	'month_aug',	'month_dec',	'month_jul',	'month_jun',	'month_mar',	'month_may',	'month_nov',	'month_oct',	'month_sep',	'day_of_week_fri',	'day_of_week_mon',	'day_of_week_thu',	'day_of_week_tue',	'day_of_week_wed',	'poutcome_failure', 'poutcome_nonexistent',	'poutcome_success'])
ss_data.head()
ss_data.info()

"""---


# **Evaluate Algorithms**
---
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ss_data, target, test_size = 0.3)
scores = []

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
scores.append(('logistic regression', model_lr.score(x_test, y_test)))

from xgboost import XGBClassifier
model_xg = XGBClassifier()
model_xg.fit(x_train, y_train)
scores.append(('xgboost', model_xg.score(x_test, y_test)))

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)
scores.append(('knn', model_knn.score(x_test, y_test)))

from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
scores.append(('decision tree', model_dt.score(x_test, y_test)))

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
scores.append(('random forest', model_rf.score(x_test, y_test)))

from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(x_train, y_train)
scores.append(('naive bayes', model_nb.score(x_test, y_test)))

from sklearn.svm import SVC
model_svm = SVC()
model_svm.fit(x_test, y_test)
scores.append(('SVM', model_svm.score(x_test, y_test)))

from sklearn.ensemble import AdaBoostClassifier
model_ada = AdaBoostClassifier()
model_ada.fit(x_train, y_train)
scores.append(('ada boost', model_ada.score(x_test, y_test)))

from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier()
model_gb.fit(x_test, y_test)
scores.append(('gradient boost', model_svm.score(x_test, y_test)))

scores