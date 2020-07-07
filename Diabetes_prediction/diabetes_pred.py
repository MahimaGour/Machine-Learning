"""

**Predict the onset of diabetes based on diagnostic measures**

*Source: Pima Indians Diabetes Database*
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('diabetes.csv')
df.head()

x = df.iloc[:, 0:7]
y = df.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5, min_samples_split=3)
model.fit(x_train, y_train)
print('DecisionTree score')
model.score(x_test, y_test)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

model2 = LogisticRegression(max_iter=200)
model2.fit(x_train, y_train)
print('LogisticRegression score')
model2.score(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(n_estimators=100,criterion='entropy', max_depth=3)
model3.fit(x_train, y_train)
print('RandomForest score')
model3.score(x_test, y_test)

from sklearn.svm import SVC

model4 = SVC()
model4.fit(x_train, y_train)
print('SVC score')
model4.score(x_test, y_test)

