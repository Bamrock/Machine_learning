import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


df = pd.read_csv("NAFLD.csv",encoding='gbk')
X = df.iloc[:,3:-1]
y = df.iloc[:,2]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
estimator=RandomForestClassifier(oob_score=True,random_state=1)
estimator.fit(X_train,y_train)
print(estimator.oob_score_)

"""
search best n_estimators hyperparameter, 
other hyperparameters are default
"""
param_test1={'n_estimators':range(1,101,10)}
grid_search=GridSearchCV(estimator=RandomForestClassifier(random_state=1),param_grid=param_test1,scoring='roc_auc',cv=10)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

"""
search best max_features hyperparameter, 
n_estimators are the best resutlt abouver
other hyperparameters are default 
"""
param_test2={'max_features':range(1,21,1)}
grid_search_1=GridSearchCV(estimator=RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],random_state=1),param_grid=param_test2,scoring='roc_auc',cv=10)
grid_search_1.fit(X_train,y_train)
print(grid_search_1.best_params_)
print(grid_search_1.best_score_)

"""
train the model with the best parameters
"""
rfl=RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],max_features=grid_search_1.best_params_['max_features'],oob_score=True,random_state=1)
rfl.fit(X_train,y_train)
print(rfl.oob_score_)

