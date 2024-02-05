# import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('heart.csv')

train, test = train_test_split(data, test_size=0.2, random_state=122)

# segregate the feature matrix and target vector
Xtrain = train.drop(columns=['target'], axis=1)
ytrain = train['target']

Xtest = test.drop(columns=['target'], axis=1)
ytest = test['target']
# scale the training and test data


scaler = MinMaxScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# create the SVM classifier
svc = SVC(random_state=122)

# define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],                 # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],    # Choice of kernel
    'gamma': ['scale', 'auto', 0.1, 1],     # Kernel coefficient for 'rbf' and 'poly'
    'degree': [2, 3, 4]                     # Degree of the polynomial kernel
}

# create a GridSearchCV object with SVM and the parameter grid
grid_search_svc = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

# fit the model with the training data
grid_search_svc.fit(Xtrain_scaled,ytrain)

# print the best parameters and the corresponding accuracy
print('Best Parameters: ', grid_search_svc.best_params_)
print('Best Accuracy: ', grid_search_svc.best_score_)

# get the best model
best_svc = grid_search_svc.best_estimator_