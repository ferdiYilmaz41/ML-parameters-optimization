# import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
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

# create a Gradient Boosting Classifier
model = GradientBoostingClassifier()

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [1, 0.5],
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4]
}

# create the GridSearchCV object
grid_search_gb = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# fit the grid search to the data
grid_search_gb.fit(Xtrain_scaled, ytrain)

# print the best parameters and the corresponding accuracy
print('Best Parameters: ', grid_search_gb.best_params_)
print('Best Accuracy: ', grid_search_gb.best_score_)

# get the best model
best_tree = grid_search_gb.best_estimator_