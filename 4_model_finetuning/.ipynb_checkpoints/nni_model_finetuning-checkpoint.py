# IMPORTS
import nni
import os
import numpy as np
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# DATA
data_path = '/home/lorenzo/skl-repo/0_data/california_housing.csv'
df = pd.read_csv(data_path)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=3542)
print(f'Train set length: {len(train_X)}')
print(f'Test set length: {len(test_X)}')

trainX_cat = train_X.select_dtypes(exclude=np.number)
trainX_num = train_X.select_dtypes(include=np.number)
testX_cat = test_X.select_dtypes(exclude=np.number)
testX_num = test_X.select_dtypes(include=np.number)

num_columns = list(trainX_num.columns)
cat_columns = list(trainX_cat.columns)

# combine some attributes
rooms_ix, beds_ix, pop_ix, hh_ix = 3,4,5,6
class CombineAttributes(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_hh = X[:, rooms_ix] / X[:, hh_ix]
        avg_hh_size = X[:, pop_ix] / X[:, hh_ix]
        if self.add_bedrooms_per_room:
            beds_per_room = X[:, beds_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_hh, avg_hh_size, beds_per_room]
        else:
            return np.c_[X, rooms_per_hh, avg_hh_size]

# pipeline for numerical columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('cstm_attribs', CombineAttributes()),
    ('std_scaler', StandardScaler()),
])

# pipeline for categorical columns
cat_pipeline = Pipeline([
    ('onehot_enc', OneHotEncoder()),
])

# full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns),
])

# Apply pipeline transformation to features (X)
train_X = full_pipeline.fit_transform(train_X)
test_X = full_pipeline.transform(test_X)

# Convert target to numpy array (y)
train_y, test_y = train_y.values, test_y.values



### NNI TUNER ###

# 1) set default parameters to be used when not tuning
DEFAULT_PARAMETERS = {
    'n_estimators': 10,
    'max_depth': None,
    'min_samples_leaf': 1,
    'bootstrap': True
}

# 2) get updated parameters when tuning
RECEIVED_PARAMS = nni.get_next_parameter()
PARAMS = DEFAULT_PARAMETERS
PARAMS.update(RECEIVED_PARAMS)

# 3) fit model
rf = RandomForestRegressor(n_estimators = PARAMS['n_estimators'],
                           max_depth = PARAMS['max_depth'],
                           min_samples_leaf = PARAMS['min_samples_leaf'],
                           bootstrap = PARAMS['bootstrap']
                          )

rf.fit(train_X, train_y)
rf_pred = rf.predict(test_X)
rf_mse = mean_squared_error(test_y, rf_pred)
rf_rmse = np.sqrt(rf_mse)
print(f'RMSE: {rf_rmse}')

# 4) report back the score to the tuner
nni.report_final_result(rf_rmse)

### HOW TO RUN ### 
# from terminal execute the following code
# cd path/to/folder
# nnictl create --config ./config.yml