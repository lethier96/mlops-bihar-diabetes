from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import common


X_train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')
Y_train = pd.read_csv('data/y_train.csv')
Y_test = pd.read_csv('data/y_test.csv')

#num_features = X_train['hour']
#cat_features = X_train['weekday', 'month']
#train_features = num_features + cat_features

model = Ridge()

model = model.fit(X_train[['weekday', 'month', 'hour']], Y_train)
y_pred = model.predict(X_test[['weekday', 'month', 'hour']])

common.persist_model(model, common.MODEL_PATH)