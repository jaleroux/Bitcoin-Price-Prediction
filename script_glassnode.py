#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:32:54 2018
Glassnode Solution
@author: james.leroux
"""

import sys
sys.path.insert(0, '/glassnode_ds_challenge')

import functions_glassnode as glass

import numpy as np 
import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt   # plots

import statsmodels.api as sm # for dickey-fuller test
import statsmodels.tsa.api as smt # for acf/pacf (wasneeded for regression)
from sklearn.preprocessing import StandardScaler # independent variable scaling
from sklearn.externals import joblib # save for standardscaler
from sklearn.metrics import f1_score # measures

from sklearn.linear_model import LogisticRegression  # ml imports
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit # preserving time-series ordering

from keras.models import model_from_json # to export model
from sklearn.externals import joblib # save for standardscaler

# GLOBAL VARIABLES
# api
HOURS_LIMIT = 2000 # number of hours returned limit
API_PARAMS = {'fsym': 'BTC', 
              'tsym': 'USD', 
              'limit': HOURS_LIMIT, 
              'aggregate': 1,
              'api_key': 'e8d00b797e32ab2eae591187aee752bb628a427577e22e9fdc7c98f50129135f'}
YEARS_OF_DATA = 2 # 2 years

# ML params
TARGET = 'close'
PRICEVOL_COLUMNS = ['high','low','open','volumefrom','volumeto']
TEST_SPLIT = 0.3

# get data
data = glass.get_data(HOURS_LIMIT, API_PARAMS, YEARS_OF_DATA) 

# duration
# min/max hours check
min_datetime = datetime.utcfromtimestamp(data['time'].min())
max_datetime = datetime.utcfromtimestamp(data['time'].max()) # .strftime("%d/%m/%Y %H:%M:%S")

duration_of_data = max_datetime - min_datetime
duration_months = (duration_of_data.days + (duration_of_data.seconds/(60*60*24)))/30.42

# plot time-series of USD-BTC
plt.figure()
plt.plot(data[TARGET])
plt.title('Bitcoin Price (hourly data)')
plt.grid(True)
plt.show()

# basic feature engineering
# 1) differencing
# split into different variables (dependent, independent)
Y = data[(TARGET)] 
X_pricevol = data[PRICEVOL_COLUMNS]
X_time = data[['date_time', TARGET]]

# check for stationarity in feature variables
# differencing needed as Dickey-Fueller null hypotheses not rejected (p-value 0.47) on open_price
sm.tsa.stattools.adfuller(X_pricevol.iloc[:,2])[1]

# therefore difference all price/volume variables by lagging by 1
Y_diff = (Y - Y.shift(1)).dropna()
X_pricevol_diff = (X_pricevol - X_pricevol.shift(1)).dropna()

sm.tsa.stattools.adfuller(X_pricevol_diff.iloc[:,2])[1] # now made stationary (to preserve the integrity of constant mean, variance through series)

X_time = X_time[X_time > X_pricevol_diff.index.min()] # cut-off 1st obs from remainder feature matrics, due to X,Y on the price features differencing

# 2) feature creation and lags
X_features, Y_diff = glass.feature_creation(Y_diff, TARGET,
                                            X_pricevol_diff, PRICEVOL_COLUMNS,
                                            X_time, 
                                            1, 1, TEST_SPLIT,
                                            status='build')

# 3) Create Binary Target Variable (on price direction)
Y_cat = pd.Series([1 if obs > 1 else 0 for obs in Y_diff], index=Y_diff.index)

# 4) train test split - basic 70/30, preserve ordering of time series
X_train, X_test, Y_train, Y_test = glass.train_test_split(X_features, Y_cat, TEST_SPLIT)


# 5) feature scaling
# parameterise feature scaling with training sample only to re-scale both training and test variables
# scaling is necessary to standardise contribtutions a variable makes to the final output/quicken convergence of optimiser
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'feature_scaling.pkl')

###############################################################################

## baseline model - persistence model y_hat = y_{t-1}
#  - high auto-correlation occurs at this lag and is a strong 
#  - but is not real prediction  
#smt.graphics.plot_acf(Y_train[Y_train.index],lags=64)
#smt.graphics.plot_pacf(Y[Y_train.index], lags=12)

# measure predictions by taking prediction with previous obs. as input 
validate_Y = [y_ix for y_ix in Y_cat[Y_train.index]]
predictions = list()
for test_ix in range(Y_test.index.min(),Y_test.index.max()+1):
    # predict (take last observation)
    predictions.append(validate_Y[-1])
    # observation
    validate_Y.append(Y_cat[test_ix])
# performance 
#accuracy_persis = accuracy_score(Y_cat[Y_test.index], predictions)
f1score_persistent  = f1_score(Y_cat[Y_test.index], predictions)
print('Persistence Model F1 Score : ',f1score_persistent)

## Regression Models 
# - ridge rergession - to offset overfitting reduce model size (in terms of features)
# - for time-series cross-validation set 5 folds
#  (preserves order of time-series and only reveals next tranche as validation set, 
#   whilst training tranche growing in size)
tscv = TimeSeriesSplit(n_splits=5) # static and incremental train/test split ; 5 different sets

model1 = LogisticRegression()
cv_av_f1_m1, cv_deviation_f1_m1, test_f1_m1, model1 = \
    glass.model_results(model1, 
                        X_train_scaled, 
                        Y_train,
                        X_test_scaled, 
                        Y_test,
                        tscv)
print('Logistic Regression Model F1 Score : ', test_f1_m1)
# plot coeffs
plt.plot(pd.DataFrame(model1.coef_, columns=X_train.columns).transpose())
plt.xticks(rotation='vertical')

# for possible overfitting, regularisation, reduces coefficient values
model2 = RidgeClassifier()
cv_av_f1_m2, cv_deviation_f1_m2, test_f1_m2, model2 = \
    glass.model_results(model2, 
                        X_train_scaled, 
                        Y_train,
                        X_test_scaled, 
                        Y_test,
                        tscv)
    
print('Ridge Classifier F1 Score : ', test_f1_m2)    
plt.plot(pd.DataFrame(model2.coef_, columns=X_train.columns).transpose())
plt.xticks(rotation='vertical')

# RNN Model - Long-Short-Term-Memory Model
# highest f1_score 
train_f1_m3, test_f1_m3, model3 = glass.keras_lstm(X_train_scaled, 
                                                   Y_train.as_matrix(), 
                                                   X_test_scaled,
                                                   Y_test.as_matrix(),
                                                   batch_size = 1, 
                                                   epoch_num = 10, 
                                                   neurons = 12)

# save model to json
model_lstm_json = model3.to_json()
with open("model_lstm.json", "w") as json_file:
    json_file.write(model_lstm_json)
# serialize weights to HDF5
model3.save_weights("model_lstm.h5")
print("Saved lstm model to disk")