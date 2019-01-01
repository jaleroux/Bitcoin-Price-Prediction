#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:49:09 2018

@author: james.leroux
"""

import sys
import os

import functions_glassnode as glass

from keras.models import model_from_json # to export model
from sklearn.externals import joblib # save for standardscaler
from datetime import timedelta

from math import ceil   # for api call rounding

from flask import Flask # for api 

app = Flask(__name__)

@app.route('/')
def price_direction_prediction():
    ## Global variables
    # api
    HOURS_LIMIT = 24 # number of hours returned limit
    API_PARAMS = {'fsym': 'BTC', 
                  'tsym': 'USD', 
                  'limit': HOURS_LIMIT, 
                  'aggregate': 1,
                  'api_key': 'e8d00b797e32ab2eae591187aee752bb628a427577e22e9fdc7c98f50129135f'}
    YEARS_OF_DATA = 1/365 # 1 day
    
    # ML params
    TARGET = 'close'
    PRICEVOL_COLUMNS = ['high','low','open','volumefrom','volumeto']
    
    ## LOAD model and feature params
    # LSTM model
    json_file = open('./model_lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_lstm.h5") # load weights into new model
    print("Loaded model from disk")
    
    # Feature scaling
    scaler = joblib.load('./feature_scaling.pkl') 
    
    ## Direction Prediction
    # Get most current data
    data_current = glass.get_data(HOURS_LIMIT, API_PARAMS, YEARS_OF_DATA) 
    
    Y = data_current[(TARGET)] 
    X_pricevol = data_current[PRICEVOL_COLUMNS]
    X_time = data_current[['date_time', TARGET]]
    
    # difference data and filter na's off 1st observation
    Y_diff = (Y - Y.shift(1)).dropna()
    X_pricevol_diff = (X_pricevol - X_pricevol.shift(1)).dropna()
    X_time = X_time[X_time > X_pricevol_diff.index.min()] # cut-off 1st obs, due to X,Y differencing
    
    X_features, _ = glass.feature_creation(Y_diff, TARGET, 
                                           X_pricevol_diff, PRICEVOL_COLUMNS,
                                           X_time, 1, 1, status = 'live')
    
    
    X_obs = X_features.values[-1,:] # take most recent observation
    
    X_obs_scaled = scaler.transform(X_obs.reshape(1,X_obs.shape[0])) # scale transform
    
    X_obs_scaled_reshape= X_obs_scaled.reshape(1, 1, X_obs_scaled.shape[1]) # get into right form for lstm
    
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    Y_forecast = loaded_model.predict_classes(X_obs_scaled_reshape, batch_size=1)
    
    if Y_forecast == 1:
        hour_direction = 'UP' 
    else:
        hour_direction = 'DOWN'
    
    next_hour = X_time['date_time'].iloc[-1] + timedelta(hours=1)
    
    Results = 'BTC ' + TARGET + ' price will go ' + hour_direction + ' in the following hour of ' + str(next_hour)
    
    return(Results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)