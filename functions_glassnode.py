#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 02:16:02 2019
Functions file for model build and deploy.
@author: james.leroux
"""

import sys
import os

sys.path.insert(0, '/glassnode_ds_challenge')

import numpy as np 
import pandas as pd

from datetime import datetime

import requests # for cryptocomapre data 
import json
import pickle # export dict's

from math import ceil   # for api call rounding

from sklearn.metrics import f1_score # measures
from sklearn.model_selection import cross_val_score

from keras.models import Sequential # need keras 2.1.2
from keras.layers import Dense
from keras.layers import LSTM

def get_data(hours_limit, api_params, years_of_data):
    
    # api call amount number
    hours_of_data = years_of_data * (24*365)
    num_api_calls = ceil(hours_of_data / hours_limit) # amount of calls to api
    
    #1st call in num_api_call:
    request = requests.get("https://min-api.cryptocompare.com/data/histohour?", params=api_params)
    if request.content != '':
       response = request.content
       response = json.loads(response)
    #   print(response)
    
    #dayIndex = len(response['Data']) - 1
    timestamp_to_end = str(response['Data'][0]['time'] - (60*60))
    
    # loop call 2nd to NUM_API_CALLS
    results_df = pd.DataFrame(response['Data'])
    for calls in range(num_api_calls-1):
        # get data from crypto server,  using toTs param as batch delimiter (time at first_call is toTs at next call)
        request = requests.get("https://min-api.cryptocompare.com/data/histohour?toTs=" + timestamp_to_end
                                , params=api_params)
    
        print('fetching data, ', calls + 1,' out of ', num_api_calls)
    
        if request.content != '':
            response = request.content
            response = json.loads(response)
            # load into dataframe
            response_df = pd.DataFrame(response['Data'])
            results_df = pd.concat([results_df, response_df], ignore_index=True)
            
            # for next iteration 
            #dayIndex = len(response['Data']) - 1
            timestamp_to_end = str(response['Data'][0]['time'] - (60*60))
        else:
            print('error at this call ', calls)
    # end api call loop
    print('fetching data, ', num_api_calls,' out of ', num_api_calls)    
    results_df = results_df.sort_values(by = ['time'], ascending=True)
    results_df = results_df.reset_index()
    
    results_df['date_time'] = pd.to_datetime(results_df['time'], unit='s')
    
    return results_df

# simple train test split function for time-series (preserve ordeR)
#def difference(df, lag):
#    diff_df = (df - df.shift(lag)).dropna()
#    diff_df = diff_df.dropna()
#    return diff_df

# don't use one hot encoding -> use mean encoding for catergorical
def mean_encoding(df, cat_feature, target):
    ''' mean price for categorical group '''
    return dict(df.groupby(cat_feature)[target].mean())


def feature_creation(target_df, target_col_name, 
                     price_features_df, feature_col_names, 
                     time_features_df, 
                     lag_start, lag_end, status='build', test_split=0.3):
    # lagset
    lags_set = range(lag_start, lag_end + 1)
    ## lags of prices
    # lag of target
    lags_target = [target_df.shift(i) for i in lags_set] # get lags
    lags_target = pd.concat(lags_target, axis = 1)
    #lags_target.fillna(0, inplace=True)
    lags_target = lags_target.dropna()
    lags_target.columns = [target_col_name + '_lag_' + str(lag_ix) for lag_ix in lags_set] # rename columns
    
    # lag of rest of price/vol features
    lags_other = [price_features_df.shift(i) for i in lags_set]
    lags_other = pd.concat(lags_other, axis=1)
    #lags.fillna(0, inplace=True)
    lags_other = lags_other.dropna()
    
    lags_other_columns = [] # rename columns
    for name, lag_ix in [(name, lag_ix) for lag_ix in lags_set for name in feature_col_names]:
        lags_other_columns.append(name + '_lag_' + str(lag_ix))
    lags_other.columns = lags_other_columns
        
    
    # time features (removing na rows from lags above)
    time_features_df = time_features_df[time_features_df.index >= lags_target.index.min()].copy()
    time_features_df['hour'] = pd.DatetimeIndex(time_features_df['date_time']).hour
    time_features_df['day'] = pd.DatetimeIndex(time_features_df['date_time']).day
    time_features_df['month'] = pd.DatetimeIndex(time_features_df['date_time']).month
    
    # mean prices  for certain time_features_df features (store them for live scoring)
    if status == 'build':
        
        test_ix = int(len(time_features_df)*(1-test_split))
        
        average_hour_dict = mean_encoding(time_features_df[:test_ix], 'hour', target_col_name)
        average_day_dict  = mean_encoding(time_features_df[:test_ix], 'day', target_col_name)
        average_month_dict= mean_encoding(time_features_df[:test_ix], 'month', target_col_name)
        
  
        pickle.dump(average_hour_dict, open("average_hour.pkl","wb"))
        pickle.dump(average_day_dict, open("average_day.pkl","wb"))
        pickle.dump(average_month_dict, open("average_month.pkl","wb"))
        
        time_features_df['hour_average'] = list(map(average_hour_dict.get, time_features_df.hour))
        time_features_df['day_average'] = list(map(average_day_dict.get, time_features_df.day))
        time_features_df['month_average'] = list(map(average_month_dict.get, time_features_df.month))
        
    if status == 'live':

        average_hour_dict = pickle.load(open("average_hour.pkl","rb")) 
        average_day_dict = pickle.load(open("average_day.pkl","rb")) 
        average_month_dict = pickle.load(open("average_month.pkl","rb")) 
        
        time_features_df['hour_average'] = list(map(average_hour_dict.get, time_features_df.hour))
        time_features_df['day_average'] = list(map(average_day_dict.get, time_features_df.day))
        time_features_df['month_average'] = list(map(average_month_dict.get, time_features_df.month))
        
    # drop categorical 
    time_features_df.drop(['date_time','hour','day','month', 'close'], axis = 1, inplace= True)
    
    ## check with plot price by hour
    #pd.DataFrame.from_dict(average_hour, orient='index')[0].plot()
    
    # concaetenate all
    X = pd.concat([lags_target, lags_other, time_features_df], axis=1)
    Y = target_df[target_df.index >= lags_target.index.min()].copy()
    
    return X, Y

# train-test split function
def train_test_split(X,Y, test_split):
    # get the index after which test set starts
    test_ix = int(len(X)*(1-test_split))
    
    X_train = X.iloc[:test_ix]
    Y_train = Y.iloc[:test_ix]
    X_test = X.iloc[test_ix:]
    Y_test = Y.iloc[test_ix:]
    
    return X_train, X_test, Y_train, Y_test

def model_results(model, X_train, Y_train, X_test, Y_test,cv_type):
    
    # fit model
    model.fit(X_train, Y_train)

    # predict 
    prediction = model.predict(X_test)

    cv_f1 = cross_val_score(model, X_train, Y_train, 
                             cv=cv_type, 
                             scoring="f1")
    cv_av_f1 = cv_f1.mean() * (-1)
    cv_deviation_f1 = cv_f1.std()

    test_f1 = f1_score(Y_test, prediction)    
        
    return cv_av_f1, cv_deviation_f1, test_f1, model

def keras_lstm(X_array_train, Y_vec_train, X_array_test, Y_vec_test,
               batch_size, epoch_num, neurons):
    # function that intialises the LSTM network for a cliassification problem, 
    # and trains the lstm over a number of epocs
    # we do our own epoch training loop so as to reset the internal state of the 
    # lstm at the beginning of each full, new training run
    X_array_train_reshape = X_array_train.reshape(X_array_train.shape[0], 1, X_array_train.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X_array_train_reshape.shape[1], X_array_train_reshape.shape[2]), stateful=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for i in range(epoch_num): # our own definied epoch training loop
        model.fit(X_array_train_reshape, Y_vec_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        # model.fit(X_array_train, Y_vec_train, epochs=1, validation_split=0.33, batch_size=batch_size, verbose=1, shuffle=False)
        #train_f1 = f1_score(Y_vec_train, model.predict_classes(X_array_train_reshape, batch_size)) # training set     
        #print(train_f1)
        model.reset_states() # reset internal state
        #print('f1 score on train', train_f1) # print f1 score of the validation set
    train_f1 = f1_score(Y_vec_train, model.predict_classes(X_array_train_reshape, batch_size))
    
    X_array_test = X_array_test.reshape(X_array_test.shape[0], 1, X_array_test.shape[1])
    test_f1 = f1_score(Y_vec_test, model.predict_classes(X_array_test, batch_size)) # training set     
    return train_f1 , test_f1, model