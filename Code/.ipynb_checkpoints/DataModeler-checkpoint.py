#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas
import math
import json
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class DataModeler():
    
    def __init__(self, input_df, forecast_len = 10): 
        self.df = input_df 
        self.forecast_len = forecast_len
        self.model = None
    
    def prep_input(self, data, scaler, n_in = 1, n_out =1, dropna = True, key = 'ambinet', time_var = 'LogTime'):
        # adopt and modified from  https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        # convert series to supervised learning
        '''
        Variables: 
            1. data: input dataframe to convert to LSTM ready input
            2. key: used to indicate and generate the response
            3. time_var: the column that is specified to be the timestamp
            4. scaler: the method requires a scaler to normalize the data set.
        Usage:
            input must be a dataframe with a time_var indicating the DateTime index to be used, 
            will perform the follwing actions: 
                1. create response column by integrating the values containing keyword `key` and drop those.
                2. drop columns contain NaN to avoid scaler error
                3. use MinMaxScaler to scale all values to [0,1]
                4. create input sequence(t-n, ...., t-1) and response requence(t,...t+n) 
        '''
        ## first create response columns
        df = pd.DataFrame(data).set_index(time_var)
        response_var = [x for x in data.columns if key in x]
        df['res'] = df[response_var].apply(lambda x: x.mean(), axis = 1).values
        df.drop(columns = response_var, inplace = True)

        ## drop columnms that contain Nan to avoid error 

        df.dropna(axis =1, inplace = True)

        ## scale the vars to be in [0,1] range

        df = pd.DataFrame(scaler.fit_transform(df.values.astype(np.float32)), columns = df.columns)

        n_vars = 0 if type(df) is list else df.shape[1] - 1
        cols, names = [], []
        ## input sequence (t-n, ....t-1) 
        for t in range(n_in, 0, -1):
            cols.append(df.drop(columns = 'res').shift(t)) ## explanatory vars
            cols.append(df['res'].shift(t)) ## response var
            names += ['var{}(t-{})'.format(j, t) for j in range(n_vars) ]
            names += ['res(t-{})'.format(t)]
        ## response sequence (t,... t+n)
        for t in range(0, n_out):
            cols.append(df['res'].shift(-t))
            if t ==0:
                names += ['res(t)']
            else: 
                names += ['res(t+{})'.format(t)]
        agg = pd.concat(cols, axis = 1)
        agg.columns = names

        if dropna: 
            ## first drop the rows that contains Nan produced by the shift.
            agg.dropna(axis = 0, thresh = max(n_in, n_out) + 1, inplace = True)
            ## then the columns with Nan values should be dropped
            agg.dropna(axis = 1, inplace = True)
        return agg
    
    def ts_train_test_split(self, data, perc = 0.8):
        '''
        Usage: 
            Analog of scipy train test split, while applying on time series data the data has to be continuous, thus the customized function.
            note that the response has to be in the last column
        '''
        length = len(data) if type(data) is list else data.shape[0]
        index = int(length * perc)
        train = data[:index, :]
        test = data[index: , :]
        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        ## importnat the X need to be reshape into 3D array for LSTM input [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        print("Using {}% as training and {}% test".format(100*perc, round(100*(1-perc))) )
        print("X reshape to have 3D: [samples, timesteps, features]")
        for i in ['X_train', 'X_test', 'y_train','y_test']:
            print('{}: shape{}'.format(i, eval(i).shape))
        return X_train, X_test, y_train, y_test
    
    def LSTM_model(self, X_train, X_val, y_train, y_val):
        '''
        Usage: 
            build LSTM model for temperature prediction purpose
        caution: 
            the hyperparameters in this setup has been pre-tuned, whenever a setting changes such as input variable or output 
            variable this should use grid search again to find the optimal hyperparameter settings.
        '''
        model = Sequential()
        model.add(LSTM(41, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        RMS = optimizers.RMSprop(learning_rate= 0.0001, rho=0.9)
        model.compile(loss='mse',
                      optimizer= RMS,
                      metrics=['mae'])
        # fit network
        history = model.fit(X_train, y_train, epochs= 60, batch_size= 100, validation_data=(X_val, y_val),verbose = 2)

        return history, model
    
    def run(self):
        
        '''
        Usage:
            Main member function for users, will execute a chain of functions defined above and produce a predictive model.
        
        Chain: 
            1) First call `prep_input`to convert the pandas data frame into desire setup len using the parameter 'forecast_len'
            2) create scaler
            3) call `ts_train_test_split` to split the training and validating data 
            4) build LSTM model [TODO:] automated tunning? 
        '''
        np.random.seed(420)
        self.scaler = MinMaxScaler(feature_range= (0,1))
        
        ## prepare input for lstm
        lstm_in = self.prep_input(self.df, self.scaler ,n_out = self.forecast_len)
        
        ## setup inverse scaler
        y_inv_scaler = MinMaxScaler()
        y_inv_scaler.min_,y_inv_scaler.scale_ = self.scaler.min_[-1], self.scaler.scale_[-1]
        self.y_inv_scaler = y_inv_scaler
        
        # split data 
        self.X_train, self.X_test, self.y_train, self.y_test = self.ts_train_test_split(lstm_in.values)
        
        ## train model
        self.history, self.model = self.LSTM_model(self.X_train, self.X_test, self.y_train, self.y_test)
        
    def forecast(self, *new_X):
        if self.model is not None: 
            ## check if model has been calculate for prediction 
            pred = self.model.predict(self.X_test[-20:,:])
            ## apply inverse scalar 
            y_forecast = self.y_inv_scaler.inverse_transform(pred).reshape(-1)
            
            return json.dumps(y_forecast.tolist())
        