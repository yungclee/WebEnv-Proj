#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random
from copy import copy
from sklearn.metrics import r2_score
import missingno as msno
import functools 
import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer():
    
    def __init__(self, input_df, time_var = 'LogTime'): 
        self.df = input_df[sorted(input_df.columns.values)]
        self.time_var = time_var
    def construct_corr(self):
        '''
        Usage: 
            the method will construct a correlation matrix for plotting correlation heatmap.
            columns contains missing values, and has no variation will be dropped.
        '''
        
        ##first drop the columns that contains missing values.
        col_missingval = self.df.columns[self.df.isna().any()].tolist()
        col_novar = self.df.drop(columns = self.time_var).columns[self.df.var() == 0].tolist()
        self.corr_matrix = self.df.drop(columns = col_missingval + col_novar).corr()
        
        return self.corr_matrix
    
    def plt_corr(self):
        '''
        Usage: 
            the method will first call construct_corr and then plot the correlation matrix. 
        '''

        fig, ax = plt.subplots(figsize=(15,15))  
        ax.set_aspect('equal')
        sns.heatmap(self.construct_corr(), cmap="YlGnBu")
        
    def var_heatmap(self, target_var, smoothing = True, cmap = "inferno"):
        
        w = np.hanning(30)
        groups = self.df.set_index(self.time_var)[target_var].groupby(pd.Grouper(freq='D'))
        if smoothing: 
            df = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1).apply(lambda x : np.convolve(w/w.sum(),x,mode='valid'))
        else: 
            df = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
        df = pd.DataFrame(df)
        df = df.T

        fig = plt.figure(figsize=(30,5))
        ax = fig.add_subplot(111)
        cax = ax.matshow(df, interpolation=None, cmap= cmap, aspect='auto')
        fig.colorbar(cax)
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Minutes')
        plt.ylabel('Day')
        plt.title('{} heatmap plot'.format(target_var))
        plt.show()