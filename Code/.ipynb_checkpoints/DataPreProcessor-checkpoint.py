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

class DataPreProcessor():
    
    def __init__(self, input_df, time_var, verbose = False, show_graph = True):
        self.df = copy(input_df)
        self.ori_df = copy(input_df)
        self.time_var = time_var
        self.verbose = verbose
        self.show_graph = True
    
    def get_processed_data(self): 
        return self.df
        
    def drop_empty_rows(self): 
        '''
        Usage: 
            this method will call pandas.dropna(), setting thresh = 2 (need at least one non nan other than the time variable presented in the data set)
            and row-wise dropping. 
        Return: 
            None, this will be an inplace operation. Will update the self.df
        '''
        self.df.dropna(axis = 0, thresh = 2, inplace = True)
    
    def drop_duplicates(self):
        '''Usage: 
            The function will call pandas.drop_duplicate() and it updates `self.df`. 
            will use groupby and mean to drop the duplicated row that has the same timestamp
        '''
        self.df.drop_duplicates()
        self.df = self.df.groupby(self.time_var).mean().reset_index()
        
    def create_imputed_flag(self): 
        '''
        Usage: 
            if the row contains missing values to be filled or imputed, the value will be true, flase o.w
        '''
        self.df['imputed'] = self.df.isnull().any(axis=1)
        
        
    def get_impute(self, target_var:str, missing_ratio = [9, 1]):
        
        '''
        Variable:
            1. target_var: the variable of which to be impute
            2.missing_ratio : [True, False] ratio
        Usage: 
            the method will first perform forward filling followed by backward filling with limit=1, 
            reasoning being that the timestamp was rounded to the closest minute.
            then it will sample a slice of continuous time and create simple random missingness 
            and compare the R-square of the imputation method from imputation methods with itself. \
        Return: 
            this will update the df
        '''
        if target_var not in self.df.columns: 
            print("Error: Unexpeceted column selected.")
            return 
        
        df = copy(self.df[[self.time_var, target_var]]).set_index(self.time_var)
        na_count = sum(self.df[target_var].isnull())
        ## set imputation flag
        missing_perc = round(100* na_count / len(df[target_var]),3)
        if not na_count:
            print('Warning: column %s has no missing values.' %target_var)
            return 
        elif na_count / len(df[target_var]) > 0.5: 
            print('Warning: [{}] {}% data are missing, should not be imputed. Exiting...'.format(target_var, missing_perc))
            return 
        else: 
            if self.verbose: print('Imputing: [{}] {}% data are missing.'.format(target_var, missing_perc))
            
        ## will first fill forward and backward one entry
        df = df.fillna(method = 'ffill', limit = 1)
        forward_fill = na_count - sum(df[target_var].isnull())
        df = df.fillna(method = 'bfill', limit = 1)
        backward_fill = na_count - sum(df[target_var].isnull()) - forward_fill
        if self.verbose: print("Preprocess forward and backward filling filled {}, {} entries".format(forward_fill, backward_fill))    
            
            
        ##this will create a numpy array 
        arr = df[target_var].values

        ## follwing chunk referenced:
        ## https://stackoverflow.com/questions/41494444/pandas-find-longest-stretch-without-nan-values
        m = np.concatenate(( [True], np.isnan(arr), [True] ))  # Mask
        ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
        start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits
        ##------------------------------
        ## set the first and final elmt to true to avoid spline error producing NAs
        mask = [True] + random.choices([True, False], weights = missing_ratio, k= stop-start-2 ) + [True]
        df_tmp = pd.DataFrame({'reference' : arr[start : stop], \
                               'target': arr[start : stop] * [x if x else np.nan for x in mask ], \
                               self.time_var :self.df[self.time_var][start: stop]}).set_index(self.time_var)
        reference = arr[start:stop]
        ## find best method to fill nan values
        ## Reference: https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460
        if self.verbose: print('\n\tComupting best imputing method....')
        df_tmp = df_tmp.assign(FillMean = df_tmp.target.fillna(df_tmp.target.mean()))
        df_tmp = df_tmp.assign(FillMedian = df_tmp.target.fillna(df_tmp.target.median()))
        df_tmp = df_tmp.assign(linear = df_tmp.target.interpolate(method='linear'))
        df_tmp = df_tmp.assign(time = df_tmp.target.interpolate(method='time'))
        df_tmp = df_tmp.assign(slinear = df_tmp.target.interpolate(method='slinear'))
        df_tmp = df_tmp.assign(quadratic = df_tmp.target.interpolate(method= 'quadratic'))
        
        ##calculate R-square
        results = [(method, r2_score(df_tmp.reference, df_tmp[method])) for method in list(df_tmp)[3:]]
        results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
        results_df = results_df.sort_values(by='R_squared', ascending=False).reset_index()

        if self.verbose: print('\tBest imputing method :%s' %results_df.Method[0])

        if results_df.Method[0] in ['time', 'linear', 'slinear', 'quadratic']:
            df[target_var] = df[target_var].interpolate(method = results_df.Method[0])
            df.reset_index()
        elif results_df.Method[0] == 'FillMedian':
            df[target_var] = df[target_var].fillna(df[target_var].median())
        elif results_df.Method[0] == 'FillMean':
            df[target_var] = df[target_var].fillna(df[target_var].mean())
        
        self.df[target_var] = df[target_var].values
        before_count = len(self.df[target_var])
        imputed_count = na_count - sum(df[target_var].isnull()) - forward_fill - backward_fill
        imputed_perc = round(100 * imputed_count / before_count, 3)
        if self.verbose: print( "\tImputed {} entries ({}%)\n".format(imputed_count, imputed_perc) )
        return (target_var, results_df.Method[0],na_count, forward_fill, backward_fill, imputed_count)
    
    def impute_all(self):
        '''
        Usage:
            the method will utilize the self.get_impute function to impute all the columns
        '''
        self.create_imputed_flag()
        if self.show_graph: msno.matrix(self.df)
        report = []
        for col in self.df.columns: 
            if col != self.time_var: 
                out = self.get_impute(col)
                if out: 
                    report.append(out)
                else: 
                    report.append([col, None, None, None, None, None])
        
        report_df = pd.DataFrame(np.array(report), columns = ['Columns', 'Impute_method', 'Nan_count', \
                                                              'Forward_filled', 'Backward_filled', 'Imputed_count'])
        if self.show_graph: msno.matrix(self.df)
        return report_df 
