#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:18:49 2021

@author: ananya
"""

import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler #no longer applicable
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def get_model_pred(filename, train_data, train_labels, true_ind):
    
    keys = ['time', 'lstride', 'rstride', 'lswing', 'rswing', 'lswing_int', 'rswing_int', 'lstance', 
        'rstance', 'lstance_int', 'rstance_int', 'dsupport', 'dsupport_int']
    
    file = 'gait_data/'+filename
    #read in file as table
    X = pd.read_table(file,header=None) 
    X.columns = keys
    
    f_keys = ['lstride', 'rstride', 'lswing', 'rswing', 'lswing_int', 'rswing_int', 'lstance', 
        'rstance', 'lstance_int', 'rstance_int', 'dsupport', 'dsupport_int']
    
    for key in f_keys:
        temp = np.asarray(X[key])
        #find absolute z scores for each value
        z = np.abs(stats.zscore(temp)) 
        ind = np.where(z > 3)
        #replace outliers with the mean value
        temp[ind] = np.mean(temp)
        X[key] = temp
        
    #get time since start for each patient
    normTime = []
    time1 = np.asarray(X['time'])
    for j in range(len(X)):
        normTime.append(time1[j]-time1[0])
    X['normTime'] = normTime
    
    aggfeats = pd.DataFrame()
    windows = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

    #find statistics for entirety of time series data
    for signal in X.columns[1:13]:
        aggfeats['agg_mean_'+signal] = [np.mean(X[signal])]
        aggfeats['agg_var_'+signal] = [np.var(X[signal])]
        aggfeats['agg_med_'+signal] = [np.median(X[signal])]
        aggfeats['agg_range_'+signal] = [np.amax(X[signal]) - np.amin(X[signal])]
    
    #find statistics for 30-second data windows
    for wind in range(len(windows)-1):
        winddat = X.loc[(X.normTime >= windows[wind]) & (X.normTime<windows[wind+1])]
        for signal in winddat.columns[1:13]:
            aggfeats['mean'+str(wind+1)+'_'+signal] = [np.mean(X[signal])]
            aggfeats['var'+str(wind+1)+'_'+signal] = [np.var(X[signal])]
            aggfeats['med'+str(wind+1)+'_'+signal] = [np.median(X[signal])]
            aggfeats['range'+str(wind+1)+'_'+signal] = [np.amax(X[signal]) - np.amin(X[signal])]
        
    #train model on established best train data
    rf = RandomForestClassifier(max_features='auto', n_estimators=200, max_depth=5, criterion='gini')
    rf.fit(train_data,train_labels)
    
    true_ind = np.asarray(true_ind, dtype=int)
    test_data = aggfeats[aggfeats.columns[true_ind]]
    #test_data = StandardScaler().fit_transform(test_data) #no longer applicable
    
    pred = rf.predict(test_data)
    
    if pred == 0:
        return "is healthy"
    elif pred == 1:
        return "has ALS"
    elif pred == 2:
        return "has Huntington's"
    elif pred == 3:
        return "has Parkinson's"
    else:
        return " "
