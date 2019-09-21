# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:46:33 2019

@author: Administrator
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.base import clone

class Bagger:
    
    """A Stacked Bagging regressor.
    
    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.
    
    When parameters are such that there is a single estimator with no stacking,
    and no sampling, it performs the same as that single estimator
        stacking=False, num_bags=1, ratio=1.0, clone_estimator=False

    Parameters
    ----------
    estimators : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
        
    Checks
    ----------
    1. Make sure each permutation is doing what it's suppose to be doing
    with regards to parameter settings
    2. Does calling sklearn fit() multiple times continue fitting the function?
    warm_starts:
        When sklearn.BaggingRegressor.warm_start=True, var = 0.0000
        Simulate a warm_start
        clone_estimator=False, refit=False ensures that subsequent calls to fit
        will leverage previously fit data
    cloning:
        when clone_estimate=False
        
    outstanding:
        Bagger(bag_estimators, stacking=False, intercept=False, 
                                num_bags=1, ratio=1.0, bootstrap=True, 
                                clone_estimator=True, refit=False)        
    """
    
    def __init__(self, estimators=None, stacking=True, intercept=True, 
                 num_bags=1, ratio=0.8, bootstrap=True, clone_estimator=False,
                 refit=True):
        
        if (estimators is None):
            estimators = {'rf':RandomForestRegressor()}
        self.estimators = [model for name, model in estimators.items()]
        self.estimator_names = list(estimators.keys())            
        
        self.models = []
        
        # stacking function, include intercept in final regression weighting
        self.stacking = stacking
        self.intercept = intercept
        self.num_bags = num_bags
        self.sample_ratio = ratio
        self.bootstrap = bootstrap
        self.clone_estimator_ = clone_estimator
        self.refit = refit
        
        self.lr_ = None
        self.weights = None
        self.bags = None
        self.summary_ = None
        
    
    ###############################################################################
    # Functions
    def sample_(self, arr, ratio=0.2):
        # select
        n = arr.shape[0]
        sz = int(ratio * n)
        idx = np.random.choice(n, sz, replace=self.bootstrap)
        return arr[idx]    
    
   
    def fit(self, X, y):
        
        # continue using the same models for multiple fit() calls if spec
        if (self.refit):
            self.models = []

        bags = []
        y_results = None
        y_samples = []
        K = len(self.estimators)

        for i in range(K):
            for j in range(self.num_bags):
                # TODO - The sample rate could be changed as a hyper parameter
                # data sample
                if (self.sample_ratio < 1.0):
                    Xy = self.sample_(np.column_stack((X,y)), self.sample_ratio)
                    Xt = Xy[:,:-1]
                    yt = Xy[:,-1]
                else:
                    Xt = X
                    yt = y
                        
        
                # save sample set to compare
                y_samples.append(yt)
                
                # get model, clone it and save it and bog it
                # clone yields a new estimator with the same parameters that 
                # has not been fit on any data.
                if (self.clone_estimator_):
                    reg = clone(self.estimators[i])
                else:
                    reg = self.estimators[i]

                self.models.append(reg)
                reg.fit(Xt, yt)
                y_pred = reg.predict(X)
                
                # capture metrics
                mse = metrics.mean_squared_error(y, y_pred)
                rsq = metrics.r2_score(y, y_pred)
                
                # save estimatation results
                bags.append([y_pred, mse, rsq])
                
                # append to input set
                if (y_results is None):
                    y_results = y_pred
                else:
                    y_results = np.column_stack((y_results, y_pred))
                
    
        # check results dimension
        if (y_results.ndim == 1):
            y_results = y_results[:,None]
            
        # Stacking - Voting with final weights on each model result
        if (self.stacking):
            reg = LinearRegression(fit_intercept=self.intercept)
            reg.fit(y_results, y)
            print(y_results.shape)
            y_pred = reg.predict(y_results)
            self.lr_ = reg
        else:
            if (len(self.models) > 1):
                y_pred = np.mean(y_results, axis=1)
                # print("{0:.4f} (mean), {1:.4f} (var), {2:d}".format(np.mean(y_pred), np.var(y_pred),len(self.models)))                
            else:
                y_pred = y_results
                
            self.weights = None
            self.lr_ = None

        # get metrics
        mse = metrics.mean_squared_error(y, y_pred)
        rsq = metrics.r2_score(y, y_pred)
        final = [y_pred, mse, rsq]
        
        # get residual
        y_resid = y - y_pred
        
        # save fit summary
        self.summary =  {'weights':self.weights,
                'y_samples':y_samples, # samples used to fit models
                'y_results':y_results, # model ouptuts from samples
                'y_resid':y_resid, # this is the next y_train in the iteration
                'bags':bags,
                'final':final}

    def predict(self, X):
        '''
        Using the fit models and the voting weights, predict the output
        using the input information
        '''
        y_results = None
        K = len(self.models)
        idx = 0
        for i in range(K):
            for j in range(self.num_bags):
                # get model and bog it
                reg = self.models[idx]
                idx += 1
                y_pred = reg.predict(X)
            
                # append to input set
                if (y_results is None):
                    y_results = y_pred
                else:
                    y_results = np.column_stack((y_results, y_pred))
                
        # check results dimension...needed with regression intercept
        if (y_results.ndim == 1):
            y_results = y_results[:,None]
            

        # Voting with final weights and voting
        if (self.stacking):
            print(y_results.shape)
            y_pred = self.lr_.predict(y_results)
        else:
            if (len(self.models) > 1):
                y_pred = np.mean(y_results, axis=1).flatten()
            else:
                y_pred = y_results.flatten()

        return y_pred