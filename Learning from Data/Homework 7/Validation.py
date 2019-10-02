#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:18:59 2019

@author: markashworth
"""

"""
Program to answer Q's 1 - 5 of Homework 7.
"""

import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import pandas as pd

class VNL_linearRegression(object):
    """
    Linear regression with non-linear transform and validation class.
    """
    def __init__(self, data, k, tSamples):
        """
        data are the training data, and k denotes the VC dimension (or complexity)
        of the method. We split our data into N = tSamples of data and V = data - N.
        """
        self.data = data
        self.k = k
        self.tSamples = tSamples
        self.X = np.vstack((self.data['x1'], self.data['x2'])).transpose()
        self.splitY() # splits and binds Y data to train (or test) and validation sets
        self.mapZ()   # splits and binds non-linear transform data to train (or test) and validation sets
        self.w = self.calcW()
    
    def splitY(self):
        """
        Function to split data according to training (or test) and validation sets.
        """
        try:
            self.tY = np.array(self.data['y'][0:self.tSamples])
            if self.tSamples == len(self.data):
                raise Exception
            else:
                self.valY = np.array(self.data['y'][self.tSamples:len(self.data)])
        except Exception:
            self.tY = np.array(self.data['y'])
            
    
    def mapZ(self):
        """
        Function to first perform non-linear mapping, and then to split data into 
        training and validation sets.
        """
        x1 = self.X[:,0]
        x2 = self.X[:,1]
        Z = np.vstack((np.ones(np.size(x1)), x1, x2, x1**2, x2**2, x1*x2, \
                         np.abs(x1 - x2), np.abs(x1 + x2))).transpose()
        try:
            self.tZ = Z[0:self.tSamples, 0:(self.k+1)]
            if self.tSamples == len(self.data):
                raise Exception
            else:
                self.valZ = Z[self.tSamples:len(self.data), 0:(self.k+1)]
        except:
            self.tZ = Z[:, 0:(self.k+1)]
    
    
    def calcW(self):
        """
        Calculate weights associated with our tZ data.
        """
        self.pinvtZ = np.linalg.pinv(self.tZ)
        return np.dot(self.pinvtZ, self.tY)
        
    @staticmethod
    def classificationError(w, Z, Y):
        """
        Calculate the classification error
        """
        sign = lambda x: x and (-1 if x < 0 else 1)
        Error = 0
        N = np.size(Y)
        for i in range(N):
            if sign(np.dot(w, Z[i])) != Y[i]:
                Error += 1
        return (1/N)*Error
        

def data_processing(name, url):
    """
    Take .dat file and convert it to a pandas data frame
    e.g data_processing('in', 'http://work.caltech.edu/data/in.dta')
    """
    urllib.request.urlretrieve(url, \
                               '/Users/markashworth/PythonTings/Learning from' \
                               + ' Data/Homework 7/' + name + '.dat')
    return pd.read_csv(name + '.dat', sep = ' ', header=None, skipinitialspace=True,\
                       names=['x1', 'x2', 'y'])
    
# Q(1) & Q(3)
def errorOnVal(data, K, train):
    """
    Function to evaluate the validation error for a range of non-lineary transformed
    linear regression models.
    K - list of model complexities
    data - data on which models are trained and validated
    train - no. of training data
    """
    weights = dict()
    validationError = dict()
    for k in K:
        nl_LR = VNL_linearRegression(data, k, train)
        weights.update({k:nl_LR.w})
        validationError.update({k:nl_LR.classificationError(nl_LR.w, nl_LR.valZ, nl_LR.valY)})
    return weights, validationError
        
        
# Q(2)
def errorOnOut(data, weights):
    """
    Function to evaluate the out of sample error for each of the 5 models. 
    data - the data that we want to evaluate on
    weights - (or models) that we want to evaluate on
    """
    outError = dict()
    for k in weights.keys():
        testObject = VNL_linearRegression(data, k, len(data))
        ZoutData = testObject.tZ 
        YoutData = testObject.tY
        outError.update({k:testObject.classificationError(weights[k], ZoutData, YoutData)})
    return outError
        
        
    
    
    
    
    
    
    
    
    
        
        
        