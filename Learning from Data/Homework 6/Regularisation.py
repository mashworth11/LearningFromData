#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:46:14 2019

@author: markashworth
"""

"""
Program to answer Q's 2 - 6 of Homework 6.
"""
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import pandas as pd

class NL_linearRegression(object):
    """
    Linear regression class with a non-linear transformation applied
    """
    def __init__(self, data):
        """
        data is the training data, data frame
        """
        self.data = data
        self.X = np.vstack((self.data['x1'], self.data['x2'])).transpose()
        self.Y = np.array(self.data['y'])
        self.Z = self.mapZ()
        self.w = self.calcW()        
        
    def mapZ(self):
        """
        Function to map to non-linear z coordinates
        """
        x1 = self.X[:,0]
        x2 = self.X[:,1]
        return np.vstack((np.ones(np.size(x1)), x1, x2, x1**2, x2**2, x1*x2, \
                         np.abs(x1 - x2), np.abs(x1 + x2))).transpose()
        
    def calcW(self):
        """
        Function to calculate the weights according to the linear regression algoridem
        """
        pinvZ = np.linalg.pinv(self.Z)
        return np.dot(pinvZ, self.Y)
    
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
    
    
class RNL_linearRegression(NL_linearRegression):
    """
    Regularised linear regression with a non-linear transformation applied. 
    We use a weight-decay regulisation method. 
    """
    def __init__(self, data, k):
        """
        k is the amount of regularisation exponent
        """
        self.lam = 10**k
        super().__init__(data)
        
    
    def calcW(self):
        """
        Function to calculate the weights according to the weight decay linear 
        regression algoridem
        """
        N = np.shape(self.Z)[1]
        return np.dot(
                      np.linalg.inv(np.dot(self.Z.transpose(), self.Z) \
                                    + self.lam*np.identity(N)), \
                      np.dot(self.Z.transpose(), self.Y))
    
    
def data_processing(name, url):
    """
    Take .dat file and convert it to a pandas data frame
    """
    urllib.request.urlretrieve(url, \
                               '/Users/markashworth/PythonTings/Learning from' \
                               + ' Data/Homework 6/' + name + '.dat')
    return pd.read_csv(name + '.dat', sep = ' ', header=None, skipinitialspace=True,\
                       names=['x1', 'x2', 'y'])

# Q(5)   
def findSmallestEout(train_dat, test_dat, K):
    """
    Function to find smallest Eout given a range of k's, K 
    """
    EoutDict = {}
    for k in K:
        Train = RNL_linearRegression(train_dat, k)
        w = Train.w
        Test = RNL_linearRegression(test_dat, k)
        Eout = Train.classificationError(w, Test.Z, Test.Y)
        EoutDict.update({k:Eout})
    return EoutDict

def visualiseSmallestEout(EoutDict):
    """
    Visualisation to accompany the findSmallestEout function
    """
    plt.figure(0)
    for i in EoutDict.keys():
        plt.plot(i, EoutDict[i], 'bo')
    plt.xlabel('Exponent k')
    plt.ylabel('E_out')
        
    

    
    
    
    
    