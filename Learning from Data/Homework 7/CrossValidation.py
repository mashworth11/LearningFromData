#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:17:13 2019

@author: markashworth
"""

"""
Program to answer Q 7 of Homework 7.
"""
import numpy as np
import itertools

class XV_linearRegression(object):
    """
    Linear regression class with a cross-validation error check
    """
    def __init__(self, rho):
        """
        rho denotes a variation in the independent parameter
        """
        self.samples = [(-1,0), (rho,1), (1,0)]
        self.splitData()
        self.W = self.calcW()
        self.XVerror = self.calcXVerror()
        
    def splitData(self):
        """
        Method to split the data 
        """
        # first generate data sets for training
        dataSets = list(itertools.combinations(self.samples,2))
        dataSetDict = {}
        # for each associated data set find the data set that you've left out
        leftOut = {}      
        for i in range(len(self.samples)):
            leftOut.update({i:np.array(list(set(self.samples).\
                                            symmetric_difference(set(dataSets[i])))[0])})
            dataSetDict.update({i:np.array(dataSets[i])})
        
        self.leftOut = leftOut
        self.dataSetDict = dataSetDict
        
    def calcW(self):
        """
        Create a weights dictionary associated with weights calculated with 
        a data set, Dn. e.g. {0:[w1,w2]}, weights associated with model calculated
        with self.samples[0] removed.:
        """
        weightsDict = {}
        for k in self.dataSetDict.keys():
            X = np.array([np.ones(2), self.dataSetDict[k][:,0]]).transpose()
            Y = self.dataSetDict[k][:,1]
            weightsDict.update({k:np.dot(np.linalg.pinv(X),Y)})
        return weightsDict
        
    def calcXVerror(self):
        """
        Program to calculate the cross validation error associated with a given
        model
        """
        XVError = 0
        n = 0
        for k in self.W.keys():
            x = np.array([1, self.leftOut[k][0]])
            y = self.leftOut[k][1]
            e = (np.dot(x,self.W[k]) - y)**2
            XVError += e
            n += 1
            
        return (1/n)*XVError
       
        
class XV_constantRegression(XV_linearRegression):
    """
    Subclass where we just consider the hypothesis set {h(x)=b}
    """
    def __init__(self, rho):
        super().__init__(rho)

    def calcW(self):
        """
        Redifine calcW method
        """   
        weightsDict = {}
        for k in self.dataSetDict.keys():
            X = np.array([np.ones(2), np.zeros(2)]).transpose()
            Y = self.dataSetDict[k][:,1]
            weightsDict.update({k:np.dot(np.linalg.pinv(X),Y)})
        return weightsDict
        
        
#Q(7)
def XVerrorComparison(rho):
    """
    Function to compute and compare the XV error for different linear models.
    rho denotes a list of input values that will be the parameter that is 
    varied within the comparison.
    """
    XVerrorDict = {}
    for p in rho:
        XV_lR = XV_linearRegression(p)
        XV_cR = XV_constantRegression(p)
        XVerrorDict.update({p:[XV_lR.XVerror, XV_cR.XVerror]})
    return XVerrorDict
        
    
  
    
    
    
    
        