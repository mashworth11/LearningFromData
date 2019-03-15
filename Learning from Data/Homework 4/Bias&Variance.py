#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:40:23 2019

@author: markashworth
"""
"""
Program to answer Q's 4 - 7 of Homework 4.
"""
import numpy as np
import matplotlib.pyplot as plt

class generateExamples(object):
    def __init__(self, N):
        self.N = N
        self.x = np.asarray(self.pickInputs())
        self.y = self.generateOutputs()
        
    def pickInputs(self):
        """
        Method to generate inputs for examples.
        """
        return [np.random.choice([-1,1])*np.random.random() for i in range(self.N)]
    
    def generateOutputs(self):
        """
        Method to generate outputs for given inputs (self.x), a.k.a this is the
        target function.
        """
        return np.sin(np.pi*self.x)
    
    def visualisation(self):
        """
        Method for visualisation.
        """
        plt.plot(self.x, self.y, 'o')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
      
        
class generalLinear(object):
    """
    Class for creating general linear regression model
    """
    def __init__(self, N = 2):
        self.N = N
        self.examples = generateExamples(self.N)
        self.x = self.examples.x
        self.X = self.createX()
        self.y = self.examples.y
        self.w = self.compute_w() # used here to computer coefficient a
    
    def createX(self):
        """
        Method for creating the X matrix used in the regression algorithm. 
        """
        X = np.vstack((np.ones(self.N), self.x))
        return X.transpose()
    
    def compute_w(self):
        """
        Method to compute the weighting parameter using linear regression algoridem
        """
        self.pinvX = np.linalg.pinv(self.X)
        return np.dot(self.pinvX, self.y)
    
    def visualisation(self):
        """
        Method to visualise our predicted hypothesis.
        """
        plt.plot(self.x, self.y, 'o', label = 'Example data')
        plt.plot(self.x, np.dot(self.w, self.X), label = 'Model')
        plt.xlim([-1,1])
        plt.ylim([-1,1])


class axHypothesis(generalLinear):
    """
    Subclass of generalLinear class that contains hypotheses h(x) = ax 
    """
    def __init__(self, N = 2):
        generalLinear.__init__(self, N)
        
    def createX(self):
        X = np.vstack((np.zeros(self.N), self.x)) # Stacks rows upon rows
        return X.transpose()
    
class bHypothesis(generalLinear):
    """
    Subclass of generalLinear class that contains hypotheses h(x) = b 
    """
    def __init__(self, N = 2):
        generalLinear.__init__(self, N)
    
    def createX(self):
        X = np.vstack((np.ones(self.N), np.zeros(self.N)))
        return X.transpose()
    
class ax2Hypothesis(generalLinear):
    """
    Subclass of generalLinear class that contains hypotheses h(x) = ax2
    """
    def __init__(self, N = 2):
        generalLinear.__init__(self, N)
        
    def createX(self):
        X = np.vstack((np.zeros(self.N), self.x**2))
        return X.transpose()

class ax2bHypothesis(generalLinear):
    """
    Subclass of generalLinear class that contains hypotheses h(x) = ax2 + b
    """
    def __init__(self, N = 2):
        generalLinear.__init__(self, N)
        
    def createX(self):
        X = np.vstack((np.ones(self.N), self.x**2))
        return X.transpose()
    
    
# Q) 4    
def calcAverage_g(model_type, N = 2, k = 1000):
    """
    Function to calculate the average g(x). Averaged over k experiments
    """
    a = 0
    for i in range(k):
        hyp = model_type(N);
        a += hyp.w    
    return a/k


# Q) 5 & 6 
def calcStatistics(model_type, k = 1000, x = np.linspace(-1,1,1000), N = 2):  
    """
    Function to calculate bias and varias statistics. For the bias we calculate
    it as an average over a discrete set from the input space [-1, 1]. For the
    variance we calculate it over k experiments and then take the average.
    """
    # av_g
    av_g = calcAverage_g(model_type)
    
    # create X
    X = np.vstack((np.ones(len(x)),x)).transpose()
    
    # bias, average across whole input space
    b = np.mean((np.dot(X, av_g)-np.sin(np.pi*x))**2)
    
    # variance, average across whole input space for each data set, then average
    # this average across multiple data sets
    v = 0
    for i in range(k):
        hyp = model_type(N) # generate new hypothesis function
        v += np.mean((np.dot(X, hyp.w) - np.dot(X, av_g))**2) # average this across the entire inpute spacede
    v = v/k
    
    return b, v

# Q) 7
def calcEout4Models(models_list):
    """
    Calculates Eout for each of the models in Q) 7 over a number of experiments I. 
    """
    models_dict = {}
    for model in models_list:
        models_dict.update({model.__name__:calcEout(model)})
    
    return models_dict
    

def calcEout(model_type):
    """
    Helper function to calcualte Eout for a given model over a number of experiments. 
    """    
    b, v = calcStatistics(model_type)
    return b + v
        
        
    
























        
        
        
        
        
        
        
        

