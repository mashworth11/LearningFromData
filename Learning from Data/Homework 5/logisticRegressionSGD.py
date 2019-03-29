#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:52:16 2019

@author: markashworth
"""
"""
Program used to the logistic regression question of homework 5 of the Learning
from Data course.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class exampleGenerator(object):
    """
    Example generator class. Inputs taken from [-1, 1] x [-1, 1], outputs 
    generated according to classification across a linear function, that is the
    straight line joining two randomly selected points.
    """
    def __init__(self, N):
        """
        N being the number of examples
        """
        self.N = N
        self.randomSelector = lambda: np.random.random()*np.random.choice([-1, 1])
        self.X = self.generateInputs()
        [self.m, self.c] = self.generateLinearEqn()
        self.Y = self.generateOutputs(self.X)
        
    def generateInputs(self, N = None):
        """
        Function to generate inputs.
        """
        if N == None:
            N = self.N
        x1 = np.asarray([self.randomSelector() for i in range(N)])
        x2 = np.asarray([self.randomSelector() for i in range(N)])
        return np.column_stack((x1,x2))
    
    def generateLinearEqn(self):
        """
        Generates parameters used to describe the equation of a line joining 
        two randomly selected points. 
        """
        point1 = np.array([self.randomSelector(), self.randomSelector()])
        point2 = np.array([self.randomSelector(), self.randomSelector()])
        m = (point1[1]-point2[1])/(point1[0]-point2[0])
        c = point1[1]-m*point1[0]
        return m, c
    
    def generateOutputs(self, X):
        """
        Function to generate outputs according to our target function. In this
        case the target function is entirely determinitistic. 
        """           
        Y = np.array([])
        for x in X:
            if (self.m*x[0] + self.c) >= x[1]:
                Y = np.append(Y,[1])
            else:
                Y = np.append(Y,[-1])
        return Y
    
    def visualise(self):
        """
        Visualisation for the examples.
        """
        for i in range(self.N):
            if self.Y[i] > 0:
                plt.plot(self.X[i,0], self.X[i,1], 'bo', label = '+1')
            else:
                plt.plot(self.X[i,0], self.X[i,1], 'rx', label = '-1')
        pos_one = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          label='+1')
        neg_one = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          label='-1')
        plt.legend(handles = [pos_one, neg_one])
        

class logisticReg(object):
    """
    Class to create logistic regression object.
    """
    def __init__(self, N, eta = 0.01):
        """
        N is the sample size, X is the matrix of inputs, w is the weighting
        vector and eta is the learning rate.
        """
        self.N = N
        self.examples = exampleGenerator(self.N)
        self.eta = eta
        self.X = np.column_stack((np.ones(self.N),self.examples.X))
        [self.w, self.t] = self.calcWeights()
        
    def calcWeights(self):
        """
        Calculate the weights using the stochastic gradient descent algoridem.
        Also calculates the number of epochs required to reach the termination
        criteria. 
        """
        w_cur = np.ones(3) # current, t-1
        w_new = np.zeros(3) # new, t
        t = 0 
        
        while np.linalg.norm(w_new-w_cur) > 0.01:
            w_cur = w_new
            epoch = np.random.permutation(self.N)
            for i in epoch:
                gradw0 = -(self.examples.Y[i]*self.X[i])/(1 + \
                         np.exp(self.examples.Y[i]*np.dot(w_cur,self.X[i,:])))
                w_new = w_new - self.eta*gradw0
            t += 1
        return w_new, t

    def calcEout(self, newSample = 100):
        """
        Calculate Eout on a random new sample of points.
        """
        newX = self.examples.generateInputs(N = newSample)
        newY = self.examples.generateOutputs(newX)
        newX = np.column_stack((np.ones(newSample), newX))
        pointwiseError = 0
        for i in range(newSample):
            pointwiseError += np.log(1+np.exp(-newY[i]*np.dot(newX[i],self.w)))
        return pointwiseError/newSample
        
    
def EoutAverage(logisticReg, I):
    """
    Function to calculate the average out of sample error across a number of
    experiments I.
    """
    Eout = 0
    for i in range(I):
        lR = logisticReg(100)
        Eout += lR.calcEout()
    
    return Eout/I
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    