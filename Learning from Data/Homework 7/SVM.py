#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:55:00 2019

@author: markashworth
"""

"""
Program to answer Q's 8 - 10 of Homework 7.
"""

import numpy as np
from cvxopt import matrix, solvers

class eval_target_function(object):
    """
    Target function evaluator class
    """
    def __init__(self, N):
        self.X = {key:[np.random.uniform(-1, 1) for i in range(2)] for key in ['x1', 'x2']}
        self.N = N
        self.training_data = self.create_training_set()
        self.test_data = self.create_test_set()
                
    def classify_point(self, z):
        """
        Method to classify point based on target function
        """
        grad = (self.X['x2'][1]-self.X['x1'][1])/(self.X['x2'][0]-self.X['x1'][0])
        c = self.X['x1'][1] - grad*self.X['x1'][0]
        if z[1] >= grad*z[0] + c:
            return 1
        else: 
            return -1

    def create_training_set(self):
        """
        Method to create training data set 
        """
        training_data = {}
        for i in range(self.N):
            z = [np.random.uniform(-1,1), np.random.uniform(-1,1)]
            training_data.update({tuple(z): self.classify_point(z)})
        return training_data
                                                       
    def create_test_set(self):
        """
        Method to create test set
        """
        test_data = {}
        for i in range(500):
            z = [np.random.uniform(-1,1), np.random.uniform(-1,1)]
            test_data.update({tuple(z): self.classify_point(z)})
        return test_data
    
        
class PLA(object):
    """
    Perceptron learning algorithm
    """
    def __init__(self, classifier_object):
        self.classifier_object = classifier_object
        self.W = np.zeros(3)
    
    def PLA_classify(self, data_type = 'training'):
        """
        Method to classify points using PLA 
        """
        misclassified_points = []
        if data_type == 'training':
            data_dictionary = self.classifier_object.training_data
        elif data_type == 'testing':
            data_dictionary = self.classifier_object.test_data
        for key in data_dictionary.keys():
            if  np.sign(np.dot(self.W, np.array([1] + list(key)))) != data_dictionary[key]:
                        misclassified_points.append(np.array(key))
        return misclassified_points
    
    def learning(self):
        """
        Method to carry out the learning algorithm
        """
        misclassified_points = self.PLA_classify()
        while misclassified_points:
            point = tuple(misclassified_points[np.random.randint(0,10)])
            self.W += self.classifier_object.training_data[point]*np.array([1] + list(point))
            misclassified_points = self.PLA_classify()
            

class SVM(object):
    """
    Support vector machine algorithem
    """
    def __init__(self, classifier_object):
        """
        Need to create the required matrices for the QP solver to leverage
        """
        self.classifier_object = classifier_object
        self.X, self.Y = self.extractXY()
        self.alpha = self.solve_QuadProg()
        self.W = self.calcW()
        self.b = self.calcb()
        
    def extractXY(self):
        """
        Method to create X and Y arrays 
        """
        X = []
        Y = []
        for key,value in self.classifier_object.training_data.items():
            X.append(list(key))
            Y.append(value)
        return np.array(X), np.array([Y])
            
    
    def create_Qd(self):
        """
        Method to create Qd matrix
        """
        Xs = self.X*self.Y.transpose()
        return np.dot(Xs, Xs.transpose())
    
    def create_Ad(self):
        """
        Method to create Qd matrix
        """
        return np.vstack([self.Y, -self.Y, np.identity(self.Y.size)])
    
    def solve_QuadProg(self):
        """
        Method to solve optimisation problem with quadratic programming solver
        """
        Qd = self.create_Qd()
        Ad = self.create_Ad()
        alpha = solvers.qp(matrix(Qd), matrix(-np.ones(Qd.shape[0])), \
                           matrix(Ad), matrix(np.zeros(Ad.shape[0])))
        return alpha
            
    def calcW(self):
        """
        Method to calculate weights
        """
        
         
    
    def calcb(self):
        """
        Method to calculate b
        """
         
    
    def evaluate_points(self):
        """
        Method to classify new points e.g. test points
        """
    
                    
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        