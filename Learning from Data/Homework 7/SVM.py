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

class eval_target_function(object):
    """
    Target function evaluator class
    """
    def __init__(self):
        self.X = {key:[np.random.uniform(-1, 1) for i in range(2)] for key in ['x1', 'x2']}
        
    def target_function(self):
        grad = (self.X['x2'][1]-self.X['x1'][1])/(self.X['x2'][0]-self.X['x1'][0])
        c = self.X['x1'][1] - grad*self.X['x1'][0]
        
    
    def classify_point(self, z):
        """
        Method to classify point based on target function
        """
        