#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:36:43 2019

@author: markashworth
"""
"""
Program to answer Q's 1 - 3 of Homework 4.
"""

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

def fixedPoint(func, x = 1, error = 0.001):
    """
    Function to perform a fixed point iteration on a function. 
    func is the function g in the fixed point algorithm  x_(n+1) = g(x_n),
    x is the starting value, and error is the tolerance error.  
    """
    x_new = func(x) 
    while abs(x_new - x) > error:
            x = x_new
            x_new = func(x)           
    return x    

def VC_bound(N, dvc = 50, delta = 0.05):
    """
    Function to define the VC bound.
    """
    return np.sqrt((8/N)*np.log(4*((2*N)**dvc + 1)/delta))
    
        
def RadePen_bound(N, dvc = 50, delta = 0.05):
    """
    Function to define the Rademacher penalty bound.
    """
    return np.sqrt((2/N)*np.log(2*N*((N)**dvc + 1))) \
           + np.sqrt((2/N)*np.log(1/delta)) + 1/N
    
    
def ParVdB_bound(N, dvc = 50, delta = 0.05):
    """
    Function to define the Parrondo and Van de Broek bound.
    We ask our function to return a handle in terms of epsilon, e.
    """
    return lambda e: np.sqrt((1/N)*(2*e + np.log(6*((2*N)**dvc + 1)/delta)))


def Dev_bound(N, dvc = 50, delta = 0.05):
    """
    Function to define the Devroye bound.
    We ask our function to return a handle in terms of epsilon, e. 
    """
    return lambda e: np.sqrt((1/(2*N))*(4*e*(1+e) \
                     + float((4*(Decimal(N)**(2*dvc) + 1)/Decimal(delta)).ln())))


def computeFPArray(N, func, dvc = 50, delta = 0.05):
    """
    Function that returns an array of fixed point evalutions on the function, func, 
    e.g. ParVdB_bound, for values in a sample size array, N.
    """   
    epsilon = np.array([])
    for i, v in enumerate(N): # no need to use enumerate here, just wanted to try it out
        f = func(v, dvc, delta)
        e = fixedPoint(f)
        epsilon = np.append(epsilon, [e])
    return epsilon


def boundPlot(VC_bound, RadePen_bound, ParVdB_bound, Dev_bound, N):
    """
    Function to answer Q 2 of ze homework. Plots 4 different bounds
    as functions of N. Requires the bounds to be input as functions.
    """
    assert len(N) > 1, 'N must be an array of sample sizes'
    plt.plot(N, VC_bound(N), label = 'VC Bound')
    plt.plot(N, RadePen_bound(N), label = 'Rademacher penalty bound')
    plt.plot(N, computeFPArray(N, ParVdB_bound), label = 'Parrando and Van den Broek bound')
    plt.plot(N, computeFPArray(N, Dev_bound), label = 'Devroye bound')
    plt.xlabel('Sample Size - N')
    plt.ylabel('Generalisation Error - Bound')
    plt.ylim([0, 6])
    plt.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    