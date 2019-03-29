#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:42:25 2019

@author: markashworth
"""
"""
Program used to answer the gradient descent question of homework 5.
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class gradDescent(object):
    """
    Class to compute the gradient descent algoridem on a function. 
    """
    def __init__(self, f, u = 1, v = 1, tC = 10**-14, eta = 0.1):
        """
        tC is the termination criteria, eta is
        the learning rate, f is a function that requires symbolic functions
        where necessary e.g. 
        f = lambda u,v: (u*sym.exp(v)-2*v*sym.exp(-u))**2
        """
        self.u = u
        self.v = v
        self.tC = tC
        self.eta = eta
        self.f = f
        self.grad = self.gradient()
        
    def gradient(self):
        """
        Calculate the gradient of the function f. Note we don't evaluate at 
        this step just generate the symbolic gradient expression which can
        then be evaluated as necessary.
        """
        u, v = sym.symbols('u v')
        return sym.lambdify((u,v), [sym.diff(self.f(u,v),u), sym.diff(self.f(u,v),v)])
    
    def update(self):
        """
        Update step for setting new position coordinates. Also check whether 
        we have reched our convergence. 
        """
        if self.f(self.u, self.v).evalf() <= self.tC:
            raise Exception('Termination criteria reached')
        else:
            evaluatedGrad = -np.asarray(self.grad(self.u, self.v))
            self.u += self.eta*evaluatedGrad[0] 
            self.v += self.eta*evaluatedGrad[1] 


class coordinateGradDescent(gradDescent):
    """
    Class to compute the coordinate gradient descent algorithem on a function.
    """
    def __init__(self, f, u = 1, v = 1, tC = 10**-14, eta = 0.1):
        gradDescent.__init__(self, f, u, v, tC, eta)
        
    def update(self):
        """
        Update step for coordinate coordinate gradient descent.
        """
        evaluatedGrad = -np.asarray(self.grad(self.u, self.v))
        if self.f(self.u, self.v).evalf() <= self.tC:
            raise Exception('Termination criteria reached')
        else:
            # step in u direction
            evaluatedGrad = -np.asarray(self.grad(self.u, self.v))
            self.u += self.eta*evaluatedGrad[0]
            # step in v direction using update u
            evaluatedGrad = -np.asarray(self.grad(self.u, self.v))
            self.v += self.eta*evaluatedGrad[1]            
        
    def checkError(self):
        """
        Function to return the current error.
        """
        return self.f(self.u, self.v).evalf()    
            
            
# Q) 5 and 6
def itCounter(function):
    """
    Function to count the number of iterations needed by the gradient descent
    method to reach the required termination tolerance given a function.
    Returns the number of iterations required to reach the termination criteria,
    and the gradient descent object containing the final values of the independent
    variables for which the termination criteria has been reached. 
    """
    gD = gradDescent(function)
    k = 0
    while True:
        try:
            gD.update()
            k += 1
        except Exception as err:
            print(err)
            break
    return gD, k


# Q) 7
def tolCounter(function):
    """
    Function to count the error after 15 iterations 
    """
    k = 0
    cGD = coordinateGradDescent(function, tC = 0)
    while k < 15:
        cGD.update()
        k += 1
    return cGD.checkError()
    
            
        
        
        
