#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:20:08 2019

@author: markashworth
"""
"""
Program to be used to answer the Hoeffding inequality question (Q1) and (Q2)
from Homework 2 of the learning from data course
"""

import numpy as np

class coinFlipExperiment(object):
    def __init__(self):
        """
        c1 is the first set of coin flips
        cRand is a random choice of a set of coin flips
        cMin is the set of coin flips which has the minimum number of heads
        """
        self.coinFlips = self.generateFlips()
        self.c1 = self.coinFlips[0]
        self.cRand = self.coinFlips[np.random.choice(list(self.coinFlips))]
        self.cMin = self.findcMin()
                
    def generateFlips(self):
        """
        Method to generate a dictionary of coin flips for which we have 1000
        coins (keys) and for each key 10 flips.
        """
        return {i:np.random.choice([0,1],10) for i in range(1000)}
        
    def findcMin(self):
        """
        Find the coin that has the minimum frequency of heads. We consider a 
        head to be equal to 1 and a tails to be equal to 0
        """
        frequency = 10 # start with all heads
        for i in self.coinFlips.keys():
            if sum(self.coinFlips[i]) < frequency:
                frequency = sum(self.coinFlips[i])
                coin = self.coinFlips[i]                   
        return coin
    
    
def coinFlipSimulation(N):
    """
    Runs the coin flip experiment N times and uses it to establish statistics
    on the number of heads for each of the coins defined by the coinFlipExperiment.
    Returns fractions of heads v1, vRand, vMin for the 3 coin selections resp-
    ectively. 
    """
    v1, vRand, vMin = [], [], []
    
    for i in range(N):
        coinExp = coinFlipExperiment()
        v1.append(sum(coinExp.c1)/len(coinExp.c1))
        vRand.append(sum(coinExp.cRand)/len(coinExp.cRand))
        vMin.append(sum(coinExp.cMin)/len(coinExp.cMin))        
    
    return v1, vRand, vMin
        
    