#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:11:26 2021

@author: oster
"""
import numpy as np
from misc import random_homogenous_polynomial_sum,  legendre_measures,legendre_measures_grad, Gramian, HkinnerLegendre  #, hermite_measures



def V(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure = legendre_measures(x.T,deg)
    measure =  np.concatenate([measure, np.ones((1,samples,deg+1))], axis=0)
    res = v.evaluate(measure)
    assert res.shape ==  (samples,)
    return res

def gradV(x,v):
    order,samples = x.shape
    deg = v.dimensions[0]-1
    measure_grad = legendre_measures_grad(x.T,deg)
    measure_grad = [np.concatenate([measure_grad(k), np.ones((1,samples,deg+1))], axis=0) for k in range(order)] 
    res = np.array([v.evaluate(m) for m in measure_grad])
    assert res.shape ==  (order,samples) 
    return res