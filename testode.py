#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:55:56 2021

@author: goette
"""

import numpy as np
from helpers import testode,lennardjones,lennardjonesenergy, magneticDipolesParam
from scipy.integrate import RK45
import matplotlib.pyplot as plt

n = 4
t0 = 0
y0 = np.array([-4,-2,2,4,0,0,0,0])
t_bound = 10
first_step = 0.01
max_step = 0.01
sigma = np.ones([n,n])
M = 4*np.ones(n)
I = 2*np.ones(n)
x = np.linspace(0,2*(n-1),n)
print(M.shape, I.shape, x.shape)

print(f"energy: {lennardjonesenergy(y0,sigma)}")

def lennardjonessigma(t,x):
    return lennardjones(t,x,sigma)

def magneticDipoles(t,y):
    return magneticDipolesParam(t,y,M,x,I)


sol = RK45(magneticDipoles,t0,y0,t_bound,first_step=first_step,vectorized=True,max_step=max_step)
t = []
state = []
for step in range(int(t_bound/max_step)):
    t.append(sol.t)
    state.append(sol.y)
    sol.step()
    
y = np.column_stack(state)
print(y.shape)

plt.figure()
plt.plot(t,y[0,:],label='x0')
plt.plot(t,y[1,:],label='x1')
plt.plot(t,y[2,:],label='x2')
plt.plot(t,y[3,:],label='x3')

plt.legend()
plt.show()

plt.figure()

plt.plot(t,y[4,:],label='v0')
plt.plot(t,y[5,:],label='v1')
plt.plot(t,y[6,:],label='v2')
plt.plot(t,y[7,:],label='v3')
plt.legend()
plt.show()