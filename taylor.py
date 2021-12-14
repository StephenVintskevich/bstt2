#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:02:49 2021

@author: goette
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial

#SeriesData[x, 1.5, {-0.0533894, 0.228598, -0.520531, 0.763979, -0.580588, -0.626251, 3.62636, -9.27498, 18.3974, -31.666, 49.4915, -71.9437, 98.7162, -129.136, 162.214, -196.729, 231.33, -264.646, 295.383}, 0, 19, 1]
t = np.linspace(1.0,2,200)
degree = 8

y = 1/(t)**13-1/(t)**7
f = lambda t : 1/(t)**13-1/(t)**7
taylor = approximate_taylor_polynomial(f, 1.5, degree, 10,order=degree+2)
coeff = np.flip(np.array([-0.0533894, 0.228598, -0.520531, 0.763979, -0.580588, -0.626251, 3.62636, -9.27498, 18.3974, -31.666, 49.4915, -71.9437, 98.7162, -129.136, 162.214, -196.729, 231.33, -264.646, 295.383]))
coeff = coeff[0:]
print(len(coeff))
print(np.polyval(coeff, 0))

plt.figure()
plt.plot(t,y)


plt.plot(t,np.polyval(coeff, t-1.5))
plt.show()

plt.figure()


plt.plot(t,y-np.polyval(coeff, t-1.5))
plt.show()