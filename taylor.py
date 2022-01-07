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
degree = 7
exp = 3
f = lambda t : 1/(t)**(2*exp+1)-1/(t)**(exp+1)
taylor = approximate_taylor_polynomial(f, 1.5, degree,0.5,order=degree+2)




plt.figure()
plt.plot(t,f(t))
plt.plot(t,taylor(t-1.5))
plt.show()

# plt.figure()
# plt.plot(t,f(t)-taylor(t-2.0))
# plt.show()