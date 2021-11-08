import os
from math import comb, factorial
from itertools import product

import numpy as np

from misc import random_homogenous_polynomial_sum,  legendre_measures, Gramian, HkinnerLegendre  #, hermite_measures
from als import ALS





numSteps = 200
numTrials = 200
maxSweeps = 2000
maxIter = 100

order = 6
degree = 7
maxGroupSize = 1  #NOTE: This is the block size needed to represent an arbitrary polynomial.

f = lambda xs: np.exp(-np.linalg.norm(xs, axis=1)**2)  #NOTE: This functions gets peakier for larger M!
sampleSizes = np.unique(np.geomspace(1e1, 1e6, numSteps).astype(int))
testSampleSize = int(1e4)


test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
test_values = f(test_points)

bstt_sum = random_homogenous_polynomial_sum([degree]*order, degree, maxGroupSize)

localGramians = [Gramian(degree+1,HkinnerLegendre(1)) for i in range(order)]
localGramians.append(np.ones([degree+1,degree+1]))
solver = ALS(bstt_sum, augmented_test_measures,  test_values,_localGramians=localGramians,_verbosity=1)
solver.maxSweeps = maxSweeps
solver.targetResidual = 1e-16
solver.run()
values = bstt_sum.evaluate(augmented_test_measures)
print(np.linalg.norm(values -  test_values) / np.linalg.norm(test_values))
