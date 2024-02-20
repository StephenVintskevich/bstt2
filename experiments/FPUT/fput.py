import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import numpy as np 
from misc import  __block,legendre_measures,random_homogenous_polynomial_sum_system2
from helpers import fermi_pasta_ulam,fermi_pasta_ulam2,SMat
from als import ALSSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

folder = "experiments/FPUT/"

order = 50  
degree = 3
maxGroupSize = 2
#interaction = [3]+ [4] + [5]*(order-4) + [4] + [3]
interaction = 5
#interaction = [2]*order

trainSampleSize = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600]
maxSweeps=10
ranks = [4]*(order-1)

S = SMat(interaction,order)
res = np.zeros(len(trainSampleSize))

for (ii, sampleSize) in enumerate(trainSampleSize):        
    kappa = 2 * np.random.rand(order)
    beta = 1.4 * np.random.rand(order)
    #train_points,train_values = fermi_pasta_ulam(order,sampleSize)
    train_points,train_values = fermi_pasta_ulam2(order,sampleSize,kappa,beta)
    print(train_points.shape)
    print(train_values.shape)
    train_measures = legendre_measures(train_points, degree)
    print(train_measures.shape)
    augmented_train_measures = np.concatenate([train_measures, np.ones((1,sampleSize,degree+1))], axis=0)

    bstt = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
    print(f"DOFS: {bstt.dofs()}")
    print(f"Ranks: {bstt.ranks}")
    print(f"Interaction: {bstt.interactions}")


   
    solver = ALSSystem2(bstt, augmented_train_measures,  train_values,_verbosity=1)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-6
    solver.maxGroupSize=maxGroupSize
    solver.run()

    testSampleSize = int(2e4)
    test_points,test_values =fermi_pasta_ulam2(order,testSampleSize,kappa,beta)
    test_measures = legendre_measures(test_points, degree)
    augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)


    values = bstt.evaluate(augmented_test_measures)
    values2 = bstt.evaluate(augmented_train_measures)
    newres = np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)
    print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
    
    res[ii] = newres

np.save(folder+"50ptcls_2-12h_smpls_even_noneven.data",res)





