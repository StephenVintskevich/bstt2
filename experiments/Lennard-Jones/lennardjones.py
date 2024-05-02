import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import numpy as np 
from misc import  __block, random_homogenous_polynomial_sum_system,random_homogenous_polynomial_sum_system2,zeros_homogenous_polynomial_sum_system,monomial_measures,legendre_measures,Gramian, HkinnerLegendre,random_full_system
from helpers import lennardJonesSamples,lennardJonesSamplesMod, SMat
from als import ALSSystem2
block = __block()

import warnings
warnings.filterwarnings("ignore")

folder = "experiments/Lennard-Jones/"

# Parameters
orders = [10] # the number of particles
degrees = [8] # max polynomial degree of the basis
#maxGroupSize = [1]+[2] +[3]*(order-4)+[2]+[1]
maxGroupSizes = [4,6,8]
interactions = [5]
trainSampleSizes = [1000*i for i in range(1,4)]
reps = 6
maxSweeps=8
c = 1.0
res = np.zeros((len(orders),len(trainSampleSizes),len(interactions), len(maxGroupSizes), len(degrees), reps))
#Model Parameters
exp = 2
mod = 1
for (ii,order) in enumerate(orders):
    for (jj,trainSampleSize) in enumerate(trainSampleSizes):
        for (kk,interaction) in enumerate(interactions):
            for (ll,maxGroupSize) in enumerate(maxGroupSizes):
                for (mm,degree) in enumerate(degrees):
                    for nn in range(reps):
                        print(f'Starting Order" {order} Samples {trainSampleSize} Interaction {interaction} MaxGroupSize {maxGroupSize} Rep {nn+1}')
                        sigma = np.ones([order,order])
        
                        #Selection Matrix
                        S = SMat(interaction,order)
                        print(S)
                    
                    
                        coeffs = random_homogenous_polynomial_sum_system2([degree]*order,degree,maxGroupSize,interaction,S)
                        print(f"DOFS: {coeffs.dofs()}")
                        print(f"Ranks: {coeffs.ranks}")
                        print(f"Interaction: {coeffs.interactions}")
                        
                        
                        #train_points,train_values = lennardJonesSamples(order,trainSampleSize,c,sigma,exp)
                        train_points,train_values = lennardJonesSamplesMod(order,trainSampleSize,c,exp,mod)
                        print(f"Finished drawing samples {trainSampleSize}")
                        #train_measures = legendre_measures(train_points, degree,np.float(-c*order),np.float(c*order))
                        train_measures = legendre_measures(train_points, degree)
                        augmented_train_measures = np.concatenate([train_measures, np.ones((1,trainSampleSize,degree+1))], axis=0)
                        
                        
                        
                        solver = ALSSystem2(coeffs, augmented_train_measures,  train_values,_verbosity=1)
                        solver.maxSweeps = maxSweeps
                        solver.targetResidual = 1e-6
                        solver.maxGroupSize=maxGroupSize
                        solver.run()
                        
                        testSampleSize = int(2e4)
                        test_points,test_values = lennardJonesSamplesMod(order,testSampleSize,c,exp,mod)
                        #test_measures =  legendre_measures(test_points, degree,np.float(-c*order),np.float(c*order))
                        test_measures =  legendre_measures(test_points, degree)
                        augmented_test_measures = np.concatenate([test_measures, np.ones((1,testSampleSize,degree+1))], axis=0)  # measures.shape == (order,N,degree+1)
                        
                        
                        values = coeffs.evaluate(augmented_test_measures)
                        values2 = coeffs.evaluate(augmented_train_measures)
                        print("L2: ",np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)," on training data: ",np.linalg.norm(values2 -  train_values) / np.linalg.norm(train_values))
                        res[ii,jj,kk,ll,mm,nn] = np.linalg.norm(values -  test_values) / np.linalg.norm(test_values)
                        for k in range(order):
                            print("L2: ",np.linalg.norm(values[:,k] -  test_values[:,k]) / np.linalg.norm(test_values[:,k]))
                            
np.save(folder+'data/10ptcls_gp4-8ev_int5_smpls1-3_nostop.data',res)
                            
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    