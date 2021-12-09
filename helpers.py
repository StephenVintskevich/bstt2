#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:35:59 2021

@author: goette
"""
import numpy as np


def fermi_pasta_ulam(number_of_oscillators, number_of_snapshots):
    """Fermi–Pasta–Ulam problem.
    Generate data for the Fermi–Pasta–Ulam problem represented by the differential equation
        d^2/dt^2 x_i = (x_i+1 - 2x_i + x_i-1) + 0.7((x_i+1 - x_i)^3 - (x_i-x_i-1)^3).
    See [1]_ for details.
    Parameters
    ----------
    number_of_oscillators: int
        number of oscillators
    number_of_snapshots: int
        number of snapshots
    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives
    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define random snapshot matrix
    snapshots = 2 * np.random.rand(number_of_oscillators, number_of_snapshots) - 1

    # compute derivatives
    derivatives = np.zeros((number_of_oscillators, number_of_snapshots))
    for j in range(number_of_snapshots):
        derivatives[0, j] = snapshots[1, j] - 2 * snapshots[0, j] + 0.7 * (
                (snapshots[1, j] - snapshots[0, j]) ** 3 - snapshots[0, j] ** 3)
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = snapshots[i + 1, j] - 2 * snapshots[i, j] + snapshots[i - 1, j] + 0.7 * (
                    (snapshots[i + 1, j] - snapshots[i, j]) ** 3 - (snapshots[i, j] - snapshots[i - 1, j]) ** 3)
        derivatives[-1, j] = - 2 * snapshots[-1, j] + snapshots[-2, j] + 0.7 * (
                -snapshots[-1, j] ** 3 - (snapshots[-1, j] - snapshots[-2, j]) ** 3)

    return snapshots, derivatives

def massive_particles(number_of_equations, number_of_samples,G,m,r):
    # define random snapshot matrix
    x = r*(2 * np.random.rand(number_of_equations, number_of_samples) - 1)

    # compute derivatives
    xdot = np.zeros((number_of_equations, number_of_samples))
    for j in range(number_of_samples):
        for i in range(number_of_equations):
            for k in range(number_of_equations):
                if k!=i:
                    diff = x[i,j]-x[k,j]
                    xdot[i, j] -= G*m[i]*m[k]/(np.abs(diff)**3)*diff

    return x, xdot

def testode(t,x):
    n = len(x)// 2
    assert 2*n == len(x)
    y = x[:n]
    v = x[n:]
    return np.concatenate([v,np.sin(y)])


def lennardJonesParam(t,x,sigma):
    n = len(x)// 2
    assert 2*n == len(x)
    y = x[:n]
    v = x[n:]
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += np.sign(y[i]-y[j])*6/sigma[i,j]*((sigma[i,j]/np.abs(y[i]-y[j]))**13 -(sigma[i,j]/np.abs(y[i]-y[j]))**7  )
    v = v.reshape(-1)
    return np.concatenate([v,res])

def lennardJonesParam2(x,sigma):
    n = len(x)
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += np.sign(x[i]-x[j])*6/sigma[i,j]*((sigma[i,j]/np.abs(x[i]-x[j]))**13 -(sigma[i,j]/np.abs(x[i]-x[j]))**7  )
    return res

def randNum(a,b):
    return (b-a)*np.random.rand()+a

def lennardJonesSamples(order,number_of_samples,c,sigma):
    samples = []
    count = 0
    while(count < number_of_samples):
        
        sample = np.sort(c*order*(2 * np.random.rand(order) - 1))
        if np.all(np.diff(sample)>=1.05):
            samples.append(sample)
            count+=1
    samples = np.column_stack(samples)
    derivatives = []
    for k in range(number_of_samples):
        derivatives.append(lennardJonesParam2(samples[:,k],sigma))
    derivatives = np.column_stack(derivatives)
    
    assert samples.shape == derivatives.shape
    return samples,derivatives


def lennardjonesenergy(x,sigma):
    n = len(x)// 2
    assert 2*n == len(x)
    y = x[:n]
    v = x[n:]
    energy = 0
    for i in range(n):
        energy += 0.5 *v[i]**2
        for j in range(i):
            energy+= (sigma[i,j]/(y[i]-y[j]))**12 - (sigma[i,j]/(y[i]-y[j]))**6
    return energy

def magneticDipolesParam(t,y,M,x,I):
    n = len(y)// 2
    assert 2*n == len(y)
    phi = y[:n]
    J = y[n:]
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += M[i]*M[j]/(np.abs(x[i]-x[j])**3)*np.sin(phi[j]-phi[i])
    J = J.reshape(-1)
    return np.concatenate([J/I,res])

def magneticDipolesParam2(phi,M,x,I):
    n = len(phi)
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += M[i]*M[j]/(np.abs(x[i]-x[j])**3)*np.sin(phi[j]-phi[i])
    return res

def magneticDipolesSamples(order,number_of_samples,M,x,I):
    samples = np.pi*(2 * np.random.rand(order, number_of_samples) - 1)
    derivatives = []
    for k in range(number_of_samples):
        derivatives.append(magneticDipolesParam2(samples[:,k],M,x,I))
    derivatives = np.column_stack(derivatives)
    
    assert samples.shape == derivatives.shape
    return samples,derivatives

def selectionMatrix3(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[2,0] = 1
        Smat[1,1] = 1
        for i in range(2,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 1:
        Smat = np.zeros([4,_numberOfEquations])
        Smat[3,0] = 1
        Smat[2,1] = 1
        Smat[1,2] = 1
        for i in range(3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 2:
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-1):
            Smat[3,i] = 1
        Smat[2,k-1] = 1
        Smat[1,k] = 1
        Smat[0,k+1] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k-1):
            Smat[2,i] = 1
        Smat[1,k-1] = 1
        Smat[0,k] = 1
    elif k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([5,_numberOfEquations])
        for i in range(0,k-1):
            Smat[4,i] = 1
        Smat[3,k-1] = 1
        Smat[2,k] = 1
        Smat[1,k+1] = 1
        for i in range(k+2,_numberOfEquations):
            Smat[0,i] = 1
    return Smat

def selectionMatrix4(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([4,_numberOfEquations])
        Smat[3,k] = 1
        Smat[2,k+1] = 1
        Smat[1,k+2] = 1
        for i in range(k+3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 1:
        Smat = np.zeros([5,_numberOfEquations])
        Smat[4,k-1] = 1
        Smat[3,k] = 1
        Smat[2,k+1] = 1
        Smat[1,k+2] = 1
        for i in range(k+3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 2:
        Smat = np.zeros([6,_numberOfEquations])
        Smat[5,k-2] = 1
        Smat[4,k-1] = 1
        Smat[3,k] = 1
        Smat[2,k+1] = 1
        Smat[1,k+2] = 1
        for i in range(k+3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 3:
        Smat = np.zeros([6,_numberOfEquations])
        for i in range(0,k-2):
            Smat[5,i] = 1
        Smat[4,k-2] = 1
        Smat[3,k-1] = 1
        Smat[2,k] = 1
        Smat[1,k+1] = 1
        Smat[0,k+2] = 1
    elif k == _numberOfEquations - 2:
         Smat = np.zeros([5,_numberOfEquations])
         for i in range(0,k-2):
             Smat[4,i] = 1
         Smat[3,k-2] = 1
         Smat[2,k-1] = 1
         Smat[1,k] = 1
         Smat[0,k+1] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-2):
            Smat[3,i] = 1
        Smat[2,k-2] = 1
        Smat[1,k-1] = 1
        Smat[0,k] = 1
    elif k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([7,_numberOfEquations])
        for i in range(0,k-2):
            Smat[6,i] = 1
        Smat[5,k-2] = 1
        Smat[4,k-1] = 1
        Smat[3,k] = 1
        Smat[2,k+1] = 1
        Smat[1,k+2] = 1
        for i in range(k+3,_numberOfEquations):
            Smat[0,i] = 1
    return Smat
