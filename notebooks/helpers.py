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

    return snapshots.T, derivatives.T

def fermi_pasta_ulam2(number_of_oscillators, number_of_snapshots, kappa, beta):
    """"This is the non-symmetrical version of the problem with
    spring constants kappa and non-linearity coefficients beta varying from site to site"""
    
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
        derivatives[0, j] = kappa[1] * (snapshots[1, j] - snapshots[0, j]) - kappa[0] * snapshots[0, j] + beta[1] * (
            snapshots[1, j] - snapshots[0, j]) ** 3 - beta[0] * snapshots[0, j] ** 3
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = kappa[i+1] * (snapshots[i + 1, j] - snapshots[i, j]) - kappa[i] * (snapshots[i, j] - snapshots[i - 1, j]) + beta[i+1] * (
                    snapshots[i + 1, j] - snapshots[i, j]) ** 3 - beta[i] * (snapshots[i, j] - snapshots[i - 1, j]) ** 3
        derivatives[-1, j] = - kappa[-1] * snapshots[-1, j] - kappa[-2] * (snapshots[-1, j] - snapshots[-2, j]) + beta[-1] * (
                -snapshots[-1, j]) ** 3 - beta[-2] * (snapshots[-1, j] - snapshots[-2, j]) ** 3

    return snapshots.T, derivatives.T


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


def lennardJonesParam(t,x,sigma,exp):
    n = len(x)// 2
    assert 2*n == len(x)
    y = x[:n]
    v = x[n:]
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += np.sign(y[i]-y[j])*6/sigma[i,j]*((sigma[i,j]/np.abs(y[i]-y[j]))**(2*exp+1) -(sigma[i,j]/np.abs(y[i]-y[j]))**(exp+1)  )
    v = v.reshape(-1)
    return np.concatenate([v,res])

def lennardJonesParam2(x,sigma,exp):
    n = len(x)
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] += np.sign(x[i]-x[j])*6/sigma[i,j]*((sigma[i,j]/np.abs(x[i]-x[j]))**(2*exp+1) -(sigma[i,j]/np.abs(x[i]-x[j]))**(exp+1)  )
    return res

def lennardJonesParam2Mod(x,exp):
    n = len(x)
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                #res[i] +=((1/(x[i]-x[j]))**(2*exp+1) -(1/(x[i]-x[j]))**(exp+1)  )
                res[i] += np.sign(x[i]-x[j])*((1/np.abs(x[i]-x[j]))**(2*exp+1) -(1/np.abs(x[i]-x[j]))**(exp+1)  )
        if i > 0: 
            res[i]*= (x[i]-x[i-1])**(2*exp+1)
        #if i > 1: 
        #    res[i]*= (x[i]-x[i-2])**(2*exp+1)
        if i < n-1:
            res[i]*= (x[i]-x[i+1])**(2*exp+1)
        #if i < n-2:
        #    res[i]*= (x[i]-x[i+2])**(2*exp+1)
    return res


def lennardJonesParam3Mod(x,exp):
    n = len(x)
    res = np.zeros([n])
    for i in range(n):
        for j in range(n):
            if i != j:
                res[i] +=((1/(x[i]-x[j]))**(2*exp+1) -(1/(x[i]-x[j]))**(exp+1)  )
        if i > 0: 
            res[i]*= (x[i]-x[i-1])**(2*exp+1)
        #if i > 1: 
        #    res[i]*= (x[i]-x[i-2])**(2*exp+1)
        if i < n-1:
            res[i]*= (x[i]-x[i+1])**(2*exp+1)
        #if i < n-2:
        #    res[i]*= (x[i]-x[i+2])**(2*exp+1)
    return res


def randNum(a,b):
    return (b-a)*np.random.rand()+a


def lennardJonesSamplesMod(order,number_of_samples,c,exp,mod=0):
    samples = []
    count = 0
    a=1.0
    b=2.0
    while(count < number_of_samples):        
        sample = c*order/2*np.random.rand(1) - c*order
        for i in range(0,order-1):
            nextSample = sample[-1]+a+np.random.rand(1)*(b-a)
            sample = np.append(sample,nextSample)
        samples.append(sample)
        count+=1
    samples = np.column_stack(samples)
    #samples =  (2 * np.random.rand(order,number_of_samples) - 1)
    derivatives = []
    for k in range(number_of_samples):
        if mod == 0:
            derivatives.append(lennardJonesParam2Mod(samples[:,k],exp))
        else:
            derivatives.append(lennardJonesParam3Mod(samples[:,k],exp))
    derivatives = np.column_stack(derivatives)
    
    assert samples.shape == derivatives.shape
    return samples.T,derivatives.T

def lennardJonesSamples(order,number_of_samples,c,sigma,exp):
    samples = []
    count = 0
    a=1.05
    b=1.95
    while(count < number_of_samples):        
        sample = c*order*(2 * np.random.rand(1) - 1)
        for i in range(0,order-1):
            nextSample = sample[-1]+a+np.random.rand(1)*(b-a)
            sample = np.append(sample,nextSample)
        samples.append(sample)
        count+=1

    samples = np.column_stack(samples)
    derivatives = []
    for k in range(number_of_samples):
        derivatives.append(lennardJonesParam2(samples[:,k],sigma,exp))
    derivatives = np.column_stack(derivatives)
    
    assert samples.shape == derivatives.shape
    return samples.T,derivatives.T


def lennardjonesenergy(x,sigma,exp):
    n = len(x)// 2
    assert 2*n == len(x)
    y = x[:n]
    v = x[n:]
    energy = 0
    for i in range(n):
        energy += 0.5 *v[i]**2
        for j in range(i):
            energy+= (sigma[i,j]/(y[i]-y[j]))**(2*exp) - (sigma[i,j]/(y[i]-y[j]))**exp
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
    return samples.T,derivatives.T

def selectionMatrix0(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([2,_numberOfEquations])
        for i in range(0,k):
            Smat[0,i] = 1
        Smat[1,k] = 1
        for i in range(k+1,_numberOfEquations):
            Smat[0,i] = 1
    return Smat

def selectionMatrix1(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([2,_numberOfEquations])
        Smat[1,0] = 1
        for i in range(1,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([2,_numberOfEquations])
        for i in range(0,k):
            Smat[1,i] = 1
        Smat[0,k] = 1
    elif k == _numberOfEquations:
         Smat = np.zeros([1,_numberOfEquations])
         for i in range(0,_numberOfEquations):
             Smat[0,i] = 1
    else:        
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k):
            Smat[2,i] = 1
        Smat[1,k] = 1
        for i in range(k+1,_numberOfEquations):
            Smat[0,i] = 1
    return Smat

def selectionMatrix2(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[2,0] = 1
        Smat[1,1] = 1
        for i in range(2,_numberOfEquations):
            Smat[0,i] = 1
    elif k == 1:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[1,0] = 1
        Smat[2,1] = 1
        Smat[1,2] = 1
        for i in range(3,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 2:
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k-1):
            Smat[2,i] = 1
        Smat[0,k-1] = 1
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
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-1):
            Smat[3,i] = 1
        Smat[1,k-1] = 1
        Smat[2,k] = 1
        Smat[1,k+1] = 1
        for i in range(k+2,_numberOfEquations):
            Smat[0,i] = 1
    return Smat
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

def selectionMatrix3dense(k,_numberOfEquations):
    assert k >= 0 and k < _numberOfEquations+1
    if k == 0:
        Smat = np.zeros([3,_numberOfEquations])
        Smat[2,0] = 1
        Smat[1,1] = 1
        for i in range(2,_numberOfEquations):
            Smat[0,i] = 1
    elif k == _numberOfEquations - 1:
        Smat = np.zeros([3,_numberOfEquations])
        for i in range(0,k-1):
            Smat[2,i] = 1
        Smat[1,k-1] = 1
        Smat[0,k] = 1
    else:        
        Smat = np.zeros([4,_numberOfEquations])
        for i in range(0,k-1):
            Smat[0,i] = 1
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

def SMat(interaction,order):
    # interaction is the number of different components per mode
    assert interaction >= 3
    lower = (interaction - 2)//2
    upper = (interaction - 3)//2
    S = np.zeros([order,order+1])
    decrease = np.linspace(interaction-2,1,interaction-2).astype(int)
    for eq in range(order):
        for pos in range(order+1):
            if pos < eq-lower:
                S[eq,pos] = interaction - 1 
            elif pos > eq + upper:
                S[eq,pos] = 0
            else:
                S[eq,pos] = decrease[pos-eq+lower]
        S[eq,order] = 0
    return S