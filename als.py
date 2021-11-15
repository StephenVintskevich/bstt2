#NOTE: This implementation is not meant to be memory efficient or fast but rather to test the approximation capabilities of the proposed model class.
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from scipy.linalg import block_diag,null_space
from bstt import Block, BlockSparseTensor
import sys
from matplotlib import pyplot as plt


class ALS(object):
    def __init__(self, _bstt, _measurements, _values, _localL2Gramians=None, _localH1Gramians=None,_maxGroupSize=3, _verbosity=0):
        self.bstt = _bstt
        assert isinstance(_measurements, np.ndarray) and isinstance(_values, np.ndarray)
        assert _maxGroupSize > 0
        assert len(_measurements) == self.bstt.order
        assert all(compMeas.shape == (len(_values), dim) for compMeas, dim in zip(_measurements, self.bstt.dimensions))
        self.measurements = _measurements
        self.values = _values
        self.verbosity = _verbosity
        self.maxSweeps = 100
        self.targetResidual = 1e-8
        self.minDecrease = 1e-4
        self.increaseRanks = False
        self.smin = 0.01
        self.sminFactor = 0.01
        self.maxGroupSize = _maxGroupSize
        if (not _localH1Gramians): 
            self.localH1Gramians = [np.eye(d) for d in self.bstt.dimensions]
        else:
            assert isinstance(_localH1Gramians,list) and len(_localH1Gramians) == self.bstt.order
            for i in range(len(_localH1Gramians)):
                lG = _localH1Gramians[i]
                assert isinstance(lG, np.ndarray) and lG.shape == (self.bstt.dimensions[i],self.bstt.dimensions[i])
                eigs_tmp = np.around(np.linalg.eigvals(lG),decimals=14)
                assert np.all(eigs_tmp >=0) and np.allclose(lG, lG.T, rtol=1e-14, atol=1e-14)
            self.localH1Gramians =_localH1Gramians 
        
        if (not _localL2Gramians): 
             self.localL2Gramians = [np.eye(d) for d in self.bstt.dimensions]
        else:
            assert isinstance(_localL2Gramians,list) and len(_localL2Gramians) == self.bstt.order
            for i in range(len(_localL2Gramians)):
                lG = _localL2Gramians[i]
                assert isinstance(lG, np.ndarray) and lG.shape == (self.bstt.dimensions[i],self.bstt.dimensions[i])
                eigs_tmp = np.around(np.linalg.eigvals(lG),decimals=14)
                assert np.all(eigs_tmp >=0) and np.allclose(lG, lG.T, rtol=1e-14, atol=1e-14)
            self.localL2Gramians =_localL2Gramians
    
        self.leftStack = [np.ones((len(self.values),1))] + [None]*(self.bstt.order-1)
        self.rightStack = [np.ones((len(self.values),1))]
        self.leftH1GramianStack = [np.ones([1,1])]  + [None]*(self.bstt.order-1)
        self.rightH1GramianStack = [np.ones([1,1])]
        self.leftL2GramianStack = [np.ones([1,1])]  + [None]*(self.bstt.order-1)
        self.rightL2GramianStack = [np.ones([1,1])]

        self.bstt.assume_corePosition(self.bstt.order-1)
        while self.bstt.corePosition > 0:
            self.move_core('left')


    def move_core(self, _direction):
        assert len(self.leftStack) + len(self.rightStack) == self.bstt.order+1
        assert len(self.leftH1GramianStack) + len(self.rightH1GramianStack) == self.bstt.order+1
        assert len(self.leftL2GramianStack) + len(self.rightL2GramianStack) == self.bstt.order+1
        valid_stacks = all(entry is not None for entry in self.leftStack + self.rightStack)
        if self.verbosity >= 2 and valid_stacks:
            pre_res = self.residual()
        singValues = self.bstt.move_core(_direction)
        if _direction == 'left':
            self.leftStack.pop()
            self.leftH1GramianStack.pop()
            self.leftL2GramianStack.pop()
            self.rightStack.append(np.einsum('ler, ne, nr -> nl', self.bstt.components[self.bstt.corePosition+1], self.measurements[self.bstt.corePosition+1], self.rightStack[-1]))
            self.rightH1GramianStack.append(np.einsum('ijk, lmn, jm,kn -> il', self.bstt.components[self.bstt.corePosition+1],  self.bstt.components[self.bstt.corePosition+1], self.localH1Gramians[self.bstt.corePosition+1], self.rightH1GramianStack[-1]))
            self.rightL2GramianStack.append(np.einsum('ijk, lmn, jm,kn -> il', self.bstt.components[self.bstt.corePosition+1],  self.bstt.components[self.bstt.corePosition+1], self.localL2Gramians[self.bstt.corePosition+1], self.rightL2GramianStack[-1]))
            if self.verbosity >= 2:
                if valid_stacks:
                    print(f"move_core {self.bstt.corePosition+1} --> {self.bstt.corePosition}.  (residual: {pre_res:.2e} --> {self.residual():.2e})")
                else:
                    print(f"move_core {self.bstt.corePosition+1} --> {self.bstt.corePosition}.")
        elif _direction == 'right':
            if self.increaseRanks:
                slices =  self.bstt.getUniqueSlices(0)
                for i,slc in zip(range(len(slices)),slices):
                    if np.min(singValues[slc]) > self.smin and  slc.stop-slc.start < self.bstt.MaxSize(i,self.bstt.corePosition-1,self.maxGroupSize):
                        u = self.calculate_update(slc,'left')
                        self.bstt.increase_block(i,u,np.zeros([self.bstt.components[self.bstt.corePosition]]).shape[1:3],'left')

            self.rightStack.pop()
            self.rightH1GramianStack.pop()
            self.rightL2GramianStack.pop()
            self.leftStack.append(np.einsum('nl, ne, ler -> nr', self.leftStack[-1], self.measurements[self.bstt.corePosition-1], self.bstt.components[self.bstt.corePosition-1]))
            self.leftH1GramianStack.append(np.einsum('ijk, lmn, jm,il -> kn', self.bstt.components[self.bstt.corePosition-1],  self.bstt.components[self.bstt.corePosition-1], self.localH1Gramians[self.bstt.corePosition-1], self.leftH1GramianStack[-1]))
            self.leftL2GramianStack.append(np.einsum('ijk, lmn, jm,il -> kn', self.bstt.components[self.bstt.corePosition-1],  self.bstt.components[self.bstt.corePosition-1], self.localL2Gramians[self.bstt.corePosition-1], self.leftL2GramianStack[-1]))
            if self.verbosity >= 2:
                if valid_stacks:
                    print(f"move_core {self.bstt.corePosition-1} --> {self.bstt.corePosition}.  (residual: {pre_res:.2e} --> {self.residual():.2e})")
                else:
                    print(f"move_core {self.bstt.corePosition-1} --> {self.bstt.corePosition}.")
        else:
            raise ValueError(f"Unknown _direction. Expected 'left' or 'right' but got '{_direction}'")

    def residual(self):
        core = self.bstt.components[self.bstt.corePosition]
        L = self.leftStack[-1]
        E = self.measurements[self.bstt.corePosition]
        R = self.rightStack[-1]
        pred = np.einsum('ler,nl,ne,nr -> n', core, L, E, R)
        return np.linalg.norm(pred -  self.values) / np.linalg.norm(self.values)


    def calculate_update(self,slc,_direction):
        if _direction == 'left':
            Gramian = np.einsum('ij,kl->ijkl',self.leftH1GramianStack[-1],self.localH1Gramians[self.bstt.corePosition-1])
            n = Gramian.shape[0]*Gramian.shape[1]
            basis = np.eye(n)
            basis = basis.reshape(Gramian.shape)
            blocks = self.bstt.getAllBlocksOfSlice(self.corePosition-1,slc,2)
            for block in blocks:
                basis[block[0],block[1],block[0],block[1]] = 0
            basis=basis.reshape(n,n)
            left = self.bstt.components[self.bstt.corePosition-1][:,:,slc].reshape(n,-1)
            basis = np.concatenate([basis,left],axis = 1)
            ns = null_space(basis.T)
            assert ns.size > 0
            ns = ns.reshape(Gramian.shape[0:2],-1)
            projGramian = np.einsum('ijkl,ijm,kln->mn',Gramian,ns,ns)
            pGe, pGP = np.linalg.eigh(projGramian)
            return np.einsum('ijk,k->ij',ns,pGP[0])
        
        
    def microstep(self):
        if self.verbosity >= 2:
            pre_res = self.residual()

        core = self.bstt.components[self.bstt.corePosition]
        L = self.leftStack[-1]
        E = self.measurements[self.bstt.corePosition]
        R = self.rightStack[-1]
        coreBlocks = self.bstt.blocks[self.bstt.corePosition]
        N = len(self.values)
        
        LGH1 = self.leftH1GramianStack[-1]
        EGH1 = self.localH1Gramians[self.bstt.corePosition]
        RGH1 = self.rightH1GramianStack[-1]

        LGL2 = self.leftL2GramianStack[-1]
        EGL2 = self.localL2Gramians[self.bstt.corePosition]
        RGL2 = self.rightL2GramianStack[-1]
        assert np.allclose(LGL2, np.eye(LGL2.shape[0]), rtol=1e-14, atol=1e-14)

        Op_blocks = []
        Weights = []
        Tr_blocks = []
        for block in coreBlocks:
            op = np.einsum('nl,ne,nr -> nler', L[:, block[0]], E[:, block[1]], R[:, block[2]])
            Op_blocks.append(op.reshape(N,-1))      
            
            # update stacks after diagonalization of left and right gramian
            Le,LP = np.linalg.eigh(LGH1[block[0],block[0]])
            Ee,EP = np.linalg.eigh(EGH1[block[1],block[1]])
            Re,RP = np.linalg.eigh(RGH1[block[2],block[2]])
            #assert np.allclose(LP.T@LGH1[block[0],block[0]]@LP, np.diag(Le), rtol=1e-12, atol=1e-12)
            #assert np.allclose(RP.T@RGH1[block[2],block[2]]@RP, np.diag(Re), rtol=1e-12, atol=1e-12),RP.T@RGH1[block[2],block[2]]@RP
            
            RPL2 = RP.T@RGL2[block[2],block[2]]@RP
            Re = Re/np.diag(RPL2)
            
            tr = np.einsum('il,jm,kn->ijklmn',LP,EP,RP)
            Tr_blocks.append(tr.reshape(Op_blocks[-1].shape[1],Op_blocks[-1].shape[1]))
            
            Weights.extend( np.einsum('i,j,k->ijk',Le,Ee,Re).reshape(-1))
        Op = np.concatenate(Op_blocks, axis=1)
        Transform  = block_diag(*Tr_blocks)        
        assert np.allclose(Transform@Transform.T, np.eye(Transform.shape[0]), rtol=1e-14, atol=1e-14)
        
        Weights = np.sqrt(Weights)
        inverseWeightMatrix = np.diag(np.reciprocal(Weights))
 
        OpTr = Op@Transform@inverseWeightMatrix
        reg = LassoCV(eps=1e-7,cv=3, random_state=0,fit_intercept=False).fit(OpTr, self.values)
        Res = reg.coef_

        core[...] = BlockSparseTensor(Transform@inverseWeightMatrix@Res, coreBlocks, core.shape).toarray()

        if self.verbosity >= 2:
            print(f"microstep.  (residual: {pre_res:.2e} --> {self.residual():.2e})")

    def run(self):
        prev_residual = self.residual()
        self.smin = prev_residual*self.sminFactor
        if self.verbosity >= 1: print(f"Initial residuum: {prev_residual:.2e}")
        for sweep in range(self.maxSweeps):
            while self.bstt.corePosition < self.bstt.order-1:
                self.microstep()
                self.move_core('right')
            while self.bstt.corePosition > 0:
                self.microstep()
                self.move_core('left')

            residual = self.residual()
            if self.verbosity >= 1: print(f"[{sweep}] Residuum: {residual:.2e}")

            if residual < self.targetResidual:
                if self.verbosity >= 1:
                    print(f"Terminating (targetResidual reached)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            if residual > prev_residual:
                if self.verbosity >= 1:
                    print(f"Terminating (residual increases)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            if (prev_residual - residual) < self.minDecrease*residual:
                if self.verbosity >= 1:
                    print(f"Terminating (minDecrease reached)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            prev_residual = residual
            self.smin = prev_residual*self.sminFactor

        if self.verbosity >= 1: print(f"Terminating (maxSweeps reached)")
        if self.verbosity >= 1: print(f"Final residuum: {self.residual():.2e}")
