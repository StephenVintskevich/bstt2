from math import comb
import numpy as np
from scipy.sparse import block_diag, diags


class Block(tuple):
    def __init__(self, iterable):
        super(Block, self).__init__()
        for slc in self:
            assert isinstance(slc, slice) and isinstance(slc.start, int) and isinstance(slc.stop, int) and 0 <= slc.start < slc.stop and slc.step in (None,1)
            #NOTE: The final two conditions may restrict the structure of the blocks unnecessarily.

    def __str__(self):
        return "(" + ", ".join(f"{slc.start}:{slc.stop}" for slc in self) + ")"

    def __repr__(self):
        return "(" + ", ".join(f"{slc.start}:{slc.stop}" for slc in self) + ")"

    def __hash__(self):
        return hash(tuple((slc.start, slc.stop) for slc in self))

    def __eq__(self, _other):
        if not isinstance(_other, Block) or len(self) != len(_other):
            return False
        return all(s1.start == s2.start and s1.stop == s2.stop for s1,s2 in zip(self,_other))

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def shape(self):
        def slice_size(_slc):
            return _slc.stop - _slc.start
        return tuple(slice_size(slc) for slc in self)

    def disjoint(self, _other):
        assert isinstance(_other, Block) and len(self) == len(_other)
        def disjoint_slices(_slc1, _slc2):
            # return (_slc1.start <= _slc2.start and _slc1.stop <= _slc2.start) or _slc2.stop <= _slc1.start
            return _slc1.stop <= _slc2.start or _slc2.stop <= _slc1.start
        return any(disjoint_slices(slc1, slc2) for slc1,slc2 in zip(self, _other))

    def contains(self, _other):
        assert isinstance(_other, Block) and len(self) == len(_other)
        def contains_slice(_slc1, _slc2):
            return _slc1.start <= _slc2.start and _slc1.stop >= _slc2.stop
        return all(contains_slice(slc1, slc2) for slc1,slc2 in zip(self, _other))

    def coherent(self, _other):
        """
        Tests if all blocks are defined by non-overlapping slices.
        This is not a restriction since every two overlapping slices can be split into three new slices:
        The first slice contains the left non-overlapping part, the second contains the overlapping part and the third contains the right non-overlapping part.
        """
        #NOTE: The condition that the middle slices are coherent may restrict TT-Blocks unnecessarily.
        assert isinstance(_other, Block) and len(self) == len(_other)
        def disjoint_slices(_slc1, _slc2):
            return (_slc1.start <= _slc2.start and _slc1.stop <= _slc2.start) or _slc2.stop <= _slc1.start
        return all((slc1 == slc2 or disjoint_slices(slc1, slc2)) for slc1, slc2 in zip(self, _other))




class BlockSparseTensor(object):
    def __init__(self, _data, _blocks, _shape):
        assert isinstance(_data, np.ndarray) and _data.ndim == 1
        self.data = _data
        assert isinstance(_blocks, (list, tuple))
        self.blocks = [Block(block) for block in _blocks]
        self.shape = _shape
        shapeBlock = Block(slice(0,dim,1) for dim in self.shape)
        assert all(shapeBlock.contains(block) for block in self.blocks)
        assert sum(block.size for block in self.blocks) == self.data.size
        for i in range(len(self.blocks)):
            for j in range(i):
                assert self.blocks[i].disjoint(self.blocks[j]) and self.blocks[i].coherent(self.blocks[j])
        assert isinstance(self.shape, tuple) and np.all(np.array(self.shape) > 0)

    def dofs(self):
        return sum(blk.size for blk in self.blocks)

    def svd(self, _mode):
        """
        Perform an SVD along the `_mode`-th mode while retaining the the block structure.

        The considered matricisation has `_mode` as its rows and all other modes as its columns.
        If `U,S,Vt = X.svd(_mode)`, then `X == (U @ S) @[_mode] Vt` where `@[_mode]` is the contraction with the `_mode`-th mode ov Vt.
        """
        # SVD for _mode == 0
        # ==================
        # Consider a block sparse tensor X of shape (l,e,r).
        # We want to compute an SVD-like decomposition X = U @ S @[0] Vt such that the sparsity pattern is preserved.
        #
        # This means that:
        #     - U is a block-diagonal, orthogonal matrix.
        #     - The contraction U @[0] X does not modifying the sparsity structure
        #     - S is a diagonal matrix
        #     - The 0-(1,2)-matrification of Vt is orthogonal.
        #     - Vt is a block sparse tensor with the same sparsity structure as X.
        #       Equivalently, the 0-(1,2)-matrification of Vt has the same sparsity structure as the matrification of X.
        #
        # Assume that X contains non-zero blocks at the 3D-slices ((:a), scl_11[k], slc_12[k]) for k=1,...,K 
        # and ((a:), slc_21[l], scl_22[l]) for l=1,...,L. After a 1-(2,3)-matricisation we obtain a matrix 
        #    ┌       ┐
        #    │ X[:a] │
        #    │ X[a:] │
        #    └       ┘
        # of shape (l, e*r) and the slices take the form ((:a), scl_1[k]) and ((a:), slc_2[l]) where slc_1 and scl_2
        # are not proper slices but index arrays that select the non-zero columns of this matricisation.
        # Let X[:a] = UₐSₐVtₐ and m[a:] = UᵃSᵃVtᵃ. Then such a decomposition is given by
        #    ┌       ┐ ┌       ┐ ┌     ┐
        #    │ Uₐ    │ │ Sₐ    │ │ Vtₐ │
        #    │    Uᵃ │ │    Sᵃ │ │ Vtᵃ │
        #    └       ┘ └       ┘ └     ┘
        # It is easy to see that X = U @ S @[0] Vt and that U, S and Vt satisfy the first four properties.
        # To see this a permutation matrix Pₐ that sorts the the columns of X[:a] such that X[:a] Pₐ = [ 0 Y ], perform 
        # the SVD Y = Uʸ Sʸ Vtʸ and observe that X[:a] = Uʸ Sʸ [ 0 Vtʸ ] Ptₐ. Since Uʸ Sʸ is block-diagonal it preserves
        # the sparisity structure of [ 0 Vtʸ ] Ptₐ which has to be the same as the one of X[:a]. Since [ 0 Vtʸ ] Ptₐ is 
        # orthogonal we know that [ 0 Vtʸ ] Ptₐ = Vtₐ by the uniqueness of the SVD.
        # A similar argument holds true for X[a:] which proves the equivalent formulation of the fourth property in 
        # terms of the matrification of X.
        #
        # Note that this prove is constructive and provides a performant and numerically stable way to compute the SVD.
        #TODO: This can be done more efficiently.

        mSlices = sorted({(block[_mode].start, block[_mode].stop) for block in self.blocks})  #NOTE: slices are not hashable.

        # Check if the block structure can be retained.
        # It is necessary that there are no slices in the matricisation that are necessarily zero due to the block structure.
        assert mSlices[0][0] == 0, f"Hole found in mode {_mode}: (0:{mSlices[0][0]})"
        for j in range(len(mSlices)-1):
            assert mSlices[j][1] == mSlices[j+1][0], f"Hole found in mode {_mode}: ({mSlices[j][1]}:{mSlices[j+1][0]})"
        assert mSlices[-1][1] == self.shape[_mode], f"Hole found in mode {_mode}: ({mSlices[-1][1]}:{self.shape[_mode]})"
        # After matricisation the SVD is performed for each row-slice individually.
        # To ensure that the block structure is maintained the non-zero columns must outnumber the non-zero rows.
        for slc in mSlices:
            rows = slc[1]-slc[0]
            cols = sum(Block(blk).size for blk in self.blocks if blk[_mode].start == slc[0])     #NOTE: For coherent blocks blk[0].start == slc[0] implies equality of the slice.
            assert cols % rows == 0
            cols //= rows  # cols is the number of all non-zero columns of the `slc`-slice of the matricisation.
            assert rows <= cols, f"The {_mode}-matrification has too few non-zero columns (shape: {(rows, cols)}) for slice ({slc[0]}:{slc[1]})."  # of components[{m}][{reason[0].start}:{reason[0].stop}] has too few non-zero columns (rows: {reason[1][0]}, columns: {reason[1][1]})"

        def notMode(_tuple):
            return _tuple[:_mode] + _tuple[_mode+1:]

        # Store the blocks of the `_mode`-matrification (interpreted as a BlockSparseTensor).
        indices = np.arange(np.product(notMode(self.shape))).reshape(notMode(self.shape))
        mBlocks = []
        for slc in mSlices:
            idcs = [indices[notMode(blk)].reshape(-1) for blk in self.blocks if blk[_mode].start == slc[0]]
            idcs = np.sort(np.concatenate(idcs))
            mBlocks.append((slice(*slc), idcs))

        matricisation = np.moveaxis(self.toarray(), _mode, 0)
        mShape = matricisation.shape
        matricisation = matricisation.reshape(self.shape[_mode], -1)

        # Compute the row-block-wise SVD.
        U_blocks, S_blocks, Vt_blocks = [], [], []
        for block in mBlocks:
            u,s,vt = np.linalg.svd(matricisation[block], full_matrices=False)
            assert u.shape[0] == u.shape[1]  #TODO: Handle the case that a singular value is zero.
            U_blocks.append(u)
            S_blocks.append(s)
            Vt_blocks.append(vt)
        U = block_diag(U_blocks, format='bsr')
        S = diags([np.concatenate(S_blocks)], [0], format='dia')
        Vt = np.zeros(matricisation.shape)
        for block, Vt_block in zip(mBlocks, Vt_blocks):
            Vt[block] = Vt_block

        # Reshape Vt back into the original tensor shape.
        Vt = np.moveaxis(Vt.reshape(mShape), 0, _mode)
        #TODO: Is this equivalent to Vt = BlockSparseTensor(data, self.blocks, self.shape).toarray()?

        return U, S, Vt

    def toarray(self):
        ret = np.zeros(self.shape)
        slices = np.cumsum([0] + [block.size for block in self.blocks]).tolist()
        for e,block in enumerate(self.blocks):
            ret[block] = self.data[slices[e]:slices[e+1]].reshape(block.shape)
        return ret

    @classmethod
    def fromarray(cls, _array, _blocks):
        test = np.array(_array, copy=True)
        for block in _blocks:
            test[block] = 0
        assert np.all(test == 0), f"Block structure and sparsity pattern do not match."
        data = np.concatenate([_array[block].reshape(-1) for block in _blocks])
        return BlockSparseTensor(data, _blocks, _array.shape)


class BlockSparseTT(object):
    def __init__(self, _components, _blocks):
        """
        _components : list of ndarrays of order 3
            The list of component tensors for the TTTensor.
        _blocks : list of list of triples
            For the k-th component tensor _blocks[k] contains the list of its blocks of non-zero values:
                _blocks[k] --- list of non-zero blocks in the k-th component tensor
            Each block is represented by a triple of integers and slices:
                block = (slice(0,3), slice(0,4), slice(1,5))
                componentTensor[block] == componentTensor[0:3, 0:4, 1:5]
            To obtain the block this triple the slice in the component tensor:
                _blocks[k][l] --- The l-th non-zero block for the k-th component tensor.
                                  The coordinates are given by _components[k][_blocks[k][l]].

        NOTE: Later we can remove _components and augment each triple in _blocks by an array that contains the data in this block.
        """
        assert all(cmp.ndim == 3 for cmp in _components)
        assert _components[0].shape[0] == 1
        assert all(cmp1.shape[2] == cmp2.shape[0] for cmp1,cmp2 in zip(_components[:-1], _components[1:]))
        assert _components[-1].shape[2] == 1
        self.components = _components

        assert isinstance(_blocks, list) and len(_blocks) == self.order

        for m, (comp, compBlocks) in enumerate(zip(self.components, _blocks)):
            BlockSparseTensor.fromarray(comp, compBlocks)

        self.blocks = _blocks

        self.__corePosition = None
        self.verify()

    def verify(self):
        for e, (compBlocks, component) in enumerate(zip(self.blocks, self.components)):
            assert np.all(np.isfinite(component))
            cmp = np.array(component)
            for block in compBlocks:
                cmp[block] = 0
            assert np.allclose(cmp, 0), f"Component {e} does not satisfy the block structure. Error: {np.max(abs(cmp)):.2e}"

    def evaluate(self, _measures):
        assert self.order > 0 and len(_measures) == self.order
        n = len(_measures[0])
        ret = np.ones((n,1))
        for pos in range(self.order):
            ret = np.einsum('nl,ler,ne -> nr', ret, self.components[pos], _measures[pos])
        assert ret.shape == (n,1)
        return ret[:,0]

    @property
    def corePosition(self):
        return self.__corePosition

    def assume_corePosition(self, _position):
        assert 0 <= _position and _position < self.order
        self.__corePosition = _position

    @property
    def ranks(self):
        return [cmp.shape[2] for cmp in self.components[:-1]]

    @property
    def dimensions(self):
        return [cmp.shape[1] for cmp in self.components]

    @property
    def order(self):
        return len(self.components)
    
    def increase_block(self,_deg,_u,_v,_direction):
        if _direction == 'left':
            slices = self.getUniqueSlices(0)
            slc = slices[_deg]
            assert self.corePosition > 0
            assert self.MaxSize(_deg,self.corePosition-1) > slc.stop - slc.start 
            
            self.components[self.corePosition-1] = np.insert(self.components[self.corePosition-1],slc.stop,_u,axis=2)
            self.components[self.corePosition] = np.insert(self.components[self.corePosition],slc.stop,_v,axis=0)
            
            for i  in range(len(self.blocks[self.corePosition])):
                block = self.blocks[self.corePosition][i]
                if block[0] == slc:
                    self.blocks[self.corePosition][i] = Block((slice( block[0].start, block[0].stop+1),block[1],block[2]))
                if block[0].start > slc.start:
                    self.blocks[self.corePosition][i] = Block((slice( block[0].start+1, block[0].stop+1),block[1],block[2]))
            for i in range(len(self.blocks[self.corePosition-1])):
                block = self.blocks[self.corePosition-1][i]
                if block[2] == slc:
                    self.blocks[self.corePosition-1][i] = Block((block[0],block[1],slice(block[2].start,block[2].stop+1)))
                if block[2].start > slc.start:
                    self.blocks[self.corePosition-1][i] = Block((block[0],block[1],slice(block[2].start+1,block[2].stop+1)))

        elif _direction == 'right':
            slices = self.getUniqueSlices(2)
            slc = slices[_deg]
            assert self.corePosition < self.order-1
            assert self.MaxSize(_deg,self.corePosition-1) > slc.stop - slc.start 
            
            self.components[self.corePosition] = np.insert(self.components[self.corePosition],slc.stop,_u,axis=2)
            self.components[self.corePosition+1] = np.insert(self.components[self.corePosition+1],slc.stop,_v,axis=0)
            
            for i  in range(len(self.blocks[self.corePosition])):
                block = self.blocks[self.corePosition][i]
                if block[2] == slc:
                    self.blocks[self.corePosition][i] = Block((block[0],block[1],slice(block[2].start,block[2].stop+1)))
                if block[2].start > slc.start:
                    self.blocks[self.corePosition][i] = Block((block[0],block[1],slice(block[2].start+1,block[2].stop+1)))
            for i in  range(len(self.blocks[self.corePosition+1])):
                block = self.blocks[self.corePosition+1][i]
                if block[0] == slc:
                    self.blocks[self.corePosition+1][i] = Block((slice(block[0].start,block[0].stop+1),block[1],block[2]))
                if block[0].start > slc.start:
                    self.blocks[self.corePosition+1][i] = Block((slice(block[0].start+1,block[0].stop+1),block[1],block[2]))
       
        self.verify()
    
    
    
    def getUniqueSlices(self,mode):
        Blocks = self.blocks[self.corePosition]
        slices = []
        for block in Blocks:
            if block[mode] not in slices:
                slices.append(block[mode])
            if len(slices)>1:
                assert slices[-2].stop==slices[-1].start
        return sorted(slices)
    
    def getAllBlocksOfSlice(self,k,slc,mode):
        Blocks = self.blocks[k]
        blck = []
        for block in Blocks:
            if block[mode] == slc:
                blck.append(block)
        return blck

    
    
    def MaxSize(self,r,k,_maxGroupSize=np.inf):
        assert r >=0 and r < self.dimensions[0]
        assert k >=0 and k < self.order-1
        k+=1
        mr, mk = self.dimensions[0]-1-r, self.order-k
        return min(comb(k+r-1,k-1), comb(mk+mr-1, mk-1), _maxGroupSize)

    def move_core(self, _direction):
        assert isinstance(self.corePosition, int)
        assert _direction in ['left', 'right']
        S = None
        if _direction == 'left':
            assert 0 < self.corePosition

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(0)

            nextCore = self.components[self.corePosition-1]
            self.components[self.corePosition-1] = (nextCore.reshape(-1, nextCore.shape[2]) @ U @ S).reshape(nextCore.shape)
            self.components[self.corePosition] = Vt

            self.__corePosition -= 1
        else:
            assert self.corePosition < self.order-1

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(2)

            nextCore = self.components[self.corePosition+1]
            self.components[self.corePosition] = Vt
            self.components[self.corePosition+1] = (S @ U.T @ nextCore.reshape(nextCore.shape[0], -1)).reshape(nextCore.shape)

            self.__corePosition += 1
        self.verify()
        return S.diagonal()

    def dofs(self):
        return sum(BlockSparseTensor.fromarray(comp, blks).dofs() for comp, blks in zip(self.components, self.blocks))

    @classmethod
    def random(cls, _dimensions, _ranks, _blocks):
        assert len(_ranks)+1 == len(_dimensions)
        ranks = [1] + _ranks + [1]
        components = [np.zeros((leftRank, dimension, rightRank)) for leftRank, dimension, rightRank in zip(ranks[:-1], _dimensions, ranks[1:])]
        for comp, compBlocks in zip(components, _blocks):
            for block in compBlocks:
                comp[block] = np.random.randn(*comp[block].shape)
        return cls(components, _blocks)
    
    
    
class BlockSparseTTSystem(object):
    def __init__(self, _components, _blocks):
        """
        _components : list of ndarrays of order 3
            The list of component tensors for the TTTensor.
        _blocks : list of list of triples
            For the k-th component tensor _blocks[k] contains the list of its blocks of non-zero values:
                _blocks[k] --- list of non-zero blocks in the k-th component tensor
            Each block is represented by a triple of integers and slices:
                block = (slice(0,3), slice(0,4), slice(1,5))
                componentTensor[block] == componentTensor[0:3, 0:4, 1:5]
            To obtain the block this triple the slice in the component tensor:
                _blocks[k][l] --- The l-th non-zero block for the k-th component tensor.
                                  The coordinates are given by _components[k][_blocks[k][l]].

        NOTE: Later we can remove _components and augment each triple in _blocks by an array that contains the data in this block.
        """
        assert all(cmp.ndim == 4 for cmp in _components)
        assert _components[0].shape[0] == 1
        assert all(cmp1.shape[3] == cmp2.shape[0] for cmp1,cmp2 in zip(_components[:-1], _components[1:]))
        assert _components[-1].shape[3] == 1
        self.components = _components

        assert isinstance(_blocks, list) and len(_blocks) == self.order

        for m, (comp, compBlocks) in enumerate(zip(self.components, _blocks)):
            BlockSparseTensor.fromarray(comp, compBlocks)

        self.blocks = _blocks

        self.__corePosition = None
        self.verify()

    def verify(self):
        for e, (compBlocks, component) in enumerate(zip(self.blocks, self.components)):
            assert np.all(np.isfinite(component))
            cmp = np.array(component)
            for block in compBlocks:
                cmp[block] = 0
            assert np.allclose(cmp, 0), f"Component {e} does not satisfy the block structure. Error: {np.max(abs(cmp)):.2e}"

    def evaluate(self, _measures):
        assert self.order > 0 and len(_measures) == self.order
        n = len(_measures[0])
        ret = np.ones((n,1,self.order))
        for pos in range(self.order):
            ret = np.einsum('nld,lemr,md,ne -> nrd', ret, self.components[pos], self.selectionMatrix(pos), _measures[pos])
        assert ret.shape == (n,1,self.order)
        return ret[:,0,:]

    def selectionMatrix(self,k):
        assert k >= 0 and k < self.order
        interactionLength = self.interaction[k]
        Smat = np.zeros([interactionLength,self.order])
        row = 1
        for i in range(self.order):
            if np.abs(i-k) <= (interactionLength-1) // 2 and row < interactionLength:
                Smat[row,i] = 1
                row+=1
            else:
                Smat[0,i] = 1
        print(k,"\n",Smat)
        return Smat
    
    @property
    def corePosition(self):
        return self.__corePosition

    def assume_corePosition(self, _position):
        assert 0 <= _position and _position < self.order
        self.__corePosition = _position

    @property
    def ranks(self):
        return [cmp.shape[3] for cmp in self.components[:-1]]

    @property
    def dimensions(self):
        return [cmp.shape[1] for cmp in self.components]
    
    @property
    def interaction(self):
        return [cmp.shape[2] for cmp in self.components]

    @property
    def order(self):
        return len(self.components)
    
    def increase_block(self,_deg,_u,_v,_direction):
        if _direction == 'left':
            slices = self.getUniqueSlices(0)
            slc = slices[_deg]
            assert self.corePosition > 0
            assert self.MaxSize(_deg,self.corePosition-1) > slc.stop - slc.start 
            
            self.components[self.corePosition-1] = np.insert(self.components[self.corePosition-1],slc.stop,_u,axis=3)
            self.components[self.corePosition] = np.insert(self.components[self.corePosition],slc.stop,_v,axis=0)
            
            for i  in range(len(self.blocks[self.corePosition])):
                block = self.blocks[self.corePosition][i]
                if block[0] == slc:
                    self.blocks[self.corePosition][i] = Block((slice( block[0].start, block[0].stop+1),block[1],block[2],block[3]))
                if block[0].start > slc.start:
                    self.blocks[self.corePosition][i] = Block((slice( block[0].start+1, block[0].stop+1),block[1],block[2],block[3]))
            for i in range(len(self.blocks[self.corePosition-1])):
                block = self.blocks[self.corePosition-1][i]
                if block[3] == slc:
                    self.blocks[self.corePosition-1][i] = Block((block[0],block[1],block[2],slice(block[3].start,block[3].stop+1)))
                if block[3].start > slc.start:
                    self.blocks[self.corePosition-1][i] = Block((block[0],block[1],block[2],slice(block[3].start+1,block[3].stop+1)))

        elif _direction == 'right':
            slices = self.getUniqueSlices(3)
            slc = slices[_deg]
            assert self.corePosition < self.order-1
            assert self.MaxSize(_deg,self.corePosition-1) > slc.stop - slc.start 
            
            self.components[self.corePosition] = np.insert(self.components[self.corePosition],slc.stop,_u,axis=3)
            self.components[self.corePosition+1] = np.insert(self.components[self.corePosition+1],slc.stop,_v,axis=0)
            
            for i  in range(len(self.blocks[self.corePosition])):
                block = self.blocks[self.corePosition][i]
                if block[3] == slc:
                    self.blocks[self.corePosition][i] = Block((block[0],block[1],block[2],slice(block[3].start,block[3].stop+1)))
                if block[3].start > slc.start:
                    self.blocks[self.corePosition][i] = Block((block[0],block[1],block[2],slice(block[3].start+1,block[3].stop+1)))
            for i in  range(len(self.blocks[self.corePosition+1])):
                block = self.blocks[self.corePosition+1][i]
                if block[0] == slc:
                    self.blocks[self.corePosition+1][i] = Block((slice(block[0].start,block[0].stop+1),block[1],block[2],block[3]))
                if block[0].start > slc.start:
                    self.blocks[self.corePosition+1][i] = Block((slice(block[0].start+1,block[0].stop+1),block[1],block[2],block[3]))

        self.verify()
    
    
    
    def getUniqueSlices(self,mode):
        Blocks = self.blocks[self.corePosition]
        slices = []
        for block in Blocks:
            if block[mode] not in slices:
                slices.append(block[mode])
            if len(slices)>1:
                assert slices[-2].stop==slices[-1].start
        return sorted(slices)
    
    def getAllBlocksOfSlice(self,k,slc,mode):
        Blocks = self.blocks[k]
        print(Blocks)
        blck = []
        for block in Blocks:
            if block[mode] == slc:
                blck.append(block)
        return blck

    
    
    def MaxSize(self,r,k,_maxGroupSize=np.inf):
        assert r >=0 and r < self.dimensions[0]
        assert k >=0 and k < self.order-1
        k+=1
        mr, mk = self.dimensions[0]-1-r, self.order-k
        return min(comb(k+r-1,k-1), comb(mk+mr-1, mk-1), _maxGroupSize)

    def move_core(self, _direction):
        assert isinstance(self.corePosition, int)
        assert _direction in ['left', 'right']
        S = None
        if _direction == 'left':
            print(_direction)
            assert 0 < self.corePosition

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(0)

            nextCore = self.components[self.corePosition-1]
            print(self.components[self.corePosition-1].shape,self.components[self.corePosition].shape)
            print(U.shape,S.shape,Vt.shape)
            self.components[self.corePosition-1] = (nextCore.reshape(-1, nextCore.shape[3]) @ U @ S).reshape(nextCore.shape)
            self.components[self.corePosition] = Vt

            self.__corePosition -= 1
        else:
            assert self.corePosition < self.order-1

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(3)

            nextCore = self.components[self.corePosition+1]
            self.components[self.corePosition] = Vt
            self.components[self.corePosition+1] = (S @ U.T @ nextCore.reshape(nextCore.shape[0], -1)).reshape(nextCore.shape)

            self.__corePosition += 1
        self.verify()
        return S.diagonal()

    def dofs(self):
        return sum(BlockSparseTensor.fromarray(comp, blks).dofs() for comp, blks in zip(self.components, self.blocks))

    @classmethod
    def random(cls, _dimensions, _ranks, _interactionranges, _blocks):
        assert len(_ranks)+1 == len(_dimensions)
        ranks = [1] + _ranks + [1]
        components = [np.zeros((leftRank, dimension, intrange, rightRank)) for leftRank, dimension,intrange, rightRank in zip(ranks[:-1], _dimensions,_interactionranges, ranks[1:])]
        for comp, compBlocks in zip(components, _blocks):
            for block in compBlocks:
                comp[block] = np.random.randn(*comp[block].shape)
        return cls(components, _blocks)
