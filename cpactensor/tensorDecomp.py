from cpactensor.data import gen_concat_tensor
from tensorpack import Decomposition
import numpy as np 
import pandas as pd 

(tensor, variables, subjects, TvN) = gen_concat_tensor()  

'''
logicTensor = np.logical_not(np.isnan(tensor))
idxs = []
for i in range(logicTensor.shape[0]):
    for j in range(logicTensor.shape[1]):
        if np.sum(logicTensor[i][j][:]) < 1:
            idxs.append(j)
tensorNoNan = np.delete(tensor, idxs, axis=1)
'''

decomp = Decomposition(tensor,max_rr=6)
decomp.perform_tfac()
decomp.save('cpactensor/figures/tensorData.pickle')