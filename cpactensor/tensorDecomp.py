from cpactensor.data import gen_concat_tensor
from tensorpack import Decomposition
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

(tensor, variables, subjects, TvN) = gen_concat_tensor()  

logicTensor = np.logical_not(np.isnan(tensor))
idxs = []
for i in range(logicTensor.shape[0]):
    for j in range(logicTensor.shape[1]):
        if np.sum(logicTensor[i][j][:]) < 1:
            idxs.append(j)
tensorNoNan = np.delete(tensor, idxs, axis=1)

decomp = Decomposition(tensorNoNan,max_rr=8)
decomp.perform_tfac()
decomp.perform_PCA(flattenon=1)
decomp.save('cpactensor/figures/tensorData.pickle')

plt.scatter(decomp.sizePCA, decomp.PCAR2X, label='PCA')
plt.scatter(decomp.sizeT, decomp.TR2X, label='Tensor')
plt.title('Size of each')
plt.savefig('cpactensor/figures/tensorVsPCA.png')

