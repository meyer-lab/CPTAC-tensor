import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import gen_concat_tensor
from tensorpac import perform_CP, calcR2X
from statsmodels.multivariate.pca import PCA

# Comparing tensor decomp vs PCA 
#def pca_vs_tensor():
    
### Import data, into Tensor
(tensor, variables, subjects, TvN) = gen_concat_tensor()

# Convert tensor to boolean array, 1 = non_missing
logicTensor = np.logical_not(np.isnan(tensor))

# Remove rows with all NaNs
idxs = []
for i in range(logicTensor.shape[0]):
    for j in range(logicTensor.shape[1]):
        if np.sum(logicTensor[i][j][:]) < 25:
            idxs.append(j)
tensorNoNan = np.delete(tensor, idxs, axis=1)

# Unfold along dim=0, TvN
tensorShape = tensorNoNan.shape
flatTensor = np.reshape(tensorNoNan, (tensorShape[0]*tensorShape[1] , tensorShape[2]))

# Choose random idxs along vars axis
num = 300
idxss = np.random.randint(flatTensor.shape[1], size=num)

randFlatTensor = flatTensor[:,idxss]
randTensor = tensorNoNan[:,:,idxss]
randVars = np.take(variables,idxss)

comps = np.arange(1,12)
CPR2X = np.zeros(comps.shape)
PCAR2X = np.zeros(comps.shape)
CPSize = np.zeros(comps.shape)
PCASize = np.zeros(comps.shape)

for c, i in enumerate(comps):
    outt = PCA(randFlatTensor, ncomp=i, missing='fill-em', standardize=False, demean=False, normalize=False)
    recon = outt.scores @ outt.loadings.T
    PCASize[c] = sum(randFlatTensor.shape) * i
    PCAR2X[c] = calcR2X(recon, mIn=randFlatTensor)
    tfac = perform_CP(randTensor, r=i)
    CPSize[c] = sum(tfac.shape) * i 
    CPR2X[c] = tfac.R2X

plt.scatter(CPSize, CPR2X, c='blue', label='Tensor')
plt.scatter(PCASize, PCAR2X, c='red', label='PCA')
plt.xlabel("Size of reduced data")
plt.ylabel("R2X")
plt.title(num, " variables chosen at random")
plt.legend()
plt.show()
