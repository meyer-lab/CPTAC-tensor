import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import gen_concat_tensor
from tensorpac import perform_CP, calcR2X
from statsmodels.multivariate.pca import PCA

# Comparing tensor decomp vs PCA 
def pca_vs_tensor():
    
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
    idxss = np.random.randint(flatTensor.shape[1], size=500)

    randFlatTensor = flatTensor[:,idxss]
    randTensor = tensorNoNan[:,:,idxss]
    randVars = np.take(variables,idxss)

    comps = np.arange(1,5)
    CPR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)

    for c, i in enumerate(comps):
        outt = PCA(randFlatTensor, ncomp=i, missing='fill-em', standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[c] = calcR2X(recon, mIn=randFlatTensor)
        tfac = perform_CP(randTensor, r=i)
        CPR2X[c] = tfac.R2X

    plt.scatter(comps, CPR2X, c='blue')
    plt.scatter(comps, PCAR2X, c='red')
    plt.show()
    