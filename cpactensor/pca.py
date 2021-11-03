import pandas as pd
import numpy as np
from data import gen_concat_tensor
from tensorpac import perform_CP
from statsmodels.multivariate.pca import PCA

# Comparing tensor decomp vs PCA 
def pca_vs_tensor():
    
    ### Import data, into Tensor
    (tensor, variables, subjects, TvN) = gen_concat_tensor()

    '''

    tfac = perform_CP(tensor, r=3)

    return tfac
    '''

    # Perform PCA 

    # Unfold along dim=0, TvN
    tensorShape = tensor.shape
    flatTensor = np.reshape(tensor, (tensorShape[0]*tensorShape[1] , tensorShape[2]))

    comps = 3
    outt = PCA(flatTensor, ncomp=comps, missing='fill-em', standardize=False, demean=False, normalize=False)
    

    # flatten TENSOR to 2D
    # systemsSerology/figure2b for R2X 

    