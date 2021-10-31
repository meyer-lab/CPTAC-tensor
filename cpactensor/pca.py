import tensorly as tl
import pandas as pd
import numpy as np
from CPTAC_TENSOR.cpactensor.data import gen_concat_tensor
from tensorPack.tensorpac import perform_CP

# Comparing tensor decomp vs PCA 
def pca_vs_tensor():

    # Import data, into Tensor
    (tensor, variables, subjects, TvN) = gen_concat_tensor()

    tfac = perform_CP(tensor)
    
    # flatten TENSOR to 2D
    # systemsSerology/figure2b for R2X 

    