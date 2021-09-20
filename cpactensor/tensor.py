import numpy as np
from tensorpac import perform_CMTF
from .data import gen_tensor_matrix

def decomp_3d_2d():
    (tensor, geneSet, patients, _), (matrix, clust_data, _) = gen_tensor_matrix()
    tensor = np.swapaxes(tensor, 0, 1)
    for rr in np.arange(1,5):
        tFac = perform_CMTF(tensor, matrix, r=rr)
        print(rr, tFac.R2X)


if __name__ == "__main__":
    decomp_3d_2d()
