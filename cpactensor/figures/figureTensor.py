import numpy as np
from tensorly.decomposition import parafac
from .common import subplotLabel, getSetup
from ..data import generate_tensors
#add all patients in any tensors as nans

def makeFigure():
    ax, f = getSetup((12,5), (3,3))
    (mRNA_tensor, genes, patients, sample_type), (prot_tensor, proteins, patients, sample_type), (clust_tensor, clusters, patients, sample_type) = generate_tensors()
    clust_fact = parafac(np.nan_to_num(clust_tensor, 0),rank = 3, mask = np.isfinite(clust_tensor), normalize_factors = True)
    #mRNA_fact = parafac(np.nan_to_num(mRNA_tensor, 0),rank = 3, mask = np.isfinite(mRNA_tensor), normalize_factors = True)
    #prot_fact = parafac(np.nan_to_num(prot_tensor, 0),rank = 3, mask = np.isfinite(prot_tensor), normalize_factors = True)

    for fact in range(3):
        ax[fact*3].plot(clust_fact[1][0].T[fact])
        ax[fact*3].set_xticks(range(len(sample_type)))
        ax[fact*3].set_xticklabels( sample_type)

    for fact in range(3):
        ax[fact*3+1].plot(clust_fact[1][1].T[fact])
        #ax[fact*3+1].set_xticks(range(len(patients)))
        #ax[fact*3+1].set_xticklabels( patients, rotation = 90)

    for fact in range(3):
        ax[fact*3+2].plot(clust_fact[1][2].T[fact])
        ax[fact*3+2].set_xticks(range(len(clusters)))
        ax[fact*3+2].set_xticklabels(clusters)

    return f
