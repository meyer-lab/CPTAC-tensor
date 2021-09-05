from cpactensor.data import *

def test_tensor_import():
    (mRNA_tensor, genes, patients, sample_type), (prot_tensor, proteins, _, _), (clust_tensor, clusters, _, _) = generate_tensors()
    assert mRNA_tensor.shape[2] == len(genes)
    assert mRNA_tensor.shape[1] == len(patients)
    assert prot_tensor.shape[2] == len(proteins)
    assert clust_tensor.shape[2] == len(clusters)
