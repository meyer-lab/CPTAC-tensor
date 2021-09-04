import pandas as pd
import numpy as np

def generate_tensors():
    path = 'data/'
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')

    # prot import
    prot_data.index = np.array(prot_data['id'])
    prot_data = prot_data[prot_data.columns[17:]]

    # mRNA import
    mRNA_data.index = np.array(mRNA_data['gene_id'])
    mRNA_data = mRNA_data[mRNA_data.columns[6:]]

    # clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # getting complete sorted list of all patients
    m_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in mRNA_data.columns])
    p_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in prot_data.columns])
    c_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in clust_data.columns])
    patients = sorted(list(m_set.union(p_set, c_set)))

    # converting patient tumor string to normal
    n_patients = [patient + '.N' for patient in patients]

    # setting up mRNA tumor and normal df (adding missing patients)
    m_tumor = mRNA_data[m_set.intersection(set(patients))]
    for col in set(patients).difference(set(m_tumor.columns)):
        m_tumor.insert(0, col, [np.nan for _ in range(len(m_tumor.index))])

    m_nat = mRNA_data.filter(regex='N$')
    m_nat = m_nat[set(m_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(m_nat.columns)):
        m_nat.insert(0, col, [np.nan for _ in range(len(m_nat.index))])

    # setting up protein tumor and normal df (adding missing patients)
    p_tumor = prot_data[p_set.intersection(set(patients))]
    for col in set(patients).difference(set(p_tumor.columns)):
        p_tumor.insert(0, col, [np.nan for _ in range(len(p_tumor.index))])

    p_nat = prot_data.filter(regex='N$')
    p_nat = p_nat[set(p_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(p_nat.columns)):
        p_nat.insert(0, col, [np.nan for _ in range(len(p_nat.index))])

    # setting up cluster tumor and normal df (adding missing patients)
    c_tumor = clust_data[c_set.intersection(set(patients))]
    for col in set(patients).difference(set(c_tumor.columns)):
        c_tumor.insert(0, col, [np.nan for _ in range(len(c_tumor.index))])

    c_nat = clust_data.filter(regex='N$')
    c_nat = c_nat[set(c_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(c_nat.columns)):
        c_nat.insert(0, col, [np.nan for _ in range(len(c_nat.index))])

    # constructing tensors and storing indices (patients already created)
    genes = mRNA_data.index
    proteins = prot_data.index
    clusters = clust_data.index
    mRNA_tensor = np.array([m_tumor[patients].T.values, m_nat[n_patients].T.values]).astype('float')
    prot_tensor = np.array([p_tumor[patients].T.values, p_nat[n_patients].T.values]).astype('float')
    clust_tensor = np.array([c_tumor[patients].T.values, c_nat[n_patients].T.values]).astype('float')
    return ['tumor', 'normal'], patients, (mRNA_tensor, genes), (prot_tensor, proteins), (clust_tensor, clusters)


def gen_tensor_matrix():
    path = 'data/'

    #prot import
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    prot_data.index = prot_data['geneSymbol']
    prot_data = prot_data[prot_data.columns[17:]]

    #mRNA import
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')
    mRNA_data.index = mRNA_data['geneSymbol']
    mRNA_data = mRNA_data[mRNA_data.columns[6:]]

    #clust import
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # generating patient set
    patients = sorted(list(set(mRNA_data.columns).intersection(set(prot_data.columns), set(clust_data.columns))))

    # generating gene set
    geneSet = list(set(mRNA_data['geneSymbol']).intersection(set(prot_data['geneSymbol'])))

    # building  matrices
    mrna_matrix = mRNA_data.loc[geneSet][patients].values
    prot_matrix = prot_data.loc[geneSet].drop_duplicates(subset=['geneSymbol'])[patients].values
    tensor = np.array([mrna_matrix.T, prot_matrix.T], dtype=float)
    matrix = clust_data[patients].T.values

    return (tensor, list(geneSet)), (matrix, np.arange(1, 25)), patients, ['mRNA', 'Protein']


def gen_4D_3D_tensors():
    path = 'data/'
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')

    # prot import
    prot_data.index = prot_data['geneSymbol']
    prot_data = prot_data[prot_data.columns[17:]]

    # mRNA import
    mRNA_data.index = mRNA_data['geneSymbol']
    mRNA_data = mRNA_data[mRNA_data.columns[6:]]

    # clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # getting complete sorted list of all patients
    m_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in mRNA_data.columns])
    p_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in prot_data.columns])
    c_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in clust_data.columns])
    patients = sorted(list(m_set.intersection(p_set, c_set)))

    # converting patient tumor string to normal
    n_patients = [patient + '.N' for patient in patients]

    # getting list of genes in common between mRNA and prot datasets
    geneSet = list(set(mRNA_data['geneSymbol']).intersection(set(prot_data['geneSymbol'])))

    # setting up mRNA tumor and normal df
    m_tumor = mRNA_data[m_set.intersection(set(patients))].loc[geneSet]
    for col in set(patients).difference(set(m_tumor.columns)):
        m_tumor.insert(0, col, [np.nan for _ in range(len(m_tumor.index))])

    m_nat = mRNA_data.filter(regex='N$')
    m_nat = m_nat[set(m_nat.columns).intersection(set(n_patients))].loc[geneSet]
    for col in set(n_patients).difference(set(m_nat.columns)):
        m_nat.insert(0, col, [np.nan for _ in range(len(m_nat.index))])

    # setting up protein tumor and normal df
    p_tumor = prot_data.loc[geneSet]
    p_tumor = p_tumor[~p_tumor.index.duplicated(keep='first')][p_set.intersection(set(patients))]
    for col in set(patients).difference(set(p_tumor.columns)):
        p_tumor.insert(0, col, [np.nan for _ in range(len(p_tumor.index))])

    p_nat = prot_data.filter(regex='N$').loc[geneSet]
    p_nat = p_nat[~p_nat.index.duplicated(keep='first')][set(p_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(p_nat.columns)):
        p_nat.insert(0, col, [np.nan for _ in range(len(p_nat.index))])

    # setting up cluster tumor and normal df
    c_tumor = clust_data[c_set.intersection(set(patients))]
    for col in set(patients).difference(set(c_tumor.columns)):
        c_tumor.insert(0, col, [np.nan for _ in range(len(c_tumor.index))])

    c_nat = clust_data.filter(regex='N$')
    c_nat = c_nat[set(c_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(c_nat.columns)):
        c_nat.insert(0, col, [np.nan for _ in range(len(c_nat.index))])

    # building tensors
    mRNA_prot_tensor = np.array([[m_tumor[patients].values.T, m_nat[n_patients].values.T],
                                 [p_tumor[patients].values.T, p_nat[n_patients].values.T]], dtype=float)
    clust_tensor = np.array([c_tumor[patients].values.T, c_nat[n_patients].values.T], dtype=float)

    return (mRNA_prot_tensor, geneSet, patients, ['tumor', 'normal'], ['mRNA', 'protein']), (
    clust_tensor, np.arange(1, 25), patients, ['tumor', 'normal'])
