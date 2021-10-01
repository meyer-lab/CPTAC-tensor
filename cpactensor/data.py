import pandas as pd
import numpy as np

def generate_tensors():
    """
    Generates a 3 tensors which share TvN and patient axes:
        mRNA_tensor  (Tumor vs. Normal) x (patient) x (gene)
        prot_tensor  (Tumor vs. Normal) x (patient) x (protein)
        clust_tensor (Tumor vs. Normal) x (patient) x (cluster center)

    Returns:
    (
        mRNA_tensor,
        gene id corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices
    ),
    (
        prot_tensor,
        protein id corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices
    ),
    (
        clust_tensor,
        cluster center number corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices
    )

    Additional comments:
        patients is an union of patients in all 3 datasets
    """
    path = 'data/'
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')

    # prot import
    prot_data.index = np.array(prot_data['id'])
    prot_data = prot_data[prot_data.columns[2:]]

    # mRNA import
    mRNA_data.index = np.array(mRNA_data['gene_id'])
    mRNA_data = mRNA_data[mRNA_data.columns[3:]]

    # clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # getting complete sorted list of all patients
    m_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in mRNA_data.columns])
    p_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in prot_data.columns])
    c_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in clust_data.columns])
    patients = sorted(list(m_set.union(p_set, c_set))) #union

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
    return (mRNA_tensor, genes, patients, ['tumor', 'normal']), (prot_tensor, proteins, patients, ['tumor', 'normal']), (clust_tensor, clusters, patients, ['tumor', 'normal'])


def gen_tensor_matrix():
    """
    Generates a 2 tensors which share patient axis:
        tensor (mRNA vs. Protein ) x (patient) x (gene)
        matrix (patient) x (cluster center)

    Returns:
    (
        tensor,
        gene symbol corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        MvP corresponding to tensor indices
    ),
    (
        matrix,
        cluster numbers corresponding to matrix indices,
        id for patients corresponding to matrix indices, 
    )

    Additional comments:
        patients is an intersection of patients in all 3 datasets
        Because there are possibly multiple proteins to each gene, the first protein found with matching genesymbol is used
    """

    path = 'data/'

    #prot import
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    prot_data.index = prot_data['geneSymbol']
    prot_data = prot_data[prot_data.columns[2:]]

    #mRNA import
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')
    mRNA_data.index = mRNA_data['geneSymbol']
    mRNA_data = mRNA_data[mRNA_data.columns[3:]]

    #clust import
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # generating patient set
    patients = sorted(list(set(mRNA_data.columns).intersection(set(prot_data.columns), set(clust_data.columns)))) # intersection

    # generating gene set
    geneSet = list(set(mRNA_data.index).intersection(set(prot_data.index)))

    # building  matrices
    mrna_matrix = mRNA_data.loc[geneSet][patients].values
    prot_matrix = prot_data.loc[~prot_data.index.duplicated(keep='first')].loc[geneSet][patients].values
    tensor = np.array([mrna_matrix.T, prot_matrix.T], dtype=float)
    matrix = clust_data[patients].T.values

    return (tensor, geneSet, patients, ['mRNA', 'Protein']), (matrix, clust_data.index, patients)


def gen_4D_3D_tensors():
    """
    Generates a 2 tensors which share patient and TvN axes:
        mRNA_prot_tensor (mRNA vs. Protein ) x (Tumor vs. Normal) x (patient) x (gene)
        clust_tensor (Tumor vs. Normal) x (patient) x (cluster center)

    Returns:
    (
        mRNA_prot_tensor,
        gene symbol corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices,
        MvP corresponding to tensor indices
    ),
    (
       clust_tensor,
        cluster numbers corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices,
    )

    Additional comments:
        patients is an intersection of patients in all 3 datasets
        Because there are possibly multiple proteins to each gene, the first protein found with matching genesymbol is used
    """
    path = 'data/'
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')

    # prot import
    prot_data.index = prot_data['geneSymbol']
    prot_data = prot_data[prot_data.columns[2:]]

    # mRNA import
    mRNA_data.index = mRNA_data['geneSymbol']
    mRNA_data = mRNA_data[mRNA_data.columns[3:]]

    # clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

   # generating patient set
    m_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in mRNA_data.columns])
    p_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in prot_data.columns])
    c_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in clust_data.columns])
    patients = sorted(list(m_set.intersection(p_set, c_set))) #intersection
    n_patients = [patient + '.N' for patient in patients] #normal counterpart for patient ids

    # getting list of genes in common between mRNA and prot datasets
    geneSet = list(set(mRNA_data.index).intersection(set(prot_data.index)))

    # filtering dataframes and adding nans for missing samples
    m_tumor = mRNA_data[m_set.intersection(set(patients))].loc[geneSet]
    for col in set(patients).difference(set(m_tumor.columns)):
        m_tumor.insert(0, col, [np.nan for _ in range(len(m_tumor.index))])
    m_nat = mRNA_data.filter(regex='N$')
    m_nat = m_nat[set(m_nat.columns).intersection(set(n_patients))].loc[geneSet]
    for col in set(n_patients).difference(set(m_nat.columns)):
        m_nat.insert(0, col, [np.nan for _ in range(len(m_nat.index))])

    p_tumor = prot_data.loc[geneSet]
    p_tumor = p_tumor[~p_tumor.index.duplicated(keep='first')][p_set.intersection(set(patients))]
    for col in set(patients).difference(set(p_tumor.columns)):
        p_tumor.insert(0, col, [np.nan for _ in range(len(p_tumor.index))])
    p_nat = prot_data.filter(regex='N$').loc[geneSet]
    p_nat = p_nat[~p_nat.index.duplicated(keep='first')][set(p_nat.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(p_nat.columns)):
        p_nat.insert(0, col, [np.nan for _ in range(len(p_nat.index))])

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
    clust_tensor, clust_data.index, patients, ['tumor', 'normal'])

def gen_concat_tensor():    
    """
    Generates a concatenated tensor (Tumor vs. Normal) x (patients) x (observations) including:
        mRNA expression: gene-level normalized RNAseq data 
        protein expression: two-component normalized Log2 transformed protein expression
        cluster centers

    Returns:
    (
        concatenated tensor (Tumor vs. Normal) x (patients) x (observations),
        gene id, protein id, or cluster number corresponding to tensor indices,
        id for patients corresponding to tensor indices, 
        TvN corresponding to tensor indices
    )

    Additional comments:
        patients is an union of patients in all 3 datasets
    """
    path = 'data/'

    #prot import
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv')
    prot_data.index = prot_data['id']
    prot_data = prot_data[prot_data.columns[2:]]

    #mRNA import
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv')
    mRNA_data.index = mRNA_data['gene_id']
    mRNA_data = mRNA_data[mRNA_data.columns[3:]]

    #clust import
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.T

    # generating patient set
    m_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in mRNA_data.columns])
    p_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in prot_data.columns])
    c_set = set([patient[:-2] if patient[-1] == 'N' else patient for patient in clust_data.columns])
    patients = sorted(list(m_set.union(p_set, c_set))) #union
    n_patients = [patient + '.N' for patient in patients] #normal counterpart sample ids

    # Filtering and then adding nans for missing samples
    tumor = mRNA_data[set(mRNA_data.columns).intersection(set(patients))]
    for col in set(patients).difference(set(tumor.columns)):
        tumor.insert(0, col, [np.nan for _ in range(len(tumor.index))])
        
    prot_tumor = prot_data[set(prot_data.columns).intersection(set(patients))]
    for col in set(patients).difference(set(prot_tumor.columns)):
        prot_tumor.insert(0, col, [np.nan for _ in range(len(prot_tumor.index))])

    clust_tumor = clust_data[set(clust_data.columns).intersection(set(patients))]
    for col in set(patients).difference(set(clust_tumor.columns)):
        clust_tumor.insert(0, col, [np.nan for _ in range(len(clust_tumor.index))])

    tumor = tumor.append(prot_tumor).append(clust_tumor) #concatenating dataframes

    nat = mRNA_data[set(mRNA_data.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(nat.columns)):
        nat.insert(0, col, [np.nan for _ in range(len(nat.index))])

    prot_nat = prot_data[set(prot_data.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(prot_nat.columns)):
        prot_nat.insert(0, col, [np.nan for _ in range(len(prot_nat.index))])

    clust_nat = clust_data[set(clust_data.columns).intersection(set(n_patients))]
    for col in set(n_patients).difference(set(clust_nat.columns)):
        clust_nat.insert(0, col, [np.nan for _ in range(len(clust_nat.index))])

    nat = nat.append(prot_nat).append(clust_nat) #concatenating dataframes

    #reordering dataframes after nan insertion 
    tumor = tumor[patients]
    nat = nat[n_patients]

    #setup index maps
    variables = list(tumor.index)
    subjects = list(tumor.columns)
    TvN = ['Tumor', 'Normal']
    
    #build tensor
    tensor = np.array([tumor.values.T, nat.values.T], dtype = float)

    return (tensor, variables, subjects, TvN)
