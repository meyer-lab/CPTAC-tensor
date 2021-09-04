import pandas as pd
from .common import subplotLabel, getSetup
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 5), (1, 1))

    # Add subplot labels

    # subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    path = 'data/'
    clust_data = pd.read_csv(path + 'CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')

    #import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)

    sns.heatmap(data=clust_data.corr(method = 'pearson'),vmin = -1, vmax = 1, cmap="RdBu", ax = ax[0])
    ax[0].set_ylabel('Cluster')
    ax[0].set_xlabel('Cluster')

    return f
