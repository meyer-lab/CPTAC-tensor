from tensorpack import *
from tensorpack.plot import *
from .common import subplotLabel, getSetup
from ..data import gen_concat_tensor

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 5), (1, 2), multz={0:1})

    tensor, _, _, _ = gen_concat_tensor()

    ## With new figure making function in tensorpack
    t = Decomposition(tensor)
    t.perform_tfac()
    t.perform_PCA()
    tfacr2x(ax[0], t)
    reduction(ax[1], t)

    # Add subplot labels
    subplotLabel(ax)

    return f