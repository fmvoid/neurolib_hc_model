#hide
import matplotlib.pyplot as plt

import numpy as np

# Let's import all the necessary functions for the parameter
from neurolib.models.fhn import FHNModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch

# load some utilty functions for explorations
import neurolib.utils.pypetUtils as pu
import neurolib.utils.paths as paths
import neurolib.optimize.exploration.explorationUtils as eu

# The brain network dataset
from neurolib.utils.loadData import Dataset

# Some useful functions are provided here
import neurolib.utils.functions as func

# a nice color map
plt.rcParams['image.cmap'] = 'plasma'

ds = Dataset("hcp")
model = FHNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
model.params.duration = 20 * 1000 #ms
# testing: model.params.duration = 20 * 1000 #ms
# original: model.params.duration = 5 * 60 * 1000 #ms

parameters = ParameterSpace({"x_ext": [np.ones((model.params['N'],)) * a for a in  np.linspace(0, 2, 2)] # testing: 2, original: 41
                             ,"K_gl": np.linspace(0, 2, 2) # testing: 2, original: 41
                             ,"coupling" : ["additive", "diffusive"]
                            }, kind="grid")
search = BoxSearch(model=model, parameterSpace=parameters, filename="example-1.2.0.hdf")

search.run(chunkwise=True, bold=True)
