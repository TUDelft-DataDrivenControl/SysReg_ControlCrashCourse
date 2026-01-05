import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({ 'mathtext.fontset':         'cm',
                      'font.size':          12.0,               'axes.labelsize':           'medium',
                      'xtick.labelsize':    'x-small',          'ytick.labelsize':          'x-small',
                      'axes.grid':          True,               'axes.formatter.limits':    [-3, 6],
                      'grid.alpha':         0.5,                'figure.figsize':           [11.0, 4],
                      'figure.constrained_layout.use': True,    'scatter.marker':           'x',
                      'animation.html':     'jshtml'
                    })

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore")

import control as cm
from helperFunctions import *

###############################################
SYS = loopShaper()
fig = plt.figure(figsize=[15, 8])
gs = GridSpec(4,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[0,0], gs[1, 0], gs[2, 0], gs[:3, 1], gs[3,:]]]

SYS.plot_LS(ax)
plt.show()