#import numpy as np
#import setup as p
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from plotpoldir import *
import math
au = 1.49598e13     # Astronomical Unit       [cm]

# global parameters suitable for both wavelength
npix = 1000  # on each of x and y axis. so the total number of pixel is 1000^2
incl = 50
phi = 0
posang = -15.0
sizeau1300 = 4.5e5
sizeau7000 = 4.5e5
Bfield = False

#
# Make and plot an example image
#
# 1.3mm imaging
makeImage(npix=npix, incl=incl, phi=phi, posang=posang, wav=1300,
          sizeau=sizeau1300)   # This calls radmc3d
a = readImage()
a.writeFits('radmc3d_1300_I.fits', dpc=3750., stokes='I')
a.writeFits('radmc3d_1300_Q.fits', dpc=3750., stokes='Q')
a.writeFits('radmc3d_1300_U.fits', dpc=3750., stokes='U')

# 7mm imaging
makeImage(npix=npix, incl=incl, phi=phi, posang=posang, wav=7000,
          sizeau=sizeau7000)   # This calls radmc3d
a = readImage()
a.writeFits('radmc3d_7000_I.fits', dpc=3750., stokes='I')
a.writeFits('radmc3d_7000_Q.fits', dpc=3750., stokes='Q')
a.writeFits('radmc3d_7000_U.fits', dpc=3750., stokes='U')
