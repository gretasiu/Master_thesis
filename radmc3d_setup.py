#
# Importing files and packages
#
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import central_star as cs
#
# Some natural constants
#
au = 1.49598e13     # Astronomical Unit       [cm]
pc = 3.08572e18     # Parsec                  [cm]
ms = 1.98892e33     # Solar mass              [g]
Ts = 5.78e3         # Solar temperature       [K]
Ls = 3.8525e33      # Solar luminosity        [erg/s]
Rs = 6.96e10        # Solar radius            [cm]
m_H = 1.6736e-24    # mass of one H atom      [g]
gas_to_dust_ratio = 100  # MASS ratio of gas to dust
#
# Number of photons
#
nphot = 1e7
nphot_scat = 1e7
#
# spherical grid parameters
#
nr = 480  # number of radial cells
ntheta = 360
nphi = 360
r_min = 1.48 * au  # minimum radius
r_max = 2.18 * pc
#
# Density model parameters (taken from eqn 2 and table 6 from Yuxin Lin 2022)
#
rhoc = 2.39e6*(2.8*m_H)/gas_to_dust_ratio  # g cm^-3
rcc = 0.1*pc
n = 1.2
#
# Star parameters
#
mstar = cs.M_star_G31*ms
rstar = cs.R_star_G31*Rs
tstar = cs.T_star_G31
pstar = np.array([0., 0., 0.])  # posiiton of the star inside the grid; center
#
# Logarithmic radial grid
#
ri = np.logspace(np.log10(r_min), np.log10(r_max), nr+1)
thetai = np.linspace(0e0, np.pi, ntheta+1)
phii = np.linspace(0e0, 2e0*np.pi, nphi+1)
rc = 0.5 * (ri[0:nr] + ri[1:nr+1])
thetac = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta+1])
phic = 0.5 * (phii[0:nphi] + phii[1:nphi+1])
#
# Dust density model
#
qq = np.meshgrid(rc, thetac, phic, indexing='ij')
rr = qq[0]
theta_grid = qq[1]
phi_grid = qq[2]
rhod = rhoc*(rr/rcc)**(-n)
#
# wavelength which I want to calculate over
#
lammin = 1e-10
lammax = 1e4
steplam = 1000
lam = np.logspace(np.log10(lammin), np.log10(lammax), steplam)
nlam = lam.size
#
# Now the alignment vector field. This is ONLY FOR TESTING.
#
# Convert spherical to Cartesian for density model
xx = rr * np.sin(theta_grid) * np.cos(phi_grid)
yy = rr * np.sin(theta_grid) * np.sin(phi_grid)
zz = rr * np.cos(theta_grid)

alvec = np.zeros((nr, ntheta, nphi, 3))
alpha = 1e-33

# Initialize Bx, By, Bz with zeros
Bx = np.zeros_like(xx)
By = np.zeros_like(yy)
Bz = np.ones_like(zz)  # Default Bz is set to 1

# Apply conditions
box = r_max//44
condition = (xx > -box) & (xx < box) & (yy > -
                                        box) & (yy < box) & (zz > -box) & (zz < box)
Bx[condition] = alpha * (xx[condition]) * \
    (zz[condition]) * np.exp(-alpha * (zz[condition]) ** 2)
By[condition] = alpha * (yy[condition]) * \
    (zz[condition]) * np.exp(-alpha * (zz[condition]) ** 2)
Bz[condition] = 1

# Putting B vector from our model into the proper array
alvec[:, :, :, 0] = Bx
alvec[:, :, :, 1] = By
alvec[:, :, :, 2] = Bz

#
# Write the wavelength_micron.inp file
#
with open('wavelength_micron.inp', 'w+') as f:
    f.write('%d\n' % (nlam))
    for value in lam:
        f.write('%13.6e\n' % (value))
#
# write the star.inp file
#
with open('stars.inp', 'w+') as f:
    f.write('2\n')
    f.write('1 %d\n\n' % (nlam))
    f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n' %
            (rstar, mstar, pstar[0], pstar[1], pstar[2]))
    for value in lam:
        f.write('%13.6e\n' % (value))
    f.write('\n%13.6e\n' % (-tstar))
#
# Write the grid file
#
with open('amr_grid.inp', 'w+') as f:
    f.write('1\n')                       # iformat
    # AMR grid style  (0=regular grid, no AMR)
    f.write('0\n')
    f.write('100\n')                       # Coordinate system, shperical
    f.write('0\n')                       # gridinfo
    f.write('1 1 1\n')                   # Include x,y,z coordinate
    f.write('%d %d %d\n' % (nr, ntheta, nphi))     # Size of grid
    for value in ri:
        f.write('%13.6e\n' % (value))      # X coordinates (cell walls)
    for value in thetai:
        f.write('%13.6e\n' % (value))      # Y coordinates (cell walls)
    for value in phii:
        f.write('%13.6e\n' % (value))      # Z coordinates (cell walls)
#
# Write the density file
#
with open('dust_density.inp', 'w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n' % (nr*ntheta*nphi))           # Nr of cells
    f.write('1\n')                       # Nr of dust species
    # Create a 1-D view, fortran-style indexing
    data = rhod.ravel(order='F')
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
#
# Dust opacity control file
#
with open('dustopac.inp', 'w+') as f:
    f.write('2               Format number of this file\n')
    f.write('1               Nr of dust species\n')
    f.write(
        '==================\n')
    f.write('20               Way in which this dust species is read\n')  # 20
    f.write('0               0=Thermal grain\n')
    f.write('dsharp-a10um        Extension of name of dustkappa_***.inp file\n')
    f.write(
        '--------------------\n')
#
# Dust alignment direction
#
with open('grainalign_dir.inp', 'w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n' % (nr*ntheta*nphi))           # Nr of cells
    for ix in range(nr):
        for iy in range(ntheta):
            for iz in range(nphi):
                f.write('%13.6e %13.6e %13.6e\n' % (
                    alvec[ix, iy, iz, 0], alvec[ix, iy, iz, 1], alvec[ix, iy, iz, 2]))
#
# Write the radmc3d.inp control file
#
with open('radmc3d.inp', 'w+') as f:
    f.write('nphot = %d\n' % (nphot))
    f.write('nphot_scat = %d\n' % (nphot_scat))
    f.write('scattering_mode_max = 4\n')
    f.write('alignment_mode = 1\n')
    f.write('setthreads = 24\n')
    f.write('istar_sphere = 0\n')
    f.write('modified_random_walk = 1\n')
    f.write('mc_scat_maxtauabs = 10\n')
