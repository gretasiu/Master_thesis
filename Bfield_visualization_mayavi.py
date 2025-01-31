%gui qt
from mayavi import mlab
import numpy as np
pc = 3.08572e18 # cm

# Original grid for the hourglass
x_core, y_core, z_core = np.mgrid[-20:20:50j, -20:20:50j, -20:20:50j]
print("Core grid shape:", x_core.shape)

# Define hourglass magnetic field components
alpha = 0.004

Bx_core = alpha * x_core * z_core * np.exp(-alpha * z_core**2)
By_core = alpha * y_core * z_core * np.exp(-alpha * z_core**2)
Bz_core = 1

# Expanded grid for the entire region (includes outer uniform field)
x_outer, y_outer, z_outer = np.mgrid[-100:100:250j, -100:100:250j, -100:100:250j]
print("Outer grid shape:", x_outer.shape)

# Initialize larger grid for combined field
Bx = np.zeros_like(x_outer)
By = np.zeros_like(y_outer)
Bz = np.zeros_like(z_outer)

# Map the hourglass field into the larger grid
core_condition = (
    (x_outer >= -20) & (x_outer <= 20) &
    (y_outer >= -20) & (y_outer <= 20) &
    (z_outer >= -20) & (z_outer <= 20)
)
Bx[core_condition] = Bx_core.flatten()
By[core_condition] = By_core.flatten()
Bz[core_condition] = Bz_core.flatten()

# Apply uniform field to the outer region
uniform_condition = ~core_condition  # Invert the condition for the uniform field
Bx[uniform_condition] = 0 
By[uniform_condition] = 0 
Bz[uniform_condition] = 1

mlab.flow(x_outer, y_outer, z_outer, Bx, By, Bz, seedtype='sphere',integration_direction='both')
