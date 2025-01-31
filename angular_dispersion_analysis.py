def data_selection_optimized(xlower, ylower, xupper, yupper, PA, PAerr, PI, I, sigma_I):
    """
    Optimized function to select non-NaN data from the specified region in 2D arrays.

    Args:
        xlower, ylower: Lower bounds of the rectangular region.
        xupper, yupper: Upper bounds of the rectangular region.
        PA: 2D array of position angles (radians).
        err: 2D array of errors in PA.
        PI: 2D array of polarized intensities.
        I: 2D array of total intensities.

    Returns:
        valid_coords: 2D array of valid (x, y) coordinates.
        valid_PA: 1D array of valid PA values.
        valid_err: 1D array of valid PA error values.
        valid_PI: 1D array of valid PI values.
        valid_I: 1D array of valid total intensity values.
    """
    # Select the region of interest
    PA_region = PA[ylower:yupper, xlower:xupper]
    PAerr_region = PAerr[ylower:yupper, xlower:xupper]
    PI_region = PI[ylower:yupper, xlower:xupper]
    I_region = I[ylower:yupper, xlower:xupper]

    # Generate x, y coordinate grids for the region
    y_indices, x_indices = np.mgrid[ylower:yupper, xlower:xupper]

    # Flatten all arrays
    flat_x = x_indices.flatten()
    flat_y = y_indices.flatten()
    flat_PA = PA_region.flatten()
    flat_PAerr = PAerr_region.flatten()
    flat_PI = PI_region.flatten()
    flat_I = I_region.flatten()

    # Create a valid mask for non-NaN values in PA
    valid_mask = ~np.isnan(flat_PA) & ~np.isnan(flat_PAerr) & ~np.isnan(flat_PI) & (flat_I >= 10 * sigma_I)

    # print(valid_mask[:100])

    # Apply the mask to filter valid data
    valid_coords = np.vstack((flat_x[valid_mask], flat_y[valid_mask])).T
    valid_flat_PA = flat_PA[valid_mask]
    valid_flat_PAerr = flat_PAerr[valid_mask]
    valid_flat_PI = flat_PI[valid_mask]

    return valid_coords, valid_flat_PA, valid_flat_PAerr, valid_flat_PI

def calculate_unique_angle_cos_distance(valid_coords, valid_flat_PA, valid_flat_PAerr, valid_flat_PI, pixel_scale, W1):
    """
    Calculate unique absolute angle differences, cosine of differences, and distances.

    Args:
        data: 2D array of angles in radian.
        pixel_scale: Scale in arcseconds per pixel.

    Returns:
        angle_diff_cos: 1D array of cosines of unique angle differences.
        distances: 1D array of unique distances in arcseconds.
    """
    # Flatten the 2D array for pairwise calculations
    # Get upper triangular indices (excluding the diagonal)
    n = len(valid_flat_PA)
    i, j = np.triu_indices(n, k=1)

    # Compute distances in arcseconds
    flat_x = valid_coords[:, 0]
    flat_y = valid_coords[:, 1]
    distances = np.sqrt((flat_x[i] - flat_x[j])**2 + (flat_y[i] - flat_y[j])**2) * pixel_scale

    # Compute absolute differences and the cosine for unique angle pairs
    angle_diff = np.abs(valid_flat_PA[i] - valid_flat_PA[j])
    angle_diff = np.where(angle_diff > np.pi/2, np.pi - angle_diff, angle_diff)
    angle_diff_cos = np.cos(angle_diff)

    # Compute the error squared od the angle difference
    sigma2_angle_diff = valid_flat_PAerr[i]**2 + valid_flat_PAerr[j]**2 - 2*valid_flat_PAerr[i]*valid_flat_PAerr[j]*np.exp(-(distances**2)/(4*W1**2))

    # Compute PI 
    PI2 = valid_flat_PI[i]*valid_flat_PI[j]

    return distances, angle_diff_cos, sigma2_angle_diff, PI2

def write_data_file(filename, valid_coords, valid_flat_PA, valid_flat_PAerr, valid_flat_PI, distances, angle_diff_cos, sigma2_angle_diff, PI2):
    """
    Write the pairwise calculation results to a data file for verification.
    
    Args:
        filename: Name of the output file.
        valid_coords: 2D array of valid (x, y) coordinates.
        valid_flat_PA: 1D array of valid PA values (radians).
        valid_flat_PAerr: 1D array of valid PA error values (radians).
        valid_flat_PI: 1D array of valid PI values.
        distances: 1D array of unique distances in arcseconds.
        angle_diff_cos: 1D array of cosines of unique angle differences.
        sigma2_angle_diff: 1D array of errors squared for angle differences.
        PI2: 1D array of polarization intensity products.
    """
    # Get pairwise upper triangular indices
    n = len(valid_flat_PA)
    i, j = np.triu_indices(n, k=0)

    valid_flat_PA = np.degrees(valid_flat_PA)
    valid_flat_PAerr = np.degrees(valid_flat_PAerr)

    with open(filename, "w") as f:
        # Write header
        f.write("xi\tyi\txj\tyj\tPA_i\tPA_j\tPA_err_i\tPA_err_j\tPI_i\tPI_j\tdistances\tangle_diff_cos\tsigma2_angle_diff\tPI2\n")
        
        # Write data
        for k in range(len(distances)):
            f.write(f"{valid_coords[i[k], 0]}\t{valid_coords[i[k], 1]}\t"  # xi, yi
                    f"{valid_coords[j[k], 0]}\t{valid_coords[j[k], 1]}\t"  # xj, yj
                    f"{valid_flat_PA[i[k]]:.6f}\t{valid_flat_PA[j[k]]:.6f}\t"  # PA_i, PA_j
                    f"{valid_flat_PAerr[i[k]]:.6f}\t{valid_flat_PAerr[j[k]]:.6f}\t"  # PA_err_i, PA_err_j
                    f"{valid_flat_PI[i[k]]:.6f}\t{valid_flat_PI[j[k]]:.6f}\t"  # PI_i, PI_j
                    f"{distances[k]:.6f}\t"  # distances
                    f"{angle_diff_cos[k]:.6f}\t"  # angle_diff_cos
                    f"{sigma2_angle_diff[k]:.6f}\t"  # sigma2_angle_diff
                    f"{PI2[k]:.6f}\n")  # PI2

def bin_pairwise_results(distances, angle_diff_cos, sigma2_angle_diff, PI2, bin_size):
    """
    Bin pairwise calculation results according to distances.

    Args:
        distances: 1D array of pairwise distances.
        angle_diff_cos: 1D array of cosine of angle differences.
        sigma2_angle_diff: 1D array of squared errors of angle differences.
        PI2: 1D array of PI products.
        bin_size: Size of the distance bins.

    Returns:
        binned_midpoints: 1D array of midpoints for each bin.
        binned_angle_diff_cos: 1D array of mean cosine of angle differences for each bin.
        binned_sigma2_angle_diff: 1D array of mean squared errors for each bin.
        binned_PI2: 1D array of mean PI products for each bin.
    """
    # Calculate the maximum distance and define bins
    max_distance = np.max(distances)
    bins = np.arange(0, max_distance + bin_size, bin_size)
    
    # Find indices of the bins each distance belongs to
    bin_indices = np.digitize(distances, bins) - 1

    # Initialize arrays to store results
    binned_midpoints = []
    binned_angle_diff_cos = []
    binned_sigma2_angle_diff = []
    binned_PI2 = []

    # Loop over each bin and calculate statistics
    for bin_idx in range(len(bins) - 1):
        # Find elements in the current bin
        in_bin = bin_indices == bin_idx

        if np.any(in_bin):  # Check if the bin is not empty
            # Calculate the midpoint of the bin
            midpoint = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            binned_midpoints.append(midpoint)

            # Calculate mean values for the bin
            binned_angle_diff_cos.append(np.mean(angle_diff_cos[in_bin]))
            binned_sigma2_angle_diff.append(np.mean(sigma2_angle_diff[in_bin]))
            binned_PI2.append(np.mean(PI2[in_bin]))

    # Convert results to arrays
    binned_midpoints = np.array(binned_midpoints)
    binned_angle_diff_cos = np.array(binned_angle_diff_cos)
    binned_sigma2_angle_diff = np.array(binned_sigma2_angle_diff)
    binned_PI2 = np.array(binned_PI2)

    return binned_midpoints, binned_angle_diff_cos, binned_sigma2_angle_diff, binned_PI2

# importing PA, PAerr and PI data 
PA = utils.FITSimage('/Users/gretasiu/Desktop/G31/G31_JVLA_selfcal/image_FITS/rob2/G31p4_Qband_D.rob2.PA.image.tt0.miriad.rob2.dropdeg.fits')
PA.FITSdata()
PAerr = utils.FITSimage('/Users/gretasiu/Desktop/G31/G31_JVLA_selfcal/image_FITS/rob2/G31p4_Qband_D.rob2.PA_error.image.tt0.miriad.rob2.dropdeg.fits')
PAerr.FITSdata()
PI = utils.FITSimage('/Users/gretasiu/Desktop/G31/G31_JVLA_selfcal/image_FITS/rob2/G31p4_Qband_D.rob2.PI.image.tt0.miriad.rob2.dropdeg.fits')
PI.FITSdata()
I = utils.FITSimage('/Users/gretasiu/Desktop/G31/G31_JVLA_selfcal/image_FITS/rob2/G31p4_Qband_D.rob2.I.image.tt0.dropdeg.fits')
I.FITSdata()

# converting degree to radian
PA.data[(PAerr.data > 10)] = np.nan
PAerr.data[(PAerr.data > 10) ] = np.nan
PA.data = np.radians(PA.data)
PAerr.data = np.radians(PAerr.data)

# W1 and W2
W1 = np.sqrt(PA.bmaj*deg_to_arcsec*PA.bmin*deg_to_arcsec)/np.sqrt(8*math.log(2))  # beam "radius"
W2 = 51.96/2 # arcsec, resolution from the shortest baseline times 50% (Ching 2017)

# data selection: central and non-nan data
xlower = 211
ylower = 221
xupper = 323
yupper = 329
sigma_I = 1.48e-4 # Jy/beam

# pixel size
pixel_scale = 0.2 # arcsec

valid_coords, valid_flat_PA, valid_flat_PAerr, valid_flat_PI = data_selection_optimized(
    xlower, ylower, xupper, yupper, PA.data, PAerr.data, PI.data, I.data, sigma_I)

distances, angle_diff_cos, sigma2_angle_diff, PI2 = calculate_unique_angle_cos_distance(
    valid_coords, valid_flat_PA, valid_flat_PAerr, valid_flat_PI, pixel_scale, W1)

# binning the data
bin_size = 0.5 #arcsec
binned_midpoints, binned_angle_diff_cos, binned_sigma2_angle_diff, binned_PI2 = bin_pairwise_results(
    distances, angle_diff_cos, sigma2_angle_diff, PI2, bin_size
)

# correct data cos(angle_diff)
binned_angle_diff_cos_corrected = binned_angle_diff_cos/(1-0.5*binned_sigma2_angle_diff)

# dispesion function (left side of the equation)
dispersion_func = 1-binned_angle_diff_cos_corrected

# PI autocorrelation function
PI_autocorrection = binned_PI2/binned_PI2[0]

fig = plt.figure(figsize=(9,6))
plt.xlabel(r'$ \mathrm{Distance}\/\/ \ell\/\/ [\mathrm{arcsec}]$', fontsize=14)
plt.ylabel(r'$<P^2(\ell)>/<P^2(0)>$', fontsize=14)
plt.scatter(binned_midpoints*0.85, PI_autocorrection, color='black')
plt.savefig("fig/PI_JVLA.pdf", transparent=True, bbox_inches='tight')

# extracting only the first 15 data points of the dispersion fucntion to do fitting 
data_num_start = int(2/bin_size)
data_num = int(len(binned_midpoints)/2.5)
ell2_fit = binned_midpoints[data_num_start:data_num]**2
avg_cos_deltaPA_fit = dispersion_func[data_num_start:data_num]
ell2_plot = binned_midpoints[0:data_num]**2
avg_cos_deltaPA_plot = dispersion_func[0:data_num]

# effective cloud depth
W1 = np.sqrt(PA.bmaj*deg_to_arcsec*PA.bmin*deg_to_arcsec)/np.sqrt(8*math.log(2))  # beam "radius"
W2 = 51.96/2 # arcsec, resolution from the shortest baseline times 50% (Ching 2017)
Delta_prime = 9

def correlation_func(ell2, delta, PAdispersion, a_2_prime):

    N1 = ((delta**2 + 2*W1**2)*Delta_prime)/(np.sqrt(2*np.pi)*delta**3)
    N2 = ((delta**2 + 2*W2**2)*Delta_prime)/(np.sqrt(2*np.pi)*delta**3)
    N12 = ((delta**2 + W1**2 + W2**2)*Delta_prime)/(np.sqrt(2*np.pi)*delta**3)
    N = (1/N1 + 1/N2 - 2/N12)**-1

    taylor_series = a_2_prime*ell2 # keeping only the first term in taylor's series
    N_term = (N)/(1+N*(1/PAdispersion))
    N1_term = (1/N1)*(1-np.exp((-ell2)/(2*(delta**2 + 2*W1**2))))
    N2_term = (1/N2)*(1-np.exp((-ell2)/(2*(delta**2 + 2*W2**2))))
    N12_term = (1/N12)*(1-np.exp((-ell2)/(2*(delta**2 + W1**2 + W2**2))))

    correlation = taylor_series + N_term*(N1_term + N2_term - N12_term)

    return correlation

# simultaneously fitting all 3 parameters
#parameters, covariance = curve_fit(correlation_func, ell2_fit, avg_cos_deltaPA_fit, maxfev=100000)
parameters, covariance = curve_fit(correlation_func, ell2_fit, avg_cos_deltaPA_fit, bounds=([-np.inf, 0, -np.inf], [np.inf, 0.28, np.inf]), maxfev=100000)
#parameters, covariance = curve_fit(correlation_func, ell2_fit, avg_cos_deltaPA_fit, p0=(0.01,0.05,0.001), maxfev=100000)

# fitted parameters
fit_delta = parameters[0]
fit_PAdispersion4 = parameters[1]
fit_a_2_prime = parameters[2]
fit_PAdispersion4_deg = np.degrees(np.sqrt(fit_PAdispersion4))
# error
Perr = np.sqrt(np.diag(covariance))
fit_delta_err = Perr[0]
fit_PAdispersion4_err = Perr[1]
fit_a_2_prime_err = Perr[2]
fit_PAdispersion4_err_deg = np.degrees(np.sqrt(fit_PAdispersion4_err))

print("The beam radius W1 is", W1, 'arcsec')
print("The low freq filter W2 is", W2, 'arcsec')
print("The effective cloud depth is", Delta_prime, 'arcsec')
print("The sqaure of PA dispersion is fitted to be", fit_PAdispersion4, "The fitting error is", fit_PAdispersion4_err)
print("delta is fitted to be", fit_delta, 'arcsec', "The fitting error is", fit_delta_err)
print("a' is fitted to be", fit_a_2_prime, 'arcsec^-2', "The fitting error is", fit_a_2_prime_err)
print("The PA dispersion in deg", fit_PAdispersion4_deg, "The fitting error is", fit_PAdispersion4_err_deg)

correlation_func_fit = correlation_func(ell2_plot, fit_delta, fit_PAdispersion4, fit_a_2_prime)
linear_func_fit = linear_fuc(ell2_plot, fit_delta, fit_PAdispersion4, fit_a_2_prime)
fig = plt.figure(figsize=(9,5))
plt.xlabel(r'$ \mathrm{Distance}^2\/\/ \ell^2\/\/ [\mathrm{arcsec^2}]$', fontsize=14)
plt.ylabel(r'$[1 - <\mathrm{cos}(\Delta \Phi)>]$', fontsize=14)
plt.scatter(ell2_plot, avg_cos_deltaPA_plot, color='black')
plt.plot(ell2_plot, correlation_func_fit, '-', color='black', linewidth=1)
plt.plot(ell2_plot, linear_func_fit, '--', color='black', linewidth=1)
plt.xlim(0)
plt.ylim(0)
plt.savefig("fig/adf_JVLA.pdf", transparent=True, bbox_inches='tight')
