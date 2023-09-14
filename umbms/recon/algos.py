"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from functools import partial

from umbms import null_logger

from umbms.loadsave import save_pickle

from umbms.recon.fwdproj import fd_fwd_proj
from umbms.recon.optimfuncs import get_ref_derivs

from umbms.plot.imgs import plot_img

###############################################################################


def orr_recon(ini_img, freqs, m_size, fd, roi_rho, phase_fac,
              tar_xs=None, tar_ys=None, tar_rads=None,
              phant_rho=0.0,
              n_cores=2, step_size=0.03,
              out_dir='',
              logger=null_logger):
    """Perform optimization-based radar reconstruction, via grad desc

    Parameters
    ----------
    ini_img : array_like
        Initial image estimate
    freqs : array_like
        The frequencies used in the scan, in [Hz]
    m_size : int
        The number of pixels along one dimension of the reconstructed image
    fd : array_like
        The measured frequency domain data
    roi_rho : float
        The radius of the reconstructed image-space, in [cm]
    phase_fac : array_like
        Phase factor array for efficient computation
    tar_xs : list, optional
        List of the x-positions of the targets, in [cm]
    tar_ys : list, optional
        List of the y-positions of the targets, in [cm]
    tar_rads : list, optional
        List of the radii of the targets, in [cm]
    phant_rho : float, optional
        The approximate radius of the presumably circular phantom,
        for plotting only, in [cm]
    n_cores : int, optional
        The number of cores to use for parallel processing
    step_size : float, optional
        The step size to use for gradient descent
    out_dir : str, optional
        The output directory, where the figures and image estimates will
        be saved
    logger :
        Logging object

    Returns
    -------
    img : array_like
        Reconstructed image
    """

    # Get the area of each individual pixel, in [cm^2]
    dv = ((2 * roi_rho / 100)**2) / (m_size**2)

    img = ini_img  # Initialize the image

    # Forward project the current image estimate
    fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv,
                      n_cores=n_cores,
                      freqs=freqs,
                      )

    cost_funcs = []  # Init list for storing cost function values
    img_estimates = []  # Init list for storing image estimates

    # Store the initial cost function value
    cost_funcs.append(float(np.sum(np.abs(fwd - fd)**2)))

    # Initialize the number of steps performed in gradient descent
    step = 0

    # Initialize the relative change in the cost function
    cost_rel_change = 1

    logger.info('\tInitial cost value:\t%.4f' % cost_funcs[0])

    # Perform gradient descent until the relative change in the cost
    # function is less than 0.1%
    while cost_rel_change > 0.001:

        logger.info('\tStep %d...' % (step + 1))

        # Calculate the gradient of the loss function wrt the
        # reflectivities in the object space
        ref_derivs = get_ref_derivs(phase_fac=phase_fac, fd=fd, fwd=fwd,
                                    freqs=freqs, n_cores=n_cores,
                                    )

        # Update image estimate
        img -= step_size * np.real(ref_derivs)

        # Store the updated image estimate
        img_estimates.append(img * np.ones_like(img))

        # Plot the map of the loss function derivative with respect to
        # each reflectivity point
        plot_img(np.real(ref_derivs),
                 tar_xs=tar_xs,
                 tar_ys=tar_ys,
                 tar_rads=tar_rads,
                 phant_rad=phant_rho,
                 roi_rho=roi_rho,
                 title="Full Deriv Step %d" % (step + 1),
                 save_str=os.path.join(out_dir,
                                       "fullDeriv_step_%d.png"
                                       % (step + 1)),
                 save_fig=True,
                 save_close=True,
                 cbar_fmt='%.2e',
                 transparent=False)

        plot_img(np.abs(img),
                 tar_xs=tar_xs,
                 tar_ys=tar_ys,
                 tar_rads=tar_rads,
                 phant_rad=phant_rho,
                 roi_rho=roi_rho,
                 title="Image Estimate Step %d" % (step + 1),
                 save_str=os.path.join(out_dir,
                                       "imageEstimate_step_%d_abs.png"
                                       % (step + 1)),
                 save_fig=True,
                 save_close=True,
                 cbar_fmt='%.2e',
                 transparent=False)

        # Forward project the current image estimate
        fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv,
                          n_cores=n_cores,
                          freqs=freqs,
                          )

        # Normalize the forward projection
        cost_funcs.append(np.sum(np.abs(fwd - fd) ** 2))

        logger.info('\t\tCost func:\t%.4f' % (cost_funcs[step + 1]))

        # Calculate the relative change in the cost function
        cost_rel_change = ((cost_funcs[step] - cost_funcs[step + 1])
                           / cost_funcs[step])
        logger.info('\t\t\tCost Func ratio:\t%.4f%%'
                    % (100 * cost_rel_change))

        if step >= 1:  # For each step after the 0th

            # Plot the value of the cost function vs the number of
            # gradient descent steps performed
            plt.figure(figsize=(12, 6))
            plt.rc('font', family='Times New Roman')
            plt.tick_params(labelsize=18)
            plt.plot(np.arange(1, step + 2), cost_funcs[:step + 1],
                     'ko--')
            plt.xlabel('Iteration Number', fontsize=22)
            plt.ylabel('Cost Function Value', fontsize=22)
            plt.title("Optimization Performance with Gradient Descent",
                      fontsize=24)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir,
                                     "costFuncs_step_%d.png" % (
                                             step + 1)),
                        transparent=False,
                        dpi=300)
            plt.close()

        step += 1  # Increment the step counter

    # After completing image reconstruction, plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.plot(np.arange(1, len(cost_funcs) + 1), cost_funcs, 'ko--')
    plt.xlabel('Iteration Number', fontsize=22)
    plt.ylabel('Cost Function Value', fontsize=22)
    plt.title("Optimization Performance with Gradient Descent",
              fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "costFuncs.png"),
                transparent=True,
                dpi=300)
    plt.close()

    # Save the image estimates to a .pickle file
    save_pickle(img_estimates, os.path.join(out_dir, 'img_estimates.pickle'))

    return img


###############################################################################


def fd_das(fd_data, phase_fac, freqs, n_cores=2):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    n_fs = np.size(freqs)  # Find number of frequencies used

    # Correct for to/from propagation
    new_phase_fac = phase_fac**(-2)

    # Create func for parallel computation
    parallel_func = partial(_parallel_fd_das_func, fd_data, new_phase_fac,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_fs)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections,
                                  [n_fs, np.size(phase_fac, axis=1),
                                   np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Sum over all frequencies
    img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_func(fd_data, new_phase_fac, freqs, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection


###############################################################################


def fd_dmas(fd_data, phase_fac, freqs, n_cores=2):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    # Find number of antenna positions
    n_ants = np.size(phase_fac, axis=0)

    # Correct for to/from propagation
    new_phase_fac = phase_fac**(-2)

    # Create func for parallel computation
    parallel_func = partial(_parallel_fd_dmas_func, fd_data, new_phase_fac,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_ants)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections,
                                  [n_ants, np.size(phase_fac, axis=1),
                                   np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Init image to return
    img = np.zeros([np.size(phase_fac, axis=1), np.size(phase_fac, axis=1)],
                   dtype=complex)

    # Loop over each antenna position
    for aa in range(n_ants):

        # For each other antenna position
        for aa_2 in range(aa + 1, n_ants):

            # Add the pair-wise multiplication
            img += (back_projections[aa, :, :] * back_projections[aa_2, :, :])

    return img


def _parallel_fd_dmas_func(fd_data, new_phase_fac, freqs, aa):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    aa : int
        Antenna position index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this antenna position
    this_phase_fac = new_phase_fac[aa, :, :] ** freqs[:, None, None]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[:, aa, None, None],
                             axis=0)

    return this_projection
