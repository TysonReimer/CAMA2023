"""
Tyson Reimer
University of Manitoba
December 14, 2018
"""

import numpy as np
from umbms.recon.breastmodels import get_breast


###############################################################################


def get_scr(img, roi_rad, adi_rad, tum_rad=1.5):
    """Returns the SCR of an image of a healthy phantom

    Returns the signal-to-clutter ratio (SCR) of a reconstructed image.

    Parameters
    ----------
    img : array_like
        The reconstructed image
    roi_rad : float
        The radius of the central region of interest (i.e., of the
        image), in [cm]
    adi_rad : float
        The radius used to approximate the breast region as a circle,
        in [cm]
    tum_rad : float
        The maximum tumour radius to be considered, in [cm]

    Returns
    -------
    scr : float
        The SCR of the reconstructed image, in [dB]
    """

    # Convert the complex-valued image to format used for display
    img_for_iqm = np.abs(img)**2

    # Find the conversion factor to convert pixel index to distance
    pix_to_dist = 2 * roi_rad / np.size(img, 0)

    # Set any NaN values to zero
    img_for_iqm[np.isnan(img_for_iqm)] = 0

    # Find the index of the maximum response in the reconstruction
    max_loc = np.argmax(img_for_iqm)

    # Find the x/y-indices of the max response in the reconstruction
    max_y_pix, max_x_pix = np.unravel_index(max_loc, np.shape(img))

    # Convert this to the x/y-positions
    max_x_pos = (max_x_pix - np.size(img, 0) // 2) * pix_to_dist
    max_y_pos = -1 * (max_y_pix - np.size(img, 0) // 2) * pix_to_dist

    # Create a model of the reconstruction, segmented by the various
    # tissue types - create a 'tumour region' assuming that a tumour
    # was actually present
    indexing_breast = get_breast(np.size(img, 0), roi_rho=roi_rad,
                                 adi_rad=adi_rad, adi_x=0, adi_y=0,
                                 fib_rad=0, fib_x=0, fib_y=0,
                                 tum_rad=tum_rad,
                                 tum_x=max_x_pos, tum_y=max_y_pos,
                                 skin_thickness=0,
                                 adi_perm=2, fib_perm=3, tum_perm=4,
                                 skin_perm=1, air_perm=1)

    # Determine the max values in the tumor region and the clutter
    # region
    sig_val = np.max(img_for_iqm[indexing_breast == 4])
    max_clut_val = np.max(
        img_for_iqm[np.logical_and(indexing_breast != 4,
                                   indexing_breast != 1)]
    )

    # Compute the SCR value, in [dB]
    scr = 20 * np.log10(sig_val / max_clut_val)

    return scr
