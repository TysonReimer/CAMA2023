"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import time

import os
import numpy as np
import matplotlib

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows
matplotlib.use('agg')

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.recon.algos import fd_das, fd_dmas, orr_recon
from umbms.recon.extras import get_fd_phase_factor, get_pix_ts

from umbms.tdelay.propspeed import estimate_speed

from umbms.hardware.antenna import apply_ant_t_delay, to_phase_center

from umbms.plot.imgs import plot_img


###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

__O_DIR = os.path.join(get_proj_path(), 'output/differential-d2/')
verify_path(__O_DIR)

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001
__SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)

__M_SIZE = 150  # Number of pixels along 1-dimension for reconstruction
__ROI_RAD = 8  # ROI radius, in [cm]

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    'A1': 5,
    'A2': 6,
    'A3': 7,
    'A11': 6,
    'A12': 5,
    'A13': 6.5,
    'A14': 6,
    'A15': 5.5,
    'A16': 7,
}

# Polar coordinate phi of the antenna at each position during the scan
__ANT_PHIS = np.flip(np.linspace(0, 355, 72) + -136.0)

###############################################################################


def do_ref_subtr(fd, md, cal_type='emp'):
    """Perform reference subtraction

    Parameters
    ----------
    fd : array_like
        The frequency domain S11, before calibration
    md :
        Metadata list, before calibration
    cal_type : str
        Must be in ['emp', 'adi'], indicates the type of reference
        subtraction to be performed (empty chamber or adipose-only)

    Returns
    -------
    cal_s11 : array_like
        The frequency domain S11 after calibration
    cal_md :
        The metadata of the scans retained after calibration
    """

    assert cal_type in ['emp', 'adi'], "cal_type must be in ['emp', 'adi']"

    # Define var for scan IDs
    scan_ids = np.array([ii['id'] for ii in md])

    # Find number of target scans; containing adipose and fibro shells
    n_tar = np.sum(np.array(['A' in ii['phant_id'] and 'F' in ii['phant_id']
                             for ii in md]))

    # Init arrays for storing s11/md from empty-reference subtraction
    cal_s11 = np.zeros([n_tar, np.size(fd, axis=1), np.size(fd, axis=2)],
                       dtype=complex)
    cal_md = []

    cc = 0  # Counter for indexing cal arrays

    for ii in range(np.size(fd, axis=0)):  # For each scan

        # True if scan had adipose and fibro shells
        is_tar = 'A' in md[ii]['phant_id'] and 'F' in md[ii]['phant_id']

        if is_tar:  # If a target scan

            # Get the reference index
            ref_idx = np.where(
                scan_ids == md[ii]['%s_ref_id' % cal_type])[0][0]

            # Perform reference subtraction
            cal_s11[cc, :, :] = fd[ii, :, :] - fd[ref_idx, :, :]

            cal_md.append(md[ii])  # Store md for this scan

            cc += 1  # Increment counter

    return cal_s11, cal_md


def recon_imgs(s11, idx_pairs, id_pairs, do_das=True, do_dmas=False,
               do_orr=False):
    """Reconstructs images, using DAS, DMAS, and/or ORR

    Parameters
    ----------
    s11 : array_like
        S11 data to be reconstructed
    idx_pairs : array_like
        Indices of left/right breasts, with respect to s11
    id_pairs : array_like
        IDs of the left/right breasts
    do_das : bool
        If True, will reconstruct DAS images
    do_dmas : bool
        If True, will reconstruct DMAS images
    do_orr : bool
        If True, will reconstruct ORR images
    """

    # Init array for storing DAS images
    das_imgs = np.zeros([np.size(s11_pair_diffs, axis=0),
                         __M_SIZE, __M_SIZE], dtype=complex)
    dmas_imgs = np.zeros([np.size(s11_pair_diffs, axis=0),
                         __M_SIZE, __M_SIZE], dtype=complex)
    orr_imgs = np.zeros([np.size(s11_pair_diffs, axis=0),
                         __M_SIZE, __M_SIZE], dtype=complex)

    # Make dir for storing DAS results
    das_o_dir = os.path.join(__O_DIR, 'das/')
    verify_path(das_o_dir)
    dmas_o_dir = os.path.join(__O_DIR, 'dmas/')
    verify_path(dmas_o_dir)
    orr_o_dir = os.path.join(__O_DIR, 'orr/')
    verify_path(orr_o_dir)

    # For each breast pair
    for ii in range(np.size(s11_pair_diffs, axis=0)):

        logger.info('\tWorking on pair [%4d / %4d]...'
                    % (ii + 1, np.size(s11_pair_diffs, axis=0)))

        # Get the metadata of the left/right breasts
        md_left = md[idx_pairs[ii, 0]]
        md_right = md[idx_pairs[ii, 1]]

        # If left breast has a tumour
        if ~np.isnan(md_left['tum_x']):
            tum_y = md_left['tum_y']
            if id_pairs[ii, 0] < 0:  # If the breast was mirrored
                tum_x = -md_left['tum_x']  # Mirror tum position
            else:
                tum_x = md_left['tum_x']
            tum_rad = md_left['tum_diam'] / 2

        # If right breast has a tumour
        elif ~np.isnan(md_right['tum_x']):
            tum_y = md_right['tum_y']
            if id_pairs[ii, 1] < 0:  # If the breast was mirrored
                tum_x = -md_right['tum_x']  # Mirror tum position
            else:
                tum_x = md_right['tum_x']
            tum_rad = md_right['tum_diam'] / 2
        else:  # If no breast had a tumour
            tum_x = 0
            tum_y = 0
            tum_rad = 0

        # Get the scan metadata explicitly
        ant_rad = md_left['ant_rad']
        adi_rad = __ADI_RADS[md_left['phant_id'].split('F')[0]]

        # Account for phase center of the antenna
        ant_rho = to_phase_center(meas_rho=ant_rad)

        # Estimate the propagation speed in the imaging domain
        speed = estimate_speed(adi_rad=adi_rad, ant_rho=ant_rho)

        # Get the approximate pixel time delays
        pix_ts = get_pix_ts(ant_rho=ant_rho,
                            m_size=__M_SIZE,
                            roi_rad=__ROI_RAD,
                            speed=speed,
                            ini_ant_ang=-136.0)

        # Correct for antenna t delay
        pix_ts = apply_ant_t_delay(pix_ts)

        # Get the phase factor for efficient computation
        phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

        # --- DAS Below this line ---------------------------------------------
        # Reconstruct the DAS image
        if do_das:
            das_start = time.time()
            das_imgs[ii, :, :] = fd_das(fd_data=s11[ii, :, :],
                                        phase_fac=phase_fac,
                                        freqs=recon_fs,
                                        n_cores=10)
            das_t = time.time() - das_start
            logger.info('\t\tDAS completed in %.1f sec' % das_t)

            plot_img(img=np.abs(das_imgs[ii, :, :]),
                     tar_xs=[tum_x],
                     tar_ys=[tum_y],
                     tar_rads=[tum_rad],
                     phant_rad=adi_rad,
                     roi_rho=__ROI_RAD,
                     save_str=os.path.join(das_o_dir, 'id_%d-%d.png'
                                           % (id_pairs[ii, 0],
                                              id_pairs[ii, 1])),
                     transparent=False,
                     save_close=True,
                     save_fig=True,
                     )

            # Save reconstructions every 100 scans to be safe
            if ii % 20 == 0:
                save_pickle(das_imgs,
                            os.path.join(das_o_dir,
                                         'das_up_to_ii_%d.pickle' % ii))

        # --- DAS Above this line ---------------------------------------------
        # --- DMAS Below this line --------------------------------------------
        if do_dmas:
            dmas_start = time.time()
            dmas_imgs[ii, :, :] = fd_dmas(fd_data=s11[ii, :, :],
                                          phase_fac=phase_fac, freqs=recon_fs,
                                          n_cores=10)
            dmas_t = time.time() - dmas_start
            logger.info('\t\tDMAS completed in %.1f sec' % dmas_t)

            plot_img(img=np.abs(dmas_imgs[ii, :, :]),
                     tar_xs=[tum_x],
                     tar_ys=[tum_y],
                     tar_rads=[tum_rad],
                     phant_rad=adi_rad,
                     roi_rho=__ROI_RAD,
                     save_str=os.path.join(dmas_o_dir, 'id_%d-%d.png'
                                           % (id_pairs[ii, 0],
                                              id_pairs[ii, 1])),
                     transparent=False,
                     save_close=True,
                     save_fig=True,
                     )

            # Save reconstructions every 10 scans to be safe
            if ii % 20 == 0:
                save_pickle(dmas_imgs,
                            os.path.join(dmas_o_dir,
                                         'dmas_up_to_ii_%d.pickle' % ii))

        # --- ORR Below this line ---------------------------------------------

        if do_orr:
            this_out = os.path.join(orr_o_dir, 'id-%d-%d/'
                                    % (id_pairs[ii, 0], id_pairs[ii, 1]))
            verify_path(this_out)
            orr_start = time.time()
            orr_imgs[ii, :, :] = \
                orr_recon(ini_img=np.zeros([__M_SIZE, __M_SIZE],
                                           dtype=complex),
                          freqs=recon_fs,
                          m_size=__M_SIZE,
                          fd=s11[ii, :, :],
                          roi_rho=__ROI_RAD,
                          phase_fac=phase_fac,
                          tar_xs=[tum_x],
                          tar_ys=[tum_y],
                          tar_rads=[tum_rad],
                          phant_rho=adi_rad,
                          out_dir=this_out,
                          logger=logger,
                          n_cores=10,
                          )
            orr_t = time.time() - orr_start
            logger.info('\t\tORR completed in %d min %.1f sec'
                        % (orr_t // 60, orr_t % 60))

            plot_img(img=np.abs(orr_imgs[ii, :, :]),
                     tar_xs=[tum_x],
                     tar_ys=[tum_y],
                     tar_rads=[tum_rad],
                     phant_rad=adi_rad,
                     roi_rho=__ROI_RAD,
                     save_str=os.path.join(orr_o_dir, 'id_%d-%d.png'
                                           % (id_pairs[ii, 0],
                                              id_pairs[ii, 1])),
                     transparent=False,
                     save_close=True,
                     save_fig=True,
                     )

            if ii % 3 == 0:  # Save reconstructions every 10 scans to be safe
                save_pickle(orr_imgs,
                            os.path.join(orr_o_dir,
                                         'orr_up_to_ii_%d.pickle' % ii))
        # --- ORR Above this line ---------------------------------------------

    if do_das:
        save_pickle(das_imgs, os.path.join(das_o_dir, 'das_imgs.pickle'))
    if do_dmas:
        save_pickle(dmas_imgs, os.path.join(dmas_o_dir, 'dmas_imgs.pickle'))
    if do_orr:
        save_pickle(orr_imgs, os.path.join(orr_o_dir, 'orr_imgs.pickle'))


def get_breast_pair_s11_diffs(s11_data, id_pairs, md):
    """

    Parameters
    ----------
    s11_data : array_like
        S11 dataset
    id_pairs : array_like
        The IDs of the left/right 'breast' scans for each pair

    Returns
    -------
    s11_pair_diffs : array_like
        The differences in the S11 of the left/right breasts
    idx_pairs : array_like
        Indices of the left/right breasts in the S11 data
    """

    if np.sum(id_pairs < 0) == 0:  # If no mirroring occurs

        idx_pairs = id_pairs - 1  # Convert from ID to index

        # Obtain the differential S11 data in the frequency domain
        s11_pair_diffs = (s11_data[idx_pairs[:, 0], :, :]
                          - s11_data[idx_pairs[:, 1], :, :])

    else:  # If sinogram mirroring is intended...

        # Find any IDs meant to be mirrored in sinogram space
        ids_to_mirror = id_pairs < 0

        # Reset id_pairs to be all positive for indexing later
        id_pairs = np.abs(id_pairs)

        idx_pairs = id_pairs - 1  # Convert from ID to index

        # Init array for return
        s11_pair_diffs = np.zeros([np.size(id_pairs, axis=0),
                                   np.size(s11_data, axis=1),
                                   np.size(s11_data, axis=2)],
                                  dtype=complex)

        # For each breast pair
        for ii in range(np.size(s11_pair_diffs, axis=0)):

            # Get breast data, pre-cal
            left_uncal = s11_data[idx_pairs[ii, 0], :, :]
            right_uncal = s11_data[idx_pairs[ii, 1], :, :]

            # Get the empty chamber reference data
            left_ref = s11_data[md[idx_pairs[ii, 0]]['emp_ref_id'] - 1, :, :]
            right_ref = s11_data[md[idx_pairs[ii, 1]]['emp_ref_id'] - 1, :, :]

            # Calibrate the left/right data by empty-chamber subtraction
            left_cal = left_uncal - left_ref
            right_cal = right_uncal - right_ref

            if ids_to_mirror[ii, 0]:  # If mirroring left breast

                left_s11 = get_mirrored_sino(left_cal)

            else:  # If not mirroring the left breast
                left_s11 = left_cal

            if ids_to_mirror[ii, 1]:  # If mirroring right breast
                right_s11 = get_mirrored_sino(right_cal)
            else:  # If not mirroring right breast
                right_s11 = right_cal

            # Perform left/right breast subtraction
            s11_pair_diffs[ii, :, :] = left_s11 - right_s11

    return s11_pair_diffs, idx_pairs


def get_mirrored_sino(sino):
    """

    Parameters
    ----------
    sino

    Returns
    -------

    """

    # Find the index of the antenna position nearest to 90deg
    # (i.e., nearest to the y-axis)
    y_int_pos = np.argmin(np.abs(__ANT_PHIS - 90))

    # Repeat the sinogram on the left/right to facilitate next steps
    padded_sino = np.tile(sino, reps=3)

    # Crop to obtain the rotated sinogram so that the y_int_pos is
    # at the middle
    rot_sino = padded_sino[:, 72 + (y_int_pos - 36): 72 + (y_int_pos + 36) + 1]

    # Mirror and remove last element (duplicate)
    mirror_sino = np.fliplr(rot_sino)[:, :-1]

    # Pad the mirrored version
    mirror_padded = np.tile(mirror_sino, reps=3)

    # Put sinogram back into original angle indexing format
    unrot_mirror = \
        mirror_padded[:, 72 + (36 - y_int_pos): 144 + (36 - y_int_pos)]

    return unrot_mirror


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the indices of pairs
    id_pairs = np.array(load_pickle(
        os.path.join(__D_DIR, 'differential/pairs_ideal_sym_mirror.pickle')))

    # Load the frequency-domain S11 and metadata
    s11 = load_pickle(os.path.join(__D_DIR, 'g3_s11.pickle'))
    md = load_pickle(os.path.join(__D_DIR, 'g3_md.pickle'))

    # Get the scan IDs
    scan_ids = np.array([ii['id'] for ii in md])

    # Store the frequencies used for reconstruction
    recon_fs = __SCAN_FS[__SCAN_FS >= 2e9]

    # Get the S11 differences and indices of the breast pairs
    s11_pair_diffs, idx_pairs = get_breast_pair_s11_diffs(s11_data=s11,
                                                          id_pairs=id_pairs,
                                                          md=md)

    # Retain only frequencies from 2-9 GHz
    s11_pair_diffs = s11_pair_diffs[:, __SCAN_FS >= 2e9, :]

    # Reconstruct images
    recon_imgs(
        s11=s11_pair_diffs,
        idx_pairs=idx_pairs,
        id_pairs=id_pairs,
        do_das=False,
        do_dmas=False,
        do_orr=True,
    )
