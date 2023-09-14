"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import os
import numpy as np
import matplotlib

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows
matplotlib.use('tkagg')
import seaborn as sns

import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle,save_pickle

from umbms.iqms.contrast import get_scr
from umbms.iqms.accuracy import get_loc_err

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

__SCR_THRESH = 1.5

# Define colors for plotting
das_col = [0, 0, 0]
dmas_col = [80, 80, 80]
gd_col = [160, 160, 160]
das_col = [ii / 255 for ii in das_col]
dmas_col = [ii / 255 for ii in dmas_col]
gd_col = [ii / 255 for ii in gd_col]

###############################################################################


def get_breast_pair_s11_diffs(s11_data, id_pairs, md):
    """Get the breast pair S11 FD data and index pairs

    Parameters
    ----------
    s11_data : array_like
        S11 dataset
    id_pairs : array_like
        The IDs of the left/right 'breast' scans for each pair
    md :
        Metadata for each scan

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
    """Get the sinogram that would be obtained from mirrored breast scan

    Parameters
    ----------
    sino : array_like
        Sinogram (frequency or time domain) measured

    Returns
    -------
    unrot_mirror : array_like
        The sinogram that would be obtained if a mirrored (about y-axis)
        version of the breast phantom was scanned
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


def get_sens_spec(imgs, scr_thresh=__SCR_THRESH):
    """Get the sensitivity and specificity at a given SCR threshold

    Parameters
    ----------
    imgs : array_like
        Images to be analyzed
    scr_thresh : float
        The threshold of the SCR used to determine if a tumour-like
        response was in the image, in [dB]

    Returns
    -------
    sensitivity : float
        The diagnostic sensitivity, in percentage
    specificity : float
        The diagnostic specificity, in percentage
    """

    # Init arr for storing results
    unhealthy_detects = []  # Tumour detection in unhealthy cases
    healthy_detects = []  # Tumour detection in healthy cases

    # For each breast pair
    for ii in range(np.size(imgs, axis=0)):

        # Get the metadata of the left/right breasts
        md_left = md[idx_pairs[ii, 0]]
        md_right = md[idx_pairs[ii, 1]]

        # If left breast has a tumour
        if ~np.isnan(md_left['tum_x']):

            tum_y = md_left['tum_y']  # Get the y-position of the tumor

            if id_pairs[ii, 0] < 0:  # If the breast was mirrored
                tum_x = -md_left['tum_x']  # Mirror tum position

            else:  # If breast was not mirrored
                tum_x = md_left['tum_x']

            # Get the tumour radius
            tum_rad = md_left['tum_diam'] / 2

        # If right breast has a tumour
        elif ~np.isnan(md_right['tum_x']):

            tum_y = md_right['tum_y']  # Get y-position of tumour

            if id_pairs[ii, 1] < 0:  # If the breast was mirrored
                tum_x = -md_right['tum_x']  # Mirror tum position

            else:  # If breat was not mirrored
                tum_x = md_right['tum_x']

            tum_rad = md_right['tum_diam'] / 2

        else:  # If no breast had a tumour

            # Set tumour x/y/ rad to zeros
            tum_x, tum_y, tum_rad = 0, 0, 0

        # Get the scan metadata explicitly
        adi_rad = __ADI_RADS[md_left['phant_id'].split('F')[0]]

        if tum_rad != 0:  # If there was a tumour

            # Get the signal to clutter ratio (SCR)
            # NOTE: The 'healthy' function is being used to define
            # the response ROI based on where the max response actually
            # was in the imaging chamber
            scr = get_scr(img=imgs[ii, :, :],
                          roi_rad=__ROI_RAD,
                          adi_rad=adi_rad,
                          tum_rad=tum_rad,
                          )

            # Get the localization error
            loc_err = get_loc_err(img=imgs[ii, :, :],
                                  roi_rho=__ROI_RAD,
                                  tum_x=tum_x,
                                  tum_y=tum_y
                                  )

            # Determine whether the tumour was accurately detected
            # (i.e., tumour-like response was in the image and
            # corresponded to the true tumour)
            tum_detected = ((scr >= scr_thresh)
                            and loc_err <= (tum_rad + 0.5))

            unhealthy_detects.append(tum_detected)  # Store result

        else:  # If both breasts had no tumour

            # Get the SCR for the healthy breast
            scr = get_scr(img=imgs[ii, :, :],
                          roi_rad=__ROI_RAD,
                          adi_rad=adi_rad,
                          tum_rad=1.5,
                          )

            # If there was a tumour-like response in the image
            tum_detected = scr >= scr_thresh

            # Store the result
            healthy_detects.append(tum_detected)

    # Calculate the sensitivity and specificity, as percentages
    sensitivity = 100 * np.sum(unhealthy_detects) / len(unhealthy_detects)
    specificity = 100 * (np.sum(1 - np.array(healthy_detects))
                         / len(healthy_detects))

    return sensitivity, specificity


def get_sens_spec_for_roc(imgs, scr_thresholds):
    """Get the sensitivity, specificity to construct an ROC curve

    Parameters
    ----------
    imgs : array_like
        Reconstructed images to be analyzed
    scr_thresholds : array_like
        The SCR thresholds to be used to determine sensitivity and
        specificity

    Returns
    -------
    sensitivities : array_like
        The diagnostic sensitivity at each scr_threshold
    specificities : array_like
        The diagnostic specificity at each scr_threshold
    """

    # Init arrays for storing points on the ROC curve
    sensitivities = np.zeros_like(scr_thresholds)
    specificities = np.zeros_like(scr_thresholds)

    for ii in range(len(scr_thresholds)):  # For each threshold

        # Report to logger
        logger.info('SCR thresh [%4d / %4d]... ' % (ii, len(scr_thresholds)))

        # Determine sensitivity and specificity
        sensitivities[ii], specificities[ii] = \
            get_sens_spec(imgs=imgs, scr_thresh=scr_thresholds[ii])

    return sensitivities, specificities


def get_roc_auc(sens, spec):
    """Get the area under the curve (AUC) of the ROC curve

    Parameters
    ----------
    sens : array_like
        The sensitivities used to define the ROC curve
    spec : array_like
        The specificities used to define the ROC curve

    Returns
    -------
    auc : float
        The area under the curve of the ROC curve
    """

    sorted_idx = np.argsort((100 - spec))  # Get indices for sorting

    # Get the true positives and false positives
    tps = sens[sorted_idx] / 100
    fps = (100 - spec[sorted_idx]) / 100

    auc = 0  # Init value of the AUC

    for ii in range(len(tps) - 1):  # For each point on the ROC curve

        # Calculate the false positive difference
        d_fps = fps[ii + 1] - fps[ii]

        if d_fps != 0:  # If the false positive difference is nonzero

            # Add the area of this rectangle to the AUC
            auc += d_fps * tps[ii + 1]

    auc *= 100  # Convert from decimal-form to percentage
    auc = float(auc)  # Ensure float type for later

    return auc


def do_threshold_classif(imgs):
    """Do maximum image intensity-based diagnosis

    Parameters
    ----------
    imgs : array_like
        Reconstructed images to be analyzed

    Returns
    -------
    sensitivities : array_like
        The diagnostic sensitivity at each scr_threshold
    specificities : array_like
        The diagnostic specificity at each scr_threshold
    preds
    """

    # Get the max intensity in each image
    img_maxes = np.max(np.abs(imgs), axis=(1,2))

    # Init arr for storing the known diagnostic labels of each image
    img_labels = np.zeros([np.size(imgs, axis=0),], dtype=bool)

    # For each breast pair
    for ii in range(np.size(imgs, axis=0)):

        # Get the metadata of the left/right breasts
        md_left = md[idx_pairs[ii, 0]]
        md_right = md[idx_pairs[ii, 1]]

        # If a breast has a tumour
        if ~np.isnan(md_left['tum_x']) or ~np.isnan(md_right['tum_x']):
            img_labels[ii] = 1

        else:  # If no breast had a tumour
            img_labels[ii] = 0

    # Plot the distributions of max image intensities
    plt.figure(figsize=(9, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    sns.histplot(img_maxes[img_labels], label='Tumour-Containing', kde=True,
                 color='r', alpha=0.3)
    sns.histplot(img_maxes[~img_labels], label='Healthy', kde=True,
                 color='k', alpha=0.3)
    plt.legend(fontsize=16)
    plt.xlabel('Maximum Image Intensity', fontsize=22)
    plt.ylabel('Kernel Density Estimate', fontsize=22)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__O_DIR, 'max_intensity_distro.png'), dpi=300,
                transparent=False)

    # Define detection thresholds
    det_thresholds = np.linspace(0, 1, 1000) * np.max(img_maxes)

    # Init arrays for storing diagnostic metrics
    sensitivities = np.zeros_like(det_thresholds)
    specificities = np.zeros_like(det_thresholds)

    for ii in range(len(det_thresholds)):  # For each threshold

        # The predictions at this detection threshold
        preds = img_maxes >= det_thresholds[ii]

        # Calculate the true positives, false negatives,
        # true negatives, and false positives
        tp = np.sum(np.logical_and(preds, img_labels))
        fn = np.sum(np.logical_and(~preds, img_labels))
        tn = np.sum(np.logical_and(~preds, ~img_labels))
        fp = np.sum(np.logical_and(preds, ~img_labels))

        # Calculate sensitivity/specificity in percentage
        sensitivities[ii] = 100 * tp / (tp + fn)
        specificities[ii] = 100 * tn / (tn + fp)

    # ---- New 

    preds = get_best_pred(imgs=imgs, sens=sensitivities,
                          specs=specificities,
                          det_thresholds=det_thresholds)

    return sensitivities, specificities, preds


def get_best_pred(imgs, sens, specs, det_thresholds):
    """Get classif predictions based on best point on ROC curve

    Parameters
    ----------
    sens
    specs

    Returns
    -------

    """

    # Find the best index
    best_idx = np.argmin(np.sqrt((1 - sens / 100)**2 + (1 - specs / 100)**2))

    # Threshold corresponding to best point on ROC
    best_thresh = det_thresholds[best_idx]

    print('Best thresh:\t%.3f' % (best_thresh / np.max(imgs)))

    # Get the max intensity in each image
    img_maxes = np.max(np.abs(imgs), axis=(1, 2))

    preds = img_maxes >= best_thresh  # Get best predictions

    return preds



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

    # Load the image files
    # orr_imgs = load_pickle(os.path.join(__O_DIR, 'orr/orr_imgs.pickle'))
    # das_imgs = load_pickle(os.path.join(__O_DIR, 'das/das_imgs.pickle'))
    # dmas_imgs = load_pickle(os.path.join(__O_DIR, 'dmas/dmas_imgs.pickle'))
    das_imgs = load_pickle(os.path.join(__O_DIR, 'das/das_illia.pickle'))
    dmas_imgs = load_pickle(os.path.join(__O_DIR, 'dmas/dmas_illia.pickle'))

    # Get the S11 differences and indices of the breast pairs
    s11_pair_diffs, idx_pairs = get_breast_pair_s11_diffs(s11_data=s11,
                                                          id_pairs=id_pairs,
                                                          md=md)

    # Get sens/spec for each recon method using max intensity
    # threhsolding
    das_sens, das_spec, das_preds = do_threshold_classif(imgs=das_imgs[:1000, :, :])
    dmas_sens, dmas_spec, dmas_preds = do_threshold_classif(imgs=dmas_imgs[:1000, :, :])
    # orr_sens, orr_spec, orr_preds = do_threshold_classif(imgs=orr_imgs)

    # Do SCR-based analysis...
    scr_thresholds = np.linspace(0, 30, 100)
    # orr_sens, orr_spec = get_sens_spec_for_roc(imgs=orr_imgs,
    #                                            scr_thresholds=scr_thresholds)
    # das_sens, das_spec = get_sens_spec_for_roc(imgs=das_imgs[:1000, :, :],
    #                                            scr_thresholds=scr_thresholds)
    # dmas_sens, dmas_spec = get_sens_spec_for_roc(imgs=dmas_imgs[:1000, :, :],
    #                                              scr_thresholds=scr_thresholds)

    # Get the AUC for each recon method
    das_auc = get_roc_auc(sens=das_sens, spec=das_spec)
    dmas_auc = get_roc_auc(sens=dmas_sens, spec=dmas_spec)
    # orr_auc = get_roc_auc(sens=orr_sens, spec=orr_spec)

    # Make the figure
    plt.figure(figsize=(10, 8))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=(20))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(100 - das_spec, das_sens, c=das_col, linestyle='-',
             label="DAS (AUC: %.1f%%)" % (das_auc))
    plt.plot(100 - dmas_spec, dmas_sens, c=dmas_col, linestyle='--',
             label="DMAS (AUC: %.1f%%)"
                   % (dmas_auc))
    # plt.plot(100 - orr_spec, orr_sens, c=gd_col, linestyle='--',
    #          label="ORR (AUC: %.1f%%)"
    #                % (orr_auc))
    plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100),
             c=[0, 108 / 255, 255 / 255], linestyle='--',
             label='Random Classifier')
    plt.legend(fontsize=16, loc='lower right')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel("False Positive Rate (%)", fontsize=22)
    plt.ylabel("True Positive Rate (%)", fontsize=22)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__O_DIR, 'rocs_illia_MAX.png'),
                dpi=300,
                transparent=False)

    # save_pickle(das_preds, os.path.join(__O_DIR, 'das_preds.pickle'))
    # save_pickle(dmas_preds, os.path.join(__O_DIR, 'dmas_preds.pickle'))
    # save_pickle(orr_preds, os.path.join(__O_DIR, 'orr_preds.pickle'))
    # save_pickle(idx_pairs, os.path.join(__O_DIR, 'idx_pairs.pickle'))
