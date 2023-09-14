"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import seaborn as sns
import pandas as pd

import statsmodels.api as sm
from scipy.stats import norm

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows

import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle,save_pickle

from umbms.iqms.contrast import get_scr
from umbms.iqms.accuracy import get_loc_err

###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

__O_DIR = os.path.join(get_proj_path(), 'output/differential-d1/')
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


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the UM-BMID gen-3 metadata
    md = load_pickle(os.path.join(__D_DIR, 'g3_md.pickle'))

    # Load the indices of the metadata for the left/right breast pairs
    idx_pairs = load_pickle(os.path.join(__O_DIR, 'idx_pairs.pickle'))

    # Load the classifier DAS/DMAS/ORR predictions
    das_preds = load_pickle(os.path.join(__O_DIR, 'das_preds.pickle'))
    dmas_preds = load_pickle(os.path.join(__O_DIR, 'dmas_preds.pickle'))
    orr_preds = load_pickle(os.path.join(__O_DIR, 'orr_preds.pickle'))

    # --------- Creating dataframe for stat analysis... -----------------------

    # Get the metadata for the left/right breasts in each pair
    left_md = np.array(md)[idx_pairs[:, 0]]
    right_md = np.array(md)[idx_pairs[:, 1]]

    # Load phantom info file for phantom fib % vol and adi vol
    phant_info = np.genfromtxt(os.path.join(get_proj_path(), 'data/umbmid/',
                                            'phant_info.csv'),
                               delimiter=',',
                               dtype=['<U20', '<U20', float, float, float],
                               skip_footer=1)

    # All phantom IDs (ex: A2F1)
    phant_ids = np.array(['%s%s' % (ii[0], ii[1]) for ii in phant_info])

    phant_densities = dict()  # Init dict for storing fib % vol
    phant_vols = dict()  # Init dict for storing adi vol

    for ii in range(len(phant_ids)):  # For each AXFY combo...

        # Store the phantom fibro % vol
        phant_densities[phant_ids[ii]] = 100 * phant_info[ii][2]

        # Store the adipose volume in [cm^3]
        phant_vols[phant_ids[ii]] = phant_info[ii][3] / (10**3)

    unhealthy_df = pd.DataFrame()  # Init dataframe for tumour phants
    healthy_df = pd.DataFrame()  # Init dataframe for healthy phants

    tum_diams = np.zeros([len(left_md)])  # Init arr
    tum_polar_rad = np.zeros([len(left_md)])  # Init arr
    tum_zs = np.zeros([len(left_md)])  # Init arr
    tum_presence = np.zeros([len(left_md)], dtype=bool)
    densities = np.zeros([len(left_md)])
    adi_vols = np.zeros([len(left_md)])

    for ii in range(len(left_md)):  # For each scan...

        densities[ii] = phant_densities[left_md[ii]['phant_id']]
        adi_vols[ii] = phant_vols[left_md[ii]['phant_id']]

        # If the left breast had a tumour...
        if not np.isnan(left_md[ii]['tum_diam']):

            tum_presence[ii] = True
            tum_diams[ii] = left_md[ii]['tum_diam']
            tum_polar_rad[ii] = np.sqrt(left_md[ii]['tum_x']**2
                                        + left_md[ii]['tum_y']**2)
            tum_zs[ii] = left_md[ii]['tum_z']

        # If the right breast had a tumor
        elif not np.isnan(right_md[ii]['tum_diam']):

            tum_presence[ii] = 1
            tum_diams[ii] = right_md[ii]['tum_diam']
            tum_polar_rad[ii] = np.sqrt(right_md[ii]['tum_x']**2
                                        + right_md[ii]['tum_y']**2)
            tum_zs[ii] = right_md[ii]['tum_z']

        else:  # If neither breast had a tumour

            tum_presence[ii] = False

            # Set values to NaN...
            tum_diams[ii] = np.NaN
            tum_polar_rad[ii] = np.NaN
            tum_zs[ii] = np.NaN

    # Store in healthy dataframe
    healthy_df['density'] = densities[~tum_presence]
    healthy_df['adi_vol'] = adi_vols[~tum_presence]

    # Store in unhealthy dataframe
    unhealthy_df['density'] = densities[tum_presence]
    unhealthy_df['adi_vol'] = adi_vols[tum_presence]
    unhealthy_df['tum_diam'] = tum_diams[tum_presence]
    unhealthy_df['tum_polar_rad'] = tum_polar_rad[tum_presence]

    # --------- Do Stat analysis on DAS... ------------------------------------

    unhealthy_df['pred_correct'] = (orr_preds[tum_presence] == 1).astype('int')
    healthy_df['pred_correct'] = (orr_preds[~tum_presence] == 0).astype('int')

    # Create logistic regression model
    healthy_model = sm.GLM.from_formula(
        "pred_correct ~  "
        " adi_vol "
        " + density",
        family=sm.families.Binomial(),
        data=healthy_df)
    healthy_results = healthy_model.fit()

    unhealthy_model = sm.GLM.from_formula(
        "pred_correct ~  "
        " adi_vol "
        " + density"
        " + tum_diam",
        family=sm.families.Binomial(),
        data=unhealthy_df)
    unhealthy_results = unhealthy_model.fit()

    # Report results
    logger.info('HEALTHY RESULTS...')
    logger.info(healthy_results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % healthy_results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in healthy_results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = healthy_results.params[ii]
        std_err = healthy_results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)

        # Get p-val
        pval = healthy_results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3f\t(%.3f,\t%.3f)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)


    # Report results
    logger.info('UNHEALTHY RESULTS...')
    logger.info(unhealthy_results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % unhealthy_results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in unhealthy_results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = unhealthy_results.params[ii]
        std_err = unhealthy_results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)

        # Get p-val
        pval = unhealthy_results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3f\t(%.3f,\t%.3f)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)