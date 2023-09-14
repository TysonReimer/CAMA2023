"""
Fatimah Eashour, Tyson Reimer
University of Manitoba
June 19, 2023

Pairing UM-BMID Gen 3 into left/right breast pairs.
The output is 4 datasets of pair ID numbers:
1.	Same adi, same fibro, same positions. 700 samples
    (500 healthy-unhealthy / H-U, 200 healthy-healthy / H-H).
2.	Same adi, same fibro, same positions, allow sinogram-reflecting
    to generate augmented samples. 2900 samples (2000 H-U, 900 H-H).
3.	Same adi, diff fibro, diff positions. 3700 samples
    (2500 H-U, 1200, H-H).
4.	Same adi, diff fibro, diff positions, allow sinogram-reflection.
    14 900 samples (10 000 H-U, 4900 H-H).
"""

import numpy as np

# TODO: Converg this to a file meant to be *run*

###############################################################################

# TODO: Clean below for PEP8
# TODO: Rewrite for simplicity, add comments
def sample_pairing(metadata, dataset_n):
    """

    Parameters
    ----------
    metadata :
        List of metadata dictionaries for each phantom scan
    dataset_n : int
        Number of the dataset, as defined at the top-level comment
        of this file

    Returns
    -------
    pair_id_ls :
        List of the ID pairs, corresponding to 'breast pairs'
    """

    pair_id_ls = []  # initialize an array to store IDs

    for left_data in metadata:  # For each 'left' breast

        for right_data in metadata:  # For each 'right' breast

            # Ensure that we have no duplicate pairs
            if left_data['id'] <= right_data['id']:

                # Was there a tumour in this pairing
                tum_here = not (np.isnan(left_data['tum_diam'])
                                or np.isnan(right_data['tum_diam']))

                # If we have a healthy-healthy *or* healthy-unhealthy
                # pair (i.e., exclude unhealthy-unhealthy cases)
                if tum_here:

                    if (left_data['phant_id'] == right_data['phant_id']
                        and (dataset_n == 1 or dataset_n == 2)) or \
                            (left_data['phant_id'][:left_data['phant_id'].index('F')] ==
                             right_data['phant_id'][:right_data['phant_id'].index('F')] and
                             (dataset_n == 3 or dataset_n == 4)):  # same fibro for 1&2, diff fibro for 3&4
                        if left_data['id'] < right_data['id']:  # np pairing for exactly the same breast scan
                            pair_id_ls.append([left_data['id'], right_data['id']])
                            if dataset_n == 2 or dataset_n == 4:
                                pair_id_ls.append([-left_data['id'], -right_data['id']])
                        if dataset_n == 2 or dataset_n == 4:
                            # allow sinogram-reflection (one breasts reflected)
                            # CAN pair the breast scan with its reflection
                            pair_id_ls.append([left_data['id'], -right_data['id']])
                            if left_data['id'] != right_data['id']:
                                pair_id_ls.append([-left_data['id'], right_data['id']])
    return pair_id_ls
