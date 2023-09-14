"""
Tyson Reimer
University of Manitoba
September 02nd, 2023
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path

from umbms.loadsave import load_pickle

from umbms.recon.breastmodels import get_roi

###############################################################################

__O_DIR_D1 = os.path.join(get_proj_path(), 'output/differential-d1/')
__O_DIR_D2 = os.path.join(get_proj_path(), 'output/differential-d2/')
__O_DIR_D3 = os.path.join(get_proj_path(), 'output/differential-d3/')

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

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

###############################################################################

if __name__ == "__main__":


    # Load the indices of pairs
    id_pairs = np.array(load_pickle(
        os.path.join(__D_DIR, 'differential/pairs_ideal_sym.pickle')))

    # Load the frequency-domain S11 and metadata
    s11 = load_pickle(os.path.join(__D_DIR, 'g3_s11.pickle'))
    md = load_pickle(os.path.join(__D_DIR, 'g3_md.pickle'))

    # Get the scan IDs
    scan_ids = np.array([ii['id'] for ii in md])

    # Load the image files
    orr_imgs_d1 = load_pickle(os.path.join(__O_DIR_D1, 'orr/orr_imgs.pickle'))
    das_imgs_d1 = load_pickle(os.path.join(__O_DIR_D1, 'das/das_imgs.pickle'))
    dmas_imgs_d1 = load_pickle(os.path.join(__O_DIR_D1, 'dmas/dmas_imgs.pickle'))

    orr_imgs_d2 = load_pickle(os.path.join(__O_DIR_D2, 'orr/orr_imgs.pickle'))
    das_imgs_d2 = load_pickle(os.path.join(__O_DIR_D2, 'das/das_imgs.pickle'))
    dmas_imgs_d2 = load_pickle(
        os.path.join(__O_DIR_D2, 'dmas/dmas_imgs.pickle'))

    orr_imgs_d3 = load_pickle(os.path.join(__O_DIR_D3, 'orr/orr_imgs.pickle'))
    das_imgs_d3 = load_pickle(os.path.join(__O_DIR_D3, 'das/das_imgs.pickle'))
    dmas_imgs_d3 = load_pickle(
        os.path.join(__O_DIR_D3, 'dmas/dmas_imgs.pickle'))


    # -------------------------------------------------------------------------

    d1_idx = 7
    d2_idx = 2
    d3_idx = 58

    d1_imgs = np.array([
        das_imgs_d1[d1_idx, :, :],
        dmas_imgs_d1[d1_idx, :, :],
        orr_imgs_d1[d1_idx, :, :]
    ])
    d1_idxs = id_pairs[d1_idx] - 1
    d2_idxs = id_pairs[d2_idx] - 1
    d3_idxs = id_pairs[d3_idx] - 1

    d2_imgs = np.array([
        das_imgs_d2[d2_idx, :, :],
        dmas_imgs_d2[d2_idx, :, :],
        orr_imgs_d2[d2_idx, :, :]
    ])

    d3_imgs = np.array([
        das_imgs_d3[d3_idx, :, :],
        dmas_imgs_d3[d3_idx, :, :],
        orr_imgs_d3[d3_idx, :, :]
    ])

    imgs = np.concatenate((d1_imgs, d2_imgs, d3_imgs))

    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno'), cax=cax)
    cbar.ax.tick_params(labelsize=16)

    axes[0, 0].text(0.5, 1.05,
                    "DAS",
                    transform=axes[0, 0].transAxes,
                    horizontalalignment='center',
                    fontsize=24)
    axes[0, 1].text(0.5, 1.05,
                    "DMAS",
                    transform=axes[0, 1].transAxes,
                    horizontalalignment='center',
                    fontsize=24)
    axes[0, 2].text(0.5, 1.05,
                    "ORR",
                    transform=axes[0, 2].transAxes,
                    horizontalalignment='center',
                    fontsize=24)

    axes[0, 0].text(-0.35, 0.5,
                    "D1",
                    transform=axes[0, 0].transAxes,
                    horizontalalignment='center',
                    rotation='vertical',
                    verticalalignment='center',
                    fontsize=24)

    axes[1, 0].text(-0.35, 0.5,
                    "D2",
                    transform=axes[1, 0].transAxes,
                    horizontalalignment='center',
                    rotation='vertical',
                    verticalalignment='center',
                    fontsize=24)

    axes[2, 0].text(-0.35, 0.5,
                    "D3",
                    transform=axes[2, 0].transAxes,
                    horizontalalignment='center',
                    rotation='vertical',
                    verticalalignment='center',
                    fontsize=24)


    for i, ax in enumerate(axes.ravel()):

        roi = get_roi(roi_rho=8,
                      m_size=np.size(imgs[i, :, :], axis=0),
                      arr_rho=8)
        img_to_plt = imgs[i, :, :] * np.ones_like(imgs[i, :, :])
        img_to_plt = np.abs(img_to_plt) / np.max(np.abs(img_to_plt))
        img_to_plt[~roi] = np.NaN

        # TODO: Fix these place holders
        if i <= 2:
            idxs_here = d1_idxs
        elif i > 2 and i <= 5:
            idxs_here = d2_idxs
        else:
            idxs_here = d3_idxs

        if ~np.isnan(md[idxs_here[0]]['tum_diam']):
            tum_rad = md[idxs_here[0]]['tum_diam'] / 2
            tum_x = md[idxs_here[0]]['tum_x']
            tum_y = md[idxs_here[0]]['tum_y']
        elif ~np.isnan(md[idxs_here[1]]['tum_diam']):
            tum_rad = md[idxs_here[1]]['tum_diam'] / 2
            tum_x = md[idxs_here[1]]['tum_x']
            tum_y = md[idxs_here[1]]['tum_y']
        else:
            tum_rad = 0
            tum_x = 0
            tum_y = 0
        md_here = md[idxs_here[0]]

        phant_rad = __ADI_RADS[md_here['phant_id'].split('F')[0]]

        draw_angs = np.linspace(0, 2 * np.pi, 1000)
        breast_xs, breast_ys = (phant_rad * np.cos(draw_angs),
                                phant_rad * np.sin(draw_angs))

        tick_bounds = [-8, 8, -8, 8]


        ax.imshow(np.abs(img_to_plt),
                  cmap='inferno',
                  extent=tick_bounds,
                  aspect='equal')
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])
        ax.plot(breast_xs, breast_ys, 'w--', linewidth=2)

        plt_xs = tum_rad * np.cos(draw_angs) + tum_x
        plt_ys = tum_rad * np.sin(draw_angs) + tum_x
        ax.plot(plt_xs, plt_ys, 'w', linewidth=2.0)

        if i == 0:
            ax.set_xlabel('x-axis (cm)', fontsize=16,
                          labelpad=-4)
            ax.set_ylabel('y-axis (cm)', fontsize=16)
        else:
            ax.set_xticks([], [])
            ax.set_yticks([], [])


    # plt.tight_layout()
    plt.show()
    o_dir = os.path.join(get_proj_path(), 'output/differential-paper-figs/')
    verify_path(o_dir)
    plt.savefig(os.path.join(o_dir, 'typical_recons.png'),
                bbox_inches='tight')