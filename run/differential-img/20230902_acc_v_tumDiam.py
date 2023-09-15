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

from umbms import get_proj_path

###############################################################################

__O_DIR = os.path.join(get_proj_path(), 'output/differential-d1/')

__TUM_DIAMS = np.array([10, 15, 20, 25, 30])  # in [mm]

__DAS_ACC = np.array([26, 44, 97, 100, 100])  # in [%]
__DMAS_ACC = np.array([26, 44, 94, 100, 100])  # in [%]
__ORR_ACC = np.array([41, 62, 100, 100, 100])  # in [%]

__ALGO_COLS = {
    'das': [0 / 255, 0 / 255, 0 / 255],
    'dmas': [80 / 255, 80 / 255, 80 / 255],
    'orr': [160 / 255, 160 / 255, 160 / 255],
}
###############################################################################

if __name__ == "__main__":

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=16)
    plt.grid(axis='y', zorder=0)
    plt.yticks(np.arange(0, 101, 10))
    plt.bar(x=[1, 5, 9, 13, 17],
            height=__DAS_ACC,
            label="DAS",
            color=__ALGO_COLS['das'],
            edgecolor='black',
            zorder=1
            )
    plt.bar(x=[2, 6, 10, 14, 18],
            height=__DMAS_ACC,
            label="DMAS",
            color=__ALGO_COLS['dmas'],
            edgecolor='black',
            zorder=1
            )
    plt.bar(x=[3, 7, 11, 15, 19],
            height=__ORR_ACC,
            label="ORR",
            color=__ALGO_COLS['orr'],
            edgecolor='black',
            zorder=1
            )
    plt.legend(fontsize=18)
    plt.xticks([2, 6, 10, 14, 18],
               __TUM_DIAMS)
    plt.xlabel("Tumour Diameter (mm)", fontsize=22)
    plt.ylabel("Diagnostic Accuracy (%)", fontsize=22)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__O_DIR,
                             'acc_by_tum_diam.png'),
                dpi=300)
