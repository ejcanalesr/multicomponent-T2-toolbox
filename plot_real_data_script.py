#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from plot.plot_results_real_data import plot_real_data_slice
from plot.plot_mean_spectrum_slice import plot_mean_spectrum_slice

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Plot T2 spectra')

parser.add_argument("--reg_method",
                    choices=["NNLS", "X2-I", "X2-L1", "X2-L2", "L_curve-I", "L_curve-L1", "L_curve-L2", "GCV-I", "GCV-L1", "GCV-L2"],
                    required=True, type=str, default="L_curve-L-cp", help="Reconstruction algorithm")

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/MET2/", required=True)

parser.add_argument("--mask", default=None, type=str, help="Brain mask, e.g., Mask.nii.gz", required=True)

parser.add_argument("--savefig_slice", default=30, type=int, help="Axial slice to save reconstructed maps, e.g., --Slice=30", required=True)

args = parser.parse_args()
# ==============================================================================
reg_method     = args.reg_method
path_to_folder = args.path_to_folder
mask           = args.mask
Slice          = args.savefig_slice
# ==============================================================================
path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '/'
path_to_mask      = path_to_save_data + mask

# ----------- PLOT reconstructed maps and spectra for a given Slice ------------
plot_mean_spectrum_slice(path_to_save_data, path_to_mask, Slice, reg_method)
