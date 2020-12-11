#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from plot.plot_results_real_data import plot_real_data_slices
from plot.plot_mean_spectrum_slices import plot_mean_spectrum_slices

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Plot T2 spectra')

parser.add_argument("--reg_method",
                    choices=["NNLS", "T2SPARC", "X2", "L_curve", "GCV", "BayesReg"],
                    required=True, type=str, default="X2", help="Regularization algorithm")

parser.add_argument("--reg_matrix",
                    choices=["I", "L1", "L2", "InvT2"],
                    required=True, type=str, default="I", help="Regularization matrix")

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/MET2/", required=True)

parser.add_argument("--path_to_mask", default=None, type=str, help="Path to the folder where the Mask is locate", required=True)

parser.add_argument("--savefig_slice", default=30, type=int, help="Axial slice to save reconstructed maps, e.g., --Slice=30", required=True)

args = parser.parse_args()
# ==============================================================================
reg_method     = args.reg_method
reg_matrix     = args.reg_matrix
path_to_folder = args.path_to_folder
path_to_mask   = args.path_to_mask
Slice          = args.savefig_slice
# ==============================================================================
#path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '/'

if reg_method == 'NNLS' or reg_method == 'T2SPARC':
    path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '/'
else:
    path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '-' + reg_matrix +  '/'
#end

#path_to_mask      = path_to_save_data + mask

# ----------- PLOT reconstructed maps and spectra for a given Slice ------------
#plot_mean_spectrum_slices(path_to_save_data, path_to_mask, Slice, reg_method)
plot_mean_spectrum_slices(path_to_save_data, path_to_mask, Slice, reg_method)
