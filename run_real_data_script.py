#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from motor.motor_recon_met2_real_data import motor_recon_met2, obj1
from plot.plot_results_real_data import plot_real_data_slice
from plot.plot_mean_spectrum_slice import plot_mean_spectrum_slice

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Myelin Water Imaging')

parser.add_argument("--minTE", default=None, type=float, help="Minimum Echo Time (TE, units: ms)", required=True)

parser.add_argument("--nTE", default=32, type=int, help="Number of TEs", required=True)

parser.add_argument("--TR", default=None, type=float, help="Repetition Time (units: ms)", required=True)

parser.add_argument("--FA_method",
                    choices=["spline", "brute-force"],
                    required=True, type=str, default="spline", help="Method to estimate the flip angle (FA)")

parser.add_argument("--FA_smooth",
                    choices=["yes", "no"],
                    required=True, type=str, default="yes", help="Smooth FA map for a robust estimation")

parser.add_argument("--denoise",
                    choices=["TV", "NESMA", "None"],
                    required=True, type=str, default="NESMA", help="Denoise data")

parser.add_argument("--reg_method",
                    choices=["NNLS", "X2-I", "X2-L1", "X2-L2", "L_curve-I", "L_curve-L1", "L_curve-L2", "GCV-I", "GCV-L1", "GCV-L2"],
                    required=True, type=str, default="L_curve-L-cp", help="Reconstruction algorithm")

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/MET2/", required=True)

parser.add_argument("--input", default=None, type=str, help="Input data, e.g., Data.nii.gz", required=True)

parser.add_argument("--mask", default=None, type=str, help="Brain mask, e.g., Mask.nii.gz", required=True)

parser.add_argument("--savefig",
                    choices=["yes", "no"],
                    required=True, type=str, default="yes", help="Save reconstructed maps in .png")

parser.add_argument("--savefig_slice", default=30, type=int, help="Axial slice to save reconstructed maps, e.g., --Slice=30", required=True)

parser.add_argument("--numcores", default=-1, type=int, help="Number of cores used in the computation: -1 = all cores")

parser.add_argument("--myelin_T2", default=40, type=float, help="Maximum T2 for the myelin compartment: T2 threshold (units: ms)", required=True)

args = parser.parse_args()
# ==============================================================================
TE_min         = args.minTE
nTE            = args.nTE
TR             = args.TR
FA_method      = args.FA_method
FA_smooth      = args.FA_smooth
denoise        = args.denoise
reg_method     = args.reg_method
path_to_folder = args.path_to_folder
input_data     = args.input
mask           = args.mask
savefig        = args.savefig
Slice          = args.savefig_slice
num_cores      = args.numcores
myelin_T2      = args.myelin_T2
# ==============================================================================
start_time = time.time()

print ('Selected Options -> Estimation method: ' + reg_method + ', FA-estimation: ' +  FA_method +  ', Smooth-FA: ' +  FA_smooth + ', denoise: ' + denoise +  ', TR(ms): ' + repr(obj1(TR)))

path_to_data      = path_to_folder + input_data
path_to_mask      = path_to_folder + mask

path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '/'

try:
    os.mkdir(path_to_save_data)
except:
    print ('Warning: this folder already exists. Results will be overwritten')
#end try

# Define experimental parameters
TE_array = TE_min * np.arange(1,nTE+1)
TE_array = np.array(TE_array)

motor_recon_met2(TE_array, path_to_data, path_to_mask, path_to_save_data, TR, reg_method, denoise, FA_method, FA_smooth)

# ----------- PLOT reconstructed maps and spectra for a given Slice ------------
if savefig == 'yes':
    plot_real_data_slice(path_to_save_data, path_to_data, Slice, reg_method)
    plot_mean_spectrum_slice(path_to_save_data, path_to_mask, Slice, reg_method)
# end

print("--- %s seconds ---" % (time.time() - start_time))
