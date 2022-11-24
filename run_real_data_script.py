#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from tabulate import tabulate

from motor.motor_recon_met2_real_data import motor_recon_met2, obj1
from plot.plot_results_real_data import plot_real_data_slices
from plot.plot_mean_spectrum_slices import plot_mean_spectrum_slices

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Myelin Water Imaging')

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/MET2/", required=True)

parser.add_argument("--input", default=None, type=str, help="Input data, e.g., Data.nii.gz", required=True)

parser.add_argument("--mask", default=None, type=str, help="Brain mask, e.g., Mask.nii.gz", required=True)

parser.add_argument("--minTE", default=None, type=float, help="Minimum Echo Time (TE, units: ms)", required=True)

parser.add_argument("--nTE", default=32, type=int, help="Number of TEs", required=True)

parser.add_argument("--TR", default=None, type=float, help="Repetition Time (units: ms)", required=True)

parser.add_argument("--FA_method",
                    choices=["spline", "brute-force"],
                    required=True, type=str, default="spline", help="Method to estimate the flip angle (FA)")

parser.add_argument("--FA_smooth",
                    choices=["yes", "no"],
                    required=True, type=str, default="yes", help="Smooth data for estimating the FA")

parser.add_argument("--denoise",
                    choices=["TV", "NESMA", "None"],
                    required=True, type=str, default="None", help="Denoising method")

parser.add_argument("--reg_method",
                    choices=["NNLS", "T2SPARC", "X2", "L_curve", "GCV", "BayesReg"],
                    required=True, type=str, default="X2", help="Regularization algorithm")

parser.add_argument("--reg_matrix",
                    choices=["I", "L1", "L2", "InvT2"],
                    required=True, type=str, default="I", help="Regularization matrix")

parser.add_argument("--numcores", default=-1, type=int, help="Number of cores used in the computation: -1 = all cores")

parser.add_argument("--myelin_T2_cutoff", default=40, type=float, help="Maximum T2 for the myelin compartment: T2 threshold (units: ms)", required=True)

parser.add_argument("--savefig",
                    choices=["yes", "no"],
                    required=True, type=str, default="yes", help="Save reconstructed maps in .png")

parser.add_argument("--savefig_slice", default=30, type=int, help="Axial slice to save reconstructed maps, e.g., --Slice=30", required=True)

args = parser.parse_args()
# ==============================================================================
TE_min           = args.minTE
nTE              = args.nTE
TR               = args.TR
FA_method        = args.FA_method
FA_smooth        = args.FA_smooth
denoise          = args.denoise
reg_method       = args.reg_method
reg_matrix       = args.reg_matrix
path_to_folder   = args.path_to_folder
input_data       = args.input
mask             = args.mask
savefig          = args.savefig
Slice            = args.savefig_slice
num_cores        = args.numcores
myelin_T2_cutoff = args.myelin_T2_cutoff
# ==============================================================================
start_time = time.time()

path_to_data      = path_to_folder + input_data
path_to_mask      = path_to_folder + mask

if reg_method == 'NNLS' or reg_method =='T2SPARC':
    path_to_save_data = path_to_folder + 'recon_all_' + reg_method + '/'
else:
    path_to_save_data = path_to_folder + 'recon_all_' + reg_method +  '-'  + reg_matrix  +  '/'
#end

if reg_method == 'T2SPARC':
    reg_matrix = 'InvT2'
# end

headers =  [ 'Selected options             ',  '   '                   ]
table   =  [
           [ '1. Regularization method     ',  reg_method,             ],
           [ '2. Regularization matrix     ',  reg_matrix,             ],
           [ '3. FA estimation method      ',  FA_method,              ],
           [ '4. Smooth image for FA est.  ',  FA_smooth,              ],
           [ '5. Denoising method          ',  denoise,                ],
           [ '6. TR(ms)                    ',  TR                      ],
           [ '7. Min. TE                   ',  TE_min,                 ],
           [ '8. Number of TEs             ',  nTE,                    ],
           [ '9. Myelin T2-cutoff (ms)     ',  myelin_T2_cutoff,       ]
           ]

table_tabulated  = tabulate(table, headers=headers)
print ('-------------------------------')
print(table_tabulated)

try:
    os.mkdir(path_to_save_data)
except:
    print('Warning: this folder already exists. Results will be overwritten')
#end try

# Define experimental parameters
TE_array = TE_min * np.arange(1,nTE+1)
TE_array = np.array(TE_array)

motor_recon_met2(TE_array, path_to_data, path_to_mask, path_to_save_data, TR, reg_method, reg_matrix, denoise, FA_method, FA_smooth, myelin_T2_cutoff, num_cores)

# ----------- PLOT reconstructed maps and spectra for a given Slice ------------
if savefig == 'yes':
    plot_real_data_slices(path_to_save_data, path_to_data, Slice, reg_method)
    try:
        path_to_WM_mask = path_to_folder + 'Segmentation/Data_seg_pve_1.nii.gz'
        plot_mean_spectrum_slices(path_to_save_data, path_to_WM_mask, Slice, reg_method)
    except:
        path_to_WM_mask = path_to_mask
        plot_mean_spectrum_slices(path_to_save_data, path_to_WM_mask, Slice, reg_method)
        #print ('Warning: WM mask (Data_seg_pve_1.nii.gz) not available for plotting the spectra ')
    #end try
# end

print("--- %s seconds ---" % (time.time() - start_time))
