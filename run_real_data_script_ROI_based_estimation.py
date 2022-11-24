#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from tabulate import tabulate

from motor.motor_recon_met2_real_data_ROI import motor_recon_met2_ROIs

import time
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

# ======================= Define input parameters ==============================
parser = argparse.ArgumentParser(description='Myelin Water Imaging')

parser.add_argument("--path_to_folder", default=None, type=str, help="Path to the folder where the data is located, e.g., /home/Datasets/MET2/", required=True)

parser.add_argument("--input", default=None, type=str, help="Input data, e.g., Data.nii.gz", required=True)

parser.add_argument("--mask", default=None, type=str, help="Brain mask, e.g., Mask.nii.gz", required=True)

parser.add_argument("--ROIs", default=None, type=str, help="Brain ROIs, e.g., ROIs.nii.gz", required=True)

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

parser.add_argument("--reg_matrix",
                    choices=["I", "L1", "L2", "InvT2"],
                    required=True, type=str, default="I", help="Regularization matrix")

parser.add_argument("--myelin_T2_cutoff", default=40, type=float, help="Maximum T2 for the myelin compartment: T2 threshold (units: ms)", required=True)
parser.add_argument("--numcores", default=-1, type=int, help="Number of cores used in the computation: -1 = all cores")


args = parser.parse_args()
# ==============================================================================
TE_min           = args.minTE
nTE              = args.nTE
TR               = args.TR
FA_method        = args.FA_method
FA_smooth        = args.FA_smooth
denoise          = args.denoise
reg_matrix       = args.reg_matrix
path_to_folder   = args.path_to_folder
input_data       = args.input
mask             = args.mask
ROIs             = args.ROIs
myelin_T2_cutoff = args.myelin_T2_cutoff
num_cores        = args.numcores

# ==============================================================================
start_time = time.time()

path_to_data      = path_to_folder + input_data
path_to_mask      = path_to_folder + mask
path_to_ROIs      = path_to_folder + ROIs

path_to_save_data = path_to_folder + 'recon_all_X2'  +  '-'  + reg_matrix  +  '_ROI-based/'

headers =  [ 'Selected options             ',  '   '                   ]
table   =  [
           [ '1. Regularization matrix     ',  reg_matrix,             ],
           [ '2. FA estimation method      ',  FA_method,              ],
           [ '3. Smooth image for FA est.  ',  FA_smooth,              ],
           [ '4. Denoising method          ',  denoise,                ],
           [ '5. TR(ms)                    ',  TR                      ],
           [ '6. Min. TE                   ',  TE_min,                 ],
           [ '7. Number of TEs             ',  nTE,                    ],
           [ '8. Myelin T2-cutoff (ms)     ',  myelin_T2_cutoff,       ]
           ]

table_tabulated  = tabulate(table, headers=headers)
print('-------------------------------')
print(table_tabulated)

try:
    os.mkdir(path_to_save_data)
except:
    print('Warning: this folder already exists. Results will be overwritten')
#end try

# Define experimental parameters
TE_array = TE_min * np.arange(1,nTE+1)
TE_array = np.array(TE_array)

motor_recon_met2_ROIs(TE_array, path_to_data, path_to_mask, path_to_ROIs, path_to_save_data, TR, reg_matrix, denoise, FA_method, FA_smooth, myelin_T2_cutoff, num_cores)
print("--- %s seconds ---" % (time.time() - start_time))
