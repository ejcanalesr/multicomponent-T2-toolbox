#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from scipy.signal import find_peaks, deconvolve

import math
import warnings

from scipy import stats, sparse, ndimage, linalg
from scipy.stats import norm, linregress, pearsonr, pearsonr, wasserstein_distance, entropy
from scipy.linalg import cholesky
from scipy.spatial import distance

from tabulate import tabulate
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import skill_metrics as sm

import joypy
import pdb

import os
import sys
from sys import version_info
#sys.path.append("..")  #one level
sys.path.append("../..") #two levels

from epg.epg import *
from flip_angle_algorithms.fa_estimation import *
from intravoxel_algorithms.algorithms import *
from motor.motor_recon_met2_real_data import create_Laplacian_matrix, obj0

#____________________________ functions _______________________________________#
def Signal_MET2_dist(x, nEchoes, tau, T2grid, T1grid, dist, TR):
    # -------------------------- Parameters ------------------------------------
    # ------- Related to the multi-echo signal
    # x[0]  = f_intra_extra        (volume)
    # x[1]  = T2_intra_extra       (T2)
    # x[2]  = f_myelin             (volume)
    # x[3]  = T2_myelin            (T2)
    # x[4]  = f_csf                (volume)
    # x[5]  = Km                   (Spin-density x global constant)
    # x[6]  = flip_angle
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    rad         =  np.pi/180.0  # constant to convert degrees to radians
    signal_all  =  (1.0 - np.exp(-TR/T1grid)) * epg_signal(nEchoes, tau, np.array(1.0/T1grid), np.array(1.0/T2grid), x[6] * rad, x[6]/2.0 * rad)
    Ms          =  x[5] * signal_all * dist
    Ms          =  np.sum(Ms, axis=1)
    return Ms
# end function

def estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i):
    km_sol    = np.sum(x_sol)
    x_sol     = x_sol/km_sol
    fM_sol    = np.sum(x_sol[ind_m])
    ft_sol    = np.sum(x_sol[ind_t])
    km_sol    = km_sol*km_i # unnormalization
    T2m_sol   = np.sum(x_sol[ind_m] * T2s[ind_m])/(np.sum(x_sol[ind_m])  + epsilon)
    T2t_sol   = np.sum(x_sol[ind_t] * T2s[ind_t])/(np.sum(x_sol[ind_t])  + epsilon)
    # ---- peaks
    peaks, _     = find_peaks(x_sol, height = 1e-5*np.max(x_sol))
    N_p_sol      = peaks.size   # number of peaks
    # ---- absolute diference between distributions
    dis_abs_error_sol = np.mean(np.abs(dist2 - x_sol))
    js_distance       = distance.jensenshannon(dist2, x_sol)
    wasserstein_dist  = wasserstein_distance(dist2, x_sol)
    return fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, js_distance, wasserstein_dist
#end

def compute_multi_metrics (MWF_nnls, True_fM, Npeaks_nnls, Dist_abs_nnls, method, vT2m, T2M_nnls, vT2ie, T2IE_nnls, vKm, KM_nnls, Dist_js_nnls, Dist_w_nnls):
    residual            = MWF_nnls - True_fM
    # 1) MAE:     Mean Absolute Error
    MAE_fM_nnls         = np.mean(np.abs(residual))
    # 2) MARE:    Mean Absolute Relative Error
    MARE_fM_nnls        = np.mean(np.abs(residual/True_fM))
    # 3) erMAX:   maximum absolute relative error
    erMAX_nnls          = np.max(np.abs(residual/True_fM))
    # 4) cRMSE:   Centered Root Mean Squared Error
    factor              = (MWF_nnls - np.mean(MWF_nnls)) - (True_fM - np.mean(True_fM))
    cRMSE_nnls          = np.sqrt(np.mean(factor**2))
    # 5) MSS:     Model Skill Socre (Taylor: Eq.5)
    R, p                = pearsonr(MWF_nnls.flatten() ,  True_fM.flatten())
    nume                = 4.0 * ( 1.0 + R )**4
    sigmaf              = np.std(MWF_nnls)/np.std(True_fM)
    R0                  = 1.0
    deno                = ( (1.0 + R0 )**4 ) * ( sigmaf + 1.0/sigmaf )**2
    MSS_nnls            = nume/deno
    # 6) RMSE:    Root Mean Squared Error
    RMSE_nnls           = np.sqrt(np.mean(residual**2))
    # 7) U95:     Uncertainty at 95%
    SDD_nnls            = np.std(residual)
    U_95_nnls           = 1.96 * np.sqrt( SDD_nnls**2 +  RMSE_nnls**2)
    # 8) RMSRE:   Root Mean Squared Relative Error
    RMSRE_nnls          = np.sqrt( np.mean( (residual/True_fM)**2 ) )
    # 9 ) MBE:    Mean Bias Error
    MBE_nnls            = np.mean(residual)
    # 10) R:      Correlation coefficient
    corr_nnls, p_value_nnls   = pearsonr(MWF_nnls.flatten() ,  True_fM.flatten())
    # 11) Global Mean Absolute Relative Error
    MARE_fIE_nnls        = np.mean( np.abs(Mie_nnls - (1.0 - True_fM))/(1.0 - True_fM) )
    MARE_T2m_nnls        = np.mean(np.abs(T2M_nnls  -  vT2m)/vT2m)
    MARE_T2ie_nnls       = np.mean(np.abs(T2IE_nnls -  vT2ie)/vT2ie)
    MARE_Km_nnls         = np.mean(np.abs(KM_nnls   -  vKm)/vKm)
    GMARE                = MARE_fM_nnls + MARE_fIE_nnls + MARE_T2m_nnls + MARE_T2ie_nnls + MARE_Km_nnls
    # 12) Mean Absolute Error of the Numper of Peaks
    MAE_Npeaks_nnls      = np.mean(np.abs( Npeaks_nnls - 2.0 ))
    # 13) Mean Absolute Error of Distributions
    MAE_Dist_nnls        = np.mean(Dist_abs_nnls)
    # 14) MJSD:  Mean Jensen-Shannon Distance
    MJS_Dist_nnls        = np.mean(Dist_js_nnls)
    # 15) MWD:  Mean Wasserstein Distance
    MW_Dist_nnls         = np.mean(Dist_w_nnls)
    # --------------------------------------------------------------------------
    #metric_vector = [method, MAE_fM_nnls, MARE_fM_nnls, erMAX_nnls, cRMSE_nnls, MSS_nnls, RMSE_nnls, U_95_nnls, RMSRE_nnls, MBE_nnls, corr_nnls, GMARE, MAE_Npeaks_nnls, MAE_Dist_nnls, MJS_Dist_nnls, MW_Dist_nnls]
    metric_vector = [method, MAE_fM_nnls, MARE_fM_nnls, RMSE_nnls, cRMSE_nnls, RMSRE_nnls, U_95_nnls, MBE_nnls, corr_nnls, GMARE, MAE_Npeaks_nnls, MAE_Dist_nnls, MJS_Dist_nnls, MW_Dist_nnls]
    return metric_vector
#end

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

os.system('clear')
epsilon  = 1.0e-50
warnings.filterwarnings("ignore")
#______________________________________________________________________________#
#______________________________________________________________________________#
N_voxels   = 10000

path_base = 'Results/SNRs_Inf'

try:
    os.mkdir(path_base)
except:
    print('The folder cannot be created, it exists...')
#end

path      = path_base + '/All_methods_' + repr(obj0(np.round(N_voxels))) + 'iters/'
try:
    os.mkdir(path)
except:
    print('The folder cannot be created, it exists...')
#end
#______________________________________________________________________________#
#______________________________________________________________________________#

# Myelin water fraction
vMWF       = np.random.uniform(low=0.05,  high=0.25,    size=N_voxels)
# T2 myelin
vT2m       = np.random.uniform(low=15.0,   high=35.0,   size=N_voxels)
# T2 intra/extra-cellular
vT2ie      = np.random.uniform(low=60.0,  high=90.0,   size=N_voxels)
# Flip angle
vFA        = np.random.uniform(low=90.0,  high=180.0,   size=N_voxels)
# Signal-to-noise ratio
# proton density
vKm        = np.random.uniform(low=1000.0, high=1000.0, size=N_voxels)
# std of T2-myelin
vsigma_m   = np.random.uniform(low=1.0,   high=3.0,     size=N_voxels)
# std of T2-ie
vsigma_ie  = np.random.uniform(low=6.0,   high=12.0,    size=N_voxels)

#______________________________________________________________________________#
T2_csf     = 2000.0
sigma_csf  = 1.0
f_csf      = 0.0
vIEWF      = 1.0 - f_csf - vMWF
#______________________________________________________________________________#
# Grid for the high resolution pdf used to generate the data
NT2grid_sim     = 1000
T2grid, dT2grid = np.linspace(1.0, 300.0, NT2grid_sim, retstep=True)
T1grid          = 1000.0 * np.ones_like(T2grid)

#______________________________ Generate Data _________________________________#
# Experimental parameters
nTE        = 32 # number of TE
TEmin      = 10.0
TE_array   = TEmin * np.arange(1,nTE+1)
nEchoes    = TE_array.shape[0]
tau        = TE_array[1] - TE_array[0]
Data       = np.zeros((N_voxels, nTE))
#_________________________Generate Dictionary _________________________________#
Npc    = 60
T2m0   = 10.0
T2mf   = 40.0
T2tf   = 200.0
T2csf  = 2000.0

T2s     = np.logspace(math.log10(T2m0), math.log10(T2csf), num=Npc, endpoint=True, base=10.0)
ind_m   = T2s <= T2mf
ind_csf = T2s > T2tf
ind_t   = (T2s>T2mf)&(T2s<=T2tf)

T1s     = 1000.0*np.ones_like(T2s) # a constant T1=1000 is assumed for all compartments
#T1s[-1] = 4000.0 # T1 for the free water compartment with the highest T2 (i.e., T2=2000.0)

# Create multi-dimensional dictionary with multiples flip_angles
N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
alpha_values = np.linspace(90.0,  180.0,  N_alphas)
TR           = 3000
Dic_3D       = create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values, TR)

# Define regularization vectors for the L-curve method
num_l_laplac   = 50
lambda_reg     = np.zeros((num_l_laplac))
lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(100.0), num=num_l_laplac-1, endpoint=True, base=10.0)
#lambda_reg[1:] = np.logspace(math.log10(1e-7), math.log10(10.0), num=num_l_laplac-1, endpoint=True, base=10.0)

# Define initial variables to save data
# Method 1)
MWF_nnls                = np.zeros((N_voxels))
Mie_nnls                = np.zeros((N_voxels))
T2M_nnls                = np.zeros((N_voxels))
T2IE_nnls               = np.zeros((N_voxels))
KM_nnls                 = np.zeros((N_voxels))
Npeaks_nnls             = np.zeros((N_voxels))
Dist_abs_nnls           = np.zeros((N_voxels))
#XSol_nnls               = np.zeros((N_voxels, Npc))
Dist_sj_nnls            = np.zeros((N_voxels))
Dist_w_nnls             = np.zeros((N_voxels))

# ------------------------------------------------------------------------------
# X2 for all regularization matrices
# ------------------------------------------------------------------------------

# Method 2)
MWF_X2_I                = np.zeros((N_voxels))
Mie_X2_I                = np.zeros((N_voxels))
T2M_X2_I                = np.zeros((N_voxels))
T2IE_X2_I               = np.zeros((N_voxels))
KM_X2_I                 = np.zeros((N_voxels))
Npeaks_X2_I             = np.zeros((N_voxels))
Dist_abs_X2_I           = np.zeros((N_voxels))
#XSol_X2_I               = np.zeros((N_voxels, Npc))
Lambdas_X2_I            = np.zeros((N_voxels))
Dist_sj_X2_I            = np.zeros((N_voxels))
Dist_w_X2_I             = np.zeros((N_voxels))

# Method 3)
MWF_X2_L1               = np.zeros((N_voxels))
Mie_X2_L1               = np.zeros((N_voxels))
T2M_X2_L1               = np.zeros((N_voxels))
T2IE_X2_L1              = np.zeros((N_voxels))
KM_X2_L1                = np.zeros((N_voxels))
Npeaks_X2_L1            = np.zeros((N_voxels))
Dist_abs_X2_L1          = np.zeros((N_voxels))
#XSol_X2_L1              = np.zeros((N_voxels, Npc))
Lambdas_X2_L1           = np.zeros((N_voxels))
Dist_sj_X2_L1           = np.zeros((N_voxels))
Dist_w_X2_L1            = np.zeros((N_voxels))

# Method 4)
MWF_X2_L2               = np.zeros((N_voxels))
Mie_X2_L2               = np.zeros((N_voxels))
T2M_X2_L2               = np.zeros((N_voxels))
T2IE_X2_L2              = np.zeros((N_voxels))
KM_X2_L2                = np.zeros((N_voxels))
Npeaks_X2_L2            = np.zeros((N_voxels))
Dist_abs_X2_L2          = np.zeros((N_voxels))
#XSol_X2_L2              = np.zeros((N_voxels, Npc))
Lambdas_X2_L2           = np.zeros((N_voxels))
Dist_sj_X2_L2           = np.zeros((N_voxels))
Dist_w_X2_L2            = np.zeros((N_voxels))

# ------------------------------------------------------------------------------
# L_curve for all regularization matrices
# ------------------------------------------------------------------------------

# Method 5)
MWF_Lcurve_I            = np.zeros((N_voxels))
Mie_Lcurve_I            = np.zeros((N_voxels))
T2M_Lcurve_I            = np.zeros((N_voxels))
T2IE_Lcurve_I           = np.zeros((N_voxels))
KM_Lcurve_I             = np.zeros((N_voxels))
Npeaks_Lcurve_I         = np.zeros((N_voxels))
Dist_abs_Lcurve_I       = np.zeros((N_voxels))
XSol_Lcurve_I           = np.zeros((N_voxels, Npc))
Lambdas_Lcurve_I        = np.zeros((N_voxels))
Dist_sj_Lcurve_I        = np.zeros((N_voxels))
Dist_w_Lcurve_I         = np.zeros((N_voxels))

# Method 6)
MWF_Lcurve_L1            = np.zeros((N_voxels))
Mie_Lcurve_L1            = np.zeros((N_voxels))
T2M_Lcurve_L1            = np.zeros((N_voxels))
T2IE_Lcurve_L1           = np.zeros((N_voxels))
KM_Lcurve_L1             = np.zeros((N_voxels))
Npeaks_Lcurve_L1         = np.zeros((N_voxels))
Dist_abs_Lcurve_L1       = np.zeros((N_voxels))
#XSol_Lcurve_L1           = np.zeros((N_voxels, Npc))
Lambdas_Lcurve_L1        = np.zeros((N_voxels))
Dist_sj_Lcurve_L1        = np.zeros((N_voxels))
Dist_w_Lcurve_L1         = np.zeros((N_voxels))

# Method 7)
MWF_Lcurve_L2            = np.zeros((N_voxels))
Mie_Lcurve_L2            = np.zeros((N_voxels))
T2M_Lcurve_L2            = np.zeros((N_voxels))
T2IE_Lcurve_L2           = np.zeros((N_voxels))
KM_Lcurve_L2             = np.zeros((N_voxels))
Npeaks_Lcurve_L2         = np.zeros((N_voxels))
Dist_abs_Lcurve_L2       = np.zeros((N_voxels))
#XSol_Lcurve_L2           = np.zeros((N_voxels, Npc))
Lambdas_Lcurve_L2        = np.zeros((N_voxels))
Dist_sj_Lcurve_L2        = np.zeros((N_voxels))
Dist_w_Lcurve_L2         = np.zeros((N_voxels))

# ------------------------------------------------------------------------------
# GCV for all regularization matrices
# ------------------------------------------------------------------------------

# Method 8)
MWF_GCV_I                = np.zeros((N_voxels))
Mie_GCV_I                = np.zeros((N_voxels))
T2M_GCV_I                = np.zeros((N_voxels))
T2IE_GCV_I               = np.zeros((N_voxels))
KM_GCV_I                 = np.zeros((N_voxels))
Npeaks_GCV_I             = np.zeros((N_voxels))
Dist_abs_GCV_I           = np.zeros((N_voxels))
#XSol_GCV_I               = np.zeros((N_voxels, Npc))
Lambdas_GCV_I            = np.zeros((N_voxels))
Dist_sj_GCV_I            = np.zeros((N_voxels))
Dist_w_GCV_I             = np.zeros((N_voxels))

# Method 9)
MWF_GCV_L1               = np.zeros((N_voxels))
Mie_GCV_L1               = np.zeros((N_voxels))
T2M_GCV_L1               = np.zeros((N_voxels))
T2IE_GCV_L1              = np.zeros((N_voxels))
KM_GCV_L1                = np.zeros((N_voxels))
Npeaks_GCV_L1            = np.zeros((N_voxels))
Dist_abs_GCV_L1          = np.zeros((N_voxels))
#XSol_GCV_L1              = np.zeros((N_voxels, Npc))
Lambdas_GCV_L1           = np.zeros((N_voxels))
Dist_sj_GCV_L1           = np.zeros((N_voxels))
Dist_w_GCV_L1            = np.zeros((N_voxels))

# Method 10)
MWF_GCV_L2               = np.zeros((N_voxels))
Mie_GCV_L2               = np.zeros((N_voxels))
T2M_GCV_L2               = np.zeros((N_voxels))
T2IE_GCV_L2              = np.zeros((N_voxels))
KM_GCV_L2                = np.zeros((N_voxels))
Npeaks_GCV_L2            = np.zeros((N_voxels))
Dist_abs_GCV_L2          = np.zeros((N_voxels))
#XSol_GCV_L2              = np.zeros((N_voxels, Npc))
Lambdas_GCV_L2           = np.zeros((N_voxels))
Dist_sj_GCV_L2           = np.zeros((N_voxels))
Dist_w_GCV_L2            = np.zeros((N_voxels))

True_fM                  = np.zeros((N_voxels))
FA_est                   = np.zeros((N_voxels))
True_dist                = np.zeros((N_voxels, Npc))

# Create Laplacian matrix for regularization: I, L1, L2
order0 = 0
order1 = 1
order2 = 2
Laplac0 = create_Laplacian_matrix(Npc, order0)
Laplac1 = create_Laplacian_matrix(Npc, order1)
Laplac2 = create_Laplacian_matrix(Npc, order2)

# -----------------------------------------------------------------------------#
for iter in range(0, N_voxels):
    print ('Voxel: ', iter, ' of ' , N_voxels)
    MWF      = vMWF[iter]
    T2m      = vT2m[iter]
    T2ie     = vT2ie[iter]
    FA       = vFA[iter]
    Km       = vKm[iter]
    sigma_m  = vsigma_m[iter]
    sigma_ie = vsigma_ie[iter]
    IEWF     = vIEWF[iter]
    # --- create noise-free signal
    x        = np.array([IEWF, T2ie, MWF, T2m, f_csf, Km, FA])
    dist     = MWF*norm.pdf(T2grid, T2m, sigma_m) + IEWF*norm.pdf(T2grid, T2ie, sigma_ie) + f_csf*norm.pdf(T2grid, T2_csf, sigma_csf)
    dist     = dist/np.sum(dist)
    Signal   = Signal_MET2_dist(x, nEchoes, tau, T2grid, T1grid, dist, TR)
    Data[iter, :] = Signal
    # ----------------------------- RECONSTRUCTION -----------------------------
    Signal        = Data[iter, :]
    # ------------------------ computing optimal FA ----------------------------
    index_i, FA_i, km_i, SSE, f_nnls = compute_optimal_FA(Signal, Dic_3D, alpha_values)
    Kernel = Dic_3D[:,:,index_i]
    # --------------------------------------------------------------------------
    km_i = Signal[0] # normalizing by the first echo
    S    = Signal/km_i #normalization
    # --------------------------------------------------------------------------
    # --- Transform the original high resolution pdf to the low resolution pdf
    dist2       = np.zeros((Npc))
    deltaT2     = np.zeros((Npc))
    # --- ind[0]
    T2_delta_max = T2s[0] + (T2s[1] - T2s[0])/2.0
    ind_0        = T2grid < T2_delta_max
    dist2[0]     = np.sum( dist[ind_0]*dT2grid )
    deltaT2[0]   = T2_delta_max
    # --- ind from 1 to Npc-1
    for it in range(1, Npc-1):
        T2_delta_min = T2s[it-1] + (T2s[it]   - T2s[it-1])/2.0
        T2_delta_max = T2s[it]   + (T2s[it+1] - T2s[it])/2.0
        ind_i = (T2grid >= T2_delta_min) & (T2grid < T2_delta_max)
        dist2[it]   = np.sum( dist[ind_i]*dT2grid )
        deltaT2[it] = T2_delta_max - T2_delta_min
    #end for it
    # ind[N]
    T2_delta_min   = T2s[Npc-2] + (T2s[Npc-1] - T2s[Npc-2])/2.0
    ind_Npc        = T2grid >= T2_delta_min
    dist2[Npc-1]   = np.sum( dist[ind_Npc]*dT2grid )
    deltaT2[Npc-1] = (T2s[Npc-1] - T2s[Npc-2])/2.0
    # normalization
    dist2 = dist2/np.sum(dist2)
    # myelin fraction of this low-resolution pdf
    fM_d2 = np.sum(dist2[ind_m])
    # --------------------------------------------------------------------------
    True_fM[iter]      = fM_d2 # MWF computed from the distribution
    FA_est[iter]       = FA_i
    True_dist[iter,:]  = dist2
    # -------------------------------------------------------------------------#

    # -------------------------------------------------------------------------#
    #                      1. Reconstruction using NNLS                        #
    # --------------------------------------------------------------------------
    x_sol, kk   = nnls(Kernel, S)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save data
    MWF_nnls[iter]      = fM_sol
    Mie_nnls[iter]      = ft_sol
    T2M_nnls[iter]      = T2m_sol
    T2IE_nnls[iter]     = T2t_sol
    KM_nnls[iter]       = km_sol
    Npeaks_nnls[iter]   = N_p_sol
    Dist_abs_nnls[iter] = dis_abs_error_sol
    #XSol_nnls[iter,:]   = x_sol
    Dist_sj_nnls[iter]  = dis_js_error_sol
    Dist_w_nnls[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                      2. Reconstruction using X2-I                        #
    # -------------------------------------------------------------------------#
    k          = 1.02
    x_sol, reg_opt, k_est = nnls_x2(Kernel, S, Laplac0, k)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save data
    MWF_X2_I[iter]      = fM_sol
    Mie_X2_I[iter]      = ft_sol
    T2M_X2_I[iter]      = T2m_sol
    T2IE_X2_I[iter]     = T2t_sol
    KM_X2_I[iter]       = km_sol
    Npeaks_X2_I[iter]   = N_p_sol
    Dist_abs_X2_I[iter] = dis_abs_error_sol
    #XSol_X2_I[iter,:]   = x_sol
    Lambdas_X2_I[iter]  = reg_opt
    Dist_sj_X2_I[iter]  = dis_js_error_sol
    Dist_w_X2_I[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                    3. Reconstruction using X2-L1                          #
    # -------------------------------------------------------------------------#
    k          = 1.02
    x_sol, reg_opt, k_est = nnls_x2(Kernel, S, Laplac1, k)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_X2_L1[iter]      = fM_sol
    Mie_X2_L1[iter]      = ft_sol
    T2M_X2_L1[iter]      = T2m_sol
    T2IE_X2_L1[iter]     = T2t_sol
    KM_X2_L1[iter]       = km_sol
    Npeaks_X2_L1[iter]   = N_p_sol
    Dist_abs_X2_L1[iter] = dis_abs_error_sol
    #XSol_X2_L1[iter,:]   = x_sol
    Lambdas_X2_L1[iter]  = reg_opt
    Dist_sj_X2_L1[iter]  = dis_js_error_sol
    Dist_w_X2_L1[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                    4. Reconstruction using X2-L2                          #
    # -------------------------------------------------------------------------#
    k          = 1.02
    x_sol, reg_opt, k_est = nnls_x2(Kernel, S, Laplac2, k)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_X2_L2[iter]      = fM_sol
    Mie_X2_L2[iter]      = ft_sol
    T2M_X2_L2[iter]      = T2m_sol
    T2IE_X2_L2[iter]     = T2t_sol
    KM_X2_L2[iter]       = km_sol
    Npeaks_X2_L2[iter]   = N_p_sol
    Dist_abs_X2_L2[iter] = dis_abs_error_sol
    #XSol_X2_L2[iter,:]   = x_sol
    Lambdas_X2_L2[iter]  = reg_opt
    Dist_sj_X2_L2[iter]  = dis_js_error_sol
    Dist_w_X2_L2[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                 5. Reconstruction using L-curve-I                        #
    # -------------------------------------------------------------------------#
    reg_opt    = nnls_lcurve_wrapper(Kernel, S, Laplac0, lambda_reg)
    x_sol      = nnls_tik(Kernel, S, Laplac0, reg_opt)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_Lcurve_I[iter]      = fM_sol
    Mie_Lcurve_I[iter]      = ft_sol
    T2M_Lcurve_I[iter]      = T2m_sol
    T2IE_Lcurve_I[iter]     = T2t_sol
    KM_Lcurve_I[iter]       = km_sol
    Npeaks_Lcurve_I[iter]   = N_p_sol
    Dist_abs_Lcurve_I[iter] = dis_abs_error_sol
    #XSol_Lcurve_I[iter,:]   = x_sol
    Lambdas_Lcurve_I[iter]  = reg_opt
    Dist_sj_Lcurve_I[iter]  = dis_js_error_sol
    Dist_w_Lcurve_I[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                 6. Reconstruction using L-curve-L1                      #
    # -------------------------------------------------------------------------#
    reg_opt    = nnls_lcurve_wrapper(Kernel, S, Laplac1, lambda_reg)
    x_sol      = nnls_tik(Kernel, S, Laplac1, reg_opt)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_Lcurve_L1[iter]      = fM_sol
    Mie_Lcurve_L1[iter]      = ft_sol
    T2M_Lcurve_L1[iter]      = T2m_sol
    T2IE_Lcurve_L1[iter]     = T2t_sol
    KM_Lcurve_L1[iter]       = km_sol
    Npeaks_Lcurve_L1[iter]   = N_p_sol
    Dist_abs_Lcurve_L1[iter] = dis_abs_error_sol
    #XSol_Lcurve_L1[iter,:]   = x_sol
    Lambdas_Lcurve_L1[iter]  = reg_opt
    Dist_sj_Lcurve_L1[iter]  = dis_js_error_sol
    Dist_w_Lcurve_L1[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                 7. Reconstruction using L-curve-L2                      #
    # -------------------------------------------------------------------------#
    reg_opt    = nnls_lcurve_wrapper(Kernel, S, Laplac2, lambda_reg)
    x_sol      = nnls_tik(Kernel, S, Laplac2, reg_opt)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_Lcurve_L2[iter]      = fM_sol
    Mie_Lcurve_L2[iter]      = ft_sol
    T2M_Lcurve_L2[iter]      = T2m_sol
    T2IE_Lcurve_L2[iter]     = T2t_sol
    KM_Lcurve_L2[iter]       = km_sol
    Npeaks_Lcurve_L2[iter]   = N_p_sol
    Dist_abs_Lcurve_L2[iter] = dis_abs_error_sol
    #XSol_Lcurve_L2[iter,:]   = x_sol
    Lambdas_Lcurve_L2[iter]  = reg_opt
    Dist_sj_Lcurve_L2[iter]  = dis_js_error_sol
    Dist_w_Lcurve_L2[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                    8. Reconstruction using GCV-I                        #
    # -------------------------------------------------------------------------#
    x_sol, reg_opt = nnls_gcv(Kernel, S, Laplac0)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save data
    MWF_GCV_I[iter]      = fM_sol
    Mie_GCV_I[iter]      = ft_sol
    T2M_GCV_I[iter]      = T2m_sol
    T2IE_GCV_I[iter]     = T2t_sol
    KM_GCV_I[iter]       = km_sol
    Npeaks_GCV_I[iter]   = N_p_sol
    Dist_abs_GCV_I[iter] = dis_abs_error_sol
    #XSol_GCV_I[iter,:]   = x_sol
    Lambdas_GCV_I[iter]  = reg_opt
    Dist_sj_GCV_I[iter]  = dis_js_error_sol
    Dist_w_GCV_I[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                 9. Reconstruction using GCV-L1                          #
    # -------------------------------------------------------------------------#
    x_sol, reg_opt = nnls_gcv(Kernel, S, Laplac1)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_GCV_L1[iter]      = fM_sol
    Mie_GCV_L1[iter]      = ft_sol
    T2M_GCV_L1[iter]      = T2m_sol
    T2IE_GCV_L1[iter]     = T2t_sol
    KM_GCV_L1[iter]       = km_sol
    Npeaks_GCV_L1[iter]   = N_p_sol
    Dist_abs_GCV_L1[iter] = dis_abs_error_sol
    #XSol_GCV_L1[iter,:]   = x_sol
    Lambdas_GCV_L1[iter]  = reg_opt
    Dist_sj_GCV_L1[iter]  = dis_js_error_sol
    Dist_w_GCV_L1[iter]   = dis_w_error_sol

    # -------------------------------------------------------------------------#
    #                 10. Reconstruction using GCV-L2                          #
    # -------------------------------------------------------------------------#
    x_sol, reg_opt = nnls_gcv(Kernel, S, Laplac2)

    fM_sol, ft_sol, T2m_sol, T2t_sol, km_sol, N_p_sol, dis_abs_error_sol, dis_js_error_sol, dis_w_error_sol = estimate_error_metrics(x_sol, dist2, ind_m, ind_t, km_i)
    # ------  save dara
    MWF_GCV_L2[iter]      = fM_sol
    Mie_GCV_L2[iter]      = ft_sol
    T2M_GCV_L2[iter]      = T2m_sol
    T2IE_GCV_L2[iter]     = T2t_sol
    KM_GCV_L2[iter]       = km_sol
    Npeaks_GCV_L2[iter]   = N_p_sol
    Dist_abs_GCV_L2[iter] = dis_abs_error_sol
    #XSol_GCV_L2[iter,:]   = x_sol
    Lambdas_GCV_L2[iter]  = reg_opt
    Dist_sj_GCV_L2[iter]  = dis_js_error_sol
    Dist_w_GCV_L2[iter]   = dis_w_error_sol
# -----------------------------------------------------------------------------#
#end for
print "Done"

# -----------------------------------------------------------------------------#
#  -------------------------- Evaluation --------------------------------------#
# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
#                         1. Evaluation - NNLS                                 #
# -----------------------------------------------------------------------------#
metric_vector_nnls = compute_multi_metrics (MWF_nnls, True_fM, Npeaks_nnls, Dist_abs_nnls, '1. NNLS', vT2m, T2M_nnls, vT2ie, T2IE_nnls, vKm, KM_nnls, Dist_sj_nnls, Dist_w_nnls)

# -----------------------------------------------------------------------------#
#                         2. Evaluation - X2-I                                 #
# -----------------------------------------------------------------------------#
metric_vector_X2_I = compute_multi_metrics (MWF_X2_I, True_fM, Npeaks_X2_I, Dist_abs_X2_I, '2. X2-I', vT2m, T2M_X2_I, vT2ie, T2IE_X2_I, vKm, KM_X2_I, Dist_sj_X2_I, Dist_w_X2_I)
mean_lambda_X2_I   = np.mean(Lambdas_X2_I)
std_lambda_X2_I    = np.std(Lambdas_X2_I)

# -----------------------------------------------------------------------------#
#                         3. Evaluation - X2-L1                                 #
# -----------------------------------------------------------------------------#
metric_vector_X2_L1 = compute_multi_metrics (MWF_X2_L1, True_fM, Npeaks_X2_L1, Dist_abs_X2_L1, '3. X2-L1', vT2m, T2M_X2_L1, vT2ie, T2IE_X2_L1, vKm, KM_X2_L1, Dist_sj_X2_L1, Dist_w_X2_L1)
mean_lambda_X2_L1   = np.mean(Lambdas_X2_L1)
std_lambda_X2_L1    = np.std(Lambdas_X2_L1)

# -----------------------------------------------------------------------------#
#                         4. Evaluation - X2-L2                                 #
# -----------------------------------------------------------------------------#
metric_vector_X2_L2 = compute_multi_metrics (MWF_X2_L2, True_fM, Npeaks_X2_L2, Dist_abs_X2_L2, '4. X2-L2', vT2m, T2M_X2_L2, vT2ie, T2IE_X2_L2, vKm, KM_X2_L2, Dist_sj_X2_L2, Dist_w_X2_L2)
mean_lambda_X2_L2   = np.mean(Lambdas_X2_L2)
std_lambda_X2_L2    = np.std(Lambdas_X2_L2)

# -----------------------------------------------------------------------------#
#                    5. Evaluation - L_curve-I                                 #
# -----------------------------------------------------------------------------#
metric_vector_L_curve_I = compute_multi_metrics (MWF_Lcurve_I, True_fM, Npeaks_Lcurve_I, Dist_abs_Lcurve_I, '5. Lcurve-I', vT2m, T2M_Lcurve_I, vT2ie, T2IE_Lcurve_I, vKm, KM_Lcurve_I,  Dist_sj_Lcurve_I, Dist_w_Lcurve_I)
mean_lambda_Lcurve_I    = np.mean(Lambdas_Lcurve_I)
std_lambda_Lcurve_I     = np.std(Lambdas_Lcurve_I)

# -----------------------------------------------------------------------------#
#                    6. Evaluation - L_curve-L1                                 #
# -----------------------------------------------------------------------------#
metric_vector_L_curve_L1 = compute_multi_metrics (MWF_Lcurve_L1, True_fM, Npeaks_Lcurve_L1, Dist_abs_Lcurve_L1, '6. Lcurve-L1', vT2m, T2M_Lcurve_L1, vT2ie, T2IE_Lcurve_L1, vKm, KM_Lcurve_L1, Dist_sj_Lcurve_L1, Dist_w_Lcurve_L1)
mean_lambda_Lcurve_L1    = np.mean(Lambdas_Lcurve_L1)
std_lambda_Lcurve_L1     = np.std(Lambdas_Lcurve_L1)

# -----------------------------------------------------------------------------#
#                    7. Evaluation - L_curve-L2                                 #
# -----------------------------------------------------------------------------#
metric_vector_L_curve_L2 = compute_multi_metrics (MWF_Lcurve_L2, True_fM, Npeaks_Lcurve_L2, Dist_abs_Lcurve_L2, '7. Lcurve-L2', vT2m, T2M_Lcurve_L2, vT2ie, T2IE_Lcurve_L2, vKm, KM_Lcurve_L2, Dist_sj_Lcurve_L2, Dist_w_Lcurve_L2)
mean_lambda_Lcurve_L2    = np.mean(Lambdas_Lcurve_L2)
std_lambda_Lcurve_L2     = np.std(Lambdas_Lcurve_L2)

# -----------------------------------------------------------------------------#
#                       8. Evaluation - GCV-I                                 #
# -----------------------------------------------------------------------------#
metric_vector_GCV_I = compute_multi_metrics (MWF_GCV_I, True_fM, Npeaks_GCV_I, Dist_abs_GCV_I, '8. GCV-I', vT2m, T2M_GCV_I, vT2ie, T2IE_GCV_I, vKm, KM_GCV_I, Dist_sj_GCV_I, Dist_w_GCV_I)
mean_lambda_GCV_I   = np.mean(Lambdas_GCV_I)
std_lambda_GCV_I    = np.std(Lambdas_GCV_I)

# -----------------------------------------------------------------------------#
#                      9. Evaluation - GCV-L1                                 #
# -----------------------------------------------------------------------------#
metric_vector_GCV_L1 = compute_multi_metrics (MWF_GCV_L1, True_fM, Npeaks_GCV_L1, Dist_abs_GCV_L1, '9. GCV-L1', vT2m, T2M_GCV_L1, vT2ie, T2IE_GCV_L1, vKm, KM_GCV_L1, Dist_sj_GCV_L1, Dist_w_GCV_L1)
mean_lambda_GCV_L1   = np.mean(Lambdas_GCV_L1)
std_lambda_GCV_L1    = np.std(Lambdas_GCV_L1)

# -----------------------------------------------------------------------------#
#                      10. Evaluation - GCV-L2                                 #
# -----------------------------------------------------------------------------#
metric_vector_GCV_L2 = compute_multi_metrics (MWF_GCV_L2, True_fM, Npeaks_GCV_L2, Dist_abs_GCV_L2, '10. GCV-L2', vT2m, T2M_GCV_L2, vT2ie, T2IE_GCV_L2, vKm, KM_GCV_L2, Dist_sj_GCV_L2, Dist_w_GCV_L2)
mean_lambda_GCV_L2   = np.mean(Lambdas_GCV_L2)
std_lambda_GCV_L2    = np.std(Lambdas_GCV_L2)

# ------------------------------------------------------------------------------
# -------------------------------- PRINT TABLE ---------------------------------
# ------------------------------------------------------------------------------
    # 1) MAE :      Mean Absolute Error
    # 2) MARE :     Mean absolute relative error
    # 3) RMSE:      Root Mean Squared Error
    # 4) cRMSE:     Centered Root Mean Squared Error
    # 5) RMSRE:     Root Mean Squared Relative Error
    # 6) U95 :      Uncertainty at 95%
    # 7 ) MBE:      Mean Bias Error
    # 8) R:        Correlation coefficient
    # 9) GMARE :   Global Mean Absolute Relative Error
    # 10) MAE-k:    Mean Absolute Error of the Numper of Peaks
    # 11) MAE-S :   Mean Absolute Error of Spectra
    # 12) MJSD-S:   Mean Jensen-Shannon distance of Spectra
    # 13) MWD-S:    Mean Wasserstein distance of Spectra
# ------------------------------------------------------------------------------

headers  = [ 'Method',  '1. MAE', '2. MARE', '3. RMSE', '4. cRMSE', '5. RMSRE', '6. U95' , '7. MBE', '8. R', '9. GMARE', '10. MAE-k', '11. MAE-S', '12. MJSD-S', '13. MWD-S' ]
table    = [
           metric_vector_nnls,
           metric_vector_X2_I,
           metric_vector_X2_L1,
           metric_vector_X2_L2,
           metric_vector_L_curve_I,
           metric_vector_L_curve_L1,
           metric_vector_L_curve_L2,
           metric_vector_GCV_I,
           metric_vector_GCV_L1,
           metric_vector_GCV_L2,
           ]

table_tabulated = tabulate(table, headers=headers)
print table_tabulated

f1 = open(path + 'table_errors.txt', 'w')
f1.write(table_tabulated)
f1.close()

np.savetxt(path + 'table_errors.csv', table, delimiter=",", fmt='%s')

print '  '
print '------------------------------------------------------------------------'
print '  '

# Regularization values

headers =  [ 'Method                ', 'mean Lambda',                 'STD'                         ]
table   =  [
           [ '1. NNLS               ',  0,                             0                            ],
           [ '2. X2-I               ',  mean_lambda_X2_I,              std_lambda_X2_I              ],
           [ '3. X2-L1              ',  mean_lambda_X2_L1,             std_lambda_X2_L1             ],
           [ '4. X2-L2              ',  mean_lambda_X2_L2,             std_lambda_X2_L2             ],
           [ '5. Lcurve-I           ',  mean_lambda_Lcurve_I,          std_lambda_Lcurve_I          ],
           [ '6. Lcurve-L1          ',  mean_lambda_Lcurve_L1,         std_lambda_Lcurve_L1         ],
           [ '7. Lcurve-L2          ',  mean_lambda_Lcurve_L2,         std_lambda_Lcurve_L2         ],
           [ '8. GCV-I              ',  mean_lambda_GCV_I,             std_lambda_GCV_I             ],
           [ '9. GCV-L1             ',  mean_lambda_GCV_L1,            std_lambda_GCV_L1            ],
           [ '10. GCV-L2            ',  mean_lambda_GCV_L2,            std_lambda_GCV_L2            ]
           ]
table_tabulated  = tabulate(table, headers=headers)
print table_tabulated

f2 = open(path + 'table_regularization.txt', 'w')
f2.write(table_tabulated)
f2.close()

np.savetxt(path + 'table_regularization.csv', table, delimiter=",", fmt='%s')

# ------------------------------------------------------------------------------
# -------------------------------- PLOT ----------------------------------------
# ------------------------------------------------------------------------------

# Set the figure properties (optional)
rcParams["figure.figsize"] = [10.0, 8]
rcParams['lines.linewidth'] = 1.25 # line width for plots
rcParams.update({'font.size': 13}) # font size of axes text

# Close any previously open graphics windows
# ToDo: fails to work within Eclipse
plt.close('all')
taylor_stats1  = sm.taylor_statistics(MWF_nnls, True_fM)
taylor_stats2  = sm.taylor_statistics(MWF_X2_I, True_fM)
taylor_stats3  = sm.taylor_statistics(MWF_X2_L1, True_fM)
taylor_stats4  = sm.taylor_statistics(MWF_X2_L2, True_fM)
taylor_stats5  = sm.taylor_statistics(MWF_Lcurve_I, True_fM)
taylor_stats6  = sm.taylor_statistics(MWF_Lcurve_L1, True_fM)
taylor_stats7  = sm.taylor_statistics(MWF_Lcurve_L2, True_fM)
taylor_stats8  = sm.taylor_statistics(MWF_GCV_I, True_fM)
taylor_stats9  = sm.taylor_statistics(MWF_GCV_L1, True_fM)
taylor_stats10 = sm.taylor_statistics(MWF_GCV_L2, True_fM)

target_stats1  = sm.target_statistics(MWF_nnls, True_fM)
target_stats2  = sm.target_statistics(MWF_X2_I, True_fM)
target_stats3  = sm.target_statistics(MWF_X2_L1, True_fM)
target_stats4  = sm.target_statistics(MWF_X2_L2, True_fM)
target_stats5  = sm.target_statistics(MWF_Lcurve_I, True_fM)
target_stats6  = sm.target_statistics(MWF_Lcurve_L1, True_fM)
target_stats7  = sm.target_statistics(MWF_Lcurve_L2, True_fM)
target_stats8  = sm.target_statistics(MWF_GCV_I, True_fM)
target_stats9  = sm.target_statistics(MWF_GCV_L1, True_fM)
target_stats10 = sm.target_statistics(MWF_GCV_L2, True_fM)

# Store statistics in arrays
sdev   = np.array( [ taylor_stats1['sdev'][0],  taylor_stats1['sdev'][1],  taylor_stats2['sdev'][1],  taylor_stats3['sdev'][1],  taylor_stats4['sdev'][1],
                     taylor_stats5['sdev'][1],  taylor_stats6['sdev'][1],  taylor_stats7['sdev'][1],  taylor_stats8['sdev'][1],  taylor_stats9['sdev'][1],
                     taylor_stats10['sdev'][1] ] )

crmsd  = np.array( [ taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1], taylor_stats4['crmsd'][1],
                     taylor_stats5['crmsd'][1], taylor_stats6['crmsd'][1], taylor_stats7['crmsd'][1], taylor_stats8['crmsd'][1], taylor_stats9['crmsd'][1],
                     taylor_stats10['crmsd'][1] ] )

ccoef  = np.array( [ taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1],  taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], taylor_stats4['ccoef'][1],
                     taylor_stats5['ccoef'][1], taylor_stats6['ccoef'][1],  taylor_stats7['ccoef'][1], taylor_stats8['ccoef'][1], taylor_stats9['ccoef'][1],
                     taylor_stats10['ccoef'][1]  ] )

bias   = np.array( [target_stats1['bias'], target_stats2['bias'],  target_stats3['bias'],  target_stats4['bias'],  target_stats5['bias'],  target_stats6['bias'],
                    target_stats7['bias'],  target_stats8['bias'],  target_stats9['bias'],  target_stats10['bias'] ] )

crmsd2 = np.array([target_stats1['crmsd'], target_stats2['crmsd'], target_stats3['crmsd'], target_stats4['crmsd'], target_stats5['crmsd'], target_stats6['crmsd'],
                   target_stats7['crmsd'], target_stats8['crmsd'], target_stats9['crmsd'], target_stats10['crmsd'] ] )

rmsd   = np.array([target_stats1['rmsd'],  target_stats2['rmsd'],  target_stats3['rmsd'],  target_stats4['rmsd'],  target_stats5['rmsd'],  target_stats6['rmsd'],
                   target_stats7['rmsd'] , target_stats8['rmsd'],  target_stats9['rmsd'],  target_stats10['rmsd'] ] )

label1 = ['Non-Dimensional Observation', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
label2 = ['Non-Dimensional Observation', 'M1: NNLS', 'M2: X2-I', 'M3: X2-L1', 'M4: X2-L2','M5: L-curve-I', 'M6: L-curve-L1', 'M7: L-curve-L2', 'M8: GCV-I', 'M9: GCV-L1', 'M10: GCV-L2']

sm.taylor_diagram(sdev,crmsd,ccoef, MarkerDisplayed='marker', markerLabel = label1, markerColor='red', colRMS='green', colSTD='blue', colCOR='black', alpha=0.5, markerSize=13, titleOBS = 'observation', styleOBS = '-')
plt.savefig(path + 'taylor_SNR.png', dpi=300)
plt.close('all')

sm.taylor_diagram(sdev,crmsd,ccoef, MarkerDisplayed='marker', markerLabel = label2, colRMS='green', colSTD='blue', colCOR='black', alpha=0.5, markerLegend='on', markerSize=6)
plt.savefig(path + 'taylor_SNR_legend.png', dpi=300)
plt.close('all')

sm.target_diagram(100*bias, 100*crmsd2, 100*rmsd, markerLabel = label1[1:], markerLabelColor = 'r',
                 circleLineSpec = 'b-.', circleLineWidth = 1.5, alpha=0.5, markerSize=13)

plt.savefig(path + 'target_bias_SNR.png', dpi=300)
plt.close('all')

MAE_S = np.array([
           metric_vector_nnls[11],
           metric_vector_X2_I[11],
           metric_vector_X2_L1[11],
           metric_vector_X2_L2[11],
           metric_vector_L_curve_I[11],
           metric_vector_L_curve_L1[11],
           metric_vector_L_curve_L2[11],
           metric_vector_GCV_I[11],
           metric_vector_GCV_L1[11],
           metric_vector_GCV_L2[11],
           ])

MWD_S = np.array([
           metric_vector_nnls[13],
           metric_vector_X2_I[13],
           metric_vector_X2_L1[13],
           metric_vector_X2_L2[13],
           metric_vector_L_curve_I[13],
           metric_vector_L_curve_L1[13],
           metric_vector_L_curve_L2[13],
           metric_vector_GCV_I[13],
           metric_vector_GCV_L1[13],
           metric_vector_GCV_L2[13],
           ])

# Bias  -> MAE_S
# crmsd- > MWD_S
sm.target_diagram(100*MAE_S, 100*MWD_S, np.sqrt((100*MAE_S)**2 + (100*MWD_S)**2), markerLabel = label1[1:], markerLabelColor = 'r',
                 circleLineSpec = 'b-.', circleLineWidth = 1.5, alpha=0.5, markerSize=13)

plt.savefig(path + 'target_distance_SNR_modify_axis_title.png', dpi=300)
plt.close('all')

# Write plot to file
# Show plot
# plt.show()

# ------------------------------------------------------------------------------
# https://github.com/PeterRochford/SkillMetrics/wiki

# Metric 	Description
# ----------------------
# bias 	    Mean error
# r 	    Correlation coefficient
# CRMSD 	Centered root-mean-square error deviation
# RMSD 	    root-mean-square error deviation
# SDEV 	    standard deviation

# Other mettrics
# ---------------------
# RI  — the reliability index
# AAE — the average absolute error
# MEF — the modeling efficiency
