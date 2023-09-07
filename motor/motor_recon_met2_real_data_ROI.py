#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Developers:
# Erick Jorge Canales-Rodríguez (EPFL, Lausanne, Switzerland)
# Date: 09/11/2020
#===============================================================================

from __future__ import division
import scipy
from   scipy import sparse, linalg
import scipy.ndimage.filters as filt
from   scipy.linalg import cholesky
from   skimage.restoration import estimate_sigma, denoise_tv_chambolle

import os
import numpy as np
from   joblib import Parallel, delayed
import multiprocessing
import nibabel as nib

import time
import warnings
import progressbar

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

import joypy
from tabulate import tabulate
import pandas as pd

# import pdb
import sys
sys.path.append("..")

from epg.epg import *
from flip_angle_algorithms.fa_estimation import *
from intravoxel_algorithms.algorithms import *
#===============================================================================

class obj5:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.5f" %(self.float)
    #end
#end

class obj1:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.1f" %(self.float)
    #end
#end

class obj0:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.0f" %(self.float)
    #end
#end

warnings.filterwarnings("ignore",category=FutureWarning)
epsilon=1.0e-16

#===============================================================================
#                                FUNCTIONS
#===============================================================================
# ----------------------- Plot T2 spectra from all voxels ----------------------
def color_gradient(x):
    if x==0:
        color = 'k'
    else:
        # default blue color in matplotlib
        color = None
    #end
    return color
#end

def plot_T2spectra(N_voxels, XSol, T2s, path, figure_name, factor):
    data_xsol_true={}
    #maxval = 28
    maxval = 60
    for it in range(0, N_voxels):
        num_obj =  repr(obj0(it+1))
        data_xsol_true[num_obj] = XSol[it, 0:maxval]
    #end for
    num_obj =  repr(obj0(0))
    mean_Xsol = np.mean(XSol, axis=0)
    data_xsol_true[num_obj] = mean_Xsol[0:maxval] * factor # scaling the mean T2 hist, to get scaled spectra
    data_dist_true = pd.DataFrame(data_xsol_true)
    T2s = T2s[0:maxval]
    T2sr   = np.int16(np.round(T2s))
    labels = [ y for y in list(T2sr) ]

    overlap = 1.0
    fig10, ax10 = joypy.joyplot( data_dist_true, kind="values", x_range=[0, maxval], fade=True, linewidth=1,
                                 figsize=(8,8), labels=labels, ylabels=True, overlap=overlap, range_style='own', grid='both',
                                 title=figure_name, colormap=lambda x: color_gradient(x) )

    xlocs_old, xlabels_old = plt.xticks()
    xlabels_new = [10, 20, 30, 40, 70, 200, 500, 1000, 2000]

    xlocs_new   = np.zeros((len(xlabels_new)))
    for iter in range(0, len(xlabels_new)):
        xlocs_new[iter] = np.interp(xlabels_new[iter], T2s, range(0, maxval))
    #

    plt.xticks(xlocs_new, xlabels_new)
    plt.gca().set_xlabel('T2 (ms)', size=20)
    plt.gca().set_ylabel('Mean T_2 of non-dominant lobe (ms)', size=20)
    plt.savefig(path + figure_name + '.png', dpi=600)
    # plt.show()
    plt.close('all')
#end figure

# Create Laplacian matrix for regularization: I, L1, L2
def create_Laplacian_matrix(Npc, order):
    if order == 2:
        # Second-order derivative
        Zerosm    = np.zeros((Npc,), dtype=np.double)
        main_diag = np.ones(Npc,     dtype=np.double)*(2.0)
        side_diag = -1.0 * np.ones(Npc-1, dtype=np.double)
        diagonals = [main_diag, side_diag, side_diag]
        laplacian = sparse.diags(diagonals, [0, -1, 1], format="csr")
        Laplac    = laplacian.toarray()
        # Newman boundary conditions
        Laplac [0,0]        = 1.0
        Laplac [-1,-1]      = 1.0
    elif order == 1:
        # First-order derivative
        Zerosm    = np.zeros((Npc,), dtype=np.double)
        main_diag = np.ones(Npc,     dtype=np.double)*(1.0)
        side_diag = -1.0 * np.ones(Npc-1, dtype=np.double)
        diagonals = [main_diag, side_diag]
        laplacian = sparse.diags(diagonals, [0, -1], format="csr")
        Laplac    = laplacian.toarray()
    elif order == 0:
        # Identity matrix
        Laplac = np.eye(Npc)
    #end if
    return Laplac
#end fun

#_______________________________________________________________________________
def motor_recon_met2_ROIs(TE_array, path_to_data, path_to_mask, path_to_ROIs, path_to_save_data, TR, reg_matrix, denoise, FA_method, FA_smooth, myelin_T2, num_cores):
    # Load Data, Mask, and ROIs
    img      = nib.load(path_to_data)
    data     = img.get_fdata()
    data     = data.astype(np.float64, copy=False)

    img_mask = nib.load(path_to_mask)
    mask     = img_mask.get_fdata()
    mask     = mask.astype(np.int64, copy=False)

    img_ROIs = nib.load(path_to_ROIs)
    ROIs     = img_ROIs.get_fdata()
    ROIs     = ROIs.astype(np.int64, copy=False)

    print('--------- Data shape -----------------')
    nx, ny, nz, nt = data.shape
    print(data.shape)
    print('--------------------------------------')

    nROIs_values = np.unique(ROIs)
    index_zero   = np.where(nROIs_values == 0)
    nROIs_values = np.delete(nROIs_values, index_zero)
    nROIs        = nROIs_values.size
    print(' ')
    print(' Number of ROIs: ', nROIs)
    print(' List of labels: ')
    print(nROIs_values)
    print (' ')

    for c in range(nt):
        data[:,:,:,c] = np.squeeze(data[:,:,:,c]) * mask
    #end
    ROIs      = ROIs * mask

    nEchoes   = TE_array.shape[0]
    tau       = TE_array[1] - TE_array[0]

    FA        = np.zeros((nx, ny, nz))
    Ktotal    = FA.copy()
    FA_index  = FA.copy()

    # ==============================================================================
    # Inital values for the dictionary
    Npc    = 60
    T2m0   = 10.0
    T2mf   = myelin_T2
    T2tf   = 200.0
    T2csf  = 2000.0

    T2s     = np.logspace(math.log10(T2m0), math.log10(T2csf), num=Npc, endpoint=True, base=10.0)

    ind_m   = T2s <= T2mf              # myelin
    ind_t   = (T2s>T2mf)&(T2s<=T2tf)   # intra+extra
    ind_csf = T2s >= T2tf              # quasi free-water and csf

    T1s     = 1000.0*np.ones_like(T2s) # a constant T1=1000 is assumed for all compartments

    # Create multi-dimensional dictionary with multiples flip_angles
    #N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
    if FA_method == 'spline':
        N_alphas     = 91*3 # (steps = 0.333 degrees, from 90 to 180)
        #N_alphas     = 91*2 # (steps = 0.5 degrees, from 90 to 180)
        #N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
        alpha_values = np.linspace(90.0,  180.0,  N_alphas)
        Dic_3D       = create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values, TR)
        #alpha_values_spline = np.round( np.linspace(90.0, 180.0, 8) )
        alpha_values_spline = np.linspace(90.0, 180.0, 15)
        Dic_3D_LR    = create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values_spline, TR)
    #end

    if FA_method == 'brute-force':
        N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
        alpha_values = np.linspace(90.0,  180.0,  N_alphas)
        Dic_3D       = create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values, TR)
    #end

    # Define regularization vectors for the L-curve method
    num_l_laplac   = 50
    lambda_reg     = np.zeros((num_l_laplac))
    lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(100.0), num=num_l_laplac-1, endpoint=True, base=10.0)

    # --------------------------------------------------------------------------
    if reg_matrix == 'I':
        order   = 0
        Laplac  = create_Laplacian_matrix(Npc, order)
    elif reg_matrix == 'L1':
        order   = 1
        Laplac  = create_Laplacian_matrix(Npc, order)
    elif reg_matrix == 'L2':
        order   = 2
        Laplac  = create_Laplacian_matrix(Npc, order)
    elif reg_matrix == 'InvT2':
        # Regularization matrix for method: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3568216/
        #: Junyu Guo et al. 2014. Multi-slice Myelin Water Imaging for Practical Clinical Applications at 3.0 T.
        T2s_mod = np.concatenate( (np.array([T2s[0] - 1.0]), T2s[:-1]) ) # add 0.0 and remove the last one
        deltaT2 = T2s - T2s_mod
        deltaT2[0] = deltaT2[1]
        Laplac  = np.diag(1./deltaT2)
    else:
        print ('Error: Wrong reg_matrix option!')
        sys.exit()
    # end if

    # create 2D solution_ ROI x Spectrum
    fsol_ROIs = np.zeros((nROIs, T2s.shape[0]))
    MWF_ROIs  = np.zeros((nROIs))

    # correct artifacts
    data[data<0.0]  = 0.0

    number_of_cores = multiprocessing.cpu_count()
    if num_cores == -1:
        num_cores = number_of_cores
        print ('Using all CPUs: ', number_of_cores)
    else:
        print ('Using ', num_cores, ' CPUs from ', number_of_cores)
    #end if

    #___________________________________________________________________________
    #_______________________________ ESTIMATION ________________________________
    #___________________________________________________________________________

    if denoise == 'TV' :
        print('Step #1: Denoising using Total Variation:')
        for voxelt in progressbar.progressbar(range(nt), redirect_stdout=True):
            print(voxelt+1, ' volumes processed')
            data_vol  = np.squeeze(data[:,:,:,voxelt])
            sigma_est = np.mean(estimate_sigma(data_vol, multichannel=False))
            #data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=1.0*sigma_est, eps=0.0002, n_iter_max=200, multichannel=False)
            data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=2.0*sigma_est, eps=0.0002, n_iter_max=200, multichannel=False)
        #end for
    elif denoise == 'NESMA' :
        data_den  = np.zeros_like(data)
        path_size = [6,6,6] # real-size = 2*path_size + 1
        print('Step #1: Denoising using the NESMA filter:')
        for voxelx in progressbar.progressbar(range(nx), redirect_stdout=True):
            print(voxelx+1, ' slices processed')
            min_x = np.max([voxelx - path_size[0], 0])
            max_x = np.min([voxelx + path_size[0], nx])
            for voxely in range(ny):
                min_y = np.max([voxely - path_size[1], 0])
                max_y = np.min([voxely + path_size[1], ny])
                for voxelz in range(nz):
                    if mask[voxelx, voxely,voxelz] == 1:
                        min_z = np.max([voxelz - path_size[2], 0])
                        max_z = np.min([voxelz + path_size[2], nz])
                        # -----------------------------------------
                        signal_path   = data[min_x:max_x, min_y:max_y, min_z:max_z, :]
                        dim           = signal_path.shape
                        signal_path2D = signal_path.reshape((np.prod(dim[0:3]), nt))
                        signal_xyz    = data[voxelx, voxely,voxelz]
                        RE            = 100 * np.sum(np.abs(signal_path2D - signal_xyz), axis=1)/np.sum(signal_xyz)
                        ind_valid     = RE < 2.5 # (percent %)
                        data_den[voxelx, voxely, voxelz] = np.mean(signal_path2D[ind_valid,:], axis=0)
                    #end if
                #end vz
            #end vy
        #end vx
        data = data_den.copy()
        del data_den
    #end if

    print('Step #2: Estimation of flip angles:')
    if FA_smooth == 'yes':
        # Smoothing the data for a robust B1 map estimation
        data_smooth = np.zeros((nx,ny,nz,nt))
        sig_g = 2.0
        for c in range(nt):
            data_smooth[:,:,:,c] = filt.gaussian_filter(np.squeeze(data[:,:,:,c]), sig_g, 0)
        #end for
    else:
        data_smooth = data.copy()
    #end if

    mean_T2_dist = 0
    for voxelz in progressbar.progressbar(range(nz), redirect_stdout=True):
        #print('Estimation of flip angles: slice', voxelz+1)
        print(voxelz+1, ' slices processed')
        # Parallelization by rows: this is more efficient for computing a single or a few slices
        mask_slice = mask[:,:,voxelz]
        data_slice = data_smooth[:,:,voxelz,:]
        #FA_par = Parallel(n_jobs=num_cores)(delayed(fitting_slice_FA)(mask_slice[:, voxely], data_slice[:,voxely,:], nx, Dic_3D, alpha_values) for voxely in tqdm(range(ny)))
        if FA_method == 'brute-force':
            FA_par = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice_FA_brute_force)(mask_slice[:, voxely], data_slice[:,voxely,:], nx, Dic_3D, alpha_values) for voxely in range(ny))
            for voxely in range(ny):
                FA[:,voxely,voxelz]       = FA_par[voxely][0]
                FA_index[:,voxely,voxelz] = FA_par[voxely][1]
                Ktotal[:, voxely,voxelz]  = FA_par[voxely][2]
                mean_T2_dist              = mean_T2_dist + FA_par[voxely][3]
            #end for voxely
        elif FA_method == 'spline':
            FA_par = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice_FA_spline_method)(Dic_3D_LR, Dic_3D, data_slice[:,voxely,:], mask_slice[:, voxely], alpha_values_spline, nx, alpha_values) for voxely in range(ny))
            for voxely in range(ny):
                FA[:,voxely,voxelz]       = FA_par[voxely][0]
                FA_index[:,voxely,voxelz] = FA_par[voxely][1]
                Ktotal[:, voxely,voxelz]  = FA_par[voxely][2]
                mean_T2_dist              = mean_T2_dist + FA_par[voxely][3]
            #end voxely
        #end if
    #end voxelx
    del data_smooth
    # TO DO: (1) Estimate also the standard deviation of the spectrum and plot it
    #        (2) Estimate a different mean spectrum for each tissue type (using a segmentation from a T1, or any strategy to segment the raw MET2 data)
    mean_T2_dist = mean_T2_dist/np.sum(mean_T2_dist)

    total_signal = 0
    total_Kernel = 0
    nv           = 0
    for voxelx in range(nx):
        for voxely in range(ny):
            for voxelz in range(nz):
                if mask[voxelx, voxely,voxelz] == 1:
                    total_signal = total_signal + data[voxelx,voxely,voxelz, :]
                    ind_xyz      = np.int(FA_index[voxelx,voxely,voxelz])
                    total_Kernel = total_Kernel + Dic_3D[:,:,ind_xyz]
                    nv = nv + 1.0
            #end vz
        #end vy
    #end vx
    total_Kernel     = total_Kernel/nv
    total_signal     = total_signal/nv

    fmean1, SSE      = nnls(total_Kernel, total_signal)
    dist_T2_mean1    = fmean1/np.sum(fmean1)

    #factor           = 1.02
    factor           = 1.01
    order            = 0
    Id               = create_Laplacian_matrix(Npc, order)
    fmean2, reg_opt2, k_est = nnls_x2(total_Kernel, total_signal, Id, factor)
    dist_T2_mean2    = fmean2/np.sum(fmean2)

    # Save mean_T2_dist, which is the initial value for RUMBA
    fig  = plt.figure('Showing results', figsize=(8,8))
    ax0  = fig.add_subplot(1, 1, 1)
    im0  = plt.plot(T2s, mean_T2_dist,  color='b', label='Mean T2-dist from all voxels: NNLS')
    im1  = plt.plot(T2s, dist_T2_mean1, color='g', label='T2-dist from mean signals: NNLS')
    im2  = plt.plot(T2s, dist_T2_mean2, color='r', label='T2-dist from mean signals: NNLS-X2-I')

    ax0.set_xscale('log')
    plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)
    plt.title('Mean spectrum', fontsize=18)
    plt.xlabel('T2', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)
    ax0.set_xlim(T2s[0], T2s[-1])
    ax0.set_ylim(0, np.max(mean_T2_dist)*1.2)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.tick_params(axis='both', which='minor', labelsize=14)
    ax0.set_yticks([])
    plt.legend()
    plt.savefig(path_to_save_data + 'Mean_spectrum_from_all_voxels.png', dpi=600)
    plt.close('all')

    # --------------------------------------------------------------------------
    logT2 = np.log(T2s)
    print('Step #3: Estimation of ROI-based T2 spectrum:')
    for nROI_i in progressbar.progressbar(range(nROIs), redirect_stdout=True):
        print(' ROIs processed:', nROI_i + 1)
        total_signal = 0
        total_Kernel = 0
        nv           = 0
        for voxelx in range(nx):
            for voxely in range(ny):
                for voxelz in range(nz):
                    if  ROIs[voxelx, voxely,voxelz] == nROIs_values[nROI_i]:
                        total_signal = total_signal + data[voxelx,voxely,voxelz, :]
                        ind_xyz      = np.int(FA_index[voxelx,voxely,voxelz])
                        total_Kernel = total_Kernel + Dic_3D[:,:,ind_xyz]
                        nv = nv + 1.0
                    #end if
                #end vz
            #end vy
        #end vx
        total_Kernel     = total_Kernel/nv
        total_signal     = total_signal/nv

        # ---------------------------- Estimation ------------------------------
        #factor           = 1.02
        factor           = 1.01
        x_sol, reg_opt2, k_est  = nnls_x2(total_Kernel, total_signal, Laplac, factor)
        vt               = np.sum(x_sol) + epsilon
        x_sol            = x_sol/vt

        fM      = np.sum(x_sol[ind_m])
        fIE     = np.sum(x_sol[ind_t])
        fCSF    = np.sum(x_sol[ind_csf])
        # ------ T2m
        # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
        T2m     = np.exp(np.sum(x_sol[ind_m] * logT2[ind_m])/(np.sum(x_sol[ind_m])   + epsilon))
        # ------ T2IE
        # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
        T2IE    = np.exp(np.sum(x_sol[ind_t] * logT2[ind_t])/(np.sum(x_sol[ind_t])   + epsilon))

        fsol_ROIs[nROI_i, :]  = x_sol
        MWF_ROIs[nROI_i]      = fM
        # ----------------------------------------------------------------------
        fig  = plt.figure('Spectrum', figsize=(8,8))
        ax0  = fig.add_subplot(1, 1, 1)
        im0  = plt.plot(T2s, x_sol, color='r', label='T2-dist from mean signals: NNLS-X2')

        ax0.set_xscale('log')
        plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)
        plt.title('Mean spectrum', fontsize=18)
        plt.xlabel('T2', fontsize=18)
        plt.ylabel('Intensity', fontsize=18)
        ax0.set_xlim(T2s[0], T2s[-1])
        ax0.set_ylim(0, np.max(x_sol)*1.2)
        ax0.tick_params(axis='both', which='major', labelsize=16)
        ax0.tick_params(axis='both', which='minor', labelsize=14)
        ax0.set_yticks([])
        plt.legend()

        try:
            path_to_save_ROI = path_to_save_data + 'ROI_' + repr(obj0(nROIs_values[nROI_i])) + '/'
            os.mkdir(path_to_save_ROI)
        except:
            print ('Warning: folder cannot be created')
        #end try

        plt.savefig(path_to_save_ROI + 'Mean_spectrum_ROI.png', dpi=600)
        plt.close('all')
        # ----------------------------------------------------------------------
        headers =  [ 'Parameter    ', 'Mean value']
        table   =  [
                   [ '1. MWF       ',  fM         ],
                   [ '2. IEWF      ',  fIE        ],
                   [ '3. FWF       ',  fCSF       ],
                   [ '4. T2M       ',  T2m        ],
                   [ '5. T2IE      ',  T2IE       ],
                   [ '6. TWC       ',  vt         ]
                   ]

        table_tabulated  = tabulate(table, headers=headers)
        print(table_tabulated)

        f2 = open(path_to_save_ROI + 'table_values.txt', 'w')
        f2.write(table_tabulated)
        f2.close()

        np.savetxt(path_to_save_ROI + 'table_values.csv', table, delimiter=",", fmt='%s')
    #end for ROIs
    print ('Done!')
    # ----------------------- Plot T2 spectra from all voxels ------------------
    rcParams["figure.figsize"] = [10.0, 8]
    rcParams['lines.linewidth'] = 1.25 # line width for plots
    rcParams.update({'font.size': 13}) # font size of axes text

    factor1 = 1.0
    plot_T2spectra(nROIs, fsol_ROIs, T2s, path_to_save_data,  'Spectra_all_ROIs',   factor1)
    np.savetxt(path_to_save_data + 'table_MWF.csv',     MWF_ROIs,     delimiter=",", fmt='%s')
    np.savetxt(path_to_save_data + 'table_Spectra.csv', fsol_ROIs,    delimiter=",", fmt='%s')
    np.savetxt(path_to_save_data + 'ROI_labels.csv',    nROIs_values, delimiter=",", fmt='%s')
#end main function
