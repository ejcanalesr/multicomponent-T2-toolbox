#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Robust myelin water imaging from multi-echo T2 data using second-order Tikhonov regularization with control points
# ISMRM 2019, Montreal, Canada. Abstract ID: 4686
# ------------------------------------------------------------------------------
# Developers:
#
# Erick Jorge Canales-Rodríguez (EPFL, CHUV, Lausanne, Switzerland; FIDMAG Research Foundation, CIBERSAM, Barcelona, Spain)
# Marco Pizzolato               (EPFL)
# Gian Franco Piredda           (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Tom Hilbert                   (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Alessandro Daducci            (Computer Science Department, University of Verona, Italy)
# Tobias Kober                  (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Jean-Philippe Thiran          (EPFL, UNIL, CHUV, Switzerland)

# Date: 11/02/2019
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
matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True

# import pdb

import sys
sys.path.append("..")

from epg.epg import *
from flip_angle_algorithms.fa_estimation import *
from intravoxel_algorithms.algorithms import *
from intravoxel_algorithms.bayesian_interpolation import *

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

def fitting_slice_T2(mask_1d, data_1d, FA_index_1d, nx, Dic_3D, lambda_reg, T2dim, nEchoes, reg_method, Laplac, dist_x_prior):
    # --------------------------------------
    tmp_f_sol_4D      = np.zeros((nx, T2dim))
    tmp_signal        = np.zeros((nx, nEchoes))
    tmp_Reg           = np.zeros((nx))
    totVoxels_sclices = np.count_nonzero(mask_1d)
    if totVoxels_sclices > 0 :
        # ----------------------------------------------------------------------
        #                            Voxelwise estimation
        # ----------------------------------------------------------------------
        for voxelx in range(0, nx):
            if (mask_1d[voxelx] > 0.0) & (np.sum(data_1d[voxelx, :]) > 0.0):
                # ==================== Reconstruction
                M       = np.ascontiguousarray(data_1d[voxelx, :])
                index_i = np.int_(FA_index_1d[voxelx])
                Kernel  = np.ascontiguousarray(Dic_3D[:,:,index_i])
                km_i    = M[0]
                #pdb.set_trace()
                if km_i > 0:  # only if there is signal
                    M = M/km_i
                    # ----------------------------------------------------------
                    if reg_method   == 'NNLS':
                        x_sol, rnorm_fkk = nnls( Kernel, M )
                        reg_opt = 0
                    elif  reg_method == 'T2SPARC':
                        reg_opt    = 1.8
                        x_sol      = nnls_tik(Kernel, M, Laplac, reg_opt)
                    elif reg_method == 'X2':
                        k          = 1.02
                        x_sol, reg_opt, k_est = nnls_x2(Kernel, M, Laplac, k)
                        reg_opt    = k_est
                    elif  reg_method == 'L_curve':
                        reg_opt    = nnls_lcurve_wrapper(Kernel, M, Laplac, lambda_reg)
                        x_sol      = nnls_tik(Kernel, M, Laplac, reg_opt)
                    elif  reg_method == 'GCV':
                        x_sol, reg_opt = nnls_gcv(Kernel, M, Laplac)
                    elif reg_method == 'BayesReg':
                        x_sol, reg_opt   = BayesReg_nnls(Kernel, M, Laplac)
                    #end
                    # -------------
                    tmp_Reg[voxelx]        = reg_opt  # Regularization parameter
                    tmp_f_sol_4D[voxelx,:] = x_sol * km_i
                    tmp_signal[voxelx,:]   = np.dot(Kernel, x_sol) * km_i
                #end if ki
                # ---------------------------------------------------------#
            #end if mask
        #end for x
    #end if
    return tmp_f_sol_4D, tmp_signal, tmp_Reg
#end main function

#_______________________________________________________________________________
def motor_recon_met2(TE_array, path_to_data, path_to_mask, path_to_save_data, TR, reg_method, reg_matrix, denoise, FA_method, FA_smooth, myelin_T2, num_cores):
    # Load Data and Mask
    img      = nib.load(path_to_data)
    data     = img.get_fdata()
    data     = data.astype(np.float64, copy=False)

    img_mask = nib.load(path_to_mask)
    mask     = img_mask.get_fdata()
    mask     = mask.astype(np.int64, copy=False)

    print('--------- Data shape -----------------')
    nx, ny, nz, nt = data.shape
    print(data.shape)
    print('--------------------------------------')

    for c in range(nt):
        data[:,:,:,c] = np.squeeze(data[:,:,:,c]) * mask
    #end

    # Only for testing: selects a few slices
    #mask[:,:,40:-1] = 0
    #mask[:,:,0:35]  = 0

    nEchoes   = TE_array.shape[0]
    tau       = TE_array[1] - TE_array[0]

    fM        = np.zeros((nx, ny, nz))
    fIE       = fM.copy()
    fnT       = fM.copy()
    fCSF      = fM.copy()
    T2m       = fM.copy()
    T2IE      = fM.copy()
    T2nT      = fM.copy()
    Ktotal    = fM.copy()
    FA        = fM.copy()
    FA_index  = fM.copy()
    reg_param = fM.copy()
    NITERS    = fM.copy()

    # ==============================================================================
    # Inital values for the dictionary

    if reg_method == 'T2SPARC':
        # Regularization matrix for method: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3568216/
        #: Junyu Guo et al. 2014. Multi-slice Myelin Water Imaging for Practical Clinical Applications at 3.0 T.
        Npc = 96
    else:
        Npc = 60
    #end if
    # -----------
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
    # lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(100.0), num=num_l_laplac-1, endpoint=True, base=10.0)
    lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(10.0), num=num_l_laplac-1, endpoint=True, base=10.0)

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
        print('Error: Wrong reg_matrix option!')
        sys.exit()
    # end if

    # create 4D images
    f_sol_4D = np.zeros((nx, ny, nz, T2s.shape[0]))
    s_sol_4D = np.zeros((nx, ny, nz, nEchoes))

    data[data<0.0]  = 0.0 # correct artifacts

    number_of_cores = multiprocessing.cpu_count()
    if num_cores == -1:
        num_cores = number_of_cores
        print('Using all CPUs: ', number_of_cores)
    else:
        print('Using ', num_cores, ' CPUs from ', number_of_cores)
    #end if

    #_______________________________________________________________________________
    #_______________________________ ESTIMATION ____________________________________
    #_______________________________________________________________________________

    if denoise == 'TV' :
        print('Step #1: Denoising using Total Variation:')
        
        for voxelt in progressbar.progressbar(range(nt), redirect_stdout=True):
            print(voxelt+1, ' volumes processed')
            data_vol  = np.squeeze(data[:,:,:,voxelt])
            sigma_est = np.mean(estimate_sigma(data_vol, channel_axis=None))
            #data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=1.0*sigma_est, eps=0.0002, n_iter_max=200, multichannel=False)
            data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=2.0*sigma_est, eps=0.0002, max_num_iter=200, channel_axis=None)
        #end for
        outImg = nib.Nifti1Image(data, img.affine)
        nib.save(outImg, path_to_save_data + 'Data_denoised.nii.gz')
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
                    ind_xyz      = np.int_(FA_index[voxelx,voxely,voxelz])
                    total_Kernel = total_Kernel + Dic_3D[:,:,ind_xyz]
                    nv = nv + 1.0
            #end vz
        #end vy
    #end vx
    total_Kernel     = total_Kernel/nv
    total_signal     = total_signal/nv

    fmean1, SSE      = nnls(total_Kernel, total_signal)
    dist_T2_mean1    = fmean1/np.sum(fmean1)

    factor           = 1.01 # smaller than 1.02 due to the low level of noise
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
    plt.savefig(path_to_save_data + 'Mean_spectrum_unitial_iter.png', dpi=600)
    plt.close('all')
    # --------------------------------------------------------------------------

    print('Step #3: Estimation of T2 spectra:')
    for voxelz in progressbar.progressbar(range(nz), redirect_stdout=True):
        print(voxelz+1, ' slices processed')
        # Parallelization by rows: this is more efficient for computing a single or a few slices
        mask_slice = mask[:,:,voxelz]
        data_slice = data[:,:,voxelz,:]
        FA_index_slice = FA_index[:,:,voxelz]
        #T2_par = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice_T2)(mask_slice[:, voxely], data_slice[:,voxely,:], FA_index_slice[:, voxely], nx, Dic_3D, lambda_reg, alpha_values, T2s.shape[0], nEchoes, num_l_laplac, N_alphas, reg_method, Laplac1, Laplac2, Is, Laplac_mod, mean_T2_dist, Laplac2_cp_var, W_inv_deltaT2) for voxely in range(ny))
        T2_par = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice_T2)(mask_slice[:, voxely], data_slice[:,voxely,:], FA_index_slice[:, voxely], nx, Dic_3D, lambda_reg, T2s.shape[0], nEchoes, reg_method, Laplac, mean_T2_dist) for voxely in range(ny))
        for voxely in range(ny):
            f_sol_4D[:,voxely,voxelz,:] = T2_par[voxely][0]
            s_sol_4D[:,voxely,voxelz,:] = T2_par[voxely][1]
            reg_param[:,voxely,voxelz]  = T2_par[voxely][2]
        #end voxely
    #end voxelx

    print('Step #4: Estimation of quantitative metrics')
    logT2 = np.log(T2s)
    for voxelx in range(nx):
        for voxely in range(ny):
            for voxelz in range(nz):
                if mask[voxelx, voxely, voxelz] > 0.0:
                    M     = data[voxelx, voxely, voxelz, :]
                    x_sol = f_sol_4D[voxelx, voxely, voxelz,:]
                    vt    = np.sum(x_sol) + epsilon
                    x_sol = x_sol/vt
                    # fill matrices
                    # pdb.set_trace()
                    fM  [voxelx, voxely, voxelz] = np.sum(x_sol[ind_m])
                    fIE [voxelx, voxely, voxelz] = np.sum(x_sol[ind_t])
                    fCSF[voxelx, voxely, voxelz] = np.sum(x_sol[ind_csf])
                    # ------ T2m
                    # Aritmetic mean
                    # T2m [voxelx, voxely, voxelz] = np.sum(x_sol[ind_m] * T2s[ind_m])/(np.sum(x_sol[ind_m])   + epsilon)
                    # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
                    T2m[voxelx, voxely, voxelz] = np.exp(np.sum(x_sol[ind_m] * logT2[ind_m])/(np.sum(x_sol[ind_m])   + epsilon))
                    # ------ T2IE0
                    # Aritmetic mean
                    # T2IE[voxelx, voxely, voxelz] = np.sum(x_sol[ind_t] * T2s[ind_t])/(np.sum(x_sol[ind_t])   + epsilon)
                    # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
                    T2IE[voxelx, voxely, voxelz] = np.exp(np.sum(x_sol[ind_t] * logT2[ind_t])/(np.sum(x_sol[ind_t])   + epsilon))
                    Ktotal[voxelx, voxely, voxelz] = vt
                # end if
            #end for z
        # end for y
    # end for x

    # -------------------------- Save all datasets -----------------------------
    outImg = nib.Nifti1Image(fM, img.affine)
    nib.save(outImg, path_to_save_data + 'MWF.nii.gz')

    outImg = nib.Nifti1Image(fIE, img.affine)
    nib.save(outImg, path_to_save_data + 'IEWF.nii.gz')

    outImg = nib.Nifti1Image(fCSF, img.affine)
    nib.save(outImg, path_to_save_data + 'FWF.nii.gz')

    outImg = nib.Nifti1Image(T2m, img.affine)
    nib.save(outImg, path_to_save_data + 'T2_M.nii.gz')

    outImg = nib.Nifti1Image(T2IE, img.affine)
    nib.save(outImg, path_to_save_data + 'T2_IE.nii.gz')

    outImg = nib.Nifti1Image(Ktotal, img.affine)
    nib.save(outImg, path_to_save_data + 'TWC.nii.gz')

    outImg = nib.Nifti1Image(FA, img.affine)
    nib.save(outImg, path_to_save_data + 'FA.nii.gz')

    outImg = nib.Nifti1Image(f_sol_4D, img.affine)
    nib.save(outImg, path_to_save_data + 'fsol_4D.nii.gz')

    outImg = nib.Nifti1Image(s_sol_4D, img.affine)
    nib.save(outImg, path_to_save_data + 'Est_Signal.nii.gz')

    outImg = nib.Nifti1Image(reg_param, img.affine)
    nib.save(outImg, path_to_save_data + 'reg_param.nii.gz')
    #end if
    print('Done!')
#end main function
