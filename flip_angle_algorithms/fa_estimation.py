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
# Tobias Kober                  (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Jean-Philippe Thiran          (EPFL, UNIL, CHUV, Switzerland)
# Alessandro Daducci            (Computer Science Department, University of Verona, Italy)
# Date: 11/02/2019
#===============================================================================

from __future__ import division
import scipy
from   scipy.optimize import minimize_scalar
from   scipy.interpolate import interp1d

import numpy as np

import sys
sys.path.append("..")
from intravoxel_algorithms.algorithms import nnls

#===============================================================================
#                                FUNCTIONS
#===============================================================================

# ------------------- Spline-based interpolation ------------------------------#
def fitting_slice_FA_spline_method(Dic_3D_LR, Dic_3D, data_1d, mask_1d, alpha_values_spline, nx, alpha_values):
    tmp_FA            = np.zeros((nx))
    tmp_FA_index      = np.zeros((nx))
    totVoxels_sclices = np.count_nonzero(mask_1d)
    tmp_KM            = np.zeros((nx))
    tmp_Fsol          = 0.0
    # -------------------------------------------
    dim3              = Dic_3D_LR.shape[2]
    if totVoxels_sclices > 0:
        for voxelx in range(0, nx):
            if (mask_1d[voxelx] > 0.0) & (np.sum(data_1d[voxelx, :]) > 0.0):
                M        = data_1d[voxelx, :]
                # M        = M/M[0]
                residual = np.zeros((dim3))
                for i in range(dim3):
                    Dic_i        = np.ascontiguousarray(Dic_3D_LR[:,:,i])
                    f, rnorm_f   = nnls( Dic_i, M )
                    residual[i]  = rnorm_f
                #end for iter
                f2  = interp1d(alpha_values_spline, residual, kind='cubic')
                res = minimize_scalar(f2, method='Bounded', bounds=(90., 180.))
                # Find FA closest to the predefined grid alpha_values
                indexFA = np.argmin( np.abs(alpha_values - res.x) )
                tmp_FA_index[voxelx] = indexFA
                tmp_FA[voxelx]       = alpha_values[indexFA]
                # ------- estimate PD and T2 distribution
                Dic_i          = np.ascontiguousarray(Dic_3D[:,:,indexFA])
                fsol, f_sqrtn  = nnls( Dic_i, M )
                km_i           = np.sum(fsol)
                tmp_KM[voxelx] = km_i
                tmp_Fsol       = tmp_Fsol + fsol
            #end if mask
        #end for x
    #end if
    return tmp_FA, tmp_FA_index, tmp_KM, tmp_Fsol
#end fun

#-------------------------------------------------------------------------------

def compute_optimal_FA(M, Dic_3D, alpha_values):
    dim3      = Dic_3D.shape[2]
    residual  = np.zeros((dim3))
    M = np.ascontiguousarray(M)
    for iter in range(dim3):
        Dic_i          = np.ascontiguousarray(Dic_3D[:,:,iter])
        f, rnorm_f     = nnls( Dic_i, M )
        residual[iter] = rnorm_f
    #end for
    index       = np.argmin(residual)
    Dic_i       = np.ascontiguousarray(Dic_3D[:,:,index])
    f, f_sqrtn  = nnls( Dic_i, M )
    km          = np.sum(f)
    alpha       = alpha_values[index]
    SSE         = np.sum( (np.dot(Dic_i, f) - M)**2 )
    return index, alpha, km, SSE, f
#end function

def fitting_slice_FA_brute_force(mask_1d, data_1d, nx, Dic_3D, alpha_values):
    tmp_FA         = np.zeros((nx))
    tmp_FA_index   = np.zeros((nx))
    tmp_KM         = np.zeros((nx))
    tmp_Fsol       = 0.0
    totVoxels_sclices = np.count_nonzero(mask_1d)
    if totVoxels_sclices > 0:
        for voxelx in xrange(nx):
            if (mask_1d[voxelx] > 0.0) and (np.sum(data_1d[voxelx, :])) > 0.0:
                M      = data_1d[voxelx, :]
                # compute the flip angle (alpha_mean) and the proton density (km_i)
                index_i, alpha_mean, km_i, SSE, fsol = compute_optimal_FA(M, Dic_3D, alpha_values)
                tmp_FA[voxelx]         = alpha_mean
                tmp_FA_index[voxelx]   = index_i
                tmp_KM[voxelx]         = km_i
                tmp_Fsol               = tmp_Fsol + fsol
            #end if mask
        #end for x
    #end if
    return tmp_FA, tmp_FA_index, tmp_KM, tmp_Fsol
#end function
