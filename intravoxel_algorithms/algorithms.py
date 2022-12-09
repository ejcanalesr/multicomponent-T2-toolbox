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
from   scipy.optimize import minimize_scalar, fminbound, minimize
#from   scipy.signal import savgol_filter
#from scipy.stats import norm, linregress, pearsonr
#from   scipy import sparse, linalg
#from   scipy.interpolate import interp1d

#import os
import numpy as np
import math

import inspect
import sys
import os
#sys.path.insert(1, os.path.dirname(inspect.getfile(scipy.optimize)))
#import _nnls
from scipy.optimize import _nnls,__nnls

import numba as nb

#import spams
#import pdb

epsilon=1.0e-16

#===============================================================================
#                                FUNCTIONS
#===============================================================================

# Standard NNLS python in scipy.
# The default number of iterations was increased from 3n to 5n to improve
# the estimation of smooth solutions.
# The "too many iterations" error was removed.

def nnls(A, b):
    A, b = map(np.asarray_chkfinite, (A, b))

    #if len(A.shape) != 2:
    #    raise ValueError("expected matrix")
    #if len(b.shape) != 1:
    #    raise ValueError("expected vector")

    m, n = A.shape

    #if m != b.shape[0]:
    #    raise ValueError("incompatible dimensions")

    #maxiter = -1 if maxiter is None else int(maxiter)
    maxiter = -1
    #maxiter = int(5*n)

    w     = np.zeros((n,), dtype=np.double)
    zz    = np.zeros((m,), dtype=np.double)
    index = np.zeros((n,), dtype=int)
    
    #x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index, maxiter)
    x, rnorm, mode = __nnls.nnls(A, m, n, b, w, zz, index, maxiter)

    #if mode != 1:
    #    raise RuntimeError("too many iterations")
    return x, rnorm
#end

# ------------------------------------------------------------------------------
#                                  L-CURVE
# ------------------------------------------------------------------------------
# Wrapper to the augmented nnls algorithm for many different lambdas
def nnls_lcurve_wrapper(D, y, Laplac_mod, lambda_reg):
    # This script is using the fact that L_mod.T*L_mod = L.T*L+ Is.T*Is,
    # where L_mod is found via the cholesky decomposition
    m, n = D.shape
    # ------------------------------------
    # Define variables
    b            = np.concatenate((y, np.zeros((n))))
    num_l_laplac = len(lambda_reg)
    Log_error    = np.zeros((num_l_laplac))
    Log_norms    = np.zeros((num_l_laplac))
    # -------------------------------------
    for i_laplac in range(0, num_l_laplac):
        lambda_reg_i = lambda_reg[i_laplac]
        A  = np.concatenate( (D, np.sqrt(lambda_reg_i)*Laplac_mod) )
        # ---------------------  Standard NNLS - scipy -------------------------
        x, rnorm = nnls(A, b)
        # ----------------------------------------------------------------------
        # Variables for the L-curve Method
        Log_error[i_laplac] = np.log( np.sum( ( np.dot(D, x) - y  )**2.0 )     + 1e-200)
        Log_norms[i_laplac] = np.log( np.sum( ( np.dot(Laplac_mod, x) )**2.0 ) + 1e-200)
        # ---------------------------------
    #end for
    corner   = select_corner(Log_error, Log_norms)
    reg_opt  = lambda_reg[corner]
    return reg_opt
#end fun

def nnls_lcurve_wrapper_prior(D, y, L, x0, lambda_reg):
    m, n = D.shape
    # ------------------------------------
    # Define variables
    num_l_laplac = len(lambda_reg)
    Log_error    = np.zeros((num_l_laplac))
    Log_norms    = np.zeros((num_l_laplac))
    # -------------------------------------
    for i_laplac in range(0, num_l_laplac):
        lambda_reg_i = lambda_reg[i_laplac]
        A  = np.concatenate( (D, np.sqrt(lambda_reg_i) * L ) )
        b  = np.concatenate( (y, np.sqrt(lambda_reg_i) * x0) )
        # ---------------------  Standard NNLS - scipy -------------------------
        x, rnorm = nnls(A, b)
        # ----------------------------------------------------------------------
        # Variables for the L-curve Method
        Log_error[i_laplac] = np.log( np.sum( ( np.dot(D, x) - y  )**2.0 ) + 1e-200)
        Log_norms[i_laplac] = np.log( np.sum( ( np.dot(L, (x-x0)) )**2.0 ) + 1e-200)
        # ---------------------------------
    #end for
    # Smoothing
    Log_error = savgol_filter(Log_error, 9, 3)
    Log_norms = savgol_filter(Log_norms, 9, 3)

    corner   = select_corner(Log_error, Log_norms)
    reg_opt  = lambda_reg[corner]
    # ----------------------------------------
    #       Compute final solution
    # ----------------------------------------
    A  = np.concatenate( (D, np.sqrt(reg_opt) * L ) )
    b  = np.concatenate( (y, np.sqrt(reg_opt) * x0) )
    x_sol, rnorm = nnls(A, b)
    return x_sol, reg_opt
#end fun

def select_corner(x,y):
    """
    Select the corner value of the L-curve formed inversion results.
    References:
    Castellanos, J. L., S. Gomez, and V. Guerra (2002), The triangle method
    for finding the corner of the L-curve, Applied Numerical Mathematics,
    43(4), 359-373, doi:10.1016/S0168-9274(01)00179-9.

    http://www.fatiando.org/v0.5/_modules/fatiando/inversion/hyper_param.html
    """
    x, y = scale_curve(x,y)
    n = len(x)
    corner = n - 1

    def dist(p1, p2):
        "Return the geometric distance between p1 and p2"
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    #end

    cte = 7. * np.pi / 8.
    angmin = None
    c = [x[-1], y[-1]]
    for k in range(0, n - 2):
        b = [x[k], y[k]]
        for j in range(k + 1, n - 1):
            a = [x[j], y[j]]
            ab = dist(a, b)
            ac = dist(a, c)
            bc = dist(b, c)
            cosa = (ab ** 2 + ac ** 2 - bc ** 2) / (2. * ab * ac)
            cosa = max(-1.0, min(cosa, 1.0)) # valid range: [-1, 1]
            ang  = np.arccos(cosa)
            area = 0.5 * ((b[0] - a[0]) * (a[1] - c[1]) - (a[0] - c[0]) * (b[1] - a[1]))
            # area is > 0 because in the paper C is index 0
            if area > 0 and (ang < cte and (angmin is None or ang < angmin)):
                corner = j
                angmin = ang
            #end if
        #end for j
    #end for k
    return corner
#end fun

def scale_curve(x,y):
    """
    Puts the data-misfit and regularizing function values in the range
    [-10, 10].

    http://www.fatiando.org/v0.5/_modules/fatiando/inversion/hyper_param.html
    """
    def scale(a):
        vmin, vmax = a.min(), a.max()
        l, u = -10, 10
        return (((u - l) / (vmax - vmin)) * (a - (u * vmin - l * vmax) / (u - l)))
    #end fun
    return scale(x), scale(y)
#end fun

# ------------------------------------------------------------------------------
#                        X2: conventional method of Mackay
# ------------------------------------------------------------------------------
def nnls_x2(Dic_i, M, Laplac, factor):
    f0, kk      = nnls( Dic_i, M )
    SSE         = np.sum( (np.dot(Dic_i, f0) - M)**2 )
    # -----------------------
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug1      = np.concatenate((M, Zerosm))
    # factor      = 1.02
    reg_opt     = fminbound(obj_nnls_x2, 0.0, 10.0, args=(Dic_i, Laplac, M_aug1, SSE, factor, M), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*Laplac)), M_aug1 )
    #return f, reg_opt
    k_est       = np.sum( (np.dot(Dic_i, f) - M)**2 )/SSE
    return f, reg_opt, k_est
#end fun

def obj_nnls_x2(x, D, L, Signal, SSE, factor, M):
    Daux     = np.concatenate((D, np.sqrt(x)*L))
    f, kk    = nnls( Daux, Signal )
    #SSEr     = np.sum( (np.dot(Daux, f) - Signal)**2 )
    SSEr     = np.sum( (np.dot(D, f) - M)**2 )
    cost_fun = np.abs(SSEr - factor*SSE)/SSE
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
#                        X2 using an apriori estimate
# ------------------------------------------------------------------------------
def nnls_x2_prior(Dic_i, M, x0, factor):
    f0, kk      = nnls( Dic_i, M )
    SSE         = np.sum( (np.dot(Dic_i, f0) - M)**2 )
    # -----------------------
    m,n         = Dic_i.shape
    Laplac      = np.eye(n)
    #factor      = 1.02
    reg_opt     = fminbound(obj_nnls_x2_prior, 0.0, 100.0, args=(Dic_i, Laplac, M, SSE, factor, x0), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate( (Dic_i, np.sqrt(reg_opt)*Laplac) ), np.concatenate( (M, np.sqrt(reg_opt)*x0) ) )
    return f, reg_opt
#end fun

def obj_nnls_x2_prior(x, D, L, M, SSE, factor, x0):
    Daux     = np.concatenate( (D, np.sqrt(x) * L ) )
    Signal   = np.concatenate( (M, np.sqrt(x) * x0) )
    f, kk    = nnls( Daux, Signal )
    SSEr     = np.sum( (np.dot(Daux, f) - Signal)**2 )
    cost_fun = np.abs(SSEr - factor*SSE)/SSE
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Tikhonov regularization using a Laplacian matrix and a fixed reg. parameter
# ------------------------------------------------------------------------------
def nnls_tik(Dic_i, M, Laplac, reg_opt):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*Laplac)), M_aug )
    return f
#end fun

# ------------------------------------------------------------------------------
#                                 GCV
# Modified GCV that selects the subset of columns in the dictionary with
# corresponding positive coefficients
# ------------------------------------------------------------------------------
def nnls_gcv(Dic_i, M, L):
    m,n         = Dic_i.shape
    M_aug       = np.concatenate( (M, np.zeros((n))) )
    Im          = np.eye(m)
    reg_opt     = fminbound(obj_nnls_gcv, 1e-8, 10.0, args=(Dic_i, L, M_aug, m, Im), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*L)), M_aug )
    return f, reg_opt
#end fun

def obj_nnls_gcv(x, D, L, Signal, m, Im):
    Daux     = np.concatenate((D, np.sqrt(x)*L))
    f, SSEr  = nnls( Daux, Signal )
    Dr       = D[:, f>0]
    Lr       = L[f>0, f>0]
    DTD      = np.matmul(Dr.T, Dr)
    LTL      = np.matmul(Lr.T, Lr)
    #A        = np.matmul(Dr, np.matmul( inv( DTD + x*LTL ), Dr.T) )
    A        = np.matmul(Dr, np.linalg.lstsq(DTD  + x*LTL, Dr.T, rcond=None)[0] )
    cost_fun = ( (1.0/m)*(SSEr**2.0) ) / ((1.0/m) * np.trace(Im - A) )**2.0
    return np.log(cost_fun)
# end fun
