#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

# Date: 2020
#===============================================================================

from __future__ import division

import scipy
from   scipy.optimize import minimize_scalar, fminbound, minimize

import numpy as np
import math

import inspect
import sys
import os
sys.path.insert(1, os.path.dirname(inspect.getfile(scipy.optimize)))
import _nnls

import numba as nb

import spams
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

    x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index, maxiter)

    #if mode != 1:
    #    raise RuntimeError("too many iterations")
    return x, rnorm
#end

# ------------------------------------------------------------------------------
# X2 standard method of Mackay-Label usig the identity matrix
def compute_f_alpha_RNNLS_MacKay(Dic_i, M):
    f0, kk      = nnls( Dic_i, M )
    SSE         = np.sum( (np.dot(Dic_i, f0) - M)**2 )
    # -----------------------
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    Laplac      = np.eye(n)
    M_aug1      = np.concatenate((M, Zerosm))
    factor      = 1.02
    reg_opt     = fminbound(NNLSreg_obj_I, 0.0, 10.0, args=(Dic_i, Laplac, M_aug1, SSE, factor), xtol=1e-05, maxfun=500, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*Laplac)), M_aug1 )
    return f, reg_opt
#end fun

def NNLSreg_obj_I(x, D, L, Signal, SSE, factor):
    Daux     = np.concatenate((D, np.sqrt(x)*L))
    f, kk    = nnls( Daux, Signal )
    SSEr     = np.sum( (np.dot(Daux, f) - Signal)**2 )
    cost_fun = np.abs(SSEr - factor*SSE)/SSE
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# X2 method using the Laplacian
def compute_f_alpha_RNNLS_X2_L(Dic_i, M, Laplac_mod):
    f0, kk      = nnls( Dic_i, M )
    SSE         = np.sum( (np.dot(Dic_i, f0) - M)**2 )
    # -----------------------
    m,n         = Dic_i.shape
    M_aug1      = np.concatenate( (M, np.zeros((n))) )
    factor      = 1.02
    reg_opt     = fminbound(NNLSreg_obj_L, 0.0, 10.0, args=(Dic_i, Laplac_mod, M_aug1, SSE, factor), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*Laplac_mod)), M_aug1 )
    return f, reg_opt
#end fun

def NNLSreg_obj_L(x, D, L, Signal, SSE, factor):
    Daux     = np.concatenate((D, np.sqrt(x)*L))
    f, kk    = nnls( Daux, Signal )
    SSEr     = np.sum( (np.dot(Daux, f) - Signal)**2 )
    cost_fun = np.abs(SSEr - factor*SSE)/SSE
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Tikhonov regularization using the identity matrix and a fixed regularization parameter
def compute_f_alpha_RNNLS_I_fixreg(Dic_i, M, reg_opt):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # --------- Estimation
    In          = np.eye(n)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*In)), M_aug )
    return f
#end fun

# ------------------------------------------------------------------------------
# Tikhonov regularization using the Laplacian matrix and a fixed regularization parameter
def compute_f_alpha_RNNLS_L_fixreg(Dic_i, M, Laplac_mod, reg_opt):
    m,n         = Dic_i.shape
    M_aug       = np.concatenate((M, np.zeros((n))))
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*Laplac_mod)), M_aug )
    return f
#end fun

# ------------------------------------------------------------------------------
# GCV usig the identity matrix
# [A better solution is found by selecting the subset of columns in the dictionary
# with corresponding positive coefficients]

def compute_f_GCV_I(Dic_i, M):
    m,n         = Dic_i.shape
    Zerosn      = np.zeros((n))
    M_aug       = np.concatenate((M,  Zerosn))
    Im          = np.eye(m)
    In          = np.eye(n)
    reg_sol     = fminbound(obj_GCV_I, 1e-5, 10.0, args=(Dic_i, In, M_aug, m, Im), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol)*In)), M_aug )
    return f, reg_sol
#end fun

def obj_GCV_I(x, D, In, Signal, m, Im):
    Daux     = np.concatenate((D, np.sqrt(x)*In))
    f, SSEr  = nnls( Daux, Signal )
    Dr       = D[:, f>0]
    Lr       = In[f>0, f>0]
    DTD      = np.matmul(Dr.T, Dr)
    LTL      = np.matmul(Lr.T, Lr)
    #A        = np.matmul(Dr, np.matmul( inv( DTD  + x*LTL ), Dr.T) )
    A        = np.matmul(Dr, np.linalg.lstsq(DTD  + x*LTL, Dr.T, rcond=None)[0] )
    cost_fun = ( (1.0/m)*(SSEr**2.0) ) / ((1.0/m) * np.trace(Im - A) )**2.0
    return np.log(cost_fun)
# end fun

# ------------------------------------------------------------------------------
def compute_f_GCV_L(Dic_i, M, L):
    m,n         = Dic_i.shape
    M_aug       = np.concatenate( (M, np.zeros((n))) )
    Im          = np.eye(m)
    reg_opt     = fminbound(obj_GCV_L, 1e-5, 10.0, args=(Dic_i, L, M_aug, m, Im), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_opt)*L)), M_aug )
    return f, reg_opt
#end fun

def obj_GCV_L(x, D, L, Signal, m, Im):
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

# ------------------------------------------------------------------------------
# Determining the optimal regularization value using the L-curve method, which
# is determined by means of the triangle method
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
        Log_error[i_laplac] = np.log( np.sum( ( np.dot(D, x) - y  )**2.0 ) + 1e-200)
        Log_norms[i_laplac] = np.log( np.sum( ( np.dot(Laplac_mod, x) )**2.0 ) + 1e-200)
        # ---------------------------------
    #end for
    corner   = select_corner(Log_error, Log_norms)
    reg_opt  = lambda_reg[corner]
    return reg_opt
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
    for k in xrange(0, n - 2):
        b = [x[k], y[k]]
        for j in xrange(k + 1, n - 1):
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
