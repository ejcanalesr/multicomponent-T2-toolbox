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
from   scipy.special import erf
from   scipy.linalg import cholesky, det, inv


import numpy as np
import numba as nb

import inspect
import sys
import os
#sys.path.insert(1, os.path.dirname(inspect.getfile(scipy.optimize)))
#import _nnls
from scipy.optimize import _nnls,__nnls

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
#                           BayesReg
#              THIS IS THE FUNCTION USED IN THE PAPER
# ------------------------------------------------------------------------------
# For more details see:
# Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
# Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
# This code can be accelerated by using a grid of alpha-values and by precomputing the related cholesky decompositions and determinants.
# ------------------------------------------------------------------------------
def BayesReg_nnls(Dic_i, M, L):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(Dic_i, M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (M - np.dot(Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(Dic_i.T, Dic_i)
    K           = np.matmul(L.T, L)
    det_L       = det(L)
    reg_sol     = fminbound(obj_BayesReg_nnls, 1e-8, 2.0, args=(Dic_i, L, M_aug, M, m, n, B, det_L, beta, K), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol) * L)), M_aug )
    return f, reg_sol
#end fun

def obj_BayesReg_nnls(x, D, L, Signal_aug, Signal, m, n, B, det_L, beta, K):
    Daux        = np.concatenate((D, np.sqrt(x) * L))
    f, kk       = nnls( Daux, Signal_aug )
    ED          = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(L, f)**2 )
    A           = beta*B + (beta*x)*K
    # -----------------------
    # A=U.T*U
    U           = cholesky(A, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    #det_U       = det(U)
    det_U       = np.prod(np.diag(U)) # faster evaluation of the determinant

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2.) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    cost_fun1   = beta*ED + beta*x*EW + np.log(det_U) - (n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m/2.) * np.log(2.*np.pi) - (m/2.) * np.log(beta) + (n/2.) * np.log(np.pi) - (n/2.) * np.log(2*beta*x)  - np.log(det_L)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
#   *** OTHER PREVIOUS IMPLEMENTATIONS AND VARIANTS TO TEST IN THE FUTURE ***
# ------------------------------------------------------------------------------
# Original equations by Mackay: Bayesian Interpolation
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
def compute_f_alpha_RNNLS_I_evidencelog(Dic_i, M):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    In          = np.eye(n)
    # ----------------------
    # SNR = M[0]/sigma, where M[0]=1 after normalization
    # SNR_min = 50
    # SNR_max = 1000

    # beta = 1/sigma**2
    inv_sigma_min = 20.0
    inv_sigma_max = 1000.0
    inv_sigma0    = 200.0

    beta_min      = inv_sigma_min**2
    beta_max      = inv_sigma_max**2
    beta0         = inv_sigma0**2
    # ----------------------
    x0          = [beta0, 0.1*beta0] # initial estimate
    bnds        = ((beta_min, beta_max),(1e-5*beta_min, 10.0*beta_max)) # bounds
    B           = np.matmul(Dic_i.T, Dic_i)
    res         = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'gtol': 1e-20, 'disp': False, 'maxiter': 300}, bounds = bnds, args=(Dic_i, In, M_aug, M, m, n, B))
    reg_sol     = res.x
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol[1]/reg_sol[0]) * In)), M_aug )
    return f, reg_sol[0], reg_sol[1]
#end fun

def NNLSreg_obj_evidencelog(x, D, In, Signal_aug, Signal, m, n, B):
    Daux       = np.concatenate((D, np.sqrt(x[1]/x[0]) * In))
    f, kk      = nnls( Daux, Signal_aug )
    ED         = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW         = 0.5 * np.sum ( f**2 )
    ratio_dem  = np.linalg.det( x[0] * B + x[1] * In)
    # ------------------------
    cost_fun   = (x[0] * ED + x[1] * EW) + 0.5 * np.log(ratio_dem) - (m/2.) * np.log(x[0]) - (n/2.) *np.log(x[1]) + (m/2.) * np.log(2.*np.pi)
    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Modified equations by Mackay: Bayesian Interpolation.
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
# Moreover, here we assume we know beta
def compute_f_alpha_RNNLS_I_evidencelog_fast(Dic_i, M):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    In          = np.eye(n)
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk      = nnls(Dic_i, M)
    sigma       = np.sqrt( np.sum( (M - np.dot(Dic_i, x0))**2 ) / (m - 1.0) )
    beta        = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*x
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(Dic_i.T, Dic_i)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_fast, 1e-8, 10.0, args=(Dic_i, In, M_aug, M, m, n, B, beta), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol) * In)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_fast(x, D, In, Signal_aug, Signal, m, n, B, beta):
    Daux       = np.concatenate((D, np.sqrt(x) * In))
    f, kk      = nnls( Daux, Signal_aug )
    ED         = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    #EW         = 0.5 * np.sum ( np.dot(In, f)**2 )
    EW         = 0.5 * np.sum ( f**2 )
    ratio_dem  = np.linalg.det( B + x*In )
    # ------------------------
    cost_fun   = beta * (ED + x*EW) + 0.5 * np.log(ratio_dem)  - (m/2.) * np.log(beta) - (n/2.) * np.log(x) + (m/2.) * np.log(2.*np.pi)
    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun


# ------------------------------------------------------------------------------
# Modified equations by Mackay: Bayesian Interpolation
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
# Here we assume we know beta and we are using an arbitrary matrix L, instead of I
def compute_f_alpha_RNNLS_I_evidencelog_mod(Dic_i, M, L):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(Dic_i, M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (M - np.dot(Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(Dic_i.T, Dic_i)
    K           = np.matmul(L.T, L)
    #det_Linv    = np.linalg.det(np.linalg.inv(L))
    det_Linv    = np.linalg.det(np.linalg.inv(K))
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_mod, 1e-8, 10.0, args=(Dic_i, L, M_aug, M, m, n, B, det_Linv, beta, K), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol) * L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_mod(x, D, L, Signal_aug, Signal, m, n, B, det_Linv, beta, K):
    Daux       = np.concatenate((D, np.sqrt(x) * L))
    f, kk      = nnls( Daux, Signal_aug )
    ED         = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW         = 0.5 * np.sum ( np.dot(L, f)**2 )
    ratio_dem  = np.linalg.det( B + x*K )
    # ------------------------
    cost_fun   = beta * (ED + x*EW) + 0.5 * ( np.log(ratio_dem) + np.log(det_Linv) ) - (m/2.) * np.log(beta) - (n/2.) * np.log(x) + (m/2.) * np.log(2.*np.pi)

    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Here we assume we know beta and we are using an arbitrary matrix L, instead of I
# Moreover, we did some approximations to consider that f >= 0
# For more details see:
# Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
# Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
# IT IS FASTER THAN THE ORIGINAL VERSION BUT THE PERFORMANCE IS WORSE
def compute_f_alpha_RNNLS_L_evidencelog_nn_fast(Dic_i, M, L):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(Dic_i, M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (M - np.dot(Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(Dic_i.T, Dic_i)
    K           = np.matmul(L.T, L)
    det_L       = det(L)
    reg_0       = 1e-5
    UH          = cholesky(B + reg_0*K, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    diag_UH     = np.diag(UH)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_nn_fast, 1e-8, 2.0, args=(Dic_i, L, M_aug, M, m, n, B, det_L, beta, K, UH, diag_UH, reg_0), xtol=1e-05, maxfun=100, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol) * L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_nn_fast(x, D, L, Signal_aug, Signal, m, n, B, det_L, beta, K, UH, diag_UH, reg_0):
    Daux        = np.concatenate((D, np.sqrt(x) * L))
    f, kk       = nnls( Daux, Signal_aug )
    ED          = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(L, f)**2 )
    # -----------------------
    #A           = beta*B + (beta*x)*K
    # Cholesky decomposition: A=U.T*U
    #U           = cholesky(A, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    diag_U  = np.sqrt(diag_UH**2 + (x-reg_0)*np.diag(K)) - diag_UH
    U_mod   = UH + np.diag(diag_U)
    U       = np.sqrt(beta)*U_mod
    #det_U       = det(U)
    det_U       = np.prod(np.diag(U)) # faster evaluation of the determinant

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2.) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    cost_fun1   = beta*ED + beta*x*EW + np.log(det_U) - (n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m/2.) * np.log(2.*np.pi) - (m/2.) * np.log(beta) + (n/2.) * np.log(np.pi) - (n/2.) * np.log(2*beta*x)  - np.log(det_L)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# The same that in the previous function, but also considering the measured signal is non-negative
def compute_f_alpha_RNNLS_L_evidencelog_nn_nnLikelihood(Dic_i, M, L):
    m,n         = Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(Dic_i, M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (M - np.dot(Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(Dic_i.T, Dic_i)
    K           = np.matmul(L.T, L)
    det_L       = det(L)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_nnLikelihood, 1e-8, 2.0, args=(Dic_i, L, M_aug, M, m, n, B, det_L, beta, K), xtol=1e-05, maxfun=100, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol) * L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_nnLikelihood(x, D, L, Signal_aug, Signal, m, n, B, det_L, beta, K):
    Daux        = np.concatenate((D, np.sqrt(x) * L))
    f, kk       = nnls( Daux, Signal_aug )
    ED          = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(L, f)**2 )
    A           = beta*B + (beta*x)*K
    #det_A       = np.linalg.det(A)
    # -----------------------
    U           = cholesky(A, lower=False) # A=U.T*U
    det_U       = det(U)

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    error_term2  = 1.0 + erf( np.dot(D, f) * np.sqrt(beta/2.) )
    series_prod2 = np.sum( np.log( error_term2 ) )

    cost_fun1   = beta*ED + beta*x*EW + np.log(det_U) - (n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m/2.) * np.log(np.pi/2.) - (m/2.) * np.log(beta) + series_prod2 + (n/2.) * np.log(np.pi) - (n/2.) * np.log(2*beta*x)  - np.log(det_L)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
def compute_f_alpha_RNNLS_I_evidencelog_fullbayes(Dic_3D, M):
    m,n,p       = Dic_3D.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((M, Zerosm))
    In          = np.eye(n)
    # ----------------------
    SNR_min     = 10.0
    SNR_max     = 100.0
    SNR0        = 40.0
    beta_min    = SNR_min**2
    beta_max    = SNR_max**2
    beta0       = SNR0**2
    # ----------------------
    x0          = [beta0, 100.0] # initial estimate
    bnds        = ((beta_min, beta_max),(1e-5*beta_min, 2e3*beta_max)) # bounds
    # ----------------------
    x_sol       = np.zeros((n,p))
    beta_sol    = np.zeros(p)
    alpha_sol   = np.zeros(p)
    Mprop       = np.zeros(p)
    for iter in range(p):
        #print iter + 1
        Dic_i   = Dic_3D[:,:,iter]
        B       = np.matmul(Dic_i.T, Dic_i)
        # --------- Regularization parameters
        #res     = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'gtol': 1e-20, 'disp': False, 'maxiter': 500}, bounds = bnds, args=(Dic_i, In, M_aug, M, m, n, B))
        res     = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'disp': False}, bounds = bnds, args=(Dic_i, In, M_aug, M, m, n, B))
        reg_sol = res.x
        # --------- Estimation
        f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol[1]/reg_sol[0]) * In)), M_aug )
        x_sol[:,iter]   = f
        beta_sol[iter]  = reg_sol[0]
        alpha_sol[iter] = reg_sol[1]
        Mprop[iter]     = Prob_model(reg_sol, f, Dic_i, M, In, B, m, n)
    #end for
    Mprop     = Mprop/np.sum(Mprop) # normalization
    return Mprop, x_sol, beta_sol, alpha_sol
#end

def fullbayes_max(Mprop, x_sol, beta_sol, alpha_sol):
    # Model comparison: select the best model
    ind_max   = np.argmax(Mprop)
    f_opt     = x_sol[:,ind_max]
    beta_opt  = beta_sol[ind_max]
    alpha_opt = alpha_sol[ind_max]
    return f_opt, beta_opt, alpha_opt, ind_max
#end

def fullbayes_BMA(Mprop, x_sol, beta_sol, alpha_sol, FA_angles):
    # Bayesian model averaging
    n,p       = x_sol.shape
    f_opt     = 0.0
    beta_opt  = 0.0
    alpha_opt = 0.0
    FA_opt    = 0.0
    for iter in range(p):
        f_opt     = f_opt     + Mprop[iter] * x_sol[:,iter]
        beta_opt  = beta_opt  + Mprop[iter] * beta_sol[iter]
        alpha_opt = alpha_opt + Mprop[iter] * alpha_sol[iter]
        FA_opt    = FA_opt    + Mprop[iter] * FA_angles[iter]
    #end for
    return f_opt, beta_opt, alpha_opt, FA_opt
#end fun

def Prob_model(x, f, D, Signal, In, B, m, n):
    ED          = 0.5 * np.sum( (np.dot(D, f) - Signal)**2 )
    EW          = 0.5 * np.sum ( f**2 )
    (sign, logdet) = np.linalg.slogdet(x[1]*In + x[0]*B)
    log_prob    = x[0]*ED + x[1]*EW + 0.5*(sign*logdet) - (n/2.0)*np.log(x[1]) - (m/2.0)*np.log(x[0]) - (m/2.0)*np.log(2*np.pi)
    gamma_opt   = 2.0 * x[1] * EW
    prob_model  = np.exp(-1.0*log_prob) * np.sqrt(2./gamma_opt) * np.sqrt(2./(m - gamma_opt))
    return prob_model
# end fun
