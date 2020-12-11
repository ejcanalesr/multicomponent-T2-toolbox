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
# Date: 2020
#===============================================================================

from __future__ import division
import os
import numpy as np
import math
import numba as nb

# ------------------------------------------------------------------------------
# Functions to generate the Dictionary of multi-echo T2 signals using the exponential model

def create_met2_design_matrix(TEs, T2s):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    '''
    M = len(TEs)
    N = len(T2s)
    design_matrix = np.zeros((M,N))
    for row in range(M):
        for col in range(N):
            exponent = -(TEs[row] / T2s[col])
            design_matrix[row,col] = np.exp(exponent)
        # end for col
    # end for row
    return design_matrix
#end fun

# Functions to generate the Dictionary of multi-echo T2 signals using the EPG model
def create_met2_design_matrix_epg(Npc, T2s, T1s, nEchoes, tau, flip_angle, TR):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    *** Here we use the epg model to simulate signal artifacts
    '''
    design_matrix = np.zeros((nEchoes, Npc))
    rad           = np.pi/180.0  # constant to convert degrees to radians
    for cols in range(Npc):
        signal = (1.0 - np.exp(-TR/T1s[cols])) * epg_signal(nEchoes, tau, np.array([1.0/T1s[cols]]), np.array([1.0/T2s[cols]]), flip_angle * rad, flip_angle/2.0 * rad)
        #signal = (1.0 - np.exp(-TR/T1s[cols])) * epg_signal(nEchoes, tau, np.array([1.0/T1s[cols]]), np.array([1.0/T2s[cols]]), flip_angle * rad, 90.0 * rad)
        design_matrix[:, cols] = signal.flatten()
        # end for row
    return design_matrix
#end fun

def epg_signal(n, tau, R1vec, R2vec, alpha, alpha_exc):
    nRates = R2vec.shape[0]
    tau = tau/2.0

    # defining signal matrix
    H = np.zeros((n, nRates))

    # RF mixing matrix
    T = fill_T(n, alpha)

    # Selection matrix to move all traverse states up one coherence level
    S = fill_S(n)

    for iRate in range(nRates):
        # Relaxation matrix
        R2 = R2vec[iRate]
        R1 = R1vec[iRate]

        R0      = np.zeros((3,3))
        R0[0,0] = np.exp(-tau*R2)
        R0[1,1] = np.exp(-tau*R2)
        R0[2,2] = np.exp(-tau*R1)

        R = fill_R(n, tau, R0, R2)
        # Precession and relaxation matrix
        P = np.dot(R,S)
        # Matrix representing the inter-echo duration
        E = np.dot(np.dot(P,T),P)
        H = fill_H(R, n, E, H, iRate, alpha_exc)
        # end
    return H
#end fun

def fill_S(n):
    the_size = 3*n + 1
    S = np.zeros((the_size,the_size))
    S[0,2]=1.0
    S[1,0]=1.0
    S[2,5]=1.0
    S[3,3]=1.0
    for o in range(2,n+1):
        offset1=( (o-1) - 1)*3 + 2
        offset2=( (o+1) - 1)*3 + 3
        if offset1<=(3*n+1):
            S[3*o-2,offset1-1] = 1.0  # F_k <- F_{k-1}
        # end
        if offset2<=(3*n+1):
            S[3*o-1,offset2-1] = 1.0  # F_-k <- F_{-k-1}
        # end
        S[3*o,3*o] = 1.0              # Z_order
    # end for
    return S
#end fun

def fill_T(n, alpha):
    T0      = np.zeros((3,3))
    T0[0,:] = [math.cos(alpha/2.0)**2, math.sin(alpha/2.0)**2,  math.sin(alpha)]
    T0[1,:] = [math.sin(alpha/2.0)**2, math.cos(alpha/2.0)**2, -math.sin(alpha)]
    T0[2,:] = [-0.5*math.sin(alpha),   0.5*math.sin(alpha),     math.cos(alpha)]

    T = np.zeros((3*n + 1, 3*n + 1))
    T[0,0] = 1.0
    T[1:3+1, 1:3+1] = T0
    for itn in range(n-1):
        T[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = T0
    # end
    return T
#end fun

def fill_R(n, tau, R0, R2):
    R  = np.zeros((3*n + 1, 3*n + 1))
    R[0,0] = np.exp(-tau*R2)
    R[1:3+1, 1:3+1] = R0
    for itn in range(n-1):
        R[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = R0
    # end
    return R
#end fun

def fill_H(R, n, E, H, iRate, alpha_exc):
    x    = np.zeros((R.shape[0],1))
    x[0] = math.sin(alpha_exc)
    x[1] = 0.0
    x[2] = math.cos(alpha_exc)
    for iEcho in range(n):
        x = np.dot(E,x)
        H[iEcho, iRate] = x[0]
    #end for IEcho
    return H
#end fun

def create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values, TR):
    dim3   = len(alpha_values)
    Dic_3D = np.zeros((nEchoes, Npc, dim3))
    for iter in range(dim3):
        Dic_3D[:,:,iter] = create_met2_design_matrix_epg(Npc, T2s, T1s, nEchoes, tau, alpha_values[iter], TR)
    #end for
    return Dic_3D
#end fun
