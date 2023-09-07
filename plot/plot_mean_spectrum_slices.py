#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import math
import time
import os
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

def plot_mean_spectrum_slices(path_to_save_data, path_to_mask, Slice, reg_method):
    print('Plotting T2 spectra')
    path_to_data         = path_to_save_data + 'fsol_4D.nii.gz'
    path_to_save_fig_png = path_to_save_data + reg_method + '_spectrums_new.png'
    path_to_save_fig_pdf = path_to_save_data + reg_method + '_spectrums_new.pdf'

    loc  = 1

    # ------------------------------------------------------------------------------
    img      = nib.load(path_to_data)
    data     = img.get_fdata()
    data     = data.astype(np.float64, copy=False)

    img_mask = nib.load(path_to_mask)
    mask     = img_mask.get_fdata()
    mask     = mask.astype(np.float64, copy=False)

    mask2 = np.zeros_like(mask)
    mask2[:,:,Slice] = mask[:,:,Slice]
    mask = mask2

    ind_mask = mask == 1
    print ('Plotting:', np.sum(ind_mask), 'T2 distributions')
    nx, ny, nz, nt = data.shape
    fsol_2D        = np.zeros((np.sum(ind_mask), nt))
    T2s            = np.logspace(math.log10(10), math.log10(2000), num=nt, endpoint=True, base=10.0)

    for nti in range(nt):
        data_i  = data[:,:,:,nti]
        fsol_2D[:, nti] = data_i[ind_mask]
    #end

    mean_Spectrum = np.mean(fsol_2D, axis=0)
    std_Spectrum  = np.std(fsol_2D, axis=0)

    Total = np.sum(mean_Spectrum)
    mean_Spectrum = mean_Spectrum/Total
    std_Spectrum  = std_Spectrum/Total

    fsol_2D = fsol_2D/Total

    #ymax = 0.3
    #ymax = np.max(mean_Spectrum[T2s<=40])
    #ymax = np.max(fsol_2D[:,T2s<=40])
    # ------------------------------------------------------------------------------

    fig  = plt.figure('Showing results', figsize=(8,8))
    axs  = fig.add_subplot(1, 1, 1)

    plt.plot(T2s, fsol_2D.T[:,:], alpha=0.2)
    #plt.plot(T2s, 1.5*mean_Spectrum, color='k')

    plt.title('Spectra', fontsize=18)
    plt.xlabel('T2', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)
    axs.set_xscale('log')

    axs.set_xlim(T2s[0], T2s[-1])
    ymax_total = np.max(fsol_2D)*1.05
    ymax = ymax_total/3.
    axs.set_ylim(0, ymax_total)

    zoom = 1.5
    #zoom1 = ymax_total/(1.5*ymax)
    #zoom2 = np.log(T2s[-1])/(1.5*np.log(50.0))
    #zoom  = np.min([zoom1, zoom2])

    #if zoom2 > zoom1:
    #    zoom  = (1.5)*zoom
    #end if

    plt.axvline(x=40.0, ymin=0, color='k', linestyle='--')

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(axs, zoom, loc=loc) # zoom-factor: 2.5, location: upper-left
    axins.plot(T2s, fsol_2D.T[:,:], alpha=0.5)
    #axins.plot(T2s, 2.*mean_Spectrum, color='k')

    x1, x2, y1, y2 = 10, 50, 0, ymax # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    #plt.axvline(x=40.0, color='k', linestyle='--', ymin=0, ymax=ymax)
    plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)

    plt.yticks(visible=False)
    plt.yticks([])

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axs.set_yticks([])

    plt.savefig(path_to_save_fig_png, dpi=500)
    plt.savefig(path_to_save_fig_pdf, dpi=500)
    #plt.show()
    plt.close('all')
#end function
