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
matplotlib.rcParams['text.latex.unicode']=True

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_0'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_5'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_20'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_50'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_200'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_367'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_400'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_500'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_602'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_674'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_1000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_1500'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_2000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_3000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_TV_604'
#eg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_3100'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_maxiter_384'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_maxiter_3000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_maxiter_157_masked_by_hand_after_1e-3'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_maxiter_1000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_v2_maxiter_1000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_ns_maxiter_212_grad'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_denoised_ns_maxiter_3000real_grad'
#reg_method         = 'ARUMBA_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto_splineFA_0.1v4'

#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_1000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_1500'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_2000'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_2500'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_3000'

#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_TV'
#reg_method         = 'GRASE_s2FA_thr40_T2s60_rbias_RUMBA_TV_3000'

#reg_method        = 'L_curve-I_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'
#reg_method        = 'L_curve-L_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'
#reg_method        = 'L_curve-L-cp_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'

#reg_method        = 'X2-I_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'
#reg_method        = 'X2-L_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'
#reg_method        = 'X2-L-cp_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto'

#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR_30_TE100'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR_60'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR_130'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR120'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR150'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_SNR100'
#reg_method         = 'X2-prior_mcGRASE_thr40_N60T2_tvs2_FAsplines2_old'
#reg_method         = 'L_curve-L-cp_mcGRASE_thr40_N60T2_tvs2_FAsplines2'
#reg_method         = 'ARUMBA_mcGRASE_thr40_N60T2_tvs2_FAbfs2'
#reg_method         = 'ARUMBA_mcGRASE_thr40_N60T2_tvs2_FAsplines2'
reg_method         = 'ARUMBA_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto_Spline_FA_s22'


#path_to_folder   = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/'

#path_to_folder   = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/'
#path_to_folder   = '/media/Disco1T/multimodal/Siemens/Control_with_T1/MWF_CONTROLS_001/Scan_data_preproc/'
path_to_folder   = '/media/Disco1T/multimodal/Siemens/mcgrass_hummels_gab/'

path_to_data     = path_to_folder + 'recon_all_' + reg_method + '/fsol_4D.nii.gz'
#path_to_mask     = path_to_folder + 'White_matter_mask_single_slice.nii.gz'
#path_to_mask     = path_to_folder + 'mask.nii.gz'
path_to_mask     = path_to_folder + 'brain_mask.nii.gz'

path_to_save_fig_png = path_to_folder + 'recon_all_' + reg_method + '/Spectrums_new.png'
path_to_save_fig_pdf = path_to_folder + 'recon_all_' + reg_method + '/Spectrums_new.pdf'


zoom = 2.0
loc  = 1
#ymax = 2500
#ymax = 8000
ymax = 5000

#ymax = 1800
#ymax = 0.15#/2.0


# ------------------------------------------------------------------------------
img      = nib.load(path_to_data)
data     = img.get_fdata()
data     = data.astype(np.float64, copy=False)

img_mask = nib.load(path_to_mask)
mask     = img_mask.get_fdata()
mask     = mask.astype(np.float64, copy=False)

# for testing: selects a single slice
#mask[:,:,41:-1] = 0
#mask[:,:,0:39]  = 0

mask2 = np.zeros_like(mask)
mask2[:,:,40] = mask[:,:,40]
mask = mask2

#Npc      = 100
Npc      = 60
T2m0     = 10.0
T2csf    = 2000.0
T2s      = np.logspace(math.log10(T2m0), math.log10(T2csf), num=Npc, endpoint=True, base=10.0)

ind_mask = mask > 0

nx, ny, nz, nt = data.shape

fsol_2D = np.zeros((np.sum(ind_mask), nt))

for nti in range(nt):
    data_i  = data[:,:,:,nti]
    fsol_2D[:, nti] = data_i[ind_mask]
#end

mean_Spectrum = np.mean(fsol_2D, axis=0)
std_Spectrum  = np.std(fsol_2D, axis=0)

#Total = np.sum(mean_Spectrum)
#mean_Spectrum = mean_Spectrum/Total
# ------------------------------------------------------------------------------

fig  = plt.figure('Showing results', figsize=(16,8.0))
ax0  = fig.add_subplot(1, 2, 1)
im0  = plt.plot(T2s, mean_Spectrum, color='b')
ax0.set_xscale('log')
plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)
plt.title('Mean spectrum', fontsize=18)
plt.xlabel('T2', fontsize=18)
plt.ylabel('Spectrum', fontsize=18)

ax0.set_xlim(T2s[0], T2s[-1])
ax0.set_ylim(0, np.max(mean_Spectrum)*1.2)

ax0.tick_params(axis='both', which='major', labelsize=16)
ax0.tick_params(axis='both', which='minor', labelsize=14)
ax0.set_yticks([])

ax1  = fig.add_subplot(1, 2, 2)
im1  = plt.plot(T2s, fsol_2D.T[:,:], alpha=0.2)

ax1.set_xscale('log')
plt.title('All spectrums', fontsize=18)
plt.xlabel('T2', fontsize=18)
plt.ylabel('Spectrum', fontsize=18)

plt.axvline(x=40.0, ymin=0, ymax=ymax, color='k', linestyle='--')


ax1.set_xlim(T2s[0], T2s[-1])
ax1.set_ylim(0, np.max(fsol_2D)*1.2)

ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(axis='both', which='minor', labelsize=14)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(ax1, zoom, loc=loc) # zoom-factor: 2.5, location: upper-left
axins.plot(T2s, fsol_2D.T[:,:], alpha=0.5)
plt.axvline(x=40.0, color='k', linestyle='--', ymin=0, ymax=ymax)
x1, x2, y1, y2 = 10, 50, 0, ymax # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.yticks(visible=False)
plt.yticks([])

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax1.set_yticks([])

plt.savefig(path_to_save_fig_png, dpi=500)
plt.savefig(path_to_save_fig_pdf, dpi=500)

plt.show()
plt.close('all')
