import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import scipy

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
# end function

#_______________________________________________________________________________
params = {
'text.latex.preamble': r'\usepackage{gensymb}',
'image.origin': 'lower',
'image.interpolation': 'nearest',
'image.cmap': 'gray',
'axes.grid': False,
'savefig.dpi': 150,  # to adjust notebook inline plot size
'axes.labelsize': 10, # fontsize for x and y labels (was 10)
'axes.titlesize': 10,
'font.size': 10, # was 10
'legend.fontsize': 10, # was 10
'xtick.labelsize': 10,
'ytick.labelsize': 10,
'text.usetex': True,
'figure.figsize': [3.39, 2.10],
'font.family': 'serif',
}
matplotlib.rcParams.update(params)

fig1 = plt.figure('Showing all results', figsize=(11,11))

Slice=40

#_______________________________________________________________________________
# X2_I
path_to_save_data = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/recon_all_X2-I_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto/'
img  = nib.load(path_to_save_data + 'MWF.nii.gz')
fM = img.get_fdata()
fM = fM.astype(np.float64, copy=False)

plt.subplot(4, 4, 1).set_axis_off()
#im1 = plt.imshow(fM[:,:,Slice].T, cmap='gray', origin='upper', clim=(0,0.25))
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
plt.title('Myelin WF')
colorbar(im1)

img  = nib.load(path_to_save_data + 'IEWF.nii.gz')
fIE = img.get_fdata()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(4, 4, 2).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
plt.title('Intra+Extra WF')
colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_fdata()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(4, 4, 3).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
plt.title('T2-IE')
colorbar(im5)

img  = nib.load(path_to_save_data + 'TWC.nii.gz')
Ktotal = img.get_fdata()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(4, 4, 4).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
plt.title('Total Water Content')
colorbar(im6)

#_______________________________________________________________________________
# X2_L
path_to_save_data = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/recon_all_X2-L_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto/'

img  = nib.load(path_to_save_data + 'MWF.nii.gz')
fM = img.get_fdata()
fM = fM.astype(np.float64, copy=False)

plt.subplot(4, 4, 5).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
plt.title('Myelin VF')
colorbar(im1)

img  = nib.load(path_to_save_data + 'IEWF.nii.gz')
fIE = img.get_fdata()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(4, 4, 6).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
plt.title('Intra+Extra VF')
colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_fdata()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(4, 4, 7).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
plt.title('T2-tissues')
colorbar(im5)

img  = nib.load(path_to_save_data + 'TWC.nii.gz')
Ktotal = img.get_fdata()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(4, 4, 8).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
plt.title('Total Water Content')
colorbar(im6)

#_______________________________________________________________________________
# X2_Lcp
path_to_save_data = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/recon_all_X2-L-cp_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto/'
img  = nib.load(path_to_save_data + 'MWF.nii.gz')
fM = img.get_fdata()
fM = fM.astype(np.float64, copy=False)

plt.subplot(4, 4, 9).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
plt.title('Myelin VF')
colorbar(im1)

img  = nib.load(path_to_save_data + 'IEWF.nii.gz')
fIE = img.get_fdata()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(4, 4, 10).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
plt.title('Intra+Extra VF')
colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(4, 4, 11).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
plt.title('T2-tissues')
colorbar(im5)

img  = nib.load(path_to_save_data + 'TWC.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(4, 4, 12).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
plt.title('Ktotal: Proton Density')
colorbar(im6)

#_______________________________________________________________________________
# L-curve-Lcp
path_to_save_data = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/MWF_Controls_001/Scan_preproc/recon_all_L_curve-L-cp_GRASE_s2FA_thr40_T2s60_X2_rbias_tv_auto/'
img  = nib.load(path_to_save_data + 'MWF.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(4, 4, 13).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
plt.title('Myelin VF')
colorbar(im1)

img  = nib.load(path_to_save_data + 'IEWF.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(4, 4, 14).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
plt.title('Intra+Extra VF')
colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(4, 4, 15).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
plt.title('T2-tissues')
colorbar(im5)

img  = nib.load(path_to_save_data + 'TWC.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(4, 4, 16).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
plt.title('Total Water Content')
colorbar(im6)
#_______________________________________________________________________________
plt.tight_layout()
path_to_save_figure = '/media/Disco1T/multimodal/Siemens/Three_controls_scan_rescan/Final_Figures/'
plt.savefig(path_to_save_figure + 'plot_collage.png', bbox_inches='tight', dpi=600)
fig1.show()
