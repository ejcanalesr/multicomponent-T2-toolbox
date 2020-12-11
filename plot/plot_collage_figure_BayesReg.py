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
'text.latex.preamble': ['\\usepackage{gensymb}'],
'image.origin': 'lower',
'image.interpolation': 'nearest',
'image.cmap': 'gray',
'axes.grid': False,
'savefig.dpi': 150,  # to adjust notebook inline plot size
'axes.labelsize': 10, # fontsize for x and y labels (was 10)
'axes.titlesize': 12,
'font.size': 10, # was 10
'legend.fontsize': 10, # was 10
'xtick.labelsize': 10,
'ytick.labelsize': 10,
'text.usetex': True,
'figure.figsize': [3.39, 2.10],
'font.family': 'serif',
}
matplotlib.rcParams.update(params)

fig1 = plt.figure('Showing all results', figsize=(10,7))

Slice   =  42
subject = '001_scan'

#_______________________________________________________________________________
#1) NNLS
method='NNLS'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'
img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 1).set_axis_off()
#im1 = plt.imshow(fM[:,:,Slice].T, cmap='gray', origin='upper', clim=(0,0.25))
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
plt.title('MWF')
#colorbar(im1)

img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 2).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
plt.title('IEWF')
#colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 3).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
plt.title('T2')
#colorbar(im5)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 4).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
plt.title('TWC')
#colorbar(im6)

#_______________________________________________________________________________
#2) X2-I
method='X2-I'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 5).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)

img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 6).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 7).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 8).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#plt.title('Ktotal: Proton Density')
#colorbar(im6)

#_____________-
#3) L_curve-I
method='L_curve-I'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 9).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)

img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 10).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 11).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 12).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#plt.title('Ktotal: Proton Density')
#colorbar(im6)
#_______________________________________________________________________________

#4) BayesReg-I
method='BayesReg-I'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 13).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)

img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 14).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)

img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 15).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 16).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#plt.title('Ktotal: Proton Density')
#colorbar(im6)
#_______________________________________________________________________________

#5) X2-Iw
method='X2-Iw'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 17).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)
#fig1.colorbar(im1, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 18).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)
#fig1.colorbar(im2, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 19).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)
#fig1.colorbar(im5, orientation="horizontal", fraction=0.046, pad=0.04)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 20).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#fig1.colorbar(im6, orientation="horizontal", fraction=0.046, pad=0.04)
#plt.title('Ktotal: Proton Density')
#colorbar(im6)
#_______________________________________________________________________________

#6) L_curve-Iw
method='L_curve-Iw'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 21).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)
#fig1.colorbar(im1, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 22).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)
#fig1.colorbar(im2, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 23).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)
#fig1.colorbar(im5, orientation="horizontal", fraction=0.046, pad=0.04)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 24).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#fig1.colorbar(im6, orientation="horizontal", fraction=0.046, pad=0.04)
#plt.title('Ktotal: Proton Density')
#colorbar(im6)

#_______________________________________________________________________________

#7) BayesReg-Iw
method='BayesReg-Iw'
path_to_save_data = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/' + subject + '/recon_all_' + method + '/'

img  = nib.load(path_to_save_data + 'fM.nii.gz')
fM = img.get_data()
fM = fM.astype(np.float64, copy=False)

plt.subplot(7, 4, 25).set_axis_off()
im1 = plt.imshow(fM[:,:,Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
#plt.title('MWF')
#colorbar(im1)
#fig1.colorbar(im1, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'fIE.nii.gz')
fIE = img.get_data()
fIE = fIE.astype(np.float64, copy=False)

plt.subplot(7, 4, 26).set_axis_off()
im2 = plt.imshow(fIE[:,:,Slice].T, cmap='magma', origin='upper', clim=(0,1))
#plt.title('IEWF')
#colorbar(im2)
#fig1.colorbar(im2, orientation="horizontal", fraction=0.046, pad=0.04)


img  = nib.load(path_to_save_data + 'T2_IE.nii.gz')
T2IE = img.get_data()
T2IE = T2IE.astype(np.float64, copy=False)

plt.subplot(7, 4, 27).set_axis_off()
im5 = plt.imshow(T2IE[:,:,Slice].T, cmap='gnuplot2', origin='upper', clim=(50,100))
#plt.title('T2')
#colorbar(im5)
#fig1.colorbar(im5, orientation="horizontal", fraction=0.046, pad=0.04)

img  = nib.load(path_to_save_data + 'Ktotal.nii.gz')
Ktotal = img.get_data()
Ktotal = Ktotal.astype(np.float64, copy=False)

plt.subplot(7, 4, 28).set_axis_off()
im6 = plt.imshow(Ktotal[:,:,Slice].T, cmap='gray', origin='upper')
#fig1.colorbar(im6, orientation="horizontal", fraction=0.046, pad=0.04)
#plt.title('Ktotal: Proton Density')
#colorbar(im6)

#__________________________________________________________________
#plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
path_to_save_figure = '/media/Disco1T/multimodal/MET2_relaxometry_codes/Recon_folder/scan_rescan/Figuras_finales_paper_BayesReg/'
plt.savefig(path_to_save_figure + 'Figure_collage.png', bbox_inches='tight', dpi=600)
plt.close()

fig2 = plt.figure('Showing all results', figsize=(7,10))
plt.subplot(1, 4, 1).set_axis_off()
fig2.colorbar(im1, orientation="horizontal", fraction=0.046, pad=0.2, extend='both', ticks=[0, 0.125, 0.25])
plt.subplot(1, 4, 2).set_axis_off()
fig2.colorbar(im2, orientation="horizontal", fraction=0.046, pad=0.2, extend='both', ticks=[0, 0.5, 1])
plt.subplot(1, 4, 3).set_axis_off()
fig2.colorbar(im5, orientation="horizontal", fraction=0.046, pad=0.2, extend='both')
plt.subplot(1, 4, 4).set_axis_off()
Image = Ktotal[:,:,Slice]
max_I = np.max(Image)
min_I = np.min(Image)
fig2.colorbar(im6, orientation="horizontal", fraction=0.046, pad=0.2, extend='both', ticks=[min_I, 0.5*(min_I+max_I), max_I])
plt.subplots_adjust(wspace=0.05, hspace=0.01)
plt.savefig(path_to_save_figure + 'Figure_colorbars.png', bbox_inches='tight', dpi=600)
