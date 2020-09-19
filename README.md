## Multi-component T2 relaxometry methods for myelin water quantification

**Implementation of the algorithms described here:**

> **Comparison of multi-component T2 relaxometry methods for myelin water quantification. Under review (2020).**
Erick Jorge Canales-Rodríguez, Marco Pizzolato, Gian Franco Piredda, Tom Hilbert, Nicolas Kunz, Caroline Pot, Thomas Yu, Raymond Salvador, Edith Pomarol-Clotet, Tobias Kober, Jean-Philippe Thiran, Alessandro Daducci.

Ten non-parametric estimation algorithms are currently implemented, including the individual combinations of three penalty terms (i.e., **I** = identity matrix, **L1** = first-order Laplacian derivative, and **L2** = second-order Laplacian derivative) with three criteria to estimate the optimal regularization weight (i.e., Chi-square residual fitting (**X2**), **L-curve**, and Generalized Cross-Validation (**GCV**)), as well as the non-regularized Non-Negative Least Squares (NNLS). All algorithms were named based on the resulting combination: ``X2-I, X2-L1, X2-L2, L-curve-I, L-curve-L1, L-curve-L2, GCV-I, GCV-L1, GCV-L2, and NNLS``.

The current implementation is written in Python 2.7 (we plan to switch to Python 3.x.)

**We are using the MRI acquisition sequence described here:**

> **Fast and high‐resolution myelin water imaging: Accelerating multi‐echo GRASE with CAIPIRINHA.**
Gian Franco Piredda, Tom Hilbert, Erick Jorge Canales‐Rodríguez, Marco Pizzolato, Constantin von Deuster, Reto Meuli, Josef Pfeuffer, Alessandro Daducci, Jean‐Philippe Thiran, Tobias Kober. **Magnetic Resonance in Medicine**, 2020, https://doi.org/10.1002/mrm.28427

## Install dependencies
```
- numpy
- nibabel
- numba
- matplotlib
- scipy
- skimage
- joblib
- multiprocessing
- progressbar2
```

## Help

Open a terminal and write:

```
$ python run_real_data_script.py -h

usage: run_real_data_script.py [-h] --minTE MINTE --nTE NTE --TR TR
                               --FA_method {spline,brute-force} --FA_smooth
                               {yes,no} --denoise {TV,NESMA,None} --reg_method
                               {NNLS,X2-I,X2-L1,X2-L2,L_curve-I,L_curve-L1,L_curve-L2,GCV-I,GCV-L1,GCV-L2}
                               --path_to_folder PATH_TO_FOLDER --input INPUT
                               --mask MASK --savefig {yes,no} --savefig_slice
                               SAVEFIG_SLICE [--numcores NUMCORES] --myelin_T2
                               MYELIN_T2

Myelin Water Imaging

optional arguments:
  -h, --help            show this help message and exit
  --minTE MINTE         Minimum Echo Time (TE, units: ms)
  --nTE NTE             Number of TEs
  --TR TR               Repetition Time (units: ms)
  --FA_method {spline,brute-force}
                        Method to estimate the flip angle (FA)
  --FA_smooth {yes,no}  Smooth FA map for a robust estimation
  --denoise {TV,NESMA,None}
                        Denoise data
  --reg_method {NNLS,X2-I,X2-L1,X2-L2,L_curve-I,L_curve-L1,L_curve-L2,GCV-I,GCV-L1,GCV-L2}
                        Reconstruction algorithm
  --path_to_folder PATH_TO_FOLDER
                        Path to the folder where the data is located, e.g.,
                        /home/Datasets/MET2/
  --input INPUT         Input data, e.g., Data.nii.gz
  --mask MASK           Brain mask, e.g., Mask.nii.gz
  --savefig {yes,no}    Save reconstructed maps in .png
  --savefig_slice SAVEFIG_SLICE
                        Axial slice to save reconstructed maps, e.g.,
                        --Slice=30
  --numcores NUMCORES   Number of cores used in the computation: -1 = all
                        cores
  --myelin_T2 MYELIN_T2
                        Maximum T2 for the myelin compartment: T2 threshold
                        (units: ms)
```

For more details see the script: **example_script_run_MET2_preproc_and_recon.sh**. 
We included some optional pre- and post-processing steps using FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) and MRtrix3 (https://www.mrtrix.org/):

```
- Brain extraction for obtaining the brain mask (i.e., bet, FSL)
- Denoising (i.e., dwidenoise, MRtrix3)
- Remove Gibbs Ringing Artifacts (i.e., mrdegibbs, MRtrix3)
- Bias-field correction of the estimated Proton Density map, and tissue segmentation to obtain WM, GM, and CSF probabilistic maps (i.e., fast, FSL)
- Create images for visualizing and inspecting the estimated maps (i.e., slices, FSL)
```

## Estimated maps

The 

## Copyright and license

**GNU Lesser General Public License v2.1**

Primarily used for software libraries, the GNU LGPL requires that derived works be licensed under the same license, but works that only link to it do not fall under this restriction.
