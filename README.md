## Multi-component T2 relaxometry methods for myelin water quantification

**Implementation of the algorithms described here:**

> **Comparison of multi-component T2 relaxometry methods for myelin water quantification. Under review (2020).**
Erick Jorge Canales-Rodríguez, Marco Pizzolato, Gian Franco Piredda, Tom Hilbert, Nicolas Kunz, Caroline Pot, Thomas Yu, Raymond Salvador, Edith Pomarol-Clotet, Tobias Kober, Jean-Philippe Thiran, Alessandro Daducci.

**We are using the MRI acquisition sequence described here:**

> **Fast and high‐resolution myelin water imaging: Accelerating multi‐echo GRASE with CAIPIRINHA.**
Gian Franco Piredda, Tom Hilbert, Erick Jorge Canales‐Rodríguez, Marco Pizzolato, Constantin von Deuster, Reto Meuli, Josef Pfeuffer, Alessandro Daducci, Jean‐Philippe Thiran, Tobias Kober. Magnetic Resonance in Medicine, 2020, https://doi.org/10.1002/mrm.28427

The current implementation is written in Python 2.7 (we plan to switch to Python 3.x.)

Ten estimation algorithms are currently implemented, including the individual combinations of three penalty terms (i.e., **I** = identity matrix, **L1** = first-order Laplacian derivative, and **L2** = second-order Laplacian derivative) with three criteria to estimate the optimal regularization weight (i.e., Chi-square residual fitting (**X2**), **L-curve**, and Generalized Cross-Validation (**GCV**)), as well as the non-regularized Non-Negative Least Squares (NNLS). All algorithms were named based on the resulting combination: `X2-I, X2-L1, X2-L2, L-curve-I, L-curve-L1, L-curve-L2, GCV-I, GCV-L1, GCV-L2, and NNLS`.


## Install dependencies:

- numpy
- nibabel
- numba
- matplotlib
- scipy
- skimage
- joblib
- multiprocessing
- progressbar2


## Copyright and license

**GNU Lesser General Public License v2.1**

Primarily used for software libraries, the GNU LGPL requires that derived works be licensed under the same license, but works that only link to it do not fall under this restriction.

## Help



