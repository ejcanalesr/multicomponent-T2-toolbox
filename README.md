# Multi-component T2 relaxometry methods for myelin water quantification

**Implementation of the algorithms described here:**

> **Comparison of multi-component T2 relaxometry methods for myelin water quantification. Under review (2020).**
Erick Jorge Canales-Rodríguez, Marco Pizzolato, Gian Franco Piredda, Tom Hilbert, Nicolas Kunz, Caroline Pot, Thomas Yu, Raymond Salvador, Edith Pomarol-Clotet, Tobias Kober, Jean-Philippe Thiran, Alessandro Daducci.

The current implementation is written in Python 2.7 (and we plan to switch to Python 3.x.)

Ten estimation algorithms are currently implemented, including the individual combinations of the three penalty terms (i.e., **I** = identity matrix, **L1** = first-order Laplacian derivative, and **L2** = second-order Laplacian derivative) and three criteria to estimate the optimal regularization weight (i.e., Chi-square residual fitting (**X2**), **L-curve**, and Generalized Cross-Validation (**GCV**)), as well as the non-regularized Non-Negative Least Squares (NNLS). All algorithms were named based on the resulting combination: `X2-I, X2-L1, X2-L2, L-curve-I, L-curve-L1, L-curve-L2, GCV-I, GCV-L1, GCV-L2, and NNLS`.

**A tutorial and example data will be provided after article acceptance**

**Install dependencies:**
- nibabel
- numba
- matplotlib
