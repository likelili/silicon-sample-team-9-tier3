"""SYN-DIGITS calibration for the Silicon Sample runs.

Thread limits are set HERE, before any submodule imports numpy, because BLAS
reads them at load time and ignores later changes.

Why they matter: the calibration parallelises over anchor columns, and each
worker's linear algebra would otherwise spawn one BLAS thread per core.  With
N workers that is N x cores threads competing for cores, which on this machine
took a ~10 minute job to over two hours at 475% CPU.  One BLAS thread per
worker, parallelism from processes, is the standard joblib pattern and the only
one that scales here.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
