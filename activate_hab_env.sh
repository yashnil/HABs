# 1) Tell OpenMP it’s OK to load twice (quick workaround)
export KMP_DUPLICATE_LIB_OK=TRUE

# 2) Limit NumPy / OpenBLAS threads → 1  (prevents runaway CPU kernels)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 3) Optionally point PyTorch to the Apple-Silicon backend
export PYTORCH_ENABLE_MPS_FALLBACK=1     # keeps training on M-series GPU