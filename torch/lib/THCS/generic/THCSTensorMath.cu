#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensorMath.cu"
#else

#define ROW_PTR2(t, r) (THCTensor_(data)(THCState *state, t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THCTensor_(data)(THCState *state, t) + (c) * (t)->stride[1])

THCudaLongTensor *THCSTensor_(toCSR)(THCState *state, long const *indices, long dim, long nnz) {
  THError("WARNING: Sparse Cuda Tensor op toCSR is not implemented");
  // TODO hook up with cusparse
  return NULL;
}

void THCSTensor_(spaddmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
  THError("WARNING: Sparse Cuda Tensor op spaddmm is not implemented");
  // TODO This is just a cusparse call (gemm?)
}

void THCSTensor_(sspaddmm)(THCState *state, THCSTensor *r_, real beta, THCSTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
  THError("WARNING: Sparse Cuda Tensor op sspaddmm is not implemented");
  // TODO Write some kernels
}

void THCSTensor_(spcadd)(THCState *state, THCTensor *r_, THCTensor *dense, real value, THCSTensor *sparse) {
  THError("WARNING: Sparse Cuda Tensor op spcadd is not implemented");
  // TODO pretty sure this is also just a cusparse call (axpyi)
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
