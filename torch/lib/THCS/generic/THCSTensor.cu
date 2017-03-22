#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cu"
#else

THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self) {
  /*
  THLongStorage *storage;
  THCTensor *other;

  THCSTensor_(contiguous)(state, self);

  // set up the new tensor
  storage = THCSTensor_(newSizeOf)(state, self);
  other = THCTensor_(newWithSize)(state, storage, NULL);
  THCTensor_(zero)(state, other);

  // Some necessary dimensions and sizes
  const ptrdiff_t nnz = THCSTensor_(nnz)(state, self);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

  TensorInfo<real, unsigned long> otherInfo =
    getTensorInfo<THCTensor, unsigned long>(state, other);
  TensorInfo<long, unsigned long> indicesInfo =
    getTensorInfo<THCudaLongTensor, unsigned long>(state, self->indices);
  TensorInfo<real, unsigned long> valuesInfo =
    getTensorInfo<THCTensor, unsigned long>(state, self->values);

  THCSTensor_toDenseKernel<unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        otherInfo, indicesInfo, valuesInfo, (unsigned long)(nnz));

  THCudaCheck(cudaGetLastError());
  THLongStorage_free(storage);
  return other;
  */
  THError("WARNING: Sparse Cuda Tensor op toDense is not implemented");
  return NULL;
}

void THCSTensor_(reorder)(THCState *state, THCSTensor *self) {
  THError("WARNING: Sparse Cuda Tensor op reorder is not implemented");
}

void THCSTensor_(contiguous)(THCState *state, THCSTensor *self) {
  if (self->contiguous) return;
  THCSTensor_(reorder)(state, self);
  self->contiguous = 1;
}

// In place transpose
void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int d1, int d2) {
  /* TODO
  THCudaLongTensor *indices = THCSTensor_(indices)(state, self);
  long i;
  for (i = 0; i < THCSTensor_(nnz)(state, self); i++) {
    long tmp = THCTensor_fastGet2d(indices, d1, i);
    THCTensor_fastSet2d(indices, d1, i,
        THCTensor_fastGet2d(indices, d2, i));
    THCTensor_fastSet2d(indices, d2, i, tmp);
  }
  i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
  self->contiguous = 0;
  THFree(indices);
  */
  THError("WARNING: Sparse Cuda Tensor op transpose is not implemented");
}

int THCSTensor_(getDevice)(THCState* state, const THCSTensor* tensor) {
  if (!tensor->values || !tensor->values->storage) return -1;
  return THCStorage_(getDevice)(state, tensor->values->storage);
}

void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask) {
  THError("WARNING: Sparse Cuda Tensor op sparseMask is not implemented");
}

#endif
