#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensorMath.cu"
#else

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define ROW_PTR2(t, r) (THCTensor_(data)(THCState *state, t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THCTensor_(data)(THCState *state, t) + (c) * (t)->stride[1])

#define I_INFO(tensor) getTensorInfo<THCIndexTensor, unsigned long>(state, tensor)
#define V_INFO(tensor) getTensorInfo<THCTensor, unsigned long>(state, tensor)

THCudaIntTensor *THCSTensor_(toCSR)(THCState *state, THCIndexTensor *rowIndices, long dim, long nnz) {
  THCudaIntTensor *csr = THCudaIntTensor_newWithSize1d(state, dim + 1);
  THCudaIntTensor *rowIndicesInt = THCudaIntTensor_newWithSize1d(state, rowIndices->size[0]);
  THCudaIntTensor_copyCudaLong(state, rowIndicesInt, rowIndices);
  THCudaSparse_Xcoo2csr(
    state, THCudaIntTensor_data(state, rowIndicesInt), nnz, dim, THCudaIntTensor_data(state, csr));
  THCudaIntTensor_free(state, rowIndicesInt);
  return csr;
}

void THCSTensor_(zero)(THCState *state, THCSTensor *self) {
  if (self->indices->nDimension) {
    THCIndexTensor_(resizeNd)(state, self->indices, 0, NULL, NULL);
  }
  if (self->values->nDimension) {
    THCTensor_(resizeNd)(state, self->values, 0, NULL, NULL);
  }
  self->nnz = 0;
}

void THCTensor_(spaddcmul)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("WARNING: Sparse Cuda Tensor op spaddcmul is not implemented");
}

void THCTensor_(spaddcdiv)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("WARNING: Sparse Cuda Tensor op spaddcdiv is not implemented");
}

void THCSTensor_(spaddmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
#if defined(THCS_REAL_IS_FLOAT) || defined(THCS_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 1, 4, sparse, r_, t, dense));
  THCudaIntTensor *csr;
  THCIndexTensor *indices;
  THCTensor *values, *r__, *dense_;

  THArgCheck(sparse->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse->nDimensionI);
  THArgCheck(sparse->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse->nDimensionV);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  long m = THCSTensor_(size)(state, sparse, 0);
  long k = THCSTensor_(size)(state, sparse, 1);
  long n = THCTensor_(size)(state, dense, 1);

  THArgCheck(THCTensor_(size)(state, t, 0) == m, 1,
      "Expected dim 0 size %d, got %d", m, THCTensor_(size)(state, t, 0));
  THArgCheck(THCTensor_(size)(state, t, 1) == n, 1,
      "Expected dim 1 size %d, got %d", n, THCTensor_(size)(state, t, 1));
  THArgCheck(THCTensor_(size)(state, dense, 0) == k, 3,
      "Expected dim 0 size %d, got %d", k, THCTensor_(size)(state, dense, 0));

  THCSTensor_(contiguous)(state, sparse);

  long nnz = THCSTensor_(nnz)(state, sparse);
  indices = THCSTensor_(indices)(state, sparse);
  values = THCSTensor_(values)(state, sparse);

  THCIndexTensor *rowIndices = THCIndexTensor_(new)(state);
  THCIndexTensor *colIndices = THCIndexTensor_(new)(state);
  THCIndexTensor_(select)(state, rowIndices, indices, 0, 0);
  THCIndexTensor_(select)(state, colIndices, indices, 0, 1);
  csr = THCSTensor_(toCSR)(state, rowIndices, m, nnz);
  THCudaIntTensor *colIndicesInt = THCudaIntTensor_newWithSize1d(state, colIndices->size[0]);
  THCudaIntTensor_copyCudaLong(state, colIndicesInt, colIndices);

  char transpose_dense;

  if(t != r_)
  {
    THCTensor_(resizeAs)(state, r_, t);
    THCTensor_(copy)(state, r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1 && r_->stride[1] != 0) {
    r__ = r_;
  } else {
    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, r_, 0, 1);
    r__ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, r__, NULL, 0, 1);
  }

  /* dense */
  if(dense->stride[0] == 1 && dense->stride[1] != 0) {
    transpose_dense = 'n';
    dense_ = dense;
  } else if(dense->stride[1] == 1 && dense->stride[0] != 0) {
    transpose_dense = 't';
    dense_ = dense;
  } else {
    transpose_dense = 't';
    dense_ = THCTensor_(newContiguous)(state, dense);
  }
#if defined(THCS_REAL_IS_FLOAT)
  THCudaSparse_Scsrmm2(
#elif defined(THCS_REAL_IS_DOUBLE)
  THCudaSparse_Dcsrmm2(
#endif
    state,
    'n',
    transpose_dense,
    m,
    n,
    k,
    nnz,
    alpha,
    THCTensor_(data)(state, values),
    THCudaIntTensor_data(state, csr),
    THCudaIntTensor_data(state, colIndicesInt),
    THCTensor_(data)(state, dense_),
    (transpose_dense == 'n' ? dense_->stride[1] : dense_->stride[0]),
    beta,
    THCTensor_(data)(state, r__),
    r__->stride[1]);

  /* free intermediate variables */
  if(dense_ != dense) {
    THCTensor_(free)(state, dense_);
  }

  if(r__ != r_) {
    THCTensor_(freeCopyTo)(state, r__, r_);
  }

  THCudaIntTensor_free(state, colIndicesInt);
  THCudaIntTensor_free(state, csr);
  THCIndexTensor_(free)(state, indices);
  THCIndexTensor_(free)(state, rowIndices);
  THCIndexTensor_(free)(state, colIndices);
  THCTensor_(free)(state, values);
#else
  THError("unimplemented data type");
#endif
}

void THCSTensor_(sspaddmm)(THCState *state, THCSTensor *r_, real beta, THCSTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
  THError("WARNING: Sparse Cuda Tensor op sspaddmm is not implemented");
  // TODO Write some kernels
}

void THCSTensor_(hspmm)(THCState *state, THCSTensor *r_, real alpha, THCSTensor *sparse, THCTensor *dense) {
#if CUDA_VERSION >= 7000
  THCThrustAllocator thrustAlloc(state);
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 2, 3, r_, sparse, dense));

  THArgCheck(sparse->nDimensionI == 2, 3,
      "matrices expected, got %dD tensor", sparse->nDimensionI);
  THArgCheck(sparse->nDimensionV == 0, 3,
      "scalar values expected, got %dD values", sparse->nDimensionV);
  THArgCheck(dense->nDimension == 2, 4,
      "matrices expected, got %dD tensor", dense->nDimension);

  long m = THCSTensor_(size)(state, sparse, 0);
  long k = THCSTensor_(size)(state, sparse, 1);
  long n = THCTensor_(size)(state, dense, 1);

  THArgCheck(THCTensor_(size)(state, dense, 0) == k, 4,
      "Expected dim 0 size %d, got %d", k, THCTensor_(size)(state, dense, 0));
  long size[2] = {m, n};
  THCSTensor_(rawResize)(state, r_, 1, 1, size);

  THCSTensor_(contiguous)(state, sparse);

  long nnz = THCSTensor_(nnz)(state, sparse);
  THCIndexTensor *indices = THCIndexTensor_(newWithSize2d)(state, 1, nnz);
  // create values in column-major format to avoid copying in spaddmm
  THCTensor *values = THCTensor_(newWithSize2d)(state, n, nnz);
  THCTensor_(transpose)(state, values, NULL, 0, 1);

  THCSTensor *newSparse = THCSTensor_(newClone)(state, sparse);
  THCIndexTensor *spIndices = THCSTensor_(indices)(state, newSparse);
  THCIndexTensor *dstIndices = THCIndexTensor_(new)(state);
  THCIndexTensor_(select)(state, dstIndices, spIndices, 0, 0);
  // Save destination indices to output hybrid tensor
  THCIndexTensor_(copy)(state, indices, dstIndices);
  // Replace destination indices with 0, 1, 2, 3, ... and compute output values
  // tensor with sparse * dense multiplication
  thrust::device_ptr<integer> indicesIter(THCIndexTensor_(data)(state, dstIndices));
  THRUST_EXEC(thrust::sequence, indicesIter, indicesIter + nnz);
  newSparse->size[0] = nnz;
  THCSTensor_(spaddmm)(state, values, ScalarConvert<int, real>::to(0), values, alpha, newSparse, dense);
  THCSTensor_(_move)(state, r_, indices, values);

  THCSTensor_(free)(state, newSparse);
  THCIndexTensor_(free)(state, spIndices);
  THCIndexTensor_(free)(state, dstIndices);

#undef THRUST_EXEC
}

void THCSTensor_(spcadd)(THCState *state, THCTensor *r_, THCTensor *dense, real value, THCSTensor *sparse) {
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 1, 3, sparse, r_, dense));
  THCTensor_(resizeAs)(state, r_, dense);
  THCSTensor_(contiguous)(state, sparse);

  THCIndexTensor *indices = THCSTensor_(indices)(state, sparse);
  THCTensor *values = THCSTensor_(values)(state, sparse);
  long nDim = THCTensor_(nDimension)(state, dense);
  long nDimI = THCSTensor_(nDimensionI)(state, sparse);

  if (r_ != dense) {
    THCTensor_(resizeAs)(state, r_, dense);
    THCTensor_(copy)(state, r_, dense);
  } else {
    if(!THCTensor_(isContiguous)(state, r_)) {
      THCTensor* r_contiguous = THCTensor_(newContiguous)(state, r_);
      THCTensor_(copy)(state, r_, r_contiguous);
      THCTensor_(free)(state, r_contiguous);
    }
  }

  // TODO more benchmarking
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (sparse->nDimensionV == 0) {
    THArgCheck(getApplyGrid(state, sparse->nnz, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_sparseElementwiseKernelScalar<TensorCAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        TensorCAddOp<real>(value),
        V_INFO(r_), I_INFO(indices), V_INFO(values),
        (unsigned long)sparse->nnz);
  } else {
    THArgCheck(getApplyGrid(state, sparse->nnz * block.x, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_sparseElementwiseKernel<TensorCAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        TensorCAddOp<real>(value),
        V_INFO(r_), I_INFO(indices), V_INFO(values),
        (unsigned long)sparse->nnz);
  }
  THCudaCheck(cudaGetLastError());

  THCIndexTensor_(free)(state, indices);
  THCTensor_(free)(state, values);
}

void THCSTensor_(mul)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  if (r_ == t) {
    THCTensor *r_values_ = THCSTensor_(values)(state, r_);
    THCTensor_(mul)(state, r_values_, r_values_, value);
    THCTensor_(free)(state, r_values_);
  } else {
    THCSTensor_(resizeAs)(state, r_, t);

    THCIndexTensor *r_indices_ = THCSTensor_(indices)(state, r_);
    THCTensor *r_values_ = THCSTensor_(values)(state, r_);
    THCIndexTensor *t_indices_ = THCSTensor_(indices)(state, t);
    THCTensor *t_values_ = THCSTensor_(values)(state, t);

    THCIndexTensor_(resizeAs)(state, r_indices_, t_indices_);
    THCIndexTensor_(copy)(state, r_indices_, t_indices_);
    THCTensor_(mul)(state, r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->contiguous = t->contiguous;

    THCIndexTensor_(free)(state, r_indices_);
    THCTensor_(free)(state, r_values_);
    THCIndexTensor_(free)(state, t_indices_);
    THCTensor_(free)(state, t_values_);
  }
}

void THCSTensor_(div)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  if (r_ == t) {
    THCTensor *r_values_ = THCSTensor_(values)(state, r_);
    THCTensor_(div)(state, r_values_, r_values_, value);
    THCTensor_(free)(state, r_values_);
  } else {
    THCSTensor_(resizeAs)(state, r_, t);

    THCIndexTensor *r_indices_ = THCSTensor_(indices)(state, r_);
    THCTensor *r_values_ = THCSTensor_(values)(state, r_);
    THCIndexTensor *t_indices_ = THCSTensor_(indices)(state, t);
    THCTensor *t_values_ = THCSTensor_(values)(state, t);

    THCIndexTensor_(resizeAs)(state, r_indices_, t_indices_);
    THCIndexTensor_(copy)(state, r_indices_, t_indices_);
    THCTensor_(div)(state, r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->contiguous = t->contiguous;

    THCIndexTensor_(free)(state, r_indices_);
    THCTensor_(free)(state, r_values_);
    THCIndexTensor_(free)(state, t_indices_);
    THCTensor_(free)(state, t_values_);
  }
}

void THCSTensor_(cadd)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src) {
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 3, 3, r_, t, src));
  if(!THCSTensor_(isSameSizeAs)(state, t, src)) {
    THError("cadd operands have incompatible sizes or dimension types");
  }
  THCSTensor_(contiguous)(state, t);
  THCSTensor_(contiguous)(state, src);

  if (src->nnz == 0) {
    THCSTensor_(copy)(state, r_, t);
    return;
  }
  if (t->nnz == 0) {
    THCSTensor_(mul)(state, r_, src, value);
    return;
  }

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-contiguous result.
  // This trade-off is preferable for the common use-case of gradient accumulation.
  // TODO have two distinct functions? The other option is commented out below
  THCIndexTensor *t_indices_ = THCSTensor_(indices)(state, t);
  THCTensor *t_values_ = THCSTensor_(values)(state, t);
  THCIndexTensor *s_indices_ = THCSTensor_(indices)(state, src);
  THCTensor *s_values_ = THCSTensor_(values)(state, src);
  if (value != ScalarConvert<int, real>::to(1)) {
    THCTensor *s_values_orig = s_values_;
    s_values_ = THCTensor_(new)(state);
    THCTensor_(mul)(state, s_values_, s_values_orig, value);
    THCTensor_(free)(state, s_values_orig);
  }
  THCIndexTensor *r_indices_ = THCIndexTensor_(new)(state);
  THCTensor *r_values_ = THCTensor_(new)(state);
  THCIndexTensor_(cat)(state, r_indices_, t_indices_, s_indices_, 1);
  THCTensor_(cat)(state, r_values_, t_values_, s_values_, 0);
  THCSTensor_(resizeAs)(state, r_, src);
  THCSTensor_(_move)(state, r_, r_indices_, r_values_);

  // // saving those because they can be overwritten when doing in-place operations
  // ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz, max_nnz = t_nnz + s_nnz;
  // long nDimI = src->nDimensionI;
  // THCIndexTensor *t_indices_ = THCSTensor_(indices)(state, t);
  // THCTensor *t_values_ = THCSTensor_(values)(state, t);
  // THCIndexTensor *s_indices_ = THCSTensor_(indices)(state, src);
  // THCTensor *s_values_ = THCSTensor_(values)(state, src);
  // THCIndexTensor *r_indices_ = THCIndexTensor_(newWithSize2d)(state, nDimI, max_nnz);
  // THCTensor *r_values_ = THCSTensor_(newValuesWithSizeOf)(state, s_values_, max_nnz);
  // THCTensor_(zero)(state, r_values_);
  // THCSTensor_(resizeAs)(state, r_, src);
  // THCSTensor_(_move)(state, r_, r_indices_, r_values_);
  //
  // const dim3 block = getApplyBlock();
  // dim3 grid;
  // THArgCheck(getApplyGrid(state, t_values_->stride[0], grid), 1, CUTORCH_DIM_WARNING);
  //
  // THCSTensor_valueSparseUnionKernel<TensorCAddOp<real>, TensorAddOp<real>, TensorCAddOp<real>, unsigned long, real>
  //   <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
  //     TensorCAddOp<real>(value),
  //     TensorAddOp<real>(),
  //     TensorCAddOp<real>(value),
  //     I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
  //     V_INFO(r_values_), V_INFO(t_values_), V_INFO(s_values_),
  //     (unsigned long)t_nnz, (unsigned long)s_nnz);
  // THCudaCheck(cudaGetLastError());
  //
  // bool freeScratchSpace = false;
  // void* scratchSpace = THCState_getCurrentDeviceScratchSpace(state);
  // if (!scratchSpace) {
  //   THCudaCheck(THCudaMalloc(state, &scratchSpace,
  //       THCState_getCurrentDeviceScratchSpaceSize(state)));
  //   freeScratchSpace = true;
  // }
  // THCSTensor_indexSparseUnionKernel<unsigned long, real>
  //   <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
  //     I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
  //     (unsigned long)t_nnz, (unsigned long)s_nnz, (unsigned long*)scratchSpace);
  // THCudaCheck(cudaGetLastError());
  // // Synchronous!
  // unsigned long nnzOut;
  // THCudaCheck(cudaMemcpy(&nnzOut, scratchSpace, sizeof(unsigned long), cudaMemcpyDeviceToHost));
  // r_->nnz = nnzOut;
  // r_->contiguous = 1;
  // if (freeScratchSpace) {
  //   THCudaCheck(THCudaFree(state, scratchSpace));
  // }

  THCIndexTensor_(free)(state, t_indices_);
  THCTensor_(free)(state, t_values_);
  THCIndexTensor_(free)(state, s_indices_);
  THCTensor_(free)(state, s_values_);
}

void THCSTensor_(csub)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src) {
  THCSTensor_(cadd)(state, r_, t, ScalarNegate<real>::to(value), src);
}

void THCSTensor_(cmul)(THCState *state, THCSTensor *r_, THCSTensor *t, THCSTensor *src) {
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 3, 3, r_, t, src));
  if(!THCSTensor_(isSameSizeAs)(state, t, src)) {
    THError("cmul operands have incompatible sizes or dimension types");
  }
  THCSTensor_(contiguous)(state, t);
  THCSTensor_(contiguous)(state, src);

  if (t->nnz == 0 || src->nnz == 0) {
    THCSTensor_(zero)(state, r_);
    return;
  }

  // saving those because they can be overwritten when doing in-place operations
  ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz;
  ptrdiff_t max_nnz = t_nnz < s_nnz ? t_nnz : s_nnz;
  long nDimI = src->nDimensionI;
  THCIndexTensor *t_indices_ = THCSTensor_(indices)(state, t);
  THCTensor *t_values_ = THCSTensor_(values)(state, t);
  THCIndexTensor *s_indices_ = THCSTensor_(indices)(state, src);
  THCTensor *s_values_ = THCSTensor_(values)(state, src);
  THCIndexTensor *r_indices_ = THCIndexTensor_(newWithSize2d)(state, nDimI, max_nnz);
  THCTensor *r_values_ = THCSTensor_(newValuesWithSizeOf)(state, s_values_, max_nnz);
  THCTensor_(zero)(state, r_values_);
  THCSTensor_(resizeAs)(state, r_, src);
  THCSTensor_(_move)(state, r_, r_indices_, r_values_);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, t_values_->stride[0], grid), 1, CUTORCH_DIM_WARNING);

  THCSTensor_valueSparseIntersectionKernel<TensorMulOp<real>, unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      TensorMulOp<real>(),
      I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
      V_INFO(r_values_), V_INFO(t_values_), V_INFO(s_values_),
      (unsigned long)t_nnz, (unsigned long)s_nnz);
  THCudaCheck(cudaGetLastError());

  THCudaLongStorage *resultNnz = THCudaLongStorage_newWithSize(state, 1);
  THCSTensor_indexSparseIntersectionKernel<unsigned long, real>
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
      I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
      (unsigned long)t_nnz, (unsigned long)s_nnz, (unsigned long*)resultNnz->data);
  THCudaCheck(cudaGetLastError());
  r_->nnz = THCudaLongStorage_get(state, resultNnz, 0);
  THCudaLongStorage_free(state, resultNnz);
  r_->contiguous = 1;

  THCIndexTensor_(free)(state, t_indices_);
  THCTensor_(free)(state, t_values_);
  THCIndexTensor_(free)(state, s_indices_);
  THCTensor_(free)(state, s_values_);
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
