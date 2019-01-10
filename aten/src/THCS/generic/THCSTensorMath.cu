#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensorMath.cu"
#else

#include "THCThrustAllocator.cuh"
#include "THCNumerics.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define ROW_PTR2(t, r) (THCTensor_(data)(THCState *state, t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THCTensor_(data)(THCState *state, t) + (c) * (t)->stride[1])

#define I_INFO(tensor) getTensorInfo<THCIndexTensor, uint64_t>(state, tensor)
#define V_INFO(tensor) getTensorInfo<THCTensor, uint64_t>(state, tensor)

THCudaIntTensor *THCSTensor_(toCSR)(THCState *state, THCIndexTensor *rowIndices, int64_t dim, int64_t nnz) {
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

void THCSTensor_(zeros)(THCState *state, THCSTensor *r_, THLongStorage *size)
{
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 1, 1, r_));
  THCSTensor_(resize)(state, r_, size);
  THCSTensor_(zero)(state, r_);
}

void THCSTensor_(zerosLike)(THCState *state, THCSTensor *r_, THCSTensor *input)
{
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 2, 2, r_, input));
  THCSTensor_(resizeAs)(state, r_, input);
  THCSTensor_(zero)(state, r_);
}

void THCTensor_(spaddcmul)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("WARNING: Sparse Cuda Tensor op spaddcmul is not implemented");
}

void THCTensor_(spaddcdiv)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2) {
  THError("WARNING: Sparse Cuda Tensor op spaddcdiv is not implemented");
}

void THCSTensor_(spaddmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCSTensor *sparse_, THCTensor *dense) {
#if defined(THCS_REAL_IS_FLOAT) || defined(THCS_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 1, 4, sparse_, r_, t, dense));
  THCudaIntTensor *csr;
  THCIndexTensor *indices;
  THCTensor *values, *r__, *dense_;

  THArgCheck(sparse_->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse_->nDimensionI);
  THArgCheck(sparse_->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse_->nDimensionV);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  int64_t m = THCSTensor_(size)(state, sparse_, 0);
  int64_t k = THCSTensor_(size)(state, sparse_, 1);
  int64_t n = THCTensor_(size)(state, dense, 1);

  THCTensor_(resize2d)(state, r_, m, n);

  THArgCheck(THCTensor_(size)(state, t, 0) == m, 1,
      "Expected dim 0 size %d, got %d", m, THCTensor_(size)(state, t, 0));
  THArgCheck(THCTensor_(size)(state, t, 1) == n, 1,
      "Expected dim 1 size %d, got %d", n, THCTensor_(size)(state, t, 1));
  THArgCheck(THCTensor_(size)(state, dense, 0) == k, 3,
      "Expected dim 0 size %d, got %d", k, THCTensor_(size)(state, dense, 0));

  THCSTensor *sparse = THCSTensor_(newCoalesce)(state, sparse_);

  int64_t nnz = THCSTensor_(nnz)(state, sparse);
  indices = THCSTensor_(newIndices)(state, sparse);
  values = THCSTensor_(newValues)(state, sparse);

  THCIndexTensor *rowIndices = THCIndexTensor_(newSelect)(state, indices, 0, 0);
  THCIndexTensor *colIndices = THCIndexTensor_(newSelect)(state, indices, 0, 1);
  csr = THCSTensor_(toCSR)(state, rowIndices, m, nnz);
  THCudaIntTensor *colIndicesInt = THCudaIntTensor_newWithSize1d(state, colIndices->size[0]);
  THCudaIntTensor_copyCudaLong(state, colIndicesInt, colIndices);

  char transpose_dense;

  if (beta == 0) {
    THCTensor_(zero)(state, r_);
  } else if (beta == ScalarConvert<int, real>::to(1)) {
    if (t != r_) {
      THCTensor_(copy)(state, r_, t);
    }
  } else {
    THCTensor_(mul)(state, r_, t, beta);
  }

  /* r_ */
  if(r_->stride[0] == 1 && r_->stride[1] == r_->size[0]) {
    r__ = r_;
    THCTensor_(retain)(state, r__);
  } else {
    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, r_, 0, 1);
    r__ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, r__, NULL, 0, 1);
  }

  /* dense */
  if(dense->stride[0] == 1 && dense->stride[1] == dense->size[0]) {
    transpose_dense = 'n';
    dense_ = dense;
    THCTensor_(retain)(state, dense_);
  } else if(dense->stride[1] == 1 && dense->stride[0] != dense->size[1]) {
    transpose_dense = 't';
    dense_ = dense;
    THCTensor_(retain)(state, dense_);
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
  THCTensor_(free)(state, dense_);
  THCTensor_(freeCopyTo)(state, r__, r_);
  THCudaIntTensor_free(state, colIndicesInt);
  THCudaIntTensor_free(state, csr);
  THCIndexTensor_(free)(state, indices);
  THCIndexTensor_(free)(state, rowIndices);
  THCIndexTensor_(free)(state, colIndices);
  THCTensor_(free)(state, values);
  THCSTensor_(free)(state, sparse);
#else
  THError("unimplemented data type");
#endif
}

void THCSTensor_(sspaddmm)(THCState *state, THCSTensor *r_, real beta, THCSTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense) {
  THError("WARNING: Sparse Cuda Tensor op sspaddmm is not implemented");
  // TODO Write some kernels
}

void THCSTensor_(hspmm)(THCState *state, THCSTensor *r_, real alpha, THCSTensor *sparse_, THCTensor *dense) {
#if CUDA_VERSION >= 7000
  THCThrustAllocator thrustAlloc(state);
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 2, 3, r_, sparse_, dense));

  THArgCheck(sparse_->nDimensionI == 2, 3,
      "matrices expected, got %dD tensor", sparse_->nDimensionI);
  THArgCheck(sparse_->nDimensionV == 0, 3,
      "scalar values expected, got %dD values", sparse_->nDimensionV);
  THArgCheck(dense->nDimension == 2, 4,
      "matrices expected, got %dD tensor", dense->nDimension);

  int64_t m = THCSTensor_(size)(state, sparse_, 0);
  int64_t k = THCSTensor_(size)(state, sparse_, 1);
  int64_t n = THCTensor_(size)(state, dense, 1);

  THArgCheck(THCTensor_(size)(state, dense, 0) == k, 4,
      "Expected dim 0 size %d, got %d", k, THCTensor_(size)(state, dense, 0));
  int64_t size[2] = {m, n};
  THCSTensor_(rawResize)(state, r_, 1, 1, size);

  THCSTensor *sparse = THCSTensor_(newCoalesce)(state, sparse_);

  int64_t nnz = THCSTensor_(nnz)(state, sparse);
  THCIndexTensor *indices = THCIndexTensor_(newWithSize2d)(state, 1, nnz);
  // create values in column-major format to avoid copying in spaddmm
  THCTensor *values = THCTensor_(newWithSize2d)(state, n, nnz);
  THCTensor_(transpose)(state, values, NULL, 0, 1);

  // why does sparse need to be cloned? If this is really necessary maybe we
  // need to fuse this with newCoalesce
  THCSTensor *newSparse = THCSTensor_(newClone)(state, sparse);
  THCIndexTensor *spIndices = THCSTensor_(newIndices)(state, newSparse);
  THCIndexTensor *dstIndices = THCIndexTensor_(newSelect)(state, spIndices, 0, 0);
  // Save destination indices to output hybrid tensor
  THCIndexTensor_(copy)(state, indices, dstIndices);
  // Replace destination indices with 0, 1, 2, 3, ... and compute output values
  // tensor with sparse * dense multiplication
  thrust::device_ptr<indexT> indicesIter(THCIndexTensor_(data)(state, dstIndices));
  THRUST_EXEC(thrust::sequence, indicesIter, indicesIter + nnz);
  newSparse->size[0] = nnz;
  THCSTensor_(spaddmm)(state, values, ScalarConvert<int, real>::to(0), values, alpha, newSparse, dense);
  THCSTensor_(_move)(state, r_, indices, values);

  THCSTensor_(free)(state, newSparse);
  THCIndexTensor_(free)(state, spIndices);
  THCIndexTensor_(free)(state, dstIndices);
  THCSTensor_(free)(state, sparse);

#undef THRUST_EXEC
}

void THCSTensor_(spcadd)(THCState *state, THCTensor *r_, THCTensor *dense, real value, THCSTensor *sparse) {
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 1, 3, sparse, r_, dense));

  const ptrdiff_t nnz = THCSTensor_(nnz)(state, sparse);
  if (nnz == 0) {
    THCTensor_(resizeAs)(state, r_, dense);
    THCTensor_(copy)(state, r_, dense);
    return;
  }

  THCTensor *r = r_;
  if (r != dense) {
    THCTensor_(retain)(state, r);
    THCTensor_(resizeAs)(state, r, dense);
    THCTensor_(copy)(state, r, dense);
  } else {
    if (!THCTensor_(isContiguous)(state, r_)) {
      THError("CUDA sparse spcadd: known bug");
    }
    r = THCTensor_(newContiguous)(state, r_);
  }

  THCIndexTensor *indices = THCSTensor_(newIndices)(state, sparse);
  THCTensor *values = THCSTensor_(newValues)(state, sparse);
  int64_t nDim = THCTensor_(nDimension)(state, dense);
  int64_t nDimI = THCSTensor_(nDimensionI)(state, sparse);

 if (THCSTensor_(isCoalesced)(state, sparse)) {
    // TODO benchmark to decide whether to remove this special case
    const dim3 block = getApplyBlock();
    dim3 grid;
    if (sparse->nDimensionV == 0) {
      THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

      THCSTensor_sparseElementwiseKernelScalar<TensorCAddOp<real>, uint64_t, real>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorCAddOp<real>(value),
          V_INFO(r_), I_INFO(indices), V_INFO(values),
          (uint64_t) nnz);
    } else {
      THArgCheck(getApplyGrid(state, nnz * block.x, grid), 1, CUTORCH_DIM_WARNING);

      THCSTensor_sparseElementwiseKernel<TensorCAddOp<real>, uint64_t, real>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorCAddOp<real>(value),
          V_INFO(r_), I_INFO(indices), V_INFO(values),
          (uint64_t) nnz);
    }
  } else {
    THCIndexTensor *indices1D = THCSTensor_(newFlattenedIndices)(state, sparse, 0);
    THCIndexTensor_(resize1d)(state, indices1D, nnz);

    if (value != ScalarConvert<int, real>::to(1)) {
      // FIXME: at some point we can wrap the scale into indexAdd
      THCTensor *scaled = THCTensor_(new)(state);
      THCTensor_(mul)(state, scaled, values, value);
      THCTensor_(free)(state, values);
      values = scaled;
    }

    int64_t view_rows = 1;
    int64_t view_columns = 1;
    THLongStorage *r_size = THCTensor_(newSizeOf)(state, r);
    for (int i = 0; i < nDimI; i++)
      view_rows *= r_size->data[i];
    for (int i = nDimI; i < nDim; i++)
      view_columns *= r_size->data[i];

    THLongStorage *r_view_size = THLongStorage_newWithSize2(view_rows, view_columns);
    THCTensor *r_view = THCTensor_(newView)(state, r, r_view_size);
    THCTensor_(resize2d)(state, values, nnz, view_columns);

    THCTensor_(indexAdd)(state, r_view, 0, indices1D, values);

    THCIndexTensor_(free)(state, indices1D);
    THLongStorage_free(r_size);
    THLongStorage_free(r_view_size);
    THCTensor_(free)(state, r_view);
  }
  THCudaCheck(cudaGetLastError());

  THCIndexTensor_(free)(state, indices);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, r);
}

void THCSTensor_(mul)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  if (r_ == t) {
    THCTensor *r_values_ = THCSTensor_(newValues)(state, r_);
    THCTensor_(mul)(state, r_values_, r_values_, value);
    THCTensor_(free)(state, r_values_);
  } else {
    THCSTensor_(resizeAs)(state, r_, t);

    THCIndexTensor *r_indices_ = THCSTensor_(newIndices)(state, r_);
    THCTensor *r_values_ = THCSTensor_(newValues)(state, r_);
    THCIndexTensor *t_indices_ = THCSTensor_(newIndices)(state, t);
    THCTensor *t_values_ = THCSTensor_(newValues)(state, t);

    THCIndexTensor_(resizeAs)(state, r_indices_, t_indices_);
    THCIndexTensor_(copy)(state, r_indices_, t_indices_);
    THCTensor_(mul)(state, r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->coalesced = t->coalesced;

    THCIndexTensor_(free)(state, r_indices_);
    THCTensor_(free)(state, r_values_);
    THCIndexTensor_(free)(state, t_indices_);
    THCTensor_(free)(state, t_values_);
  }
}

void THCSTensor_(div)(THCState *state, THCSTensor *r_, THCSTensor *t, real value) {
  if (r_ == t) {
    THCTensor *r_values_ = THCSTensor_(newValues)(state, r_);
    THCTensor_(div)(state, r_values_, r_values_, value);
    THCTensor_(free)(state, r_values_);
  } else {
    THCSTensor_(resizeAs)(state, r_, t);

    THCIndexTensor *r_indices_ = THCSTensor_(newIndices)(state, r_);
    THCTensor *r_values_ = THCSTensor_(newValues)(state, r_);
    THCIndexTensor *t_indices_ = THCSTensor_(newIndices)(state, t);
    THCTensor *t_values_ = THCSTensor_(newValues)(state, t);

    THCIndexTensor_(resizeAs)(state, r_indices_, t_indices_);
    THCIndexTensor_(copy)(state, r_indices_, t_indices_);
    THCTensor_(div)(state, r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->coalesced = t->coalesced;

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
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient accumulation.
  THCIndexTensor *t_indices_ = THCSTensor_(newIndices)(state, t);
  THCTensor *t_values_ = THCSTensor_(newValues)(state, t);
  THCIndexTensor *s_indices_ = THCSTensor_(newIndices)(state, src);
  THCTensor *s_values_ = THCSTensor_(newValues)(state, src);
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

  // FIXME: add some heuristic about when to call coalesce() here, so that
  // tensors don't totally blow up in size by concatenation; e.g.
  //   r->minUnique = max(a->minUnique + b->minUnique);
  //   if (r->nnz / r->minUnique > COMPACTION_THRESHOLD) {
  //     THCSTensor_(contiguous)(r);
  //     r->minUnique = r->nnz;
  //   }

  THCIndexTensor_(free)(state, t_indices_);
  THCTensor_(free)(state, t_values_);
  THCIndexTensor_(free)(state, s_indices_);
  THCTensor_(free)(state, s_values_);
}

void THCSTensor_(csub)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src) {
  THCSTensor_(cadd)(state, r_, t, ScalarNegate<real>::to(value), src);
}

void THCSTensor_(cmul)(THCState *state, THCSTensor *r_, THCSTensor *t_, THCSTensor *src_) {
  THCAssertSameGPU(THCSTensor_(checkGPU)(state, 3, 3, r_, t_, src_));
  if(!THCSTensor_(isSameSizeAs)(state, t_, src_)) {
    THError("cmul operands have incompatible sizes or dimension types");
  }
  THCSTensor *t = THCSTensor_(newCoalesce)(state, t_);
  THCSTensor *src = THCSTensor_(newCoalesce)(state, src_);

  if (t->nnz == 0 || src->nnz == 0) {
    THCSTensor_(zero)(state, r_);
    return;
  }

  // saving those because they can be overwritten when doing in-place operations
  ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz;
  ptrdiff_t max_nnz = t_nnz < s_nnz ? t_nnz : s_nnz;
  int64_t nDimI = src->nDimensionI;
  THCIndexTensor *t_indices_ = THCSTensor_(newIndices)(state, t);
  THCTensor *t_values_ = THCSTensor_(newValues)(state, t);
  THCIndexTensor *s_indices_ = THCSTensor_(newIndices)(state, src);
  THCTensor *s_values_ = THCSTensor_(newValues)(state, src);
  THCIndexTensor *r_indices_ = THCIndexTensor_(newWithSize2d)(state, nDimI, max_nnz);
  THCTensor *r_values_ = THCSTensor_(newValuesWithSizeOf)(state, s_values_, max_nnz);
  THCTensor_(zero)(state, r_values_);
  THCSTensor_(resizeAs)(state, r_, src);
  THCSTensor_(_move)(state, r_, r_indices_, r_values_);

  int64_t valueSize = t_values_->stride[0];
  const dim3 block = dim3(min((int64_t) getApplyBlock().x, valueSize));
  dim3 grid;
  THArgCheck(getApplyGrid(state, valueSize, grid), 1, CUTORCH_DIM_WARNING);

  THCSTensor_valueSparseIntersectionKernel<TensorMulOp<real>, uint64_t, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      TensorMulOp<real>(),
      I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
      V_INFO(r_values_), V_INFO(t_values_), V_INFO(s_values_),
      (uint64_t)t_nnz, (uint64_t)s_nnz);
  THCudaCheck(cudaGetLastError());

  THCudaLongStorage *resultNnz = THCudaLongStorage_newWithSize(state, 1);
  THCSTensor_indexSparseIntersectionKernel<uint64_t, real>
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
      I_INFO(r_indices_), I_INFO(t_indices_), I_INFO(s_indices_),
      (uint64_t)t_nnz, (uint64_t)s_nnz, (uint64_t*)resultNnz->data);
  THCudaCheck(cudaGetLastError());
  r_->nnz = THCudaLongStorage_get(state, resultNnz, 0);
  THCudaLongStorage_free(state, resultNnz);
  r_->coalesced = 1;

  THCIndexTensor_(free)(state, t_indices_);
  THCTensor_(free)(state, t_values_);
  THCIndexTensor_(free)(state, s_indices_);
  THCTensor_(free)(state, s_values_);
  THCSTensor_(free)(state, t);
  THCSTensor_(free)(state, src);
}

void THCSTensor_(pow)(THCState *state, THCSTensor *r_, THCSTensor *t_, real value) {
  if (THCNumerics<real>::eq(value, ScalarConvert<int, real>::to(0))) {
    THError("cannot raise to zeroth power on sparse tensor");
  }
  THCSTensor *t = THCSTensor_(newCoalesce)(state, t_);
  THCSTensor_(resizeAs)(state, r_, t);

  THCIndexTensor *r_indices_ = THCSTensor_(newIndices)(state, r_);
  THCTensor *r_values_ = THCSTensor_(newValues)(state, r_);
  THCIndexTensor *t_indices_ = THCSTensor_(newIndices)(state, t);
  THCTensor *t_values_ = THCSTensor_(newValues)(state, t);

  THCIndexTensor_(resizeAs)(state, r_indices_, t_indices_);
  THCIndexTensor_(copy)(state, r_indices_, t_indices_);
  THCTensor_(pow)(state, r_values_, t_values_, value);
  r_->nnz = t->nnz;
  r_->coalesced = t->coalesced;

  THCIndexTensor_(free)(state, r_indices_);
  THCTensor_(free)(state, r_values_);
  THCIndexTensor_(free)(state, t_indices_);
  THCTensor_(free)(state, t_values_);
  THCSTensor_(free)(state, t);
}

#if defined(THCS_REAL_IS_FLOAT) || defined(THCS_REAL_IS_DOUBLE) || defined(THCS_REAL_IS_HALF)
accreal THCSTensor_(normall)(THCState *state, THCSTensor *self, real value) {
  THCSTensor* self_coalesced = THCSTensor_(newCoalesce)(state, self);
  accreal result = THCTensor_(normall)(state, self_coalesced->values, value); 
  THCSTensor_(free)(state, self_coalesced);
  return result;
}
#endif

#undef ROW_PTR2
#undef COL_PTR2

#endif
