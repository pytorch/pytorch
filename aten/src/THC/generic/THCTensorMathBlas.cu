#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathBlas.cu"
#else

#include "ATen/cuda/CUDAContext.h"
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

#define ERROR_ONLY_FP_TYPES(func) \
  THError("%s for CUDA tensors only supports floating-point types. Try converting the tensors with .float()", func);

accreal THCTensor_(dot)(THCState *state, THCTensor *self, THCTensor *src)
{
  if ( (THTensor_nDimension(self) != 1) || (THTensor_nDimension(src) != 1) ) {
    THError("1D tensors expected, got %dD, %dD tensors",
       THTensor_nDimension(self), THTensor_nDimension(src));
  }

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  THArgCheck(THCTensor_(nElement)(state, self) ==
             THCTensor_(nElement)(state, src), 2, "sizes do not match");

  self = THCTensor_(newContiguous)(state, self);
  src = THCTensor_(newContiguous)(state, src);

#ifdef THC_REAL_IS_FLOAT
  accreal result = THCudaBlas_Sdot(state,
                                THCTensor_(nElement)(state, self),
                                THCTensor_(data)(state, self), 1,
                                THCTensor_(data)(state, src), 1);
#elif defined(THC_REAL_IS_DOUBLE)
  accreal result = THCudaBlas_Ddot(state,
                                THCTensor_(nElement)(state, self),
                                THCTensor_(data)(state, self), 1,
                                THCTensor_(data)(state, src), 1);
#elif defined(THC_REAL_IS_HALF)
  accreal result = THCudaBlas_Hdot(state,
                                THCTensor_(nElement)(state, self),
                                THCTensor_(data)(state, self), 1,
                                THCTensor_(data)(state, src), 1);
#endif

  THCTensor_(free)(state, src);
  THCTensor_(free)(state, self);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::check_names_for_dot(self, src);
#endif
  return result;

#else
  ERROR_ONLY_FP_TYPES("dot");
  return ScalarConvert<int, accreal>::to(0);
#endif
}

void THCTensor_(addmv)(THCState *state, THCTensor *r_, THCTensor *t, THCTensor *mat, THCTensor *vec, scalar_t beta, scalar_t alpha)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, mat, vec));
  if( (mat->dim() != 2) || (THTensor_nDimension(vec) != 1) )
    THError("2D tensor and 1D tensor expected, got %dD, %dD tensors",
       mat->dim(), THTensor_nDimension(vec));


  auto vec_size = THTensor_sizeLegacyNoScalars(vec, 0);
  auto vec_stride = THTensor_strideLegacyNoScalars(vec, 0);

  if( mat->size(1) != THTensor_sizeLegacyNoScalars(vec, 0) )
    THError("size mismatch");

  if(t->dim() != 1)
    THError("size mismatch");

  if(THTensor_sizeLegacyNoScalars(t, 0) != mat->size(0))
    THError("size mismatch");

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  if(r_ != t)
  {
#ifdef BUILD_NAMEDTENSOR
    at::NoNamesGuard guard;
#endif
    THCTensor_(resizeAs)(state, r_, t);
    THCTensor_(copy)(state, r_, t);
  }

  auto r_stride = THTensor_strideLegacyNoScalars(r_, 0);

  if(mat->stride(0) == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 'n', mat->size(0), mat->size(1),
                    alpha, THCTensor_(data)(state, mat), mat->stride(1),
                    THCTensor_(data)(state, vec), vec_stride,
                    beta, THCTensor_(data)(state, r_), r_stride);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 'n', mat->size(0), mat->size(1),
                    alpha, THCTensor_(data)(state, mat), mat->stride(1),
                    THCTensor_(data)(state, vec), vec_stride,
                    beta, THCTensor_(data)(state, r_), r_stride);
#endif
  }
  else if(mat->stride(1) == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 't',  mat->size(1), mat->size(0),
                    alpha, THCTensor_(data)(state, mat), mat->stride(0),
                    THCTensor_(data)(state, vec), vec_stride,
                    beta, THCTensor_(data)(state, r_), r_stride);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 't',  mat->size(1), mat->size(0),
                     alpha, THCTensor_(data)(state, mat), mat->stride(0),
                     THCTensor_(data)(state, vec), vec_stride,
                     beta, THCTensor_(data)(state, r_), r_stride);
#endif
  }
  else
  {
    THCTensor *cmat = THCTensor_(newContiguous)(state, mat);

#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 't',  mat->size(1), mat->size(0),
                    alpha, THCTensor_(data)(state, cmat), cmat->stride(0),
                    THCTensor_(data)(state, vec), vec_stride,
                    beta, THCTensor_(data)(state, r_), r_stride);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 't',  mat->size(1), mat->size(0),
                    alpha, THCTensor_(data)(state, cmat), cmat->stride(0),
                    THCTensor_(data)(state, vec), vec_stride,
                    beta, THCTensor_(data)(state, r_), r_stride);
#endif

    THCTensor_(free)(state, cmat);
  }

  // In cublasSgemv, cublasDgemv (x,0).mv(0) does not
  // handle beta, whereas cublasSgemm, cublasDgemm do for case where (x,0).mm(0,y).
  if (THTensor_sizeLegacyNoScalars(vec, 0) == 0 && mat->size(0) != 0) {
    if(THCNumerics<scalar_t>::eq(beta, ScalarConvert<int, scalar_t>::to(0))) {
      THCTensor_(zero)(state, r_);
    } else if(THCNumerics<scalar_t>::ne(beta, ScalarConvert<int, scalar_t>::to(1))) {
      THCTensor_(mul)(state, r_, r_, beta);
    }
  }

#elif defined(THC_REAL_IS_HALF)
    // Currently no Hgemv/SgemvEx in Cublas
    THCTensor *vecAsMatrix = THCTensor_(newWithTensor)(state, vec);
    THCTensor_(resize2d)(state, vecAsMatrix, vec_size, 1);

    THCTensor *tAsMatrix = THCTensor_(newWithTensor)(state, t);
    THCTensor_(resize2d)(state, tAsMatrix, THTensor_sizeLegacyNoScalars(tAsMatrix, 0), 1);

    THCTensor_(addmm)(state, r_, tAsMatrix, mat, vecAsMatrix, beta, alpha);

    // r_ will have answer as matrix, need to return a vector
    THCTensor_(resize1d)(state, r_, THTensor_sizeLegacyNoScalars(r_, 0));
    THCTensor_(free)(state, vecAsMatrix);
    THCTensor_(free)(state, tAsMatrix);
#endif
#else
  ERROR_ONLY_FP_TYPES("addmv");
#endif
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmv(r_, mat, vec, t);
#endif
}

void THCTensor_(addr)(THCState *state, THCTensor *r_, THCTensor *t, THCTensor *vec1, THCTensor *vec2, scalar_t beta, scalar_t alpha)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, vec1, vec2));
  if ( (THTensor_nDimension(vec1) != 1) || (THTensor_nDimension(vec2) != 1) ) {
    THError("1D tensors expected, got %dD, %dD tensors",
       THTensor_nDimension(vec1), THTensor_nDimension(vec2));
  }
  auto vec1_size = THTensor_sizeLegacyNoScalars(vec1, 0);
  auto vec2_size = THTensor_sizeLegacyNoScalars(vec2, 0);
  auto vec1_stride = THTensor_strideLegacyNoScalars(vec1, 0);
  auto vec2_stride = THTensor_strideLegacyNoScalars(vec2, 0);

  if (t->dim() != 2) {
    THError("size mismatch");
  }

  if ( (t->size(0) != vec1_size) || (t->size(1) != vec2_size) ) {
    THError("size mismatch");
  }

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  if (r_ != t) {
    THCTensor_(resizeAs)(state, r_, t);
    THCTensor_(copy)(state, r_, t);
  }

  if(THCNumerics<scalar_t>::eq(beta, ScalarConvert<int, scalar_t>::to(0))) {
    THCTensor_(zero)(state, r_);
  } else if(THCNumerics<scalar_t>::ne(beta, ScalarConvert<int, scalar_t>::to(1))) {
    THCTensor_(mul)(state, r_, r_, beta);
  }

  if(r_->stride(0) == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec1_size, vec2_size,
                   alpha, THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, r_), r_->stride(1));
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec1_size, vec2_size,
                   alpha, THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, r_), r_->stride(1));
#endif
  }
  else if(r_->stride(1) == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec2_size, vec1_size,
                   alpha, THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, r_), r_->stride(0));
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec2_size, vec1_size,
                   alpha, THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, r_), r_->stride(0));
#endif
  }
  else
  {
    THCTensor *cr = THCTensor_(newClone)(state, r_);

#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec2_size, vec1_size,
                   alpha, THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, cr), cr->stride(0));
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec2_size, vec1_size,
                   alpha, THCTensor_(data)(state, vec2), vec2_stride,
                   THCTensor_(data)(state, vec1), vec1_stride,
                   THCTensor_(data)(state, cr), cr->stride(0));
#endif

    THCTensor_(freeCopyTo)(state, cr, r_);
  }
#elif defined(THC_REAL_IS_HALF)
  // currently no Hger/SgerEx in Cublas.
  THCTensor *vec2T = THCTensor_(newWithTensor)(state, vec2);
  THCTensor_(resize2d)(state, vec2T, vec2_size, 1);
  THCTensor_(transpose)(state, vec2T, NULL, 0, 1);

  THCTensor *vec1M = THCTensor_(newWithTensor)(state, vec1);
  THCTensor_(resize2d)(state, vec1M, vec1_size, 1);

  THCTensor_(addmm)(state, r_, t, vec1M, vec2T, beta, alpha);
  THCTensor_(free)(state, vec2T);
  THCTensor_(free)(state, vec1M);
#endif
#else
  ERROR_ONLY_FP_TYPES("addr");
#endif
}

void THCTensor_(addmm)(THCState *state, THCTensor *r_, THCTensor *t, THCTensor *m1, THCTensor *m2, scalar_t beta, scalar_t alpha)
{
#ifdef BUILD_NAMEDTENSOR
  // The logic in this function changes around the pointers, so save a copy of the originals.
  THCTensor* orig_m1 = m1;
  THCTensor* orig_m2 = m2;
#endif

#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, m1, m2));
  char transpose_r, transpose_m1, transpose_m2;
  THCTensor *r__, *m1_, *m2_;

  if( (m1->dim() != 2) || (m2->dim() != 2) )
    THError("2D tensors expected, got %dD, %dD tensors", m1->dim(), m2->dim());

  if(t->dim() != 2)
    THError("2D tensor expected, got %dD tensor for t", t->dim());

  if(m1->size(1) != m2->size(0)) {
    THCDescBuff bm1 = THCTensor_(sizeDesc)(state, m1);
    THCDescBuff bm2 = THCTensor_(sizeDesc)(state, m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( (t->size(0) != m1->size(0)) || (t->size(1) != m2->size(1)) ) {
    THCDescBuff bt  = THCTensor_(sizeDesc)(state, t);
    THCDescBuff bm1 = THCTensor_(sizeDesc)(state, m1);
    THCDescBuff bm2 = THCTensor_(sizeDesc)(state, m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THCTensor_(resizeAs)(state, r_, t);
    if (ScalarConvert<scalar_t, double>::to(beta) != 0.0) {
#ifdef BUILD_NAMEDTENSOR
      at::NoNamesGuard guard;
#endif
      THCTensor_(copy)(state, r_, t);
    }
  }

  /* r_ */
  if(r_->stride(0) == 1 &&
     r_->stride(1) != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride(1) == 1 &&
          r_->stride(0) != 0)
  {
    THCTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, r_, 0, 1);
    r__ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m1->stride((transpose_r == 'n' ? 1 : 0)) != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m1->stride((transpose_r == 'n' ? 0 : 1)) != 0)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THCTensor_(newContiguous)(state, m1);
  }

  /* m2 */
  if(m2->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m2->stride((transpose_r == 'n' ? 1 : 0)) != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m2->stride((transpose_r == 'n' ? 0 : 1)) != 0)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THCTensor_(newContiguous)(state, m2);
  }

#ifdef THC_REAL_IS_HALF
  THCudaBlas_Hgemm(state,
                   transpose_m1,
                   transpose_m2,
                   r__->size((transpose_r == 'n' ? 0 : 1)),
                   r__->size((transpose_r == 'n' ? 1 : 0)),
                   m1_->size((transpose_r == 'n' ? 1 : 0)),
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride((transpose_r == 'n' ? 1 : 0)) : m1_->stride((transpose_r == 'n' ? 0 : 1))),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride((transpose_r == 'n' ? 1 : 0)) : m2_->stride((transpose_r == 'n' ? 0 : 1))),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride((transpose_r == 'n' ? 1 : 0)));
#elif defined(THC_REAL_IS_FLOAT)
  THCudaBlas_Sgemm(state,
                   transpose_m1,
                   transpose_m2,
                   r__->size((transpose_r == 'n' ? 0 : 1)),
                   r__->size((transpose_r == 'n' ? 1 : 0)),
                   m1_->size((transpose_r == 'n' ? 1 : 0)),
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride((transpose_r == 'n' ? 1 : 0)) : m1_->stride((transpose_r == 'n' ? 0 : 1))),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride((transpose_r == 'n' ? 1 : 0)) : m2_->stride((transpose_r == 'n' ? 0 : 1))),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride((transpose_r == 'n' ? 1 : 0)));
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgemm(state,
                   transpose_m1,
                   transpose_m2,
                   r__->size((transpose_r == 'n' ? 0 : 1)),
                   r__->size((transpose_r == 'n' ? 1 : 0)),
                   m1_->size((transpose_r == 'n' ? 1 : 0)),
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride((transpose_r == 'n' ? 1 : 0)) : m1_->stride((transpose_r == 'n' ? 0 : 1))),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride((transpose_r == 'n' ? 1 : 0)) : m2_->stride((transpose_r == 'n' ? 0 : 1))),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride((transpose_r == 'n' ? 1 : 0)));
#endif

  /* free intermediate variables */
  if(m1_ != m1) {
    THCTensor_(free)(state, m1_);
  }

  if(m2_ != m2) {
    THCTensor_(free)(state, m2_);
  }

  if(r__ != r_) {
    THCTensor_(freeCopyTo)(state, r__, r_);
  }
#else
  ERROR_ONLY_FP_TYPES("addmm");
#endif

#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmm(r_, orig_m1, orig_m2, t);
#endif
}

void THCTensor_(addbmm)(THCState *state, THCTensor *result, THCTensor *t,
                        THCTensor *batch1, THCTensor *batch2, scalar_t beta, scalar_t alpha) {
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, t) == 2, 4, "expected 2D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch2) == 3, 7, "expected 3D tensor");

  int64_t batchnum = THCTensor_(size)(state, batch1, 0);
  int64_t m1d1 = THCTensor_(size)(state, batch1, 1);
  int64_t innerdim = THCTensor_(size)(state, batch1, 2);
  int64_t m2d2 = THCTensor_(size)(state, batch2, 2);

  THArgCheck(batchnum == THCTensor_(size)(state, batch2, 0), 7,
      "equal number of batches expected");
  // M is t, as listed in the docs under addbmm
  THArgCheck(m1d1 == THCTensor_(size)(state, t, 0), 6,
      "first dimension must match first dimension of M");
  THArgCheck(m2d2 == THCTensor_(size)(state, t, 1), 7,
      "second dimension must match second dimension of M");
  THArgCheck(innerdim == THCTensor_(size)(state, batch2, 1), 6,
      "second dimension must match first dimension of batch2");

  if (t != result) {
    THCTensor_(resizeAs)(state, result, t);
    if (ScalarConvert<scalar_t, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, result, t);
    }
  }

  THCTensor *slice1 = THCTensor_(new)(state);
  THCTensor *slice2 = THCTensor_(new)(state);
  for (int64_t i=0; i<batchnum; i++) {
    THCTensor_(select)(state, slice1, batch1, 0, i);
    THCTensor_(select)(state, slice2, batch2, 0, i);

    THCTensor_(addmm)(state, result, result, slice1, slice2, beta, alpha);
    beta = ScalarConvert<int, scalar_t>::to(1);
  }
  THCTensor_(free)(state, slice1);
  THCTensor_(free)(state, slice2);
#else
  ERROR_ONLY_FP_TYPES("addbmm");
#endif
}

__global__ void createBatchGemmBuffer(const scalar_t** buffer, scalar_t* data,
                                      int64_t stride, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer[idx] = data + idx * stride;
  }
}

__global__ void createBatchGemmBuffer3(const scalar_t** buffer1, const scalar_t ** buffer2, const scalar_t ** buffer3, scalar_t* data1,
                                       scalar_t * data2, scalar_t * data3, int64_t stride1, int64_t stride2, int64_t stride3, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer1[idx] = data1 + idx * stride1;
    buffer2[idx] = data2 + idx * stride2;
    buffer3[idx] = data3 + idx * stride3;
  }
}

void THCTensor_(baddbmm)(THCState *state, THCTensor *result, THCTensor *t,
                         THCTensor *batch1, THCTensor *batch2,
                         scalar_t beta, scalar_t alpha) {
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch2, 0), 7,
             "equal number of batches expected");
#ifdef BUILD_NAMEDTENSOR
  auto outnames = at::namedinference::compute_baddbmm_outnames(result, batch1, batch2, t);
  {
    at::NoNamesGuard guard;
#endif
  THArgCheck(THCTensor_(size)(state, t, 1) == THCTensor_(size)(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, t, 2) == THCTensor_(size)(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, batch1, 2) == THCTensor_(size)(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THCTensor_(resizeAs)(state, result, t);
    if (ScalarConvert<scalar_t, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, result, t);
    }
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  THCTensor *result_, *batch1_, *batch2_;
  if (result->stride(1) == 1)
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride(2);
  }
  else if (result->stride(2) == 1)
  {
    transpose_result = true;

    THCTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride(1);
  }
  else
  {
    transpose_result = false;

    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, result, 1, 2);
    result_ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, result_, NULL, 1, 2);

    ldc = result_->stride(2);
  }

  if (batch1->stride(transpose_result ? 2 : 1) == 1 &&
   batch1->stride(transpose_result ? 1 : 2) != 0)
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride(transpose_result ? 1 : 2);
  }
  else if (batch1->stride(transpose_result ? 1 : 2) == 1 &&
   batch1->stride(transpose_result ? 2 : 1) != 0)
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride(transpose_result ? 2 : 1);
  }
  else
  {
    transpose_batch1 = transpose_result ? 'n' : 't';
    // batch1_ is later freed if batch1_ != batch1
    if (THCTensor_(isContiguous)(state, batch1)) {
      batch1_ = batch1;
    } else {
      batch1_ = THCTensor_(newContiguous)(state, batch1);
    }
    lda = batch1_->stride(1);
  }

  if (batch2->stride(transpose_result ? 2 : 1) == 1 &&
   batch2->stride(transpose_result ? 1 : 2) != 0)
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride(transpose_result ? 1 : 2);
  }
  else if (batch2->stride(transpose_result ? 1 : 2) == 1 &&
   batch2->stride(transpose_result ? 2 : 1) != 0)
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride(transpose_result ? 2 : 1);
  }
  else
  {
    transpose_batch2 = transpose_result ? 'n' : 't';
    // batch2_ is later freed if batch2_ != batch2
    if (THCTensor_(isContiguous)(state, batch2)) {
      batch2_ = batch2;
    } else {
      batch2_ = THCTensor_(newContiguous)(state, batch2);
    }
    ldb = batch2_->stride(1);
  }
  int64_t num_batches = result_->size(0);

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  // Compute pointers to matrices in each batch.
#if CUDA_VERSION < 8000 && !defined __HIP_PLATFORM_HCC__
  size_t matrices_size = num_batches * sizeof(scalar_t*);

//   Copy pointers to device.
  auto d_matrices1 = static_cast<const scalar_t**>(THCudaMalloc(state, matrices_size));
  auto d_matrices2 = static_cast<const scalar_t**>(THCudaMalloc(state, matrices_size));
  auto d_result_matrices = static_cast<scalar_t**>(THCudaMalloc(state, matrices_size));

  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;

  createBatchGemmBuffer3<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    d_matrices1, d_matrices2, (const scalar_t**)d_result_matrices, THCTensor_(data)(state, batch1_),
    THCTensor_(data)(state, batch2_), THCTensor_(data)(state, result_),
    batch1_->stride(0), batch2_->stride(0), result_->stride(0), num_batches);

#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_SgemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_DgemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);
#endif //THC_REAL

  THCudaFree(state, d_matrices1);
  THCudaFree(state, d_matrices2);
  THCudaFree(state, d_result_matrices);

#else
#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_SgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_DgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#endif //THC_REAL
#endif //CUDA_VERSION

#elif defined(THC_REAL_IS_HALF)

#if CUDA_VERSION < 9010
  // Currently no HgemmBatched in Cublas
  for (int64_t i = 0; i < num_batches; ++i) {
    THCudaBlas_Hgemm(
        state,
        transpose_batch1,
        transpose_batch2,
        result_->size(transpose_result ? 2 : 1),
        result_->size(transpose_result ? 1 : 2),
        batch1_->size(transpose_result ? 1 : 2),
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride(0), lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride(0), ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride(0), ldc);
  }
#else
#ifndef __HIP_PLATFORM_HCC__
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5){
#endif

  THCudaBlas_HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#ifndef __HIP_PLATFORM_HCC__
   } else {
      for (int64_t i = 0; i < num_batches; ++i) {
        THCudaBlas_Hgemm(
        state,
        transpose_batch1,
        transpose_batch2,
        result_->size(transpose_result ? 2 : 1),
        result_->size(transpose_result ? 1 : 2),
        batch1_->size(transpose_result ? 1 : 2),
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride(0), lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride(0), ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride(0), ldc);
      }
   }
#endif

#endif
#endif
  if (batch1_ != batch1) {
    THCTensor_(free)(state, batch1_);
  }

  if (batch2_ != batch2) {
    THCTensor_(free)(state, batch2_);
  }

  if (result_ != result) {
    THCTensor_(freeCopyTo)(state, result_, result);
  }
#ifdef BUILD_NAMEDTENSOR
  }
  at::namedinference::propagate_names(result, std::move(outnames), /*validate_names=*/false);
#endif

#else
  ERROR_ONLY_FP_TYPES("baddbmm");
#endif
}

#endif
