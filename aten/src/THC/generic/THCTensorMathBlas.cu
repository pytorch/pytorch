#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathBlas.cu"
#else

#define ERROR_ONLY_FP_TYPES(func) \
  THError("%s for CUDA tensors only supports floating-point types. Try converting the tensors with .float()", func);

THC_API accreal
THCTensor_(dot)(THCState *state, THCTensor *self, THCTensor *src)
{
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
  accreal result = ScalarConvert<half, accreal>::to(
                   THCudaBlas_Hdot(state,
                                THCTensor_(nElement)(state, self),
                                THCTensor_(data)(state, self), 1,
                                THCTensor_(data)(state, src), 1));
#endif

  THCTensor_(free)(state, src);
  THCTensor_(free)(state, self);
  return result;

#else
  ERROR_ONLY_FP_TYPES("dot");
  return ScalarConvert<int, accreal>::to(0);
#endif
}

THC_API void
THCTensor_(addmv)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCTensor *mat, THCTensor *vec)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, mat, vec));
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  if(r_ != t)
  {
    THCTensor_(resizeAs)(state, r_, t);
    THCTensor_(copy)(state, r_, t);
  }

  if(mat->stride[0] == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, THCTensor_(data)(state, mat), mat->stride[1],
                    THCTensor_(data)(state, vec), vec->stride[0],
                    beta, THCTensor_(data)(state, r_), r_->stride[0]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, THCTensor_(data)(state, mat), mat->stride[1],
                    THCTensor_(data)(state, vec), vec->stride[0],
                    beta, THCTensor_(data)(state, r_), r_->stride[0]);
#endif
  }
  else if(mat->stride[1] == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCTensor_(data)(state, mat), mat->stride[0],
                    THCTensor_(data)(state, vec), vec->stride[0],
                    beta, THCTensor_(data)(state, r_), r_->stride[0]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 't',  mat->size[1], mat->size[0],
                     alpha, THCTensor_(data)(state, mat), mat->stride[0],
                     THCTensor_(data)(state, vec), vec->stride[0],
                     beta, THCTensor_(data)(state, r_), r_->stride[0]);
#endif
  }
  else
  {
    THCTensor *cmat = THCTensor_(newContiguous)(state, mat);

#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCTensor_(data)(state, cmat), cmat->stride[0],
                    THCTensor_(data)(state, vec), vec->stride[0],
                    beta, THCTensor_(data)(state, r_), r_->stride[0]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCTensor_(data)(state, cmat), cmat->stride[0],
                    THCTensor_(data)(state, vec), vec->stride[0],
                    beta, THCTensor_(data)(state, r_), r_->stride[0]);
#endif

    THCTensor_(free)(state, cmat);
  }

#elif defined(THC_REAL_IS_HALF)
    // Currently no Hgemv/SgemvEx in Cublas
    THCTensor *vecAsMatrix = THCTensor_(newWithTensor)(state, vec);
    THCTensor_(resize2d)(state, vecAsMatrix, vecAsMatrix->size[0], 1);

    THCTensor *tAsMatrix = THCTensor_(newWithTensor)(state, t);
    THCTensor_(resize2d)(state, tAsMatrix, tAsMatrix->size[0], 1);

    THCTensor_(addmm)(state, r_, beta, tAsMatrix, alpha, mat, vecAsMatrix);

    // r_ will have answer as matrix, need to return a vector
    THCTensor_(resize1d)(state, r_, r_->size[0]);
    THCTensor_(free)(state, vecAsMatrix);
    THCTensor_(free)(state, tAsMatrix);
#endif
#else
  ERROR_ONLY_FP_TYPES("addmv");
#endif
}

THC_API void
THCTensor_(addr)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCTensor *vec1, THCTensor *vec2)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, vec1, vec2));
  if ( (vec1->nDimension != 1) || (vec2->nDimension != 1) ) {
    THError("vector and vector expected");
  }

  if (t->nDimension != 2) {
    THError("size mismatch");
  }

  if ( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) ) {
    THError("size mismatch");
  }

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  if (r_ != t) {
    THCTensor_(resizeAs)(state, r_, t);
    THCTensor_(copy)(state, r_, t);
  }

  if(THCNumerics<real>::eq(beta, ScalarConvert<int, real>::to(0))) {
    THCTensor_(zero)(state, r_);
  } else if(THCNumerics<real>::ne(beta, ScalarConvert<int, real>::to(1))) {
    THCTensor_(mul)(state, r_, r_, beta);
  }

  if(r_->stride[0] == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec1->size[0], vec2->size[0],
                   alpha, THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, r_), r_->stride[1]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec1->size[0], vec2->size[0],
                   alpha, THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, r_), r_->stride[1]);
#endif
  }
  else if(r_->stride[1] == 1)
  {
#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec2->size[0], vec1->size[0],
                   alpha, THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, r_), r_->stride[0]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec2->size[0], vec1->size[0],
                   alpha, THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, r_), r_->stride[0]);
#endif
  }
  else
  {
    THCTensor *cr = THCTensor_(newClone)(state, r_);

#ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sger(state, vec2->size[0], vec1->size[0],
                   alpha, THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, cr), cr->stride[0]);
#elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dger(state, vec2->size[0], vec1->size[0],
                   alpha, THCTensor_(data)(state, vec2), vec2->stride[0],
                   THCTensor_(data)(state, vec1), vec1->stride[0],
                   THCTensor_(data)(state, cr), cr->stride[0]);
#endif

    THCTensor_(freeCopyTo)(state, cr, r_);
  }
#elif defined(THC_REAL_IS_HALF)
  // currently no Hger/SgerEx in Cublas.
  THCTensor *vec2T = THCTensor_(newWithTensor)(state, vec2);
  THCTensor_(resize2d)(state, vec2T, vec2T->size[0], 1);
  THCTensor_(transpose)(state, vec2T, NULL, 0, 1);

  THCTensor *vec1M = THCTensor_(newWithTensor)(state, vec1);
  THCTensor_(resize2d)(state, vec1M, vec1M->size[0], 1);

  THCTensor_(addmm)(state, r_, beta, t, alpha, vec1M, vec2T);
  THCTensor_(free)(state, vec2T);
  THCTensor_(free)(state, vec1M);
#endif
#else
  ERROR_ONLY_FP_TYPES("addr");
#endif
}

THC_API void
THCTensor_(addmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCTensor *m1, THCTensor *m2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, r_, t, m1, m2));
  char transpose_r, transpose_m1, transpose_m2;
  THCTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

  if(t->nDimension != 2)
    THError("matrix expected, got %dD tensor for t", t->nDimension);

  if(m1->size[1] != m2->size[0]) {
    THCDescBuff bm1 = THCTensor_(sizeDesc)(state, m1);
    THCDescBuff bm2 = THCTensor_(sizeDesc)(state, m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) ) {
    THCDescBuff bt  = THCTensor_(sizeDesc)(state, t);
    THCDescBuff bm1 = THCTensor_(sizeDesc)(state, m1);
    THCDescBuff bm2 = THCTensor_(sizeDesc)(state, m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THCTensor_(resizeAs)(state, r_, t);
    if (ScalarConvert<real, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, r_, t);
    }
  }

  /* r_ */
  if(r_->stride[0] == 1 &&
     r_->stride[1] != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1 &&
          r_->stride[0] != 0)
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
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
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
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
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
                   r__->size[(transpose_r == 'n' ? 0 : 1)],
                   r__->size[(transpose_r == 'n' ? 1 : 0)],
                   m1_->size[(transpose_r == 'n' ? 1 : 0)],
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride[(transpose_r == 'n' ? 1 : 0)]);
#elif defined(THC_REAL_IS_FLOAT)
  THCudaBlas_Sgemm(state,
                   transpose_m1,
                   transpose_m2,
                   r__->size[(transpose_r == 'n' ? 0 : 1)],
                   r__->size[(transpose_r == 'n' ? 1 : 0)],
                   m1_->size[(transpose_r == 'n' ? 1 : 0)],
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride[(transpose_r == 'n' ? 1 : 0)]);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgemm(state,
                   transpose_m1,
                   transpose_m2,
                   r__->size[(transpose_r == 'n' ? 0 : 1)],
                   r__->size[(transpose_r == 'n' ? 1 : 0)],
                   m1_->size[(transpose_r == 'n' ? 1 : 0)],
                   alpha,
                   THCTensor_(data)(state, m1_),
                   (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   THCTensor_(data)(state, m2_),
                   (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   beta,
                   THCTensor_(data)(state, r__),
                   r__->stride[(transpose_r == 'n' ? 1 : 0)]);
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
}

THC_API void
THCTensor_(addbmm)(THCState *state, THCTensor *result, real beta, THCTensor *t,
                   real alpha, THCTensor *batch1, THCTensor *batch2) {
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THCTensor_(nDimension)(state, t) == 2, 4, "expected 2D tensor");
  THArgCheck(THCTensor_(nDimension)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimension)(state, batch2) == 3, 7, "expected 3D tensor");

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
    if (ScalarConvert<real, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, result, t);
    }
  }

  THCTensor *slice1 = THCTensor_(new)(state);
  THCTensor *slice2 = THCTensor_(new)(state);
  for (int64_t i=0; i<batchnum; i++) {
    THCTensor_(select)(state, slice1, batch1, 0, i);
    THCTensor_(select)(state, slice2, batch2, 0, i);

    THCTensor_(addmm)(state, result, beta, result, alpha, slice1, slice2);
    beta = ScalarConvert<int, real>::to(1);
  }
  THCTensor_(free)(state, slice1);
  THCTensor_(free)(state, slice2);
#else
  ERROR_ONLY_FP_TYPES("addbmm");
#endif
}

__global__ void createBatchGemmBuffer(const real** buffer, real* data,
                                      int64_t stride, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer[idx] = data + idx * stride;
  }
}

__global__ void createBatchGemmBuffer3(const real** buffer1, const real ** buffer2, const real ** buffer3, real* data1,
                                       real * data2, real * data3, int64_t stride1, int64_t stride2, int64_t stride3, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer1[idx] = data1 + idx * stride1;
    buffer2[idx] = data2 + idx * stride2;
    buffer3[idx] = data3 + idx * stride3;
  }
}

THC_API void
THCTensor_(baddbmm)(THCState *state, THCTensor *result, real beta, THCTensor *t,
                    real alpha, THCTensor *batch1, THCTensor *batch2) {
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THCTensor_(nDimension)(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimension)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimension)(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch2, 0), 7,
             "equal number of batches expected");
  THArgCheck(THCTensor_(size)(state, t, 1) == THCTensor_(size)(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, t, 2) == THCTensor_(size)(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, batch1, 2) == THCTensor_(size)(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THCTensor_(resizeAs)(state, result, t);
    if (ScalarConvert<real, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, result, t);
    }
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  THCTensor *result_, *batch1_, *batch2_;
  if (result->stride[1] == 1)
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride[2];
  }
  else if (result->stride[2] == 1)
  {
    transpose_result = true;

    THCTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride[1];
  }
  else
  {
    transpose_result = false;

    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, result, 1, 2);
    result_ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, result_, NULL, 1, 2);

    ldc = result_->stride[2];
  }

  if (batch1->stride[transpose_result ? 2 : 1] == 1 &&
   batch1->stride[transpose_result ? 1 : 2] != 0)
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 1 : 2];
  }
  else if (batch1->stride[transpose_result ? 1 : 2] == 1 &&
   batch1->stride[transpose_result ? 2 : 1] != 0)
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 2 : 1];
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
    lda = batch1_->stride[1];
  }

  if (batch2->stride[transpose_result ? 2 : 1] == 1 &&
   batch2->stride[transpose_result ? 1 : 2] != 0)
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 1 : 2];
  }
  else if (batch2->stride[transpose_result ? 1 : 2] == 1 &&
   batch2->stride[transpose_result ? 2 : 1] != 0)
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 2 : 1];
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
    ldb = batch2_->stride[1];
  }
  int64_t num_batches = result_->size[0];

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  // Compute pointers to matrices in each batch.
#if CUDA_VERSION < 8000
  size_t matrices_size = num_batches * sizeof(real*);

//   Copy pointers to device.
  const real **d_matrices1, **d_matrices2;
  real **d_result_matrices;
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices1, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices2, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_result_matrices, matrices_size));

  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;

  createBatchGemmBuffer3<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    d_matrices1, d_matrices2, (const real**)d_result_matrices, THCTensor_(data)(state, batch1_),
    THCTensor_(data)(state, batch2_), THCTensor_(data)(state, result_),
    batch1_->stride[0], batch2_->stride[0], result_->stride[0], num_batches);

#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_SgemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
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
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
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
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride[0],
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride[0],
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride[0],
      num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_DgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride[0],
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride[0],
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride[0],
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
        result_->size[transpose_result ? 2 : 1],
        result_->size[transpose_result ? 1 : 2],
        batch1_->size[transpose_result ? 1 : 2],
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride[0], lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride[0], ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride[0], ldc);
  }
#else
  cudaDeviceProp* prop = THCState_getCurrentDeviceProperties(state);
  if (prop->major >= 5){

  THCudaBlas_HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride[0],
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride[0],
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride[0],
      num_batches);
   } else {
      for (int64_t i = 0; i < num_batches; ++i) {
        THCudaBlas_Hgemm(
        state,
        transpose_batch1,
        transpose_batch2,
        result_->size[transpose_result ? 2 : 1],
        result_->size[transpose_result ? 1 : 2],
        batch1_->size[transpose_result ? 1 : 2],
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride[0], lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride[0], ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride[0], ldc);
      }
   }

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

#else
  ERROR_ONLY_FP_TYPES("baddbmm");
#endif
}

THC_API void THCTensor_(btrifact)(THCState *state, THCTensor *ra_, THCudaIntTensor *rpivots_, THCudaIntTensor *rinfo_, int pivot, THCTensor *a)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THAssert(THCTensor_(checkGPU)(state, 2, ra_, a));
  THArgCheck(THCTensor_(nDimension)(state, a) == 3, 3, "expected 3D tensor");
  THArgCheck(THCTensor_(size)(state, a, 1) ==
             THCTensor_(size)(state, a, 2), 3, "matrices must be square");

  if (ra_ != a) {
    THCTensor_(resizeAs)(state, ra_, a);
    // not sure if this is kosher, but things are nicer if we return in column major
    if (ra_->stride[0] == 1) {
      THCTensor_(transpose)(state, ra_, NULL, 1, 0);
    } else if (ra_->stride[2] == 1) {
      THCTensor_(transpose)(state, ra_, NULL, 1, 2);
    }
    THCTensor_(copy)(state, ra_, a);
  }


  int n = a->size[1];
  int lda;
  THCTensor *ra__;

  if (ra_->stride[1] == 1) {
    // column ordered, what BLAS wants
    lda = ra_->stride[2];
    ra__ = ra_;
  } else {
    // not column ordered, need to make it such (requires copy)
    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, ra_, 1, 2);
    ra__ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, ra__, NULL, 1, 2);
    lda = ra__->stride[2];
  }

  int64_t num_batches = ra__->size[0];

  if (!pivot) {
    THCudaIntTensor *t = THCudaIntTensor_new(state);
    THCudaIntTensor_range(state, t, 1, n, 1);
    THCudaIntTensor_unsqueeze1d(state, t, t, 0);
    THCudaIntTensor** ptrs = (THCudaIntTensor**) THAlloc(sizeof(THCudaIntTensor*)*num_batches);
    for (int64_t i=0; i<num_batches; i++) {
      ptrs[i] = t;
    }
    THCudaIntTensor_catArray(state, rpivots_, ptrs, num_batches, 0);
    THCudaIntTensor_free(state, t);
    THFree(ptrs);
  } else {
    THCudaIntTensor_resize2d(state, rpivots_, num_batches, n);
  }

  bool free_rinfo_ = !rinfo_;
  if (rinfo_ == NULL) rinfo_ = THCudaIntTensor_new(state);
  THCudaIntTensor_resize1d(state, rinfo_, num_batches);
  int *info_gpu = THCudaIntTensor_data(state, rinfo_);

  // Copy pointers to device.
  real **d_result;
  size_t matrices_size = num_batches * sizeof(real*);
  THCudaCheck(THCudaMalloc(state, (void**)&d_result, matrices_size));

  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;
  createBatchGemmBuffer<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    (const real**)d_result, THCTensor_(data)(state, ra__),
    ra__->stride[0], num_batches);

  int *pivots_gpu = NULL;
  if (pivot) {
    pivots_gpu = THCudaIntTensor_data(state, rpivots_);
  }
#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_Sgetrf(state, n, d_result, lda, pivots_gpu, info_gpu, num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgetrf(state, n, d_result, lda, pivots_gpu, info_gpu, num_batches);
#endif

  THCudaFree(state, d_result);

  if (ra__ != ra_) {
    THCTensor_(freeCopyTo)(state, ra__, ra_);
  }

  if (free_rinfo_) {
    int min = THCudaIntTensor_minall(state, rinfo_);
    int max = THCudaIntTensor_maxall(state, rinfo_);
    THCudaIntTensor_free(state, rinfo_);
    if (min != 0 || max != 0) {
      THError("failed to factorize some batch elements (min info == %d, max info == %d)",
              min, max);
    }
  }

#else
  THError("btrifact for CUDA tensors is only supported for floats and doubles");
#endif
}


THC_API void THCTensor_(btrisolve)(THCState *state, THCTensor *rb_, THCTensor *b,
                              THCTensor *atf, THCudaIntTensor *pivots)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THAssert(THCTensor_(checkGPU)(state, 3, rb_, atf, b));
  THArgCheck(THCTensor_(nDimension)(state, atf) == 3, 3, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimension)(state, b) == 3 ||
             THCTensor_(nDimension)(state, b) == 2, 4, "expected 2D or 3D tensor");
  THArgCheck(THCTensor_(size)(state, atf, 0) ==
             THCTensor_(size)(state, b, 0), 3, "number of batches must be equal");
  THArgCheck(THCTensor_(size)(state, atf, 1) ==
             THCTensor_(size)(state, atf, 2), 3, "A matrices must be square");
  THArgCheck(THCTensor_(size)(state, atf, 1) ==
             THCTensor_(size)(state, b, 1), 3, "dimensions of A and b must be equal");

  if (rb_ != b) {
    THCTensor_(resizeAs)(state, rb_, b);
    THCTensor_(copy)(state, rb_, b);
  }


  int n = atf->size[1];
  int nrhs = rb_->nDimension > 2 ? rb_->size[2] : 1;
  THCTensor *atf_;
  THCTensor *rb__;
  int lda, ldb;

  // correct ordering of A_tf
  if (atf->stride[1] == 1) {
    // column ordered, what BLAS wants
    lda = atf->stride[2];
    atf_ = atf;
  } else {
    // not column ordered, need to make it such (requires copy)
    // it would be nice if we could use the op(A) flags to automatically
    // transpose A if needed, but this leads to unpredictable behavior if the
    // user clones A_tf later with a different ordering
    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, atf, 1, 2);
    atf_ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, atf_, NULL, 1, 2);
    lda = atf_->stride[2];
  }

  // correct ordering of B
  if (rb_->stride[1] == 1) {
    // column ordered
    if (rb_->nDimension == 2 || rb_->size[2] == 1) {
      ldb = n;
    } else {
      ldb = rb_->stride[2];
    }
    rb__ = rb_;
  } else {
    // make column ordered
    if (rb_->nDimension > 2) {
      THCTensor *transp_r_ = THCTensor_(newTranspose)(state, rb_, 1, 2);
      rb__ = THCTensor_(newClone)(state, transp_r_);
      THCTensor_(free)(state, transp_r_);
      THCTensor_(transpose)(state, rb__, NULL, 1, 2);
      ldb = rb__->stride[2];
    } else {
      rb__ = THCTensor_(newClone)(state, rb_);
      ldb = n;
    }
  }

  int64_t num_batches = rb_->size[0];
  size_t matrices_size = num_batches * sizeof(real*);

  // Copy pointers to device.
  real **d_result;
  const real **d_atf;
  THCudaCheck(THCudaMalloc(state, (void**)&d_result, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_atf, matrices_size));

  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;
  createBatchGemmBuffer<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    (const real**)d_result, THCTensor_(data)(state, rb__),
    rb__->stride[0], num_batches);
  createBatchGemmBuffer<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    d_atf, THCTensor_(data)(state, atf_),
    atf_->stride[0], num_batches);

  if (!THCudaIntTensor_isContiguous(state, pivots)) {
      THError("Error: pivots is not contiguous.");
  }

  int *pivots_data = THCudaIntTensor_data(state, pivots);
  int info;

#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_Sgetrs(state, 'n', n, nrhs, d_atf, lda, pivots_data, d_result, ldb, &info, num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgetrs(state, 'n', n, nrhs, d_atf, lda, pivots_data, d_result, ldb, &info, num_batches);
#endif

  if (info < 0) {
    THError("Illegal arg %d", -info);
  }

  THCudaFree(state, d_result);
  THCudaFree(state, d_atf);

  if (atf_ != atf) {
    THCTensor_(free)(state, atf_);
  }

  if (rb__ != rb_) {
    THCTensor_(freeCopyTo)(state, rb__, rb_);
  }

#else
  THError("btrisolve for CUDA tensors is only supported for floats and doubles");
#endif
}

#endif
