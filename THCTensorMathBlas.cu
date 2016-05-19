#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

float THCudaTensor_dot(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  THArgCheck(THCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, src), 2, "sizes do not match");

  {
    self = THCudaTensor_newContiguous(state, self);
    src = THCudaTensor_newContiguous(state, src);

    float result = THCudaBlas_dot(state,
                                  THCudaTensor_nElement(state, self),
                                  THCudaTensor_data(state, self), 1,
                                  THCudaTensor_data(state, src), 1);
    THCudaTensor_free(state, src);
    THCudaTensor_free(state, self);

    return result;
  }
}

void THCudaTensor_addmv(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  THAssert(THCudaTensor_checkGPU(state, 4, r_, t, mat, vec));
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THCudaBlas_gemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, THCudaTensor_data(state, mat), mat->stride[1],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(state, mat), mat->stride[0],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cmat = THCudaTensor_newContiguous(state, mat);

    THCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(state, cmat), cmat->stride[0],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);

    THCudaTensor_free(state, cmat);
  }
}

void THCudaTensor_addmm(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *m1, THCudaTensor *m2)
{
  THAssert(THCudaTensor_checkGPU(state, 4, r_, t, m1, m2));
  char transpose_r, transpose_m1, transpose_m2;
  THCudaTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
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
    THCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    THCudaTensor *transp_r_ = THCudaTensor_newTranspose(state, r_, 0, 1);
    r__ = THCudaTensor_newClone(state, transp_r_);
    THCudaTensor_free(state, transp_r_);
    THCudaTensor_transpose(state, r__, NULL, 0, 1);
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
    m1_ = THCudaTensor_newContiguous(state, m1);
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
    m2_ = THCudaTensor_newContiguous(state, m2);
  }

  /* do the operation */
  THCudaBlas_gemm(state,
                  transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  THCudaTensor_data(state, m1_),
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  THCudaTensor_data(state, m2_),
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  THCudaTensor_data(state, r__),
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THCudaTensor_free(state, m1_);

  if(m2_ != m2)
    THCudaTensor_free(state, m2_);

  if(r__ != r_)
    THCudaTensor_freeCopyTo(state, r__, r_);
}

void THCudaTensor_addr(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  THAssert(THCudaTensor_checkGPU(state, 4, r_, t, vec1, vec2));
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
  }

  if(beta != 1)
    THCudaTensor_mul(state, r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THCudaBlas_ger(state, vec1->size[0], vec2->size[0],
                   alpha, THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cr = THCudaTensor_newClone(state, r_);

    THCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, cr), cr->stride[0]);

    THCudaTensor_freeCopyTo(state, cr, r_);
  }
}

void THCudaTensor_addbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
    float alpha, THCudaTensor *batch1, THCudaTensor *batch2) {
  THAssert(THCudaTensor_checkGPU(state, 4, result, t, batch1, batch2));
  THArgCheck(THCudaTensor_nDimension(state, t) == 2, 4, "expected 2D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch2) == 3, 7, "expected 3D tensor");

  long batchnum = THCudaTensor_size(state, batch1, 0);
  long m1d1 = THCudaTensor_size(state, batch1, 1);
  long innerdim = THCudaTensor_size(state, batch1, 2);
  long m2d2 = THCudaTensor_size(state, batch2, 2);

  THArgCheck(batchnum == THCudaTensor_size(state, batch2, 0), 7,
      "equal number of batches expected");
  // M is t, as listed in the docs under addbmm
  THArgCheck(m1d1 == THCudaTensor_size(state, t, 0), 6,
      "first dimension must match first dimension of M");
  THArgCheck(m2d2 == THCudaTensor_size(state, t, 1), 7,
      "second dimension must match second dimension of M");
  THArgCheck(innerdim == THCudaTensor_size(state, batch2, 1), 6,
      "second dimension must match first dimension of batch2");

  if (t != result) {
    THCudaTensor_resizeAs(state, result, t);
    THCudaTensor_copy(state, result, t);
  }

  THCudaTensor *slice1 = THCudaTensor_new(state);
  THCudaTensor *slice2 = THCudaTensor_new(state);
  for (long i=0; i<batchnum; i++) {
    THCudaTensor_select(state, slice1, batch1, 0, i);
    THCudaTensor_select(state, slice2, batch2, 0, i);

    THCudaTensor_addmm(state, result, beta, result, alpha, slice1, slice2);
    beta = 1;
  }
  THCudaTensor_free(state, slice1);
  THCudaTensor_free(state, slice2);
}

void THCudaTensor_baddbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
                          float alpha, THCudaTensor *batch1, THCudaTensor *batch2) {
  THAssert(THCudaTensor_checkGPU(state, 4, result, t, batch1, batch2));
  THArgCheck(THCudaTensor_nDimension(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THCudaTensor_size(state, t, 0) == THCudaTensor_size(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THCudaTensor_size(state, t, 0) == THCudaTensor_size(state, batch2, 0), 7,
             "equal number of batches expected");
  THArgCheck(THCudaTensor_size(state, t, 1) == THCudaTensor_size(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THCudaTensor_size(state, t, 2) == THCudaTensor_size(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THCudaTensor_size(state, batch1, 2) == THCudaTensor_size(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THCudaTensor_resizeAs(state, result, t);
    THCudaTensor_copy(state, result, t);
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  long lda, ldb, ldc;
  THCudaTensor *result_, *batch1_, *batch2_;
  if (result->stride[1] == 1)
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride[2];
  }
  else if (result->stride[2] == 1)
  {
    transpose_result = true;

    THCudaTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride[1];
  }
  else
  {
    transpose_result = false;

    THCudaTensor *transp_r_ = THCudaTensor_newTranspose(state, result, 1, 2);
    result_ = THCudaTensor_newClone(state, transp_r_);
    THCudaTensor_free(state, transp_r_);
    THCudaTensor_transpose(state, result_, NULL, 1, 2);

    ldc = result_->stride[2];
  }

  if (batch1->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 1 : 2];
  }
  else if (batch1->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch1 = transpose_result ? 'n' : 't';
    batch1_ = THCudaTensor_newContiguous(state, batch1);
    lda = batch1_->stride[1];
  }

  if (batch2->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 1 : 2];
  }
  else if (batch2->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch2 = transpose_result ? 'n' : 't';
    batch2_ = THCudaTensor_newContiguous(state, batch2);
    ldb = batch2_->stride[1];
  }

  // Compute pointers to matrices in each batch.
  long num_batches = result_->size[0];
  size_t matrices_size = num_batches * sizeof(float*);
  const float **matrices1 = (const float **)THAlloc(matrices_size);
  const float **matrices2 = (const float **)THAlloc(matrices_size);
  float **result_matrices = (float **)THAlloc(matrices_size);
  for (int i = 0; i < num_batches; ++i)
  {
    matrices1[i] = THCudaTensor_data(state, batch1_) + i * batch1_->stride[0];
    matrices2[i] = THCudaTensor_data(state, batch2_) + i * batch2_->stride[0];
    result_matrices[i] = THCudaTensor_data(state, result_) + i * result_->stride[0];
  }

  // Copy pointers to device.
  const float **d_matrices1, **d_matrices2;
  float **d_result_matrices;
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices1, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices2, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_result_matrices, matrices_size));

  THCudaCheck(cudaMemcpyAsync(d_matrices1, matrices1, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THCudaCheck(cudaMemcpyAsync(d_matrices2, matrices2, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THCudaCheck(cudaMemcpyAsync(d_result_matrices, result_matrices, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));

  THCudaBlas_gemmBatched(
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

  THCudaFree(state, d_matrices1);
  THCudaFree(state, d_matrices2);
  THCudaFree(state, d_result_matrices);
  THFree(matrices1);
  THFree(matrices2);
  THFree(result_matrices);

  if (batch1_ != batch1)
    THCudaTensor_free(state, batch1_);

  if (batch2_ != batch2)
    THCudaTensor_free(state, batch2_);

  if (result_ != result)
    THCudaTensor_freeCopyTo(state, result_, result);
}
