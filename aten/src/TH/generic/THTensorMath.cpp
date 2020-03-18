#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>
#include <ATen/NamedTensorUtils.h>

// HEY YOU!
//
// Looking for a function which used to be in THTensorMath.cpp, but
// can't find it anymore?  Check THTensorMoreMath.cpp and
// THTensorEvenMoreMath.cpp.  These source files have been split up
// because they were getting too big (a whopping 4669 lines at time
// of writing) and causing MSVC to run out of memory.  Did you come
// here because you saw:
//
//    fatal error C1002: compiler is out of heap space in pass 2
//
// Try splitting up the file some more.
//
// At some point, we should reorganize these files in a way that makes
// sense (rather than just having cut the file down the middle, which is
// what I did when I split these up originally).


#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

static void THTensor_(addmmImpl)(THTensor *r_, THTensor *t, THTensor *m1, THTensor *m2, scalar_t beta, scalar_t alpha)
{
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;
  int free_m1 = 0;
  int free_m2 = 0;

  if( (m1->dim() != 2) || (m2->dim() != 2))
    THError("matrices expected, got %dD, %dD tensors", m1->dim(), m2->dim());

  if(m1->size(1) != m2->size(0)) {
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( t->dim() != 2 )
    THError("matrix expected, got %dD tensor for t", t->dim());

  if( (t->size(0) != m1->size(0)) || (t->size(1) != m2->size(1)) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THTensor_(resizeAs)(r_, t);
    if (beta != 0.0) {
      at::Tensor r__wrap = THTensor_wrap(r_);
      at::Tensor t_wrap = THTensor_wrap(t);
      at::native::copy_(r__wrap, t_wrap);
    }
  }

  if((r_->size(0) == 0) || (r_->size(1) == 0))
  {
    return;
  }

  // n == 1 || ldc >= max(1, m)
  #define LDC_COND(M, N, LDC) ((N) == 1 || (LDC) >= THMax(1, M))

  /* r_ */
  if(r_->stride(0) == 1 &&
     LDC_COND(r_->size(0), r_->size(1), r_->stride(1)))
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride(1) == 1 &&
          LDC_COND(r_->size(1), r_->size(0), r_->stride(0)))
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';
    // make r__ FORTRAN contiguous
    THTensor *transp_r_ = THTensor_(newTranspose)(r_, 0, 1);
    r__ = THTensor_(newClone)(transp_r_);
    c10::raw::intrusive_ptr::decref(transp_r_);
    THTensor_(transpose)(r__, NULL, 0, 1);
  }

  #undef LDC_COND

  int64_t m = r__->size((transpose_r == 'n' ? 0 : 1));
  int64_t n = r__->size((transpose_r == 'n' ? 1 : 0));
  int64_t k = m1->size((transpose_r == 'n' ? 1 : 0));
  int64_t ldr__ = r__->stride((transpose_r == 'n' ? 1 : 0));

  /* m1 */
  /* Need ldm1_ >= max(1, (transpose_m1 == 'n' ? m : k)) */
  if(m1->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m1->stride((transpose_r == 'n' ? 1 : 0)) >= THMax(1, m))
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m1->stride((transpose_r == 'n' ? 0 : 1)) >= THMax(1, k))
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THTensor_(newContiguous)(m1);
    free_m1 = 1;
  }

  /* m2 */
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if(m2->stride((transpose_r == 'n' ? 0 : 1)) == 1 &&
     m2->stride((transpose_r == 'n' ? 1 : 0)) >= THMax(1, k))
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride((transpose_r == 'n' ? 1 : 0)) == 1 &&
          m2->stride((transpose_r == 'n' ? 0 : 1)) >= THMax(1, n))
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THTensor_(newContiguous)(m2);
    free_m2 = 1;
  }

  int64_t ldm1_ = (transpose_m1 == 'n' ? m1_->stride((transpose_r == 'n' ? 1 : 0)) : m1_->stride((transpose_r == 'n' ? 0 : 1)));
  int64_t ldm2_ = (transpose_m2 == 'n' ? m2_->stride((transpose_r == 'n' ? 1 : 0)) : m2_->stride((transpose_r == 'n' ? 0 : 1)));

  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                m,
                n,
                k,
                alpha,
                m1_->data<scalar_t>(),
                ldm1_,
                m2_->data<scalar_t>(),
                ldm2_,
                beta,
                r__->data<scalar_t>(),
                ldr__);

  /* free intermediate variables */
  if(free_m1)
    c10::raw::intrusive_ptr::decref(m1_);

  if(free_m2)
    c10::raw::intrusive_ptr::decref(m2_);

  if(r__ != r_)
    THTensor_(freeCopyTo)(r__, r_);
}

void THTensor_(addmm)(THTensor *r_, THTensor *t, THTensor *m1, THTensor *m2, scalar_t beta, scalar_t alpha) {
  {
    at::NoNamesGuard guard;
    THTensor_(addmmImpl)(r_, t, m1, m2, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmm(r_, m1, m2, t);
}

static void THTensor_(addmvImpl)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha)
{
  if( (mat->dim() != 2) || (THTensor_nDimension(vec) != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->dim(), THTensor_nDimension(vec));

  if( mat->size(1) != THTensor_sizeLegacyNoScalars(vec, 0) ) {
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THDescBuff bv = THTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(THTensor_nDimension(t) != 1)
    THError("vector expected, got t: %dD", t->dim());

  if(THTensor_sizeLegacyNoScalars(t, 0) != mat->size(0)) {
    THDescBuff bt = THTensor_(sizeDesc)(t);
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    at::Tensor r__wrap = THTensor_wrap(r_);
    at::Tensor t_wrap = THTensor_wrap(t);
    at::native::copy_(r__wrap, t_wrap);
  }

  auto r_stride = THTensor_strideLegacyNoScalars(r_, 0);

  // n == 1 || lda >= max(1, m)
  #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

  if(mat->stride(0) == 1 && LDA_COND(mat->size(0), mat->size(1), mat->stride(1)))
  {
    THBlas_(gemv)('n', mat->size(0), mat->size(1),
                  alpha, mat->data<scalar_t>(), mat->stride(1),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);
  }
  else if(mat->stride(1) == 1 && LDA_COND(mat->size(1), mat->size(0), mat->stride(0)))
  {
    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, mat->data<scalar_t>(), mat->stride(0),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, cmat->data<scalar_t>(), cmat->stride(0),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);

    c10::raw::intrusive_ptr::decref(cmat);
  }

  // In gemv (x,0).mv(0) does not
  // handle beta, whereas gemm does for case where (x,0).mm(0,y).
  if (THTensor_sizeLegacyNoScalars(vec, 0) == 0 && mat->size(0) != 0) {
    if (beta == 0) {
      THTensor_(zero)(r_);
    } else if (beta != 1) {
      THTensor_(mul)(r_, r_, beta);
    }
  }

  #undef LDA_COND
}

void THTensor_(addmv)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha) {
  {
    at::NoNamesGuard guard;
    THTensor_(addmvImpl)(r_, t, mat, vec, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmv(r_, mat, vec, t);
}

void THTensor_(addr)(THTensor *r_, THTensor *t, THTensor *vec1, THTensor *vec2, scalar_t beta, scalar_t alpha)
{
  if( (THTensor_nDimension(vec1) != 1) || (THTensor_nDimension(vec2) != 1) )
    THError("vector and vector expected, got %dD, %dD tensors",
        THTensor_nDimension(vec1), THTensor_nDimension(vec2));

  if(t->dim() != 2)
    THError("expected matrix, got %dD tensor for t", t->dim());

  auto vec1_size = THTensor_(size)(vec1, 0);
  auto vec2_size = THTensor_(size)(vec2, 0);
  auto vec1_stride = THTensor_(stride)(vec1, 0);
  auto vec2_stride = THTensor_(stride)(vec2, 0);

  if( (t->size(0) != vec1_size) || (t->size(1) != vec2_size) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bv1 = THTensor_(sizeDesc)(vec1);
    THDescBuff bv2 = THTensor_(sizeDesc)(vec2);
    THError("size mismatch, t: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    at::Tensor r__wrap = THTensor_wrap(r_);
    at::Tensor t_wrap = THTensor_wrap(t);
    at::native::copy_(r__wrap, t_wrap);
  }

  if(beta == 0) {
    THTensor_(zero)(r_);
  }
  else if(beta != 1)
    THTensor_(mul)(r_, r_, beta);

  // n == 1 || lda >= max(1, m)
  #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

  if(r_->stride(0) == 1 && LDA_COND(vec1_size, vec2_size, r_->stride(1)))
  {
    THBlas_(ger)(vec1_size, vec2_size,
                 alpha, vec1->data<scalar_t>(), vec1_stride,
                 vec2->data<scalar_t>(), vec2_stride,
                 r_->data<scalar_t>(), r_->stride(1));
  }
  else if(r_->stride(1) == 1 && LDA_COND(vec2_size, vec1_size, r_->stride(0)))
  {
    THBlas_(ger)(vec2_size, vec1_size,
                 alpha, vec2->data<scalar_t>(), vec2_stride,
                 vec1->data<scalar_t>(), vec1_stride,
                 r_->data<scalar_t>(), r_->stride(0));
  }
  else
  {
    THTensor *cr = THTensor_(newClone)(r_);

    THBlas_(ger)(vec2_size, vec1_size,
                 alpha, vec2->data<scalar_t>(), vec2_stride,
                 vec1->data<scalar_t>(), vec1_stride,
                 cr->data<scalar_t>(), cr->stride(0));

    THTensor_(freeCopyTo)(cr, r_);
  }

  #undef LDA_COND
}

#ifndef TH_REAL_IS_BFLOAT16 /* non bfloat16 only part */

void THTensor_(addbmm)(THTensor *result, THTensor *t, THTensor *batch1, THTensor *batch2, scalar_t beta, scalar_t alpha)
{
  int64_t batch;

  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(batch1) == 3, 1, "expected 3D tensor");
  THArgCheck(THTensor_(nDimensionLegacyNoScalars)(batch2) == 3, 2, "expected 3D tensor");
  THArgCheck(THTensor_(size)(batch1, 0) == THTensor_(size)(batch2, 0), 2,
             "equal number of batches expected, got %d, %d",
             THTensor_(size)(batch1, 0), THTensor_(size)(batch2, 0));
  THArgCheck(THTensor_(size)(batch1, 2) == THTensor_(size)(batch2, 1), 2,
             "wrong matrix size, batch1: %dx%d, batch2: %dx%d",
             THTensor_(size)(batch1, 1), THTensor_(size)(batch1,2),
             THTensor_(size)(batch2, 1), THTensor_(size)(batch2,2));

  int64_t dim1 = THTensor_(size)(batch1, 1);
  int64_t dim2 = THTensor_(size)(batch2, 2);
  THArgCheck(THTensor_(size)(t, 0) == dim1, 1, "output tensor of incorrect size");
  THArgCheck(THTensor_(size)(t, 1) == dim2, 1, "output tensor of incorrect size");

  if (t != result) {
    THTensor_(resizeAs)(result, t);
    if (beta != 0.0) {
      at::Tensor result_wrap = THTensor_wrap(result);
      at::Tensor t_wrap = THTensor_wrap(t);
      at::native::copy_(result_wrap, t_wrap);
    }
  }

  THTensor *matrix1 = THTensor_(new)();
  THTensor *matrix2 = THTensor_(new)();

  for (batch = 0; batch < THTensor_(size)(batch1, 0); ++batch) {
    THTensor_(select)(matrix1, batch1, 0, batch);
    THTensor_(select)(matrix2, batch2, 0, batch);

    THTensor_(addmm)(result, result, matrix1, matrix2, beta, alpha);
    beta = 1; // accumulate output once
  }

  c10::raw::intrusive_ptr::decref(matrix1);
  c10::raw::intrusive_ptr::decref(matrix2);
}

#endif /* !defined(TH_REAL_IS_BOOL) */

#endif /* !defined(TH_REAL_IS_BFLOAT16) */

#endif /* TH_GENERIC_FILE */
