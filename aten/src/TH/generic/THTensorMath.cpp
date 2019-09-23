#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.cpp"
#else

#include <ATen/core/EnableNamedTensor.h>
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

void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          rp[i] = tp[i] & sp[i];
        }
      });
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data & *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data & *src_data;);
  }
#endif
}

void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          rp[i] = tp[i] | sp[i];
        }
      });
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data | *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data | *src_data;);
  }
#endif
}

void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)src;
  return THError("cbitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          rp[i] = tp[i] ^ sp[i];
        }
      });
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data ^ *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data ^ *src_data;);
  }
#endif
}

void THTensor_(bitxor)(THTensor *r_, THTensor *t, scalar_t value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitxor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD * 100,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
        rp[i] = tp[i] ^ value;
      }
    });
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data ^ value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#endif
}

void THTensor_(bitor)(THTensor *r_, THTensor *t, scalar_t value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitor is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    scalar_t *tp = t->data<scalar_t>();
    scalar_t *rp = r_->data<scalar_t>();
    at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD * 100,
        [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
        rp[i] = tp[i] | value;
      }
    });
  } else {
    TH_TENSOR_APPLY2_PARALLEL(r_Size, r_Contig, tContig, scalar_t, r_, scalar_t, t, *r__data = *t_data | value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#endif
}

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

void THTensor_(addmm)(THTensor *r_, THTensor *t, THTensor *m1, THTensor *m2, scalar_t beta, scalar_t alpha)
{
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;
  int free_m1 = 0;
  int free_m2 = 0;

#ifdef BUILD_NAMEDTENSOR
  // The logic in this function changes these around so we save a copy of the original
  THTensor* orig_m1 = m1;
  THTensor* orig_m2 = m2;
#endif

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
#ifdef BUILD_NAMEDTENSOR
      at::NoNamesGuard guard;
#endif
      at::Tensor r__wrap = THTensor_wrap(r_);
      at::Tensor t_wrap = THTensor_wrap(t);
      at::native::copy_(r__wrap, t_wrap);
    }
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

#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmm(r_, orig_m1, orig_m2, t);
#endif
}

void THTensor_(addmv)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha)
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

#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmv(r_, mat, vec, t);
#endif
  #undef LDA_COND
}

void THTensor_(addr)(THTensor *r_, THTensor *t, THTensor *vec1, THTensor *vec2, scalar_t beta, scalar_t alpha)
{
  if( (THTensor_nDimension(vec1) != 1) || (THTensor_nDimension(vec2) != 1) )
    THError("vector and vector expected, got %dD, %dD tensors",
        THTensor_nDimension(vec1), THTensor_nDimension(vec2));

  if(t->dim() != 2)
    THError("expected matrix, got %dD tensor for t", t->dim());

  auto vec1_size = THTensor_sizeLegacyNoScalars(vec1, 0);
  auto vec2_size = THTensor_sizeLegacyNoScalars(vec2, 0);
  auto vec1_stride = THTensor_strideLegacyNoScalars(vec1, 0);
  auto vec2_stride = THTensor_strideLegacyNoScalars(vec2, 0);

  if( (t->size(0) != vec1_size) || (t->size(1) != vec2_size) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bv1 = THTensor_(sizeDesc)(vec1);
    THDescBuff bv2 = THTensor_(sizeDesc)(vec2);
    THError("size mismatch, t: %s, vec1: %s, vec2: %s", bt.str, bv1.str, bv2.str);
  }

  if(r_ != t)
  {
#ifdef BUILD_NAMEDTENSOR
    at::NoNamesGuard guard;
#endif
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

// Should wrap if the value (a) has a different sign than the divisor (b), but is not 0.
static inline bool modulo_wrap(scalar_t a, scalar_t b) {
  return (a != 0) && (a < 0) != (b < 0);
}

void THTensor_(cadd)(THTensor *r_, THTensor *t, scalar_t value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size) {
    if (r_Contig && tContig && srcContig) {
      if(r_ == t) {
        THBlas_(axpy)(THTensor_(nElement)(t), value, src->data<scalar_t>(), 1, r_->data<scalar_t>(), 1);
      } else {
        TH_TENSOR_APPLY3_CONTIG(scalar_t, r_, scalar_t, t, scalar_t, src, THVector_(cadd)(r__data, t_data, src_data, value, r__len););
      }
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data + value * *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data + value * *src_data;);
  }
}

void THTensor_(csub)(THTensor *r_, THTensor *t, scalar_t value, THTensor *src)
{
  THTensor_(cadd)(r_, t, -value, src);
}

void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      TH_TENSOR_APPLY3_CONTIG(scalar_t, r_, scalar_t, t, scalar_t, src, THVector_(cmul)(r__data, t_data, src_data, r__len););
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * *src_data;);
  }
}

void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      TH_TENSOR_APPLY3_CONTIG(scalar_t, r_, scalar_t, t, scalar_t, src, THVector_(cdiv)(r__data, t_data, src_data, r__len););
    } else {
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
  } else {
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / *src_data;);
  }
}

void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("clshift is not supported for torch.HalfTensor");
#endif
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_FLOAT)
          rp[i] = tp[i] * powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
          rp[i] = tp[i] * pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
          rp[i] = ((scalar_t) tp[i]) << sp[i];
#else
          rp[i] = ((ureal) tp[i]) << sp[i];
#endif
        }
      });
    } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * powf(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * pow(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((scalar_t)*t_data) << *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((ureal)*t_data) << *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data * pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((scalar_t)*t_data) << *src_data;);
#else
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((ureal)*t_data) << *src_data;);
#endif
  }
}

void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src)
{
#if defined(TH_REAL_IS_HALF)
  return THError("crshift is not supported for torch.HalfTensor");
#endif
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_FLOAT)
          rp[i] = tp[i] / powf(2, sp[i]);
#elif defined(TH_REAL_IS_DOUBLE)
          rp[i] = tp[i] / pow(2, sp[i]);
#elif defined(TH_REAL_IS_BYTE)
          rp[i] = ((scalar_t) tp[i]) >> sp[i];
#else
          rp[i] = ((ureal) tp[i]) >> sp[i];
#endif
        }
      });
    } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / powf(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / pow(2, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((scalar_t)*t_data) >> *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((ureal)*t_data) >> *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / powf(2, *src_data););
#elif defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data / pow(2, *src_data););
#elif defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((scalar_t)*t_data) >> *src_data;);
#else
      TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = ((ureal)*t_data) >> *src_data;);
#endif
  }
}

void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = fmod(tp[i], sp[i]);
#else
          rp[i] = tp[i] % sp[i];
#endif
        }
      });
    } else {

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig,scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = fmod(*t_data, *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = (*t_data % *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = fmod(*t_data, *src_data););
#else
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = (*t_data % *src_data););
#endif
  }
}

void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      scalar_t *tp = t->data<scalar_t>();
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = r_->data<scalar_t>();
      at::parallel_for(0, r_Size, TH_OMP_OVERHEAD_THRESHOLD,
          [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
  #if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
          rp[i] = (sp[i] == 0)? NAN : tp[i] - sp[i] * floor(tp[i] / sp[i]);
  #else
          // There is no NAN for integers
          rp[i] = tp[i] % sp[i];
          if (modulo_wrap(rp[i], sp[i]))
            rp[i] += sp[i];
  #endif
        }
      });
    } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY3_PARALLEL(r_Size, r_Contig, tContig, srcContig, scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data % *src_data;
                                                    if (modulo_wrap(*r__data, *src_data)) *r__data += *src_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
  } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data););
#else
    // There is no NAN for integers
    TH_TENSOR_APPLY3(scalar_t, r_, scalar_t, t, scalar_t, src, *r__data = *t_data % *src_data;
                                                     if (modulo_wrap(*r__data, *src_data)) *r__data += *src_data;);
#endif

  }
}

void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, scalar_t gain)
{
  int64_t N1 = m1->size(0);
  int64_t N2 = m2->size(0);
  int64_t dim;
  scalar_t *m1_p;
  scalar_t *m2_p;
  scalar_t *r_p;

  THTensor_(resize2d)(r_, N1, N2);

  m1 = THTensor_(newContiguous)(m1);
  m2 = THTensor_(newContiguous)(m2);

  THTensor_(resize2d)(m1, N1, THTensor_(nElement)(m1) / N1);
  THTensor_(resize2d)(m2, N2, THTensor_(nElement)(m2) / N2);

  dim = m1->size(1);
  THArgCheck(m1->size(1) == m2->size(1), 3, "m1 and m2 must have the same inner vector dim");

  m1_p = m1->data<scalar_t>();
  m2_p = m2->data<scalar_t>();
  r_p = r_->data<scalar_t>();

  at::parallel_for(0, N1, 0,
      [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      int64_t j, k;
      for (j = 0; j < N2; j++) {
        scalar_t sum = 0;
        for (k = 0; k < dim; k++) {
          scalar_t term = m1_p[i * dim + k] - m2_p[j * dim + k];
          sum += term * term;
        }
        r_p[i * N2 + j] = gain * sum;
      }
    }
  });

  c10::raw::intrusive_ptr::decref(m1);
  c10::raw::intrusive_ptr::decref(m2);
}

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
