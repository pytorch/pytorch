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
    THTensor_wrap(r_).zero_();
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

#endif /* !defined(TH_REAL_IS_BOOL) */

#endif /* TH_GENERIC_FILE */
