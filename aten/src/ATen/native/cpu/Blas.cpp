#include <type_traits>
#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>

#ifdef USE_BLAS
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
#endif

namespace at { namespace native {

namespace {

void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

#if defined(TH_REAL_IS_DOUBLE)
    dscal_(&i_n, &a, x, &i_incx);
#else
    sscal_(&i_n, &a, x, &i_incx);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++) {
      if (a == 0) {
        x[i*incx] = 0;
      } else {
        x[i*incx] *= a;
      }
    }
  }
}

} // anonymous namespace

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1)
    lda = m;

#ifdef USE_BLAS
  if (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value) {
    if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX) &&
        (incx > 0) && (incx <= INT_MAX) &&
        (incy > 0) && (incy <= INT_MAX) )
    {
      THArgCheck(lda >= THMax(1, m), 6,
      "lda should be at least max(1, m=%d), but have %d", m, lda);
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      if (std::is_same<scalar_t, double>::value) {
        dgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
      } else if (std::is_same<scalar_t, float>::value) {
        sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
      }
      return;
    }
  }
#endif

  {
    int64_t i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        scalar_t sum = 0;
        scalar_t *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
          if (beta == 0)
            y[i*incy] = alpha*sum;
          else
            y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        scal<scalar_t>(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        scalar_t *column_ = a+lda*j;
        scalar_t z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

namespace {

// static void THTensor_(addmvImpl)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha)
// {

//   if(r_ != t)
//   {
//     THTensor_(resizeAs)(r_, t);
//     at::Tensor r__wrap = THTensor_wrap(r_);
//     at::Tensor t_wrap = THTensor_wrap(t);
//     at::native::copy_(r__wrap, t_wrap);
//   }

//   auto r_stride = THTensor_strideLegacyNoScalars(r_, 0);

//   // n == 1 || lda >= max(1, m)
//   #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

//   if(mat->stride(0) == 1 && LDA_COND(mat->size(0), mat->size(1), mat->stride(1)))
//   {
//     THBlas_(gemv)('n', mat->size(0), mat->size(1),
//                   alpha, mat->data<scalar_t>(), mat->stride(1),
//                   vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
//                   beta, r_->data<scalar_t>(), r_stride);
//   }
//   else if(mat->stride(1) == 1 && LDA_COND(mat->size(1), mat->size(0), mat->stride(0)))
//   {
//     THBlas_(gemv)('t',  mat->size(1), mat->size(0),
//                   alpha, mat->data<scalar_t>(), mat->stride(0),
//                   vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
//                   beta, r_->data<scalar_t>(), r_stride);
//   }
//   else
//   {
//     THTensor *cmat = THTensor_(newContiguous)(mat);

//     THBlas_(gemv)('t',  mat->size(1), mat->size(0),
//                   alpha, cmat->data<scalar_t>(), cmat->stride(0),
//                   vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
//                   beta, r_->data<scalar_t>(), r_stride);

//     c10::raw::intrusive_ptr::decref(cmat);
//   }

//   // In gemv (x,0).mv(0) does not
//   // handle beta, whereas gemm does for case where (x,0).mm(0,y).
//   if (THTensor_sizeLegacyNoScalars(vec, 0) == 0 && mat->size(0) != 0) {
//     if (beta == 0) {
//       THTensor_(zero)(r_);
//     } else if (beta != 1) {
//       THTensor_(mul)(r_, r_, beta);
//     }
//   }

//   #undef LDA_COND
// }

void addmv_impl_cpu(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return;
}

} // anonymous namespace

REGISTER_DISPATCH(addmv_stub, &addmv_impl_cpu);

}} // namespace at::native