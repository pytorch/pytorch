#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>

namespace at { namespace native {

namespace blas_impl {

template <typename scalar_t>
bool scal_use_fast_path(int64_t n, int64_t incx);

template <typename scalar_t>
bool gemv_use_fast_path(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy);

template <typename scalar_t>
void scal_fast_path(int *n, scalar_t *a, scalar_t *x, int *incx);

template <typename scalar_t>
void gemv_fast_path(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy);

}  // namespace blas_impl

template <typename scalar_t>
inline void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if (n == 1) incx = 1;
  if (blas_impl::scal_use_fast_path<scalar_t>(n, incx)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    blas_impl::scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    return;
  }
  for (int64_t i = 0; i < n; i++) {
    if (a == 0) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
bool gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1) lda = m;

  if (blas_impl::gemv_use_fast_path<scalar_t>(m, n, lda, incx, incy)) {
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    blas_impl::gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return true;
  }

  if ((trans == 'T') || (trans == 't')) {
    for (int64_t i = 0; i < n; i++)
    {
      scalar_t sum = 0;
      scalar_t *row_ = a + lda * i;
      for (int64_t j = 0; j < m; j++) {
        sum += x[j * incx] * row_[j];
      }
      if (beta == 0) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    if (beta != 1) scal<scalar_t>(m, beta, y, incy);

    for (int64_t j = 0; j < n; j++) {
      scalar_t *column_ = a + lda * j;
      scalar_t z = alpha * x[j * incx];
      for (int64_t i = 0; i < m; i++) {
        y[i * incy] += z * column_[i];
      }
    }
  }
  return false;
}

#define INSTANTIATE(scalar_t, _) \
template bool gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
AT_FORALL_SCALAR_TYPES_AND(BFloat16, INSTANTIATE);
#undef INSTANTIATE

namespace {

constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda > std::max<int64_t>(1L, m);
}

void addmv_impl_cpu(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta_, Scalar alpha_) {
  auto r_stride = result.stride(0);
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, mat.scalar_type(), "addmv_impl_cpu", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    bool is_fast = false;
    if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
      is_fast = gemv<scalar_t>('n', mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
      is_fast = gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else {
      Tensor cmat = mat.contiguous();
      is_fast = gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }

    // In THE FAST PATH of gemv (x,0).mv(0) does not handle beta, whereas gemm does for case where (x,0).mm(0,y).
    // But in the naive fall back implementation, this is not the case.
    if (is_fast && vec.size(0) == 0 && mat.size(0) != 0) {
      if (beta == scalar_t(0)) {
        result.zero_();
      } else if (beta != scalar_t(1)) {
        result.mul_(beta);
      }
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(addmv_stub, &addmv_impl_cpu);

}} // namespace at::native

