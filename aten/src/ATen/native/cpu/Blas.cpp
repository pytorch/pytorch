#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>

namespace at { namespace native {

template<typename scalar_t>
bool gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

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
