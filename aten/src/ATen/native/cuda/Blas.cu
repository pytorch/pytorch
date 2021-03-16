#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>

namespace at { namespace native {

Tensor &addmv_impl_cuda(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_) {
  auto r_stride = result.stride(0);
  auto vec_stride = vec.stride(0);

  // Check for contiguity of `vec` and update `vec_stride` accordingly
  const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
  vec_stride = vec_contiguous.stride(0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    if (mat.stride(0) == 1 && mat.stride(1) >= std::max<int64_t>(1, mat.size(0))) {
      at::cuda::blas::gemv<scalar_t>('n',
        mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1), vec_contiguous.data_ptr<scalar_t>(),
        vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1 && mat.stride(0) >= std::max<int64_t>(1, mat.size(1))) {
      at::cuda::blas::gemv<scalar_t>('t',
        mat.size(1), mat.size(0), alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
        vec_contiguous.data_ptr<scalar_t>(), vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else {
      Tensor cmat = mat.contiguous();
      at::cuda::blas::gemv<scalar_t>('t',
          mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
          vec_contiguous.data_ptr<scalar_t>(), vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
  });
  return result;
}

}} // namespace at::native
