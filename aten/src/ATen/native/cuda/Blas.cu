#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>

namespace at { namespace native {

Tensor &addmv_impl_cuda(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta_, Scalar alpha_) {
  auto r_stride = result.stride(0);
  auto vec_size = vec.size(0);
  auto vec_stride = vec.stride(0);

  if (mat.scalar_type() == kHalf || mat.scalar_type() == kBFloat16) {
    // Currently no Hgemv/SgemvEx in Cublas
    Tensor vec_as_matrix = vec.reshape({vec_size, 1}).contiguous();
    Tensor self_as_matrix = self.reshape({mat.size(0), 1}).contiguous();
    at::addmm_out(result, self_as_matrix, mat, vec_as_matrix, beta_, alpha_);
    result.resize_({result.size(0)});
    return result;
  }

  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES(mat.scalar_type(), "addmv_impl_cuda", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    if (mat.stride(0) == 1) {
      at::cuda::blas::gemv<scalar_t>('n',
        mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1), vec.data_ptr<scalar_t>(),
        vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1) {
      at::cuda::blas::gemv<scalar_t>('t',
        mat.size(1), mat.size(0), alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
        vec.data_ptr<scalar_t>(), vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else {
      Tensor cmat = mat.contiguous();
      at::cuda::blas::gemv<scalar_t>('t',
          mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }

    // In cublasSgemv, cublasDgemv (x,0).mv(0) does not
    // handle beta, whereas cublasSgemm, cublasDgemm do for case where (x,0).mm(0,y).
    if (vec.size(0) == 0 && mat.size(0) != 0) {
      if (beta == scalar_t(0)) {
        result.zero_();
      } else if (beta != scalar_t(1)) {
        result.mul_(beta);
      }
    }
  });
  return result;
}

}} // namespace at::native
