#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDABlas.h>

namespace at { namespace native {

Tensor &addmv_impl_cuda(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta_, Scalar alpha_) {
  auto r_stride = result.stride(0);
  auto vec_stride = vec.stride(0);

  // Check for contiguity of `vec` and update `vec_stride` accordingly
  const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
  vec_stride = vec_contiguous.stride(0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    if (mat.stride(0) == 1) {
      at::cuda::blas::gemv<scalar_t>('n',
        mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1), vec_contiguous.data_ptr<scalar_t>(),
        vec_stride, beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1) {
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

    // In cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv (x,0).mv(0) does not
    // handle beta, whereas cublasSgemm, cublasDgemm do for case where (x,0).mm(0,y).
    // This logic could live in blas::gemv<float> and <double> if blas::gemv's interface
    // can be extended to accept result as an argument.
    if (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, c10::complex<float>>::value || std::is_same<scalar_t, c10::complex<double>>::value) {
      if (vec.size(0) == 0 && mat.size(0) != 0) {
        if (beta == scalar_t(0)) {
          result.zero_();
        } else if (beta != scalar_t(1)) {
          result.mul_(beta);
        }
      }
    }
  });
  return result;
}

Tensor dot_cuda(const Tensor& self, const Tensor& other){

  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, got, ",
      self.dim(), ", ",
      other.dim(),
      " tensors");

  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(), "] to have the same number of elements, but got ",
      self.numel(), " and ",
      other.numel(),
      " elements respectively");

  TensorArg self_arg{self, "self", 1};
  TensorArg other_arg{other, "other", 2};
  checkAllSameGPU("dot", {self_arg, other_arg});
  checkSameType("dot", other_arg, other_arg);

  auto self_contig = self.contiguous();
  auto other_contig = other.contiguous();
  return AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "dot", [&] {

    Tensor result = at::empty({}, self_contig.options());
    auto res = at::cuda::blas::dot<scalar_t>(
        self_contig.numel(),
        self_contig.data_ptr<scalar_t>(),
        self_contig.stride(0),
        other_contig.data_ptr<scalar_t>(),
        other_contig.stride(0));

    result.fill_(res);
    return result;
  });
}

}} // namespace at::native
