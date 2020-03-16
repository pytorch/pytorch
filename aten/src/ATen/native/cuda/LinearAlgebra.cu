#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/cuda/CUDABlas.h>

namespace at { namespace native {

Tensor baddbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm(self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_cuda(Tensor &result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cuda(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm_out(self, self, batch1, batch2, beta, alpha);
}

Tensor bmm_cuda(const Tensor& self, const Tensor& mat2) {
  return legacy::cuda::_th_bmm(self, mat2);
}

Tensor& bmm_out_cuda(Tensor &result, const Tensor& batch1, const Tensor& batch2) {
  return legacy::cuda::_th_bmm_out(result, batch1, batch2);
}

Tensor prepare_tensor_for_cublas(Tensor& tensor, char& transpose_tensor) {
  Tensor tensor_;

  if ((tensor.stride(0) == 1) && (tensor.stride(1) != 0)) {
    tensor_ = tensor;
    transpose_tensor = 'n';
  } else if ((tensor.stride(1) == 1) && (tensor.stride(0) != 0)) {
    tensor_ = tensor;
    transpose_tensor = 't';
  } else {
    // transpose_result = 'n';
    // result_ = result.transpose().clone(at::MemoryFormat::Contiguous).transpose();

    // This way is better, right?
    transpose_tensor = 't';
    tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
  }

  return tensor_;
}

// Check https://github.com/pytorch/pytorch/issues/22078
// for information about the bug. We don't know the exact conditions that trigger it,
// but using Sgemm or Hgemm on Maxwell or Pascal seems to be a
// necessary condition.
static void checkCuda90Bug(int i_m, int i_n, int i_k)
{
#if CUDA_VERSION < 9200 && CUDA_VERSION >= 9000
  static std::once_flag alreadyWarned;
  const int LIMIT = 1 << 21;
  if (i_m > LIMIT || i_n > LIMIT || i_k > LIMIT) {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major == 5 || prop->major == 6) {
      std::call_once(alreadyWarned, []() {
        TORCH_WARN("Matrix multiplication for dimensions larger than 2^21 has known bugs on your combination of CUDA version and device type. Please consider upgrading to CUDA 9.2 or later.");
      });
    }
  }
#endif
}

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  TORCH_CHECK(
    (mat1.dim() == 2) && (mat2.dim() == 2) &&
    (self.dim() == 2) && (result.dim() == 2),
    "tensors must be 2-D"
  );
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1 dim 1 must match mat2 dim 0");
  TORCH_CHECK(self.size(0) == mat1.size(0), "self dim 0 must match mat1 dim 0");
  TORCH_CHECK(self.size(1) == mat2.size(1), "self dim 1 must match mat2 dim 1");

  if (!result.is_set_to(self)) {
    if (beta.to<double>() != 0.0) {
      result.copy_(self);
    }
  }

  if ((result.size(0) == 0) || (result.size(1) == 0)) {
    return result;
  }

  char transpose_result;
  Tensor result_ = prepare_tensor_for_cublas(result, transpose_result);

  char transpose_mat1;
  char transpose_mat2;
  Tensor mat1_ = (transpose_result == 'n') ? mat1 : mat2.transpose(0, 1);
  Tensor mat2_ = (transpose_result == 'n') ? mat2 : mat1.transpose(0, 1);

  mat1_ = prepare_tensor_for_cublas(mat1_, transpose_mat1);
  mat2_ = prepare_tensor_for_cublas(mat2_, transpose_mat2);

  int64_t m = mat1_.size(0);
  int64_t k = mat1_.size(1);
  int64_t n = mat2_.size(1);
  int64_t mat1_ld = (transpose_mat1 == 'n') ? mat1_.stride(1) : mat1_.stride(0);
  int64_t mat2_ld = (transpose_mat2 == 'n') ? mat2_.stride(1) : mat2_.stride(0);
  int64_t result_ld = (transpose_result == 'n') ? result_.stride(1) : result_.stride(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "addmm_cuda", [&] {
    if (self.scalar_type() == at::ScalarType::BFloat16 || self.scalar_type() == at::ScalarType::Int) {
      checkCuda90Bug(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    }
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* mat1_ptr = mat1_.data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_.data_ptr<scalar_t>();
    scalar_t* result_ptr = result_.data_ptr<scalar_t>();
    at::cuda::blas::gemm<scalar_t>(
      transpose_mat1,
      transpose_mat2,
      m, n, k,
      alpha_val,
      mat1_ptr, mat1_ld,
      mat2_ptr, mat2_ld,
      beta_val,
      result_ptr, result_ld
    );
  });
  result.copy_(result_);
  return result;
}

Tensor& mm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat2) {
  result.resize_({ self.size(0), mat2.size(1) });
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

Tensor mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({ self.size(0), mat2.size(1) }, self.options());
  return addmm_out_cuda_impl(result, result, self, mat2, 0, 1);
}

} }
