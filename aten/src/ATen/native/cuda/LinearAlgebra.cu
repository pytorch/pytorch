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

Tensor prepare_matrix_for_cublas(Tensor& tensor, bool& transpose_tensor) {
  Tensor tensor_;
  IntArrayRef tensor_strides = tensor.strides();

  if ((tensor_strides[0] == 1) && (tensor_strides[1] != 0)) {
    tensor_ = tensor;
    transpose_tensor = false;
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] != 0)) {
    tensor_ = tensor;
    transpose_tensor = true;
  } else {
    transpose_tensor = true;
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
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self_sizes = self.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");
  TORCH_CHECK(self_sizes[0] == mat1_sizes[0], "self dim 0 must match mat1 dim 0");
  TORCH_CHECK(self_sizes[1] == mat2_sizes[1], "self dim 1 must match mat2 dim 1");

  // If self and result either point to the same data or if beta is zero,
  // we can avoid copying self into result. Otherwise, we need to copy.
  if (beta.to<double>() != 0.0) {
    if ((result.data_ptr() != self.data_ptr()) || (result.strides() != self.strides())) {
      result.copy_(self);
    }
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  Tensor result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  Tensor mat1_ = transpose_result ? mat2 : mat1;
  Tensor mat2_ = transpose_result ? mat1 : mat2;
  mat1_ = prepare_matrix_for_cublas(mat1_, transpose_mat1);
  mat2_ = prepare_matrix_for_cublas(mat2_, transpose_mat2);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_.sizes();
    mat2_sizes = mat2_.sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_.stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_.stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_.stride(transpose_result ? 0 : 1);
  at::ScalarType scalar_type = self.scalar_type();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "addmm_cuda", [&] {
    if (scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float) {
      checkCuda90Bug(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    }
    scalar_t alpha_val = alpha.to<scalar_t>();
    scalar_t beta_val = beta.to<scalar_t>();
    scalar_t* mat1_ptr = mat1_.data_ptr<scalar_t>();
    scalar_t* mat2_ptr = mat2_.data_ptr<scalar_t>();
    scalar_t* result_ptr = result_.data_ptr<scalar_t>();
    at::cuda::blas::gemm<scalar_t>(
      transpose_mat1 ? 't' : 'n',
      transpose_mat2 ? 't' : 'n',
      m, n, k,
      alpha_val,
      mat1_ptr, mat1_ld,
      mat2_ptr, mat2_ld,
      beta_val,
      result_ptr, result_ld
    );
  });
  if (result.data_ptr() != result_.data_ptr()) {
    result.copy_(result_);
  }
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
