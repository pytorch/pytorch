#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor &tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(at::empty(tensor.sizes(), tensor.options()));
}

void cummax_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummax_cuda_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

void cummin_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummin_cuda_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

Tensor& _logcumsumexp_out_cuda(const Tensor& self, int64_t dim, Tensor& result) {
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  TensorArg output_arg{ result, "output", 1 };
  TensorArg input_arg{ self, "input", 2 };
  checkAllSameGPU(__func__, {output_arg, input_arg});

  auto result_ = contiguous_out_arg(result);
  launch_logcumsumexp_cuda_kernel(*result_, self, dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

Tensor _logcumsumexp_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cuda(self, dim, result);
}

void cumsum_cuda_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);
  launch_cumsum_cuda_kernel(*result_, self, dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void cumprod_cuda_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);
  launch_cumprod_cuda_kernel(*result_, self, dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

REGISTER_CUDA_DISPATCH(cumsum_stub, &cumsum_cuda_kernel);
REGISTER_CUDA_DISPATCH(cumprod_stub, &cumprod_cuda_kernel);

}} // namespace at::native
