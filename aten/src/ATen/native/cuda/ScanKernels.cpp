#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/ReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

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
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
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
  launch_logcumsumexp_cuda_kernel(*result_, self, wrap_dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

Tensor _logcumsumexp_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cuda(self, dim, result);
}

int64_t canonicalize_dim(int64_t rank, int64_t idx) {
  TORCH_INTERNAL_ASSERT(rank >= 0);
  if (rank == 0) {
    rank = 1;
  }
  if (idx >= 0 && idx < rank) {
    return idx;
  }
  int64_t _idx = (idx < 0) ? (idx + rank) : idx;
  TORCH_INTERNAL_ASSERT(!(_idx < 0 || _idx >= rank));
  return _idx;
}

// This function is ported from the cumsum decomp in `torch/_refs`.
Tensor& cumsum_deterministic(Tensor& result, Tensor& self, int64_t dim) {
  int64_t ndim = self.dim();
  dim = canonicalize_dim(ndim, dim);
  if (ndim == 0) {
    return at::sum_out(result, self.unsqueeze(0), /*dim=*/IntArrayRef{0});
  }
  self = self.unsqueeze(dim + 1);
  Tensor rg = at::arange(self.size(dim), c10::TensorOptions().device(self.device()));
  Tensor mask = rg.unsqueeze(1).le(rg);
  for (int idx = 0; idx < (ndim - dim - 1); idx++) {
    mask = mask.unsqueeze(-1);
  }
  Tensor masked_self = at::where(mask, self, 0);
  return at::sum_out(result, masked_self, /*dim=*/IntArrayRef{dim});
}

void cumsum_cuda_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);
  if ((self.is_floating_point() || self.is_complex()) && globalContext().deterministicAlgorithms()) {
    // See Note [Enabling Deterministic Operations]
    cumsum_deterministic(const_cast<Tensor&>(*result_), const_cast<Tensor&>(self), dim);
  } else {
    launch_cumsum_cuda_kernel(*result_, self, dim);
  }
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

} // namespace at::native
