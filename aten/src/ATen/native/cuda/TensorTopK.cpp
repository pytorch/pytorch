#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/TensorTopK.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cuda/Sort.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sort_cuda_dispatch.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at::native {

// TODO: remove this when CUDA <11.6 is no longer supported
void topk_out_with_sort(
  const Tensor& self,
  int64_t k, int64_t dim, bool largest,
  const Tensor& values,
  const Tensor& indices
) {
  auto [sorted_values, sorted_indices] = at::cuda::sort(self, /* stable= */false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

// TODO: remove this when CUDA <11.6 is no longer supported
bool disable_sort_for_topk();
bool should_use_sort(const Tensor& self, int64_t dim) {
#if defined(USE_ROCM)
  if (self.dtype() == kBool) return false; // Bool sort not supported in ROCm: https://github.com/pytorch/pytorch/issues/139972
  return (self.numel() >= 10000 && self.numel() == self.size(dim)); // based on the experiments in https://github.com/pytorch/pytorch/pull/146387
#else
  if (disable_sort_for_topk()) return false;
  // This heuristics is based on the experiment in https://github.com/pytorch/pytorch/pull/68632
  if (self.dim() == 0) return false;
  if (self.dtype() == kBool) return false; // Bool is not support by topk
  int64_t slice_size = self.size(dim);
  if (slice_size == 0) return false;
  int64_t num_slices = self.numel() / slice_size;
  return num_slices <= 10 && slice_size >= 100000;
#endif
}

TORCH_IMPL_FUNC(topk_out_cuda)
  (const Tensor& self,
   int64_t k, int64_t dim, bool largest, bool sorted,
   const Tensor& values,
   const Tensor& indices) {
  TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2}, input_arg{self, "self", 3};
  checkAllSameGPU(__func__, {topK_arg, indices_arg, input_arg});

  dim = at::maybe_wrap_dim(dim, self);

  if (should_use_sort(self, dim)) {
    topk_out_with_sort(self, k, dim, largest, values, indices);
    return;
  }

  // If k is 0 the result is an empty tensor, so we don't need to launch a kernel.
  if (k == 0) {
    return;
  }

  launch_gather_topk_kernel(self, k, dim, largest, values, indices);

  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted && values.numel() > 1) {
    if (should_use_small_sort(values, dim)) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice

      sortKeyValueInplace(values, indices, dim, largest);
    } else {
      // Depend upon the backup sort that returns indices, which we
      // can use in conjunction with gather to produce the original
      // indices.
      // This is not the most efficient implementation, especially since
      // there are memory allocations performed here. If the user desires
      // greater performance, they should torch.gather() the results
      // themselves using the reported indices, providing previously
      // allocated tensors to receive the results.

      Tensor sortedIndices = at::empty_like(indices);
      Tensor sortedValues = at::empty_like(values);
      at::cuda::sort_outf(values, /* stable= */ false, dim, largest, sortedValues, sortedIndices);
      indices.copy_(indices.gather(dim, sortedIndices));
      values.copy_(sortedValues);
    }
  }
}

} // namespace at::native
