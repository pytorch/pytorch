#include <ATen/native/cuda/TensorTopK.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cuda/Sort.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(topk_out_cuda)
  (const Tensor& self,
   int64_t k, int64_t dim, bool largest, bool sorted,
   const Tensor& values,
   const Tensor& indices) {
  TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2}, input_arg{self, "self", 3};
  checkAllSameGPU(__func__, {topK_arg, indices_arg, input_arg});
  dim = at::maybe_wrap_dim(dim, self);

  // If k is 0 the result is an empty tensor, so we don't need to launch a kernel.
  if (k == 0) {
    return;
  }

  launch_gather_topk_kernel(self, k, dim, largest, sorted, values, indices);

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
      sort_out_cuda(values, dim, largest, sortedValues, sortedIndices);
      indices.copy_(indices.gather(dim, sortedIndices));
      values.copy_(sortedValues);
    }
  }
}

}} // namespace at::native
