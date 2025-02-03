#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/TensorModeKernel.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>

constexpr int64_t MAX_BLOCK_SIZE = AT_ROCM_ENABLED() ? 256 : 1024;

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
constexpr int64_t MAX_GRID_SIZE = 65535LL;

namespace at::native {

void mode_kernel_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t slice_size = ensure_nonempty_size(self, dim);
  int64_t slices = self.numel() / slice_size;

  // Resize output value, index Tensors to appropriate sizes (i.e. the same as
  // the input Tensor, except at dim=dimension, the size is 1)
  assert(0 <= dim && static_cast<size_t>(dim) < self_sizes.size());
  self_sizes[dim] = 1;

  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }

  at::native::resize_output(values, self_sizes);
  at::native::resize_output(indices, self_sizes);

  // If sliceSize is 1, copy input to values and set indices
  if (slice_size == 1) {
    values.copy_(self);
    indices.fill_(0);
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }
    return;
  }

  // Beginning our optimized implementation. First thing we want to do is to
  // transpose the input Tensor along the sort dimension, and then make it
  // contiguous.
  auto transposed = self.transpose(dim, ndim - 1);
  auto contiguous = transposed.contiguous();

  // We also need to view the values and indices Tensors as transposed in order
  // to properly determine the offset into the underlying storage in which to
  // place the mode and index for a particular set of dimension values.
  auto values_transposed = values.transpose(dim, ndim - 1);
  auto indices_transposed = indices.transpose(dim, ndim - 1);

  // Requirements for fused kernel implementation:
  //
  // 1. sliceSize <= 2 * max threads per block
  // 2. uses one block per slice, so number of slices must be less than the
  // maximum number of blocks for a kernel launch
  // 3. Can use 32-bit index math for indexing (mainly just for implementation
  // conciseness, could be changed)
  //
  // MAX_BLOCK_SIZE and MAX_GRID_SIZE come from:
  //     ATen/native/cuda/SortingCommon.cuh
  if (slice_size <= 2 * MAX_BLOCK_SIZE &&
      slices <= MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE &&
      canUse32BitIndexMath(self)) {
    launch_fused_mode_kernel(
        values_transposed, indices_transposed, contiguous, slice_size, slices);
  } else {
    // [Note: CUDA torch.mode clones self]
    //
    // If transposed is already contiguous, it will return a tensor with the
    // same storage. So, since we do not want to modify self, we clone it.
    if (transposed.is_same(contiguous)) {
      contiguous = contiguous.clone();
    }

    launch_apply_mode_kernel(
        values_transposed, indices_transposed, contiguous, dim, ndim);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

REGISTER_CUDA_DISPATCH(mode_stub, &mode_kernel_impl)
} // namespace at::native
