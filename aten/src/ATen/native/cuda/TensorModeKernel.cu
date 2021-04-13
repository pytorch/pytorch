#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <c10/util/Exception.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/TensorModeKernel.cuh>
#include <THC/THCThrustAllocator.cuh>

namespace at {
namespace native {

template <typename scalar_t>
void calculate_mode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    std::vector<int64_t>& position,
    int dim) {
  THCThrustAllocator thrust_allocator(globalContext().lazyInitCUDA());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto policy = thrust::cuda::par(thrust_allocator).on(stream);

  TORCH_INTERNAL_ASSERT(self.is_contiguous());

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  scalar_t* data = self.data_ptr<scalar_t>();
  for (int64_t i = 0; i < position.size(); i++) {
    data += position[i] * ensure_nonempty_stride(self, i);
  }

  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t n_element = ensure_nonempty_size(self, ndim - 1);

  scalar_t* iter_begin = data;
  scalar_t* iter_end = data + n_element;

  Tensor sort_buffer = at::arange(0, n_element, self.options().dtype(kLong));
  auto sort_buffer_ptr =
      thrust::device_pointer_cast(sort_buffer.data_ptr<int64_t>());

  // Sort the input data. The original indices of the data are stored in
  // sort_buffer_ptr
  thrust::sort_by_key(policy, iter_begin, iter_end, sort_buffer_ptr);

  // Count # of unique elements via an inner product between adjacent elements.
  // Add 1 if two neighboring element are not equal.
  int unique = 1 +
      thrust::inner_product(
                   policy,
                   iter_begin,
                   iter_end - 1,
                   iter_begin + 1,
                   0,
                   thrust::plus<int>(),
                   thrust::not_equal_to<scalar_t>());

  // Count frequency of each element
  Tensor keys = at::empty(unique, self.options());
  Tensor counts = at::empty(unique, self.options().dtype(kLong));

  auto keys_ptr = thrust::device_pointer_cast(keys.data_ptr<scalar_t>());
  auto counts_ptr = thrust::device_pointer_cast(counts.data_ptr<int64_t>());

  thrust::reduce_by_key(
      policy,
      iter_begin,
      iter_end,
      thrust::constant_iterator<int>(1),
      keys_ptr,
      counts_ptr);

  // Find index of maximum count
  auto it = thrust::max_element(policy, counts_ptr, counts_ptr + unique);
  scalar_t mode = keys_ptr[it - counts_ptr];

  // Find first index within which it occurs
  auto position_iter = thrust::find(policy, iter_begin, iter_end, mode);

  TORCH_INTERNAL_ASSERT(position_iter != iter_end);
  int64_t index = sort_buffer_ptr[position_iter - iter_begin];

  // Place mode, index in output
  scalar_t* values_data = values.data_ptr<scalar_t>();
  int64_t* indices_data = indices.data_ptr<int64_t>();

  for (int64_t i = 0; i < position.size(); i++) {
    int64_t pos = position[i];
    values_data += ensure_nonempty_stride(values, i) * pos;
    indices_data += ensure_nonempty_stride(indices, i) * pos;
  }

  AT_CUDA_CHECK(cudaMemcpyAsync(
      values_data, &mode, sizeof(scalar_t), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      indices_data, &index, sizeof(scalar_t), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename scalar_t>
void apply_mode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    std::vector<int64_t>& position,
    int dim,
    int curDim) {
  // Because we have transposed the Tensor, the data for the dimension we are
  // mode'ing along is always in the innermost dimension
  int64_t ndim = ensure_nonempty_dim(self.dim());
  if (curDim == ndim - 1) {
    calculate_mode<scalar_t>(values, indices, self, position, dim);
  } else {
    for (int i = 0; i < ensure_nonempty_size(self, curDim); ++i) {
      position[curDim] = i;
      apply_mode<scalar_t>(values, indices, self, position, dim, curDim + 1);
    }
  }
}

template <int64_t size, typename scalar_t>
void handle_fused_mode(
    dim3 grid,
    const Tensor& self,
    cuda::detail::TensorInfo<scalar_t, unsigned int>& ti_values,
    cuda::detail::TensorInfo<int64_t, unsigned int>& ti_indices,
    int64_t slice_size,
    int64_t slices) {
  const dim3 block(size / 2);
  const auto memsize =
      (sizeof(scalar_t) * size) + (2 * size * sizeof(unsigned int));
  compute_mode<scalar_t, size>
      <<<grid, block, memsize, at::cuda::getCurrentCUDAStream()>>>(
          self.data_ptr<scalar_t>(), ti_values, ti_indices, slice_size, slices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void fused_mode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t slice_size,
    int64_t slices) {
  // Set-up TensorInfo structs for passing to kernel
  auto ti_values = cuda::detail::getTensorInfo<scalar_t, unsigned int>(values);
  auto ti_indices = cuda::detail::getTensorInfo<int64_t, unsigned int>(indices);

  // The number of blocks is the number of slices that we need to calculate
  // the mode for. Each block is responsible for computing a single mode
  dim3 grid;
  getGridFromTiles(slices, grid);

  // The blocksize is two elements per thread, rounded up to the nearest power
  // of 2
  auto ceilPowerOf2 = nextHighestPowerOf2(slice_size);

  // Tradeoff between compilation time and the number of specializations.
  // Ideally we would have one handle_fused_mode for each power of 2
  switch (ceilPowerOf2) {
    case 2048:
      handle_fused_mode<2048, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1024:
    case 512:
    case 256:
      handle_fused_mode<1024, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 128:
    case 64:
      handle_fused_mode<128, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 32:
    case 16:
    case 8:
    case 4:
    case 2:
      handle_fused_mode<32, scalar_t>(
          grid, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1:
    default:
      TORCH_INTERNAL_ASSERT(false);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

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

  // Call mode
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, self.scalar_type(), "cuda_mode", [&] {
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
        cuda::detail::canUse32BitIndexMath(self)) {
      fused_mode<scalar_t>(
          values_transposed,
          indices_transposed,
          contiguous,
          slice_size,
          slices);
    } else {
      // If transposed is already contiguous, it will return a tensor with the
      // same storage. So, since we do not want to modify self, we clone it.
      if (transposed.is_contiguous()) {
        contiguous = contiguous.clone();
      }

      // Position will store the dimension values we are processing
      std::vector<int64_t> position(ndim - 1, 0);

      apply_mode<scalar_t>(
          values_transposed, indices_transposed, contiguous, position, dim, 0);
    }
  });

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

#undef MAX_GRID_SIZE
#undef MAX_BLOCK_SIZE

REGISTER_DISPATCH(mode_stub, &mode_kernel_impl);
} // namespace native
} // namespace at
