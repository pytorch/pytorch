#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
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

#define MAX_GRID_SIZE 65535
#define MAX_BLOCK_SIZE 1024

namespace at {
namespace native {

template <typename scalar_t>
void calculate_mode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    std::vector<int64_t>& position,
    thrust::device_vector<int64_t>& sort_buffer,
    int dim) {
  auto state = globalContext().getTHCState();
  auto stream = at::cuda::getCurrentCUDAStream();

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
  THCThrustAllocator thrust_allocator(state);

  // Wrap input data in Thrust device vector
  thrust::device_ptr<scalar_t> vec_ptr = thrust::device_pointer_cast(data);
  thrust::device_vector<scalar_t> iter(vec_ptr, vec_ptr + n_element);

  // Fill sort_buffer with [0, 1, 2, ... n_element - 1]
  thrust::sequence(
      thrust::cuda::par(thrust_allocator).on(stream),
      sort_buffer.begin(),
      sort_buffer.end());

  // Sort the input data. The original indices of the data are stored in
  // sort_buffer
  thrust::sort_by_key(
      thrust::cuda::par(thrust_allocator).on(stream),
      iter.begin(),
      iter.end(),
      sort_buffer.begin());

  // Count # of unique elements via an inner product between adjacent elements.
  // Add 1 if two neighboring element are not equal.
  int unique = 1 +
      thrust::inner_product(
                   thrust::cuda::par(thrust_allocator).on(stream),
                   iter.begin(),
                   iter.end() - 1,
                   iter.begin() + 1,
                   0,
                   thrust::plus<int>(),
                   thrust::not_equal_to<scalar_t>());

  // Count frequency of each element
  thrust::device_vector<scalar_t> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
      thrust::cuda::par(thrust_allocator).on(stream),
      iter.begin(),
      iter.end(),
      thrust::constant_iterator<int>(1),
      keys.begin(),
      counts.begin());

  // Find index of maximum count
  auto it = thrust::max_element(
      thrust::cuda::par(thrust_allocator).on(stream),
      counts.begin(),
      counts.end());
  scalar_t mode = keys[it - counts.begin()];

  // Find first index within which it occurs
  auto position_iter = thrust::find(
      thrust::cuda::par(thrust_allocator).on(stream),
      iter.begin(),
      iter.end(),
      mode);

  TORCH_INTERNAL_ASSERT(position_iter != iter.end());
  int64_t index = sort_buffer[position_iter - iter.begin()];

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
}

template <typename scalar_t>
void apply_mode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    std::vector<int64_t>& position,
    thrust::device_vector<int64_t>& sort_buffer,
    int dim,
    int curDim) {
  // Because we have transposed the Tensor, the data for the dimension we are
  // mode'ing along is always in the innermost dimension
  int64_t ndim = ensure_nonempty_dim(self.dim());
  if (curDim == ndim - 1) {
    calculate_mode<scalar_t>(values, indices, self, position, sort_buffer, dim);
  } else {
    for (int i = 0; i < ensure_nonempty_size(self, curDim); ++i) {
      position[curDim] = i;
      apply_mode<scalar_t>(
          values, indices, self, position, sort_buffer, dim, curDim + 1);
    }
  }
}

#define HANDLE_MODE(SIZE)                                                  \
  {                                                                        \
    const dim3 block(SIZE / 2);                                            \
    const auto memsize =                                                   \
        (sizeof(scalar_t) * SIZE) + (2 * SIZE * sizeof(unsigned int));     \
    compute_mode<scalar_t, SIZE>                                           \
        <<<grid, block, memsize, at::cuda::getCurrentCUDAStream()>>>(      \
            self.data_ptr<scalar_t>(), ti_values, ti_indices, slice_size); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
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
  // Ideally we would have one HANDLE_MODE for each power of 2
  switch (ceilPowerOf2) {
    case 2048:
      HANDLE_MODE(2048)
      break;
    case 1024:
    case 512:
    case 256:
      HANDLE_MODE(1024)
      break;
    case 128:
    case 64:
      HANDLE_MODE(128)
      break;
    case 32:
    case 16:
    case 8:
    case 4:
    case 2:
      HANDLE_MODE(32)
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

  values.resize_(self_sizes);
  indices.resize_(self_sizes);

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
  AT_DISPATCH_ALL_TYPES_AND(kHalf, self.scalar_type(), "cuda_mode", [&] {
    // Requirements for fused kernel implementation:
    //
    // 1. sliceSize <= 2 * max threads per block
    // 2. uses one block per slice, so number of slices must be less than the
    // maximum number of blocks for a kernel launch
    // 3. Can use 32-bit index math for indexing (mainly just for implementation
    // conciseness, could be changed)
    if (slice_size <= MAX_BLOCK_SIZE && slices <= MAX_GRID_SIZE &&
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

      // Sort Buffer is a Storage that will be used in the internal sort
      // required to calculate the mode efficiently
      thrust::device_vector<int64_t> sort_buffer(slice_size);

      apply_mode<scalar_t>(
          values_transposed,
          indices_transposed,
          contiguous,
          position,
          sort_buffer,
          dim,
          0);
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
