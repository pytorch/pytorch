#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorTransformations.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/roll_native.h>
#endif

#include <cstddef>
#include <vector>

namespace at::native {

template <typename scalar_t, typename IndexType>
C10_LAUNCH_BOUNDS_2(cuda::getApplyBlockSize(), cuda::getApplyBlocksPerSM())
__global__ void kernel_pointwise_flip_apply2(
    const cuda::detail::TensorInfo<scalar_t, IndexType> in_tensor_info,
    cuda::detail::TensorInfo<scalar_t, IndexType> out_tensor_info,
    IndexType N,
    int flip_dim,
    IndexType total_dims) {
  for (IndexType linear_index = blockIdx.x * blockDim.x + threadIdx.x; linear_index < N; linear_index += gridDim.x * blockDim.x) {
    IndexType dst_offset = 0;
    if (flip_dim == 0) {
      // flip 1st dim
      dst_offset = (in_tensor_info.sizes[0] - 1 - linear_index / in_tensor_info.strides[0]) * in_tensor_info.strides[0] + linear_index % in_tensor_info.strides[0];
    }
    else {
      // flip last dim
      IndexType i = total_dims - 1;
      dst_offset = linear_index / in_tensor_info.strides[0] * in_tensor_info.strides[0] + (in_tensor_info.sizes[i] - 1 - linear_index % in_tensor_info.strides[0]);
    }
    out_tensor_info.data[dst_offset] = in_tensor_info.data[linear_index];
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void flip_cuda_kernel(
    scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t N,
    int64_t* flip_dims,
    int64_t flip_dims_size,
    int64_t* strides,
    int64_t* strides_contiguous,
    int64_t* shape,
    int64_t total_dims) {
  int64_t linear_index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }

  int64_t cur_indices = linear_index, rem = 0, dst_offset = 0;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / strides_contiguous[i];
    rem = temp - cur_indices * strides_contiguous[i];
    // flip the indices if it is in flip_dims
    for (int64_t j = 0; j < flip_dims_size; j++) {
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * strides[i];
    cur_indices = rem;
  }
  out_tensor[linear_index] = in_tensor[dst_offset];
}

#if defined(USE_ROCM)

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void roll_cuda_kernel(
    const scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t N,
    int64_t roll_dim,
    int64_t start,
    int64_t size,
    int64_t stride,
    int64_t total_dims) {
  for (int64_t linear_index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
       linear_index < N; linear_index += blockDim.x*gridDim.x)
  {
    // roll dim idx is the index of linear_index along the rolling dimension.
    int64_t roll_dim_idx = linear_index % (stride * size) / stride;
    // index into the source data to find appropriate value.
    int64_t source_idx = 0;
    if( roll_dim_idx >= (size - start) ) {
      source_idx = linear_index - ((size - start) * stride);
    } else {
      source_idx = linear_index + (start * stride);
    }
    out_tensor[linear_index] = in_tensor[source_idx];
  }
}

#else

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void roll_cuda_kernel(
    const scalar_t* in_tensor,
    scalar_t* out_tensor,
    int64_t N,
    int64_t roll_dim,
    int64_t start,
    int64_t size,
    int64_t stride,
    int64_t total_dims) {
  int64_t linear_index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }
  // roll dim idx is the index of linear_index along the rolling dimension.
  int64_t roll_dim_idx = linear_index % (stride * size) / stride;
  // index into the source data to find appropriate value.
  int64_t source_idx = 0;
  if( roll_dim_idx >= (size - start) ) {
    source_idx = linear_index - ((size - start) * stride);
  } else {
    source_idx = linear_index + (start * stride);
  }
  out_tensor[linear_index] = in_tensor[source_idx];
}

#endif

// Roll a tensor along a dimension
Tensor roll_cuda(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }

  auto in_tensor = self;
  if(!self.is_contiguous()) {
    in_tensor = self.contiguous();
  }
  auto out_tensor = at::empty_like(in_tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }
  const int64_t N = in_tensor.numel();
  const int64_t dim = dims[0];
  const int64_t size = in_tensor.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if( start < 0 ) start = start + size;

  dim3 dim_block = cuda::getApplyBlock();
#if defined(USE_ROCM)
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // Given a thread block size of 512, we launch with 4 blocks per SM/CU
  dim3 dim_grid(num_mp * 4);
#else
  dim3 dim_grid;
  TORCH_CHECK(cuda::getApplyGrid(N, dim_grid, in_tensor.get_device()), "unable to get dim grid");
#endif

  auto total_dims = in_tensor.dim();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      in_tensor.scalar_type(), "roll_cuda",
      [&] {
        roll_cuda_kernel<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          in_tensor.const_data_ptr<scalar_t>(), out_tensor.mutable_data_ptr<scalar_t>(), N,
          dim, start,
          size,
          in_tensor.stride(dim),
          total_dims);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return out_tensor;
}

} // namespace at::native
