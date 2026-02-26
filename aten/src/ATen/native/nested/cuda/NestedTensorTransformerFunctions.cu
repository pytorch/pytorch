#include <cuda_fp16.h>
#include <type_traits>
#include <cmath>
#include <limits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION)
#define build_grouped_gemm
#endif

#ifdef build_grouped_gemm
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#endif

#include <ATen/NestedTensorImpl.h>

#define BLOCK_DIM 256
#define GRID_DIM_Y 16

namespace at {
namespace native {

template <typename T>
__global__ void remove_padding_transform0213_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);

    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);
    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
}

template <typename T>
__global__ void remove_padding_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset = batch_id * input_sizes[1] * input_sizes[2];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
}

template <typename T>
__global__ void remove_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
}

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  if (output_dim == 2) {
    remove_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  } else {
    remove_padding<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  }
}

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  remove_padding_transform0213_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

template void remove_padding_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template void remove_padding_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template void remove_padding_transform0213_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template void remove_padding_transform0213_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template <typename T>
__global__ void add_padding_1(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int batch_output_offset = batch_id * output_sizes_1;
  for (int ii = 0; ii < (output_sizes_1 / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && i < sizes_i[0]) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
  const int i = (output_sizes_1 / grainsize) * grainsize + tid;
  if (i < output_sizes_1) {
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && (i < sizes_i[0])) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_2(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset = batch_id * output_sizes_1 * output_sizes_2;
  const int output_numel = output_sizes_1 * output_sizes_2;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_3(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    int output_sizes_3,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset =
      batch_id * output_sizes_1 * output_sizes_2 * output_sizes_3;
  const int output_numel = output_sizes_1 * output_sizes_2 * output_sizes_3;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  dim3 grid;
  grid.x = output_batch_size;
  grid.y = GRID_DIM_Y;
  if (input_dim == 1) {
    add_padding_1<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        batch_size);
  }
  if (input_dim == 2) {
    add_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        batch_size);
  }
  if (input_dim == 3) {
    add_padding_3<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        output_sizes[3],
        batch_size);
  }
}

template void add_padding_kernelLauncher<double>(
    double* input,
    double* output,
    double padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    float padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    c10::Half padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

// NB: The following code covers jagged <-> padded dense conversions and was lifted
// from fbgemm_gpu. For more details, see
// https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/src/jagged_tensor_ops

// Passing lambda exp argument by value instead of by reference to avoid
// "internal compiler error: in maybe_undo_parenthesized_ref" error for specific
// compiler version.
#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string torch_tensor_device_name(
    const std::optional<at::Tensor>& ten) {
  if (ten.has_value()) {
    return torch_tensor_device_name(ten.value());
  } else {
    return "N/A";
  }
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor& ten) {
  return ten.is_cuda();
}

inline bool torch_tensor_on_cuda_gpu_check(
    const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cuda_gpu_check(ten.value());
}

#define TENSOR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                 \
      torch_tensor_on_cuda_gpu_check(x),                       \
      #x " must be a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))

// A wrapper class for passing dynamically sized dimension information (e.g.
// tensor.dims()) from the host to device.
constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

// Warp size
#ifdef USE_ROCM
static constexpr int32_t kWarpSize = 64;
#else
static constexpr int32_t kWarpSize = 32;
#endif
// Max thread num in one thread block
static constexpr int32_t kMaxThreads = 1024;

#define DEVICE_INLINE __device__ C10_ALWAYS_INLINE

__host__ DEVICE_INLINE int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

__host__ DEVICE_INLINE int32_t round_down(int32_t a, int32_t b) {
  return a / b * b;
}

inline std::tuple<dim3, dim3, StackArray<int64_t>> check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int threads_x =
      inner_dense_size >= kWarpSize / 2 ? kWarpSize : inner_dense_size;
  const int threads_y = kMaxThreads / kWarpSize;
  const dim3 blocks(
      div_round_up(outer_dense_size * jagged_folded_size, threads_y));

  StackArray<int64_t> jagged_dims_tensor{};
  const int num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= static_cast<int>(kStackArrayMaxDims));
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {dim3(threads_x, threads_y), blocks, jagged_dims_tensor};
}

template <int NUM_JAGGED_DIM, typename index_t>
DEVICE_INLINE bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coordinates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

// output = f(x, y) where x is jagged, y is dense, and output is dense.
// A generic elementwise operation between a jagged tensor and a dense tensor
// This kernel assumes jagged dims are clustered together, preceded by outer
// dense dimensions and followed by inner dense dimensions.
// The outer/inner dense dimensions, and jagged dimensions in between are
// assumed to be folded so physically the dense tensor is 3D and the value of
// jagged tensor is 2D.
// To support arbitrary number of jagged dimensions, we pass a vector of
// pointers to offset tensors (this is ugly and probably we can use nested
// tensor here).
// This kernel parallelizes the (folded) inner dense dimension across
// blockDim.x so the inner dense dimension should be similar to or bigger than
// warp size.
// We rely on compiler unrolling the compiler time constant NUM_JAGGED_DIM.
template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_dense_elementwise_dense_output_kernel_(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    StackArray<int64_t> jagged_dims,
    F f,
    const scalar_t padding_value) {
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  const int outer_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int outer_stride = gridDim.x * blockDim.y;
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (is_zero) {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
        output[oidx][jidx][2 * iidx + 1] =
            f(padding_value, y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
        output[oidx][jidx][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
      }
    }
  }
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (y.numel() == 0) {
    return;
  }

  dim3 threads, blocks;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                \
  {                                                                           \
    std::vector<Tensor> x_offsets_contig;                                     \
    x_offsets_contig.resize(num_jagged_dim);                                  \
    StackArray<index_t*> x_offset_ptrs;                                       \
    x_offset_ptrs.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                \
      x_offsets_contig[d] = x_offsets[d].contiguous();                        \
      x_offset_ptrs.vals[d] =                                                 \
          x_offsets_contig[d].template data_ptr<index_t>();                   \
    }                                                                         \
    jagged_dense_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>    \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
            x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            x_offset_ptrs,                                                    \
            y_reshaped                                                        \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            output_reshaped                                                   \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            jagged_dims_tensor,                                               \
            f,                                                                \
            padding_value);                                                   \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    auto [threads, blocks, jagged_dims_tensor] =                               \
        check_shape_and_partition_(x_values, x_offsets, y);                    \
    blocks.x = div_round_up(x_values.size(0), threads.y);                      \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                      \
        NUM_JAGGED_DIM,                                                        \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        [f] __device__(scalar_t x, scalar_t y, scalar_t /*unused*/)            \
            -> scalar_t { return f(x, y); });                                  \
  }

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_dense_dense_elementwise_jagged_output_kernel_(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_0,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_1,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  const int offset_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int offset_stride = gridDim.x * blockDim.y;
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx = 0;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

///@addtogroup jagged-tensor-ops-cuda
template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef INVOKE_KERNEL_WITH_DIM

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<int64_t> {
  __device__ int64_t* getPointer() {
    extern __shared__ int64_t s_int64_t[];
    return s_int64_t;
  }
};

template <>
struct SharedMemory<int32_t> {
  __device__ int32_t* getPointer() {
    extern __shared__ int32_t s_int32_t[];
    return s_int32_t;
  }
};

template <typename index_t>
__global__ void jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
    int nnz,
    int B) {
  struct SharedMemory<index_t> smem;
  index_t* offsets_sh = smem.getPointer();

  for (int i = threadIdx.x; i < B + 1; i += blockDim.x) {
    offsets_sh[i] = offsets[i];
  }
  __syncthreads();
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= nnz)
    return;
  int first = -1;
  int count = B - 1;
  first = 1;
  while (count > 0) {
    int idx = first;
    int step = count / 2;
    idx += step;
    if (offsets_sh[idx] <= row) {
      first = ++idx;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  --first;

  int dense_row = first;
  int offset = offsets_sh[dense_row];
  int dense_col = row - offset;
  rows[row] = dense_row;
  cols[row] = dense_col;
}

struct VecType128 {
  typedef float4 TType; // Transaction Type
  typedef struct __align__(16) {
    __half a, b, c, d, w, x, y, z;
  }
  half8;

  union Data {
    half8 val;
    TType mask;
  } data;

  __device__ VecType128() {
    data.mask = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
};

struct VecType64 {
  typedef float2 TType; // Transaction Type
  typedef struct __align__(8) {
    __half a, b, c, d;
  }
  half4;

  union Data {
    half4 val;
    TType mask;
  } data;

  __device__ VecType64() {
    data.mask = make_float2(0.0f, 0.0f);
  }
};

struct VecType32 {
  typedef float TType; // Transaction Type

  union Data {
    __half2 val;
    TType mask;
  } data;

  __device__ VecType32() {
    data.mask = 0.0f;
  }
};

template <typename F>
__device__ void f128(
    VecType128& v_out,
    const VecType128& x,
    const VecType128& y0,
    const VecType128& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
  v_out.data.val.w = f(x.data.val.w, y0.data.val.w, y1.data.val.w);
  v_out.data.val.x = f(x.data.val.x, y0.data.val.x, y1.data.val.x);
  v_out.data.val.y = f(x.data.val.y, y0.data.val.y, y1.data.val.y);
  v_out.data.val.z = f(x.data.val.z, y0.data.val.z, y1.data.val.z);
}

template <typename F>
__device__ void f64(
    VecType64& v_out,
    const VecType64& x,
    const VecType64& y0,
    const VecType64& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
}

template <typename F>
__device__ void f32(
    VecType32& v_out,
    const VecType32& x,
    const VecType32& y0,
    const VecType32& y1,
    F f) {
  v_out.data.val = __halves2half2(
      f(__low2half(x.data.val),
        __low2half(y0.data.val),
        __low2half(y1.data.val)),
      f(__high2half(x.data.val),
        __high2half(y0.data.val),
        __high2half(y1.data.val)));
}

template <typename index_t, typename F>
__global__ void jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_(
    at::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits>
        x_values,
    const at::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y0,
    const at::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y1,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
    const int nnz,
    const int E,
    F f) {
  int values_row = threadIdx.y + blockIdx.y * blockDim.y;
  if (values_row >= nnz)
    return;
  for (int real_row = values_row; real_row < nnz;
       real_row += blockDim.y * gridDim.y) {
    int dense_row = rows[real_row];
    int dense_col = cols[real_row];
    __half* values_ptr = reinterpret_cast<__half*>(&values[real_row][0]);
    const __half* x_ptr =
        reinterpret_cast<const __half*>(&x_values[real_row][0]);
    const __half* y0_ptr =
        reinterpret_cast<const __half*>(&y0[dense_row][dense_col][0]);
    const __half* y1_ptr =
        reinterpret_cast<const __half*>(&y1[dense_row][dense_col][0]);
    if ((dense_col < y0.size(1)) && (dense_row < y0.size(0)) &&
        (dense_col < y1.size(1)) && (dense_row < y1.size(0)) &&
        (dense_col >= 0) && (dense_row >= 0)) {
      for (int tid = threadIdx.x; tid < E / 8; tid += blockDim.x) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y1_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 8) * 8; tid < E / 4;
           tid += blockDim.x) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y1_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 4) * 4; tid < E / 2;
           tid += blockDim.x) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y1_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 2) * 2; tid < E; tid += blockDim.x) {
        auto v_x = static_cast<__half>(x_ptr[tid]);
        auto v_y0 = static_cast<__half>(y0_ptr[tid]);
        auto v_y1 = static_cast<__half>(y1_ptr[tid]);
        values_ptr[tid] = f(v_x, v_y0, v_y1);
      }
    } else {
      for (int tid = threadIdx.x; tid < E / 8; tid += blockDim.x) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 8) * 8; tid < E / 4;
           tid += blockDim.x) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 4) * 4; tid < E / 2;
           tid += blockDim.x) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 2) * 2; tid < E; tid += blockDim.x) {
        auto v_x = static_cast<__half>(x_ptr[tid]);
        values_ptr[tid] = f(v_x, __half{}, __half{});
      }
    }
  }
}

// Check to see if the inputs to the op are amenable to the fast path
inline bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
    const int& num_jagged_dim,
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0_reshaped,
    const Tensor& y_1_reshaped,
    const Tensor& output_values) {
  bool matches = true;
  matches &= (num_jagged_dim == 1);

  // Unit stride embedding dim
  matches &= (x_values.stride(-1) == 1);
  matches &= (output_values.stride(-1) == 1);
  matches &= (y_0_reshaped.stride(-1) == 1);
  matches &= (y_1_reshaped.stride(-1) == 1);

  // Each row is aligned to 128-bit
  matches &= (x_values.stride(-2) % 8 == 0);
  matches &= (output_values.stride(-2) % 8 == 0);
  matches &= (y_0_reshaped.stride(-2) % 8 == 0);
  matches &= (y_1_reshaped.stride(-2) % 8 == 0);

  // Base addresses aligned to 128-bit
  matches &= (reinterpret_cast<uint64_t>(x_values.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(output_values.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(y_0_reshaped.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(y_1_reshaped.data_ptr()) % 16 == 0);

  // Rows and col fit into int32_t
  matches &= (y_0_reshaped.size(0) < INT_MAX);
  matches &= (y_0_reshaped.size(1) < INT_MAX);

  int max_shared_bytes = 0;
#ifndef USE_ROCM
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_bytes,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      y_0_reshaped.get_device()));
#else
  // MI100 has 64 KB local memory (shared memory) per workgroup
  max_shared_bytes = 64 << 10;
#endif
  int shared_kb = max_shared_bytes >> 10;
#ifndef USE_ROCM
  // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
  int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
  TORCH_CHECK(used_shared_kb > 0);
#else
  // MI100 has independent shared mem and L1
  int used_shared_kb = shared_kb;
#endif
  auto used_shared_bytes = static_cast<size_t>(used_shared_kb << 10);
  AT_DISPATCH_INDEX_TYPES(
      x_offsets[0].scalar_type(), "check_shared_memory", [&] {
        auto B = y_0_reshaped.size(0);
        // the default shared memory on V100/A100/H100 is 48 KB from
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
        if ((B + 1) * sizeof(index_t) >= static_cast<size_t>(used_shared_bytes)) {
          matches = false;
        }
      });
  return matches;
}

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    auto [threads, blocks, jagged_dims_tensor] =                               \
        check_shape_and_partition_(x_values, x_offsets, y);                    \
    blocks.x = div_round_up(x_values.size(0), threads.y);                      \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                      \
        NUM_JAGGED_DIM,                                                        \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        [f] __device__(scalar_t x, scalar_t y, scalar_t /*unused*/)            \
            -> scalar_t { return f(x, y); });                                  \
  }

inline int calc_used_shared_bytes(const int device) {
    int max_shared_bytes;
#ifndef USE_ROCM
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shared_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    int shared_kb = max_shared_bytes >> 10;
#ifndef USE_ROCM
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    int used_shared_bytes = used_shared_kb << 10;
    return used_shared_bytes;
}

template <typename index_t>
inline void set_max_dynamic_shared_mem_size_for_opt_search_kernel(const int used_shared_bytes) {
#ifndef USE_ROCM
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
            index_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        used_shared_bytes)); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
}

///@addtogroup jagged-tensor-ops-cuda
template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_opt_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_reshaped,
          y_reshaped,
          output_values)) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(), "jagged_indices_fast_path", [=] {
          auto nnz = output_values.size(0);
          auto B = y_reshaped.size(0);
          auto E = y_reshaped.size(2);
          Tensor t_rows_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));
          Tensor t_cols_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));

          // Binary search
          size_t dynamic_smem_size = (B + 1) * sizeof(index_t);
          auto cur_max_shared_bytes =
              at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
          if (dynamic_smem_size > cur_max_shared_bytes) {
            int used_shared_bytes = calc_used_shared_bytes(y_reshaped.get_device());
            set_max_dynamic_shared_mem_size_for_opt_search_kernel<index_t>(used_shared_bytes);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK(dynamic_smem_size <= static_cast<size_t>(used_shared_bytes));
          }
          dim3 threads_bs = dim3(1024, 1, 1);
          dim3 blocks_bs = dim3(div_round_up(nnz, threads_bs.x), 1, 1);
          jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
              index_t>
              <<<blocks_bs,
                 threads_bs,
                 dynamic_smem_size,
                 at::cuda::getCurrentCUDAStream()>>>(
                  x_offsets[0]
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  B);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          // Gather kernel
          dim3 threads = dim3(16, 16, 1);
          dim3 blocks = dim3(1, div_round_up(nnz, threads.y), 1);
          if (blocks.y > 65535) {
            blocks.y = 65535;
          }
          jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
              index_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  output_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  x_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  y_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  y_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  E,
                  [f] __device__(__half x, __half y0, __half) -> __half {
                    // NB: added the static_casts here
                    return static_cast<__half>(
                        f(static_cast<scalar_t>(x), static_cast<scalar_t>(y0))
                    );
                  });
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }); // AT_DISPATCH
  } else {
    JAGGED_TENSOR_DISPATCH_DIMS();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

at::Tensor _fbgemm_jagged_to_padded_dense_forward(
    const Tensor& values,
    TensorList offsets,
    c10::IntArrayRef max_lengths,
    const double padding_value) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  AT_DISPATCH_V2(
      values.scalar_type(),
      "jagged_to_padded_dense",
      AT_WRAP([&] {
        scalar_t fill_value = _get_padding_value<scalar_t>(padding_value, values.is_floating_point());  // Clamp infinite sentinels to dtype min/max to avoid overflow
        jagged_dense_elementwise_dense_output_<scalar_t>(
            values_canonicalized,
            offsets.vec(),
            padded_values_view, // dummy not used in the lambda function
            padded_values_view,
           [] __device__(scalar_t x, scalar_t /*unused*/) -> scalar_t {
              return x;
            },
            fill_value);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool, kHalf, kBFloat16);

  return padded_values;
}

#define DISPATCH_DENSE_TO_JAGGED_CASE(TYPE)                          \
  AT_DISPATCH_CASE(TYPE, [&] {                                       \
    jagged_dense_elementwise_jagged_output_opt_<scalar_t>(           \
        values,                                                      \
        offsets.vec(),                                               \
        dense,                                                       \
        output,                                                      \
        [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t { \
          return y;                                                  \
        });                                                          \
  })

Tensor _fbgemm_dense_to_jagged_forward_symint(
    const Tensor& dense,
    TensorList offsets,
    std::optional<at::SymInt> total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dense.get_device());

  // clang-format off
  AT_DISPATCH_SWITCH(
      values.scalar_type(),
      "dense_to_jagged_gpu_op_forward",
      DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Half)
      // NB: removed this to build
      // DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Int)
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(
          at::ScalarType::Long,
          at::ScalarType::BFloat16,
          [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                values,
                offsets.vec(),
                dense,
                output,
                [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                  return y;
                }); // device lambda
          } // lambda
          ) // CASE_FLOATING_TYPES_AND
  ); // SWITCH
  // clang-format on

#undef DISPATCH_DENSE_TO_JAGGED_CASE

  return output;
}

} // namespace native
} // namespace at
