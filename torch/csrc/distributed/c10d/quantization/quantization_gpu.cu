#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/quantization/quantization_gpu.h>
#include <torch/csrc/distributed/c10d/quantization/quantization_utils.h>
#include <torch/library.h>

// TODO: The kernels are copied from fbgemm_gpu, we should dedup them later

// FP32 -> BF16 kernel
__global__ void _float_to_bfloat16_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    uint16_t* __restrict__ output) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      // Add 2^15 and right shift 16 to do round-nearest
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

// BF16 -> FP32 kernel
__global__ void _bfloat16_to_float_cuda_kernel(
    const uint16_t* __restrict__ input,
    const int nrows,
    const int ncols,
    float* __restrict__ output) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      const uint16_t* input_row = input + row * ncols;
      float* output_row = output + row * ncols;
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

namespace torch {
namespace distributed {
namespace c10d {
namespace quantization {

at::Tensor _float_to_bfloat16_cuda(const at::Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int output_columns = ncols;

  auto output = at::empty(
      {nrows, output_columns},
      input.options().dtype(at::kHalf)); // at::kHalf

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  // TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia
  // NCCL input.options().dtype(at::kBFloat16)); // at::kBFloat16

  constexpr int threads_per_block = 256;
  const int blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const int gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  const int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _float_to_bfloat16_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(),
      nrows,
      ncols,
      // TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia
      // NCCL
      reinterpret_cast<uint16_t*>(output.data_ptr<at::Half>()));
  //C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

at::Tensor _bfloat16_to_float_cuda(const at::Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);
  const int output_columns = ncols;

  auto output = at::empty(
      {nrows, output_columns}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t

  if (nrows == 0 || output_columns == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const int gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  const int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _bfloat16_to_float_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      // TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia
      // NCCL
      reinterpret_cast<uint16_t*>(input.data_ptr<at::Half>()),
      nrows,
      ncols,
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

#define DISPATCH_TO_CUDA(name, function) \
    m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))

TORCH_LIBRARY_IMPL(quantization, CUDA, m) {
    DISPATCH_TO_CUDA("_Bfloat16QuantizedToFloat", _bfloat16_to_float_cuda);
    DISPATCH_TO_CUDA("_FloatToBfloat16Quantized", _float_to_bfloat16_cuda);
}

} // namespace quantization
} // namespace c10d
} // namespace distributed
} // namespace torch
