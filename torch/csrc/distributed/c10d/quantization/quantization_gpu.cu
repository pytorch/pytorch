#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/quantization/quantization_gpu.h>
#include <torch/csrc/distributed/c10d/quantization/quantization_utils.h>
#include <torch/library.h>

// TODO: The kernels are copied from fbgemm_gpu, we should dedup them later

// FP32 -> BF16 kernel
__global__ void _float_to_bfloat16_cuda_kernel(
    const float* __restrict__ input,
    const size_t nrows,
    const size_t ncols,
    uint16_t* __restrict__ output) {
  const auto row_incre = blockDim.y * gridDim.y;
  const auto col_incre = blockDim.x * gridDim.x;
  for (auto row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;
    for (auto col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
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
    const size_t nrows,
    const size_t ncols,
    float* __restrict__ output) {
  const auto row_incre = blockDim.y * gridDim.y;
  const auto col_incre = blockDim.x * gridDim.x;
  for (auto row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    for (auto col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
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

namespace torch::distributed::c10d::quantization {

at::Tensor _float_to_bfloat16_cuda(const at::Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto nrows = input.size(0);
  const auto ncols = input.size(1);
  const size_t output_columns = ncols;

  auto output = at::empty(
      {nrows, ncols},
#if HAS_NCCL_BF16_DATATYPE
      input.options().dtype(at::kBFloat16));
#else
      input.options().dtype(at::kHalf));
#endif

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr size_t threads_per_block = 256;
  const auto blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const auto gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  const auto gridDim_y =
      std::min<size_t>((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _float_to_bfloat16_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.const_data_ptr<float>(),
      nrows,
      ncols,
#if HAS_NCCL_BF16_DATATYPE
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr<at::BFloat16>())
#else
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr<at::Half>())
#endif
      );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

at::Tensor _bfloat16_to_float_cuda(const at::Tensor& input) {
  TENSOR_ON_CUDA_GPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto nrows = input.size(0);
  const auto ncols = input.size(1);
  const size_t output_columns = ncols;

  auto output = at::empty(
      {nrows, ncols}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t

  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr size_t threads_per_block = 256;

  const auto blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const auto gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  const auto gridDim_y =
      std::min<size_t>((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _bfloat16_to_float_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
#if HAS_NCCL_BF16_DATATYPE
      reinterpret_cast<const uint16_t*>(input.const_data_ptr<at::BFloat16>()),
#else
      reinterpret_cast<const uint16_t*>(input.const_data_ptr<at::Half>()),
#endif
      nrows,
      ncols,
      output.mutable_data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

#define DISPATCH_TO_CUDA(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))

TORCH_LIBRARY_IMPL(quantization, CUDA, m) {
  DISPATCH_TO_CUDA("_Bfloat16QuantizedToFloat", _bfloat16_to_float_cuda);
  DISPATCH_TO_CUDA("_FloatToBfloat16Quantized", _float_to_bfloat16_cuda);
}

} // namespace torch::distributed::c10d::quantization
