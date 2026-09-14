#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace at::native {

// One warp computes one output element out[b, n] = sum_k x[b,k]*w[n,k] * scale[n].
// Lanes stride over K so both x and w reads are coalesced across the warp, and
// the K reduction runs in parallel and finishes with a warp shuffle. When K is a
// multiple of 4 the row is read 4 elements at a time (float4 / char4).
template <int kWarpSize = 32>
__global__ void weight_int8pack_mm_kernel(
    const float* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t B,
    int64_t K,
    int64_t N) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t output_count = B * N;
  const int64_t output_stride =
      static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t out_idx =
           static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
       out_idx < output_count;
       out_idx += output_stride) {
    const int64_t b = out_idx / N;
    const int64_t n = out_idx % N;
    const float* x_row = x + b * K;
    const int8_t* w_row = w + n * K;

    float acc = 0.0f;
    if ((K & 3) == 0 &&
        reinterpret_cast<uintptr_t>(x_row) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(w_row) % 4 == 0) {
      const int64_t k4 = K >> 2;
      const float4* x_row4 = reinterpret_cast<const float4*>(x_row);
      const char4* w_row4 = reinterpret_cast<const char4*>(w_row);
      for (int64_t j = lane; j < k4; j += kWarpSize) {
        const float4 xv = x_row4[j];
        const char4 wv = w_row4[j];
        acc += xv.x * static_cast<float>(wv.x) +
            xv.y * static_cast<float>(wv.y) +
            xv.z * static_cast<float>(wv.z) + xv.w * static_cast<float>(wv.w);
      }
    } else {
      for (int64_t k = lane; k < K; k += kWarpSize) {
        acc += x_row[k] * static_cast<float>(w_row[k]);
      }
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      acc += WARP_SHFL_DOWN(acc, offset);
    }

    if (lane == 0) {
      out[out_idx] = acc * scale[n];
    }
  }
}

void launch_weight_int8pack_mm_cuda_kernel(
    const Tensor& x,
    const Tensor& w_int8,
    const Tensor& scale,
    Tensor& out) {
  const int64_t B = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = w_int8.size(0);

  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  const dim3 block(kWarpSize * kWarpsPerBlock);
  const int64_t num_warps = B * N;
  const int64_t requested_grid_x =
      (num_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t max_grid_x =
      at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  const dim3 grid(std::min(requested_grid_x, max_grid_x));

  auto stream = at::cuda::getCurrentCUDAStream();

  weight_int8pack_mm_kernel<kWarpSize><<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      w_int8.data_ptr<int8_t>(),
      scale.const_data_ptr<float>(),
      out.data_ptr<float>(),
      B,
      K,
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Main GPU entry point
at::Tensor _weight_int8pack_mm_cuda(
    const at::Tensor& x,
    const at::Tensor& w_int8,
    const at::Tensor& scale) {
  // --- Check inputs ---
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w_int8.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_int8.dim() == 2, "w must be 2D");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D");

  TORCH_CHECK(
      x.size(1) == w_int8.size(1),
      "K dimension mismatch: x.size(1) != w.size(1)");
  TORCH_CHECK(
      w_int8.size(0) == scale.size(0),
      "Output dim mismatch: w.size(0) != scale.size(0)");

  // --- Determine shapes ---
  auto B = x.size(0); // batch size
  auto N = w_int8.size(0); // output dim

  // Ensure inputs are in the correct types for the kernel
  auto x_f32 = x.to(
      at::kFloat, /*non_blocking=*/false, /*copy=*/false, at::MemoryFormat::Contiguous);
  auto w_int8_contiguous = w_int8.contiguous();
  auto scale_f32 = scale.to(
      at::kFloat, /*non_blocking=*/false, /*copy=*/false, at::MemoryFormat::Contiguous);

  // --- Allocate output ---
  auto out = at::empty({B, N}, x_f32.options());

  // --- Launch kernel ---
  launch_weight_int8pack_mm_cuda_kernel(
      x_f32, w_int8_contiguous, scale_f32, out);

  return out.to(x.dtype());
}

} // namespace at::native
