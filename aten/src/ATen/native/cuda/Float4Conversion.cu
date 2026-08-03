#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float4_e2m1fn_x2.h>

#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/_convert_to_float4_e2m1fn_x2_native.h>
#include <ATen/ops/_convert_from_float4_e2m1fn_x2_native.h>
#endif

namespace at::native {

namespace {

__global__ void convert_to_float4_e2m1fn_x2_kernel(
    const float* in,
    uint8_t* out,
    int64_t n_out) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_out;
       i += stride) {
    uint8_t lo = c10::detail::fp4e2m1_from_fp32_value(in[2 * i]);
    uint8_t hi = c10::detail::fp4e2m1_from_fp32_value(in[2 * i + 1]);
    out[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
}

__global__ void convert_from_float4_e2m1fn_x2_kernel(
    const uint8_t* in,
    float* out,
    int64_t n_in) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_in;
       i += stride) {
    const uint8_t byte = in[i];
    out[2 * i] = c10::detail::fp4e2m1_to_fp32_value(byte & 0xF);
    out[2 * i + 1] = c10::detail::fp4e2m1_to_fp32_value(byte >> 4);
  }
}

} // namespace

Tensor _convert_to_float4_e2m1fn_x2_cuda(const Tensor& self) {
  // float64 is rejected: the kernel narrows the input through an intermediate
  // float32 (self.to(kFloat)), and a double within half a float32 ULP of an
  // exact fp4 midpoint would double-round to the wrong bucket.
  // TODO(future PR): implement cast from float64 if there is a need, for now
  // not worth the extra complexity
  TORCH_CHECK(
      isFloatingType(self.scalar_type()) &&
          self.scalar_type() != kFloat4_e2m1fn_x2 &&
          self.scalar_type() != kDouble,
      "conversion to Float4_e2m1fn_x2 is only supported from a floating point "
      "dtype other than float64, got ",
      self.scalar_type());
  // Require a contiguous input: the pack kernel assumes row-major layout (two
  // fp4 values share a byte along the last dim). Supporting arbitrary strides
  // would need stride-aware kernels, which is not worth it right now.
  TORCH_CHECK(
      self.is_contiguous(),
      "conversion to Float4_e2m1fn_x2 requires a contiguous input");
  TORCH_CHECK(
      self.dim() >= 1,
      "conversion to Float4_e2m1fn_x2 requires at least 1 dimension, got a 0-dim tensor");
  TORCH_CHECK(
      self.size(-1) % 2 == 0,
      "conversion to Float4_e2m1fn_x2 requires the last dimension to be even, got shape ",
      self.sizes());

  auto input = self.to(kFloat);
  auto sizes = input.sizes().vec();
  sizes.back() /= 2;
  auto out = at::empty(sizes, input.options().dtype(kFloat4_e2m1fn_x2));

  const int64_t n_out = out.numel();
  if (n_out == 0) {
    return out;
  }

  constexpr int threads = 256;
  const int64_t blocks = std::min<int64_t>(
      (n_out + threads - 1) / threads, static_cast<int64_t>(65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  // TODO(future PR): use hardware intrinsics instead of bitshifting on
  // CUDA 10.0+
  convert_to_float4_e2m1fn_x2_kernel<<<blocks, threads, 0, stream>>>(
      input.const_data_ptr<float>(),
      reinterpret_cast<uint8_t*>(out.data_ptr()),
      n_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor _convert_from_float4_e2m1fn_x2_cuda(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == kFloat4_e2m1fn_x2,
      "conversion from Float4_e2m1fn_x2 is only supported from a "
      "Float4_e2m1fn_x2 dtype, got ",
      self.scalar_type());
  // Require a contiguous input: the unpack kernel assumes row-major layout (two
  // fp4 values share a byte along the last dim). Supporting arbitrary strides
  // would need stride-aware kernels, which is not worth it right now.
  TORCH_CHECK(
      self.is_contiguous(),
      "conversion from Float4_e2m1fn_x2 requires a contiguous input");
  const Tensor& input = self;
  auto sizes = input.sizes().vec();
  TORCH_CHECK(
      !sizes.empty(),
      "conversion from Float4_e2m1fn_x2 requires at least 1 dimension, got shape ",
      input.sizes());
  sizes.back() *= 2;
  auto out = at::empty(sizes, input.options().dtype(kFloat));

  const int64_t n_in = input.numel();
  if (n_in == 0) {
    return out;
  }

  constexpr int threads = 256;
  const int64_t blocks = std::min<int64_t>(
      (n_in + threads - 1) / threads, static_cast<int64_t>(65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  convert_from_float4_e2m1fn_x2_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(input.const_data_ptr()),
      out.data_ptr<float>(),
      n_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

} // namespace at::native
