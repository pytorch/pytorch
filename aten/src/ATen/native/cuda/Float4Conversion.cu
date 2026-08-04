#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/Float4_e2m1fn_x2.h>
#include <c10/util/bit_cast.h>

#include <cuda.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#define AT_FP4_HAS_CVT_INTRINSIC 1
#endif

#include <array>
#include <type_traits>

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

// Pack two fp32 values into one fp4x2 byte (val_a -> low nibble, val_b -> high).
// On sm_100+ this uses the hardware fp4 convert intrinsic; the software encoder
// below matches its numerics exactly (including NaN -> positive max magnitude),
// so the two paths produce identical bytes on every architecture.
C10_HOST_DEVICE inline uint8_t pack_two_fp4(float a, float b) {
#if defined(AT_FP4_HAS_CVT_INTRINSIC) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 1000
  return __nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest);
#else
  uint8_t lo = c10::detail::fp4e2m1_from_fp32_value(a);
  uint8_t hi = c10::detail::fp4e2m1_from_fp32_value(b);
  return static_cast<uint8_t>((hi << 4) | lo);
#endif
}

} // namespace

Tensor _convert_to_float4_e2m1fn_x2_cuda(const Tensor& self) {
  // float64 is rejected: the encode narrows through an intermediate float32, and
  // a double within half a float32 ULP of an exact fp4 midpoint would double-round.
  // TODO(future PR): implement cast from float64 if there is a need, for now
  // not worth the extra complexity
  TORCH_CHECK(
      isFloatingType(self.scalar_type()) &&
          self.scalar_type() != kFloat4_e2m1fn_x2 &&
          self.scalar_type() != kDouble,
      "conversion to Float4_e2m1fn_x2 is only supported from a floating point "
      "dtype other than float64, got ",
      self.scalar_type());
  // Require a contiguous input: the pack reinterprets each pair of last-dim
  // values as one wider element, which assumes row-major layout. Supporting
  // arbitrary strides would need stride-aware kernels, which is not worth it
  // right now.
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

  auto sizes = self.sizes().vec();
  sizes.back() /= 2;
  auto out = at::empty(sizes, self.options().dtype(kFloat4_e2m1fn_x2));
  if (out.numel() == 0) {
    return out;
  }

  // Pack two consecutive inputs into each output byte. We reinterpret each input
  // pair as a single element twice as wide (fp32->fp64, bf16/fp16->fp32) so the
  // cast becomes a 1:1 elementwise map; TensorIterator then handles vectorized
  // loads/stores instead of a hand-written kernel.
  auto out_bytes = out.view(kByte);
  AT_DISPATCH_V2(
      self.scalar_type(),
      "_convert_to_float4_e2m1fn_x2_cuda",
      AT_WRAP([&] {
        constexpr bool is_4byte = std::is_same_v<scalar_t, float>;
        using wide_t = std::conditional_t<is_4byte, double, float>;
        auto input = self.view(is_4byte ? kDouble : kFloat);
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(out_bytes)
                        .add_input(input)
                        .build();
        gpu_kernel(iter, [] GPU_LAMBDA(wide_t packed) -> uint8_t {
          auto pair = c10::bit_cast<std::array<scalar_t, 2>>(packed);
          return pack_two_fp4(
              static_cast<float>(pair[0]), static_cast<float>(pair[1]));
        });
      }),
      kFloat,
      kHalf,
      kBFloat16);
  return out;
}

Tensor _convert_from_float4_e2m1fn_x2_cuda(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == kFloat4_e2m1fn_x2,
      "conversion from Float4_e2m1fn_x2 is only supported from a "
      "Float4_e2m1fn_x2 dtype, got ",
      self.scalar_type());
  // Require a contiguous input: the unpack reinterprets each pair of output
  // values as one wider element, which assumes row-major layout. Supporting
  // arbitrary strides would need stride-aware kernels, which is not worth it
  // right now.
  TORCH_CHECK(
      self.is_contiguous(),
      "conversion from Float4_e2m1fn_x2 requires a contiguous input");
  TORCH_CHECK(
      self.dim() >= 1,
      "conversion from Float4_e2m1fn_x2 requires at least 1 dimension, got shape ",
      self.sizes());

  auto sizes = self.sizes().vec();
  sizes.back() *= 2;
  auto out = at::empty(sizes, self.options().dtype(kFloat));
  if (self.numel() == 0) {
    return out;
  }

  // Unpack each byte into two floats. We reinterpret each output pair as one
  // wider float64 element so the map is 1:1 (uint8 -> float64), reusing
  // TensorIterator's vectorized path.
  auto in_bytes = self.view(kByte);
  auto out_wide = out.view(kDouble);
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(out_wide)
                  .add_input(in_bytes)
                  .build();
  gpu_kernel(iter, [] GPU_LAMBDA(uint8_t byte) -> double {
    std::array<float, 2> pair{
        c10::detail::fp4e2m1_to_fp32_value(byte & 0xF),
        c10::detail::fp4e2m1_to_fp32_value(byte >> 4)};
    return c10::bit_cast<double>(pair);
  });
  return out;
}

} // namespace at::native
