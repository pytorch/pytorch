#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/mxfp_mm_kernel.h>
#include <c10/util/bit_cast.h>
#include <c10/util/irange.h>

namespace at::native {

#if defined(CPU_CAPABILITY_DEFAULT)
const float* get_mxfp8_values() {
  static const auto values = [] {
    std::array<float, 256> result{};
    for (int encoding = 0; encoding < 256; ++encoding) {
      result[encoding] = static_cast<float>(c10::Float8_e4m3fn(
          static_cast<uint8_t>(encoding), c10::Float8_e4m3fn::from_bits()));
    }
    return result;
  }();
  return values.data();
}
#endif

namespace {

constexpr std::array<float, 16> mxfp4_values = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
C10_ALWAYS_INLINE float apply_mxfp_scale(
    float value, uint8_t scale_a, uint8_t scale_b) {
  if (scale_a == 0xff || scale_b == 0xff) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const int exponent = static_cast<int>(scale_a) +
      static_cast<int>(scale_b) - 254;
  const int biased_exponent = exponent + 127;
  if (biased_exponent > 0 && biased_exponent < 255) {
    const auto bits = static_cast<uint32_t>(biased_exponent) << 23;
    return value * c10::bit_cast<float>(bits);
  }
  return std::scalbn(value, exponent);
}

void mxfp_mm_emulated_kernel(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const Tensor& bias_data,
    Tensor& out) {
  const bool is_fp4 = mat_a.scalar_type() == ScalarType::Float4_e2m1fn_x2;
  const int64_t m = mat_a.size(0);
  const int64_t k = mat_a.size(1) * (is_fp4 ? 2 : 1);
  const int64_t n = mat_b.size(1);
  const int64_t groups = (k + 31) / 32;
  const auto* scale_a_data = scale_a.const_data_ptr<c10::Float8_e8m0fnu>();
  const auto* scale_b_data = scale_b.const_data_ptr<c10::Float8_e8m0fnu>();
  const auto* bias_f32 =
      bias_data.defined() && bias_data.scalar_type() == ScalarType::Float
      ? bias_data.const_data_ptr<float>()
      : nullptr;
  const auto* bias_f16 =
      bias_data.defined() && bias_data.scalar_type() == ScalarType::Half
      ? bias_data.const_data_ptr<c10::Half>()
      : nullptr;
  const auto* bias_bf16 =
      bias_data.defined() && bias_data.scalar_type() == ScalarType::BFloat16
      ? bias_data.const_data_ptr<c10::BFloat16>()
      : nullptr;
  auto* out_f32 = out.scalar_type() == ScalarType::Float
      ? out.mutable_data_ptr<float>()
      : nullptr;
  auto* out_bf16 = out.scalar_type() == ScalarType::BFloat16
      ? out.mutable_data_ptr<c10::BFloat16>()
      : nullptr;

  auto run = [&](auto load_a, auto load_b) {
    const int64_t grain_size =
        std::max<int64_t>(1, at::internal::GRAIN_SIZE / k);
    at::parallel_for(0, m * n, grain_size, [&](int64_t begin, int64_t end) {
      for (const auto index : c10::irange(begin, end)) {
        const int64_t row = index / n;
        const int64_t column = index % n;
        float result = 0.0f;
        for (const auto group : c10::irange(groups)) {
          float partial = 0.0f;
          const int64_t group_start = group * 32;
          const int64_t group_size = std::min<int64_t>(32, k - group_start);
          for (const auto offset : c10::irange(group_size)) {
            const int64_t inner = group_start + offset;
            partial += load_a(row, inner) * load_b(inner, column);
          }
          result += apply_mxfp_scale(
              partial,
              scale_a_data[row * groups + group].x,
              scale_b_data[column * groups + group].x);
        }
        if (bias_f32 != nullptr) {
          result += bias_f32[column];
        } else if (bias_f16 != nullptr) {
          result += static_cast<float>(bias_f16[column]);
        } else if (bias_bf16 != nullptr) {
          result += static_cast<float>(bias_bf16[column]);
        }
        if (out_f32 != nullptr) {
          out_f32[index] = result;
        } else {
          out_bf16[index] = c10::BFloat16(result);
        }
      }
    });
  };

  if (is_fp4) {
    const int64_t packed_k = k / 2;
    const auto* a = static_cast<const uint8_t*>(mat_a.const_data_ptr());
    const auto* b = static_cast<const uint8_t*>(mat_b.const_data_ptr());
    run(
        [&](int64_t row, int64_t inner) {
          const uint8_t packed = a[row * packed_k + inner / 2];
          return mxfp4_values[(packed >> (4 * (inner % 2))) & 0x0f];
        },
        [&](int64_t inner, int64_t column) {
          const uint8_t packed = b[column * packed_k + inner / 2];
          return mxfp4_values[(packed >> (4 * (inner % 2))) & 0x0f];
        });
  } else {
    const auto* a = mat_a.const_data_ptr<c10::Float8_e4m3fn>();
    const auto* b = mat_b.const_data_ptr<c10::Float8_e4m3fn>();
    const auto* mxfp8_values = get_mxfp8_values();
    run(
        [&](int64_t row, int64_t inner) {
          return mxfp8_values[a[row * k + inner].x];
        },
        [&](int64_t inner, int64_t column) {
          return mxfp8_values[b[column * k + inner].x];
        });
  }
}

} // namespace

REGISTER_DISPATCH(mxfp_mm_stub, &mxfp_mm_emulated_kernel)

} // namespace at::native
