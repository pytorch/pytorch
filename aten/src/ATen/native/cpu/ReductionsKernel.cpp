#include <ATen/native/Reductions.h>
#include <immintrin.h>
#include <limits>

namespace at {
namespace native {

namespace {

namespace op {

C10_ALWAYS_INLINE float maximum(float a, float b) noexcept {
  // Assumes IEEE 754 floating point
  return std::isnan(a) || b < a ? a : b;
}

} // namespace op

namespace simd {

C10_ALWAYS_INLINE __m512 maximum(__m512 a, __m512 b) noexcept {
  const auto nan_mask = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  return _mm512_mask_max_ps(a, nan_mask, a, b);
}

C10_ALWAYS_INLINE float reduce_maximum(__m512 a) noexcept {
  const auto nan_mask = _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q);
  return _mm512_mask2int(nan_mask) != 0
      ? std::numeric_limits<float>::quiet_NaN()
      : _mm512_reduce_max_ps(a);
}

} // namespace simd

namespace kernel {

C10_ALWAYS_INLINE void max(
    float* C10_RESTRICT out,
    const float* C10_RESTRICT in,
    std::size_t n) noexcept {
  constexpr int N = 16;

  constexpr float identity = std::numeric_limits<float>::has_infinity
      ? -std::numeric_limits<float>::infinity()
      : std::numeric_limits<float>::lowest();

  std::size_t i = 0;

  const auto load = [&]() noexcept {
    const auto result = _mm512_loadu_ps(in + i);
    i += N;
    return result;
  };

  // Vectorized loops
  if (n >= (N << 0)) {
    auto result_0 = load();

    // Unroll 2
    if (n >= (N << 1)) {
      auto result_1 = load();

      // Unroll 4
      if (n >= (N << 2)) {
        auto result_2 = load();
        auto result_3 = load();

        while (i < n - (n % (N << 2))) {
          result_0 = simd::maximum(result_0, load());
          result_1 = simd::maximum(result_1, load());
          result_2 = simd::maximum(result_2, load());
          result_3 = simd::maximum(result_3, load());
        }

        result_0 = simd::maximum(result_0, result_2);
        result_1 = simd::maximum(result_1, result_3);
      }

      if (i < n - (n % (N << 1))) {
        result_0 = simd::maximum(result_0, load());
        result_1 = simd::maximum(result_1, load());
      }

      result_0 = simd::maximum(result_0, result_1);
    }

    if (i < n - (n % (N << 0))) {
      result_0 = simd::maximum(result_0, load());
    }

    *out = simd::reduce_maximum(result_0);
  } else {
    *out = identity;
  }

  // Cleanup
  for (; i < n; ++i) {
    *out = op::maximum(*out, in[i]);
  }
}

} // namespace kernel

namespace impl {

void max(const Tensor& in, const Tensor& out) {
  kernel::max(out.data_ptr<float>(), in.data_ptr<float>(), in.numel());
}

} // namespace impl

} // namespace

REGISTER_DISPATCH(_max_stub, &impl::max);

} // namespace native
} // namespace at
