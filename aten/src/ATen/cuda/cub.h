#pragma once
#include <cstdint>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDAConfig.h>

// NOTE: These templates are intentionally not defined in this header,
// which avoids re-compiling them for each translation unit. If you get
// a link error, you need to add an explicit instantiation for your
// types in cub.cu

namespace at::cuda::cub {

inline int get_num_bits(uint64_t max_key) {
  int num_bits = 1;
  while (max_key > 1) {
    max_key >>= 1;
    num_bits++;
  }
  return num_bits;
}

namespace detail {

// radix_sort_pairs doesn't interact with value_t other than to copy
// the data, so we can save template instantiations by reinterpreting
// it as an opaque type.
// We use native integer types for 1/2/4/8-byte values to reduce
// register usage in CUDA kernels. For sizes > 8 fall back to char array.
template <int N> struct alignas(N) OpaqueType { char data[N]; };
template <> struct alignas(1) OpaqueType<1> { uint8_t data; };
template <> struct alignas(2) OpaqueType<2> { uint16_t data; };
template <> struct alignas(4) OpaqueType<4> { uint32_t data; };
template <> struct alignas(8) OpaqueType<8> { uint64_t data; };

template<typename key_t, int value_size>
void radix_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const OpaqueType<value_size> *values_in, OpaqueType<value_size> *values_out,
    int64_t n, bool descending, int64_t begin_bit, int64_t end_bit);

}  // namespace detail

template<typename key_t, typename value_t>
void radix_sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8) {
  static_assert(std::is_trivially_copyable_v<value_t> ||
                AT_ROCM_ENABLED(),  // ROCm incorrectly fails this check for vector types
                "radix_sort_pairs value type must be trivially copyable");
  // Make value type opaque, so all inputs of a certain size use the same template instantiation
  using opaque_t = detail::OpaqueType<sizeof(value_t)>;
  static_assert(sizeof(value_t) <= 8 && (sizeof(value_t) & (sizeof(value_t) - 1)) == 0,
                "This size of value_t is not instantiated. Please instantiate it in cub.cu"
                " and modify this check.");
  static_assert(sizeof(value_t) == alignof(value_t), "Expected value_t to be size-aligned");
  detail::radix_sort_pairs_impl(
      keys_in, keys_out,
      reinterpret_cast<const opaque_t*>(values_in),
      reinterpret_cast<opaque_t*>(values_out),
      n, descending, begin_bit, end_bit);
}

template<typename key_t>
void radix_sort_keys(
    const key_t *keys_in, key_t *keys_out,
    int64_t n, bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8);

// NOTE: Intermediate sums will be truncated to input_t precision
template <typename input_t, typename output_t>
void inclusive_sum_truncating(const input_t *input, output_t *output, int64_t n);

template <typename scalar_t>
void inclusive_sum(const scalar_t *input, scalar_t *output, int64_t n) {
  return inclusive_sum_truncating(input, output, n);
}

// NOTE: Sums are done is common_type<input_t, output_t>
template <typename input_t, typename output_t>
void exclusive_sum_in_common_type(const input_t *input, output_t *output, int64_t n);

template <typename scalar_t>
void exclusive_sum(const scalar_t *input, scalar_t *output, int64_t n) {
  return exclusive_sum_in_common_type(input, output, n);
}

void mask_exclusive_sum(const uint8_t *mask, int64_t *output_idx, int64_t n);
inline void mask_exclusive_sum(const bool *mask, int64_t *output_idx, int64_t n) {
  return mask_exclusive_sum(
      reinterpret_cast<const uint8_t*>(mask), output_idx, n);
}

}  // namespace at::cuda::cub
