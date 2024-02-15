#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <typename dst_t, int dst_t_N, typename src_t, int src_t_N>
struct ConvertImpl {
  static inline VectorizedN<dst_t, dst_t_N> apply(
      const VectorizedN<src_t, src_t_N>& src) {
    constexpr int count = std::min(
        VectorizedN<src_t, src_t_N>::size(),
        VectorizedN<dst_t, dst_t_N>::size());
    __at_align__ src_t src_buf[VectorizedN<src_t, src_t_N>::size()];
    src.store(src_buf);
    __at_align__ dst_t dst_buf[VectorizedN<dst_t, dst_t_N>::size()];
    for (int i = 0; i < count; i++) {
      dst_buf[i] = static_cast<dst_t>(src_buf[i]);
    }
    return VectorizedN<dst_t, dst_t_N>::loadu(dst_buf, count);
  }
};

template <typename dst_t, typename src_t>
inline Vectorized<dst_t> convert(const Vectorized<src_t>& src) {
  return ConvertImpl<dst_t, 1, src_t, 1>::apply(src);
}

template <
    typename dst_t,
    int dst_t_N,
    typename src_t,
    int src_t_N,
    std::enable_if_t<src_t_N == 1 && dst_t_N != 1, int> = 0>
inline VectorizedN<dst_t, dst_t_N> convert(const Vectorized<src_t>& src) {
  return ConvertImpl<dst_t, dst_t_N, src_t, 1>::apply(src);
}

template <
    typename dst_t,
    int dst_t_N,
    typename src_t,
    int src_t_N,
    std::enable_if_t<src_t_N != 1 && dst_t_N != 1, int> = 0>
inline VectorizedN<dst_t, dst_t_N> convert(
    const VectorizedN<src_t, src_t_N>& src) {
  return ConvertImpl<dst_t, dst_t_N, src_t, src_t_N>::apply(src);
}

template <
    typename dst_t,
    int dst_t_N,
    typename src_t,
    int src_t_N,
    std::enable_if_t<src_t_N != 1 && dst_t_N == 1, int> = 0>
inline Vectorized<dst_t> convert(const VectorizedN<src_t, src_t_N>& src) {
  return ConvertImpl<dst_t, dst_t_N, src_t, src_t_N>::apply(src);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
