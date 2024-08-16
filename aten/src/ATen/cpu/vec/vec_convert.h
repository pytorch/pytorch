#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecConvert {
  static inline VectorizedN<dst_t, dst_n> apply(
      const VectorizedN<src_t, src_n>& src) {
    constexpr int count = std::min(
        VectorizedN<src_t, src_n>::size(), VectorizedN<dst_t, dst_n>::size());
    __at_align__ src_t src_buf[VectorizedN<src_t, src_n>::size()];
    src.store(src_buf);
    __at_align__ dst_t dst_buf[VectorizedN<dst_t, dst_n>::size()];
    for (int i = 0; i < count; i++) {
      dst_buf[i] = static_cast<dst_t>(src_buf[i]);
    }
    return VectorizedN<dst_t, dst_n>::loadu(dst_buf, count);
  }
};

template <typename dst_t, typename src_t>
inline std::enable_if_t<std::is_same_v<dst_t, src_t>, Vectorized<src_t>>
convert(const Vectorized<src_t>& src) {
  return src;
}

template <typename dst_t, typename src_t>
inline std::enable_if_t<!std::is_same_v<dst_t, src_t>, Vectorized<dst_t>>
convert(const Vectorized<src_t>& src) {
  return VecConvert<dst_t, 1, src_t, 1>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    std::enable_if_t<dst_n != 1, int> = 0>
inline VectorizedN<dst_t, dst_n> convert(const VectorizedN<src_t, src_n>& src) {
  return VecConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    bool keep = false,
    std::enable_if_t<dst_n == 1, int> = 0>
inline std::conditional_t<keep, VectorizedN<dst_t, 1>, Vectorized<dst_t>>
convert(const VectorizedN<src_t, src_n>& src) {
  return VecConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
