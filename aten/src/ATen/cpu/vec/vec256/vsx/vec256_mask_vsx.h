#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_VSX)

template <int N>
struct VecMaskCast<int, N, float, N> {
  static inline VecMask<int, N> apply(const VecMask<float, N>& vec_mask) {
    VectorizedN<int, N> result;

    for (int i = 0; i < N; ++i) {
      auto tmp = vec_mask[i];
      result[i] = reinterpret_cast<const Vectorized<int>&>(tmp);
    }
    return VecMask<int, N>(result);
  }
};

template <int N>
struct VecMaskCast<float, N, int, N> {
  static inline VecMask<float, N> apply(const VecMask<int, N>& vec_mask) {
    VectorizedN<float, N> result;

    for (int i = 0; i < N; ++i) {
      auto tmp = vec_mask[i];
      result[i] = reinterpret_cast<const Vectorized<float>&>(tmp);
    }
    return VecMask<float, N>(result);
  }
};

template <int dst_n, typename mask_t, int mask_n>
struct VecMaskCast<
    int64_t,
    dst_n,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (dst_n == 2 * mask_n) &&
        (std::is_same_v<mask_t, float> || std::is_same_v<mask_t, int>)>> {
  static inline VecMask<int64_t, dst_n> apply(
      const VecMask<mask_t, mask_n>& vec_mask) {
    VectorizedN<int64_t, dst_n> result;

    auto int_mask = vec_mask.template cast<int, mask_n>();

    for (int i = 0; i < mask_n; ++i) {
      VectorizedN<int, 1> in_int_n;
      in_int_n[0] = int_mask[i];

      auto int64_vecs = convert<int64_t, 2, int, 1>(in_int_n);

      result[2 * i] = int64_vecs[0];
      result[2 * i + 1] = int64_vecs[1];
    }
    return VecMask<int64_t, dst_n>(result);
  }
};

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
