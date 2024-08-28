#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

/**
 * The `VecMask` class provides a convenient interface for working with
 * vectorized masks in SIMD operations. It encapsulates a `Vectorized<T, N>`
 * mask that can be directly usable in masked vectorized operations. It provides
 * various methods for manipulating and accessing the mask elements:
 * 1. `from` and `to`: Conversion between a vector of boolean values and a
 * vectorized mask.
 * 2. `cast`: Casts the mask to a different base type.
 * 3. `all_zero`: Checks if all mask elements are zero.
 * 4. `is_masked`: Checks if a specific element is masked.
 * 5. `loadu`: Loads data from memory using the mask.
 * 6. `all_masked`: Checks if all mask elements are masked.
 *
 * Some helper template classes are provided to simplify the specialization of
 * the `VecMask` for the specific CPU arch:
 * 1. `VecMaskLoad`: Loads data from memory using the mask.
 * 2. `VecMaskTo`: Converts the mask to boolean.
 * 3. `VecMaskCast`: Casts the mask to a different base type.
 *
 */
template <typename T, int N>
class VecMask;

template <
    typename data_t,
    int data_n,
    typename mask_t,
    int mask_n,
    typename Enabled = void>
struct VecMaskLoad {
  static inline VectorizedN<data_t, data_n> apply(
      const data_t* ptr,
      const VecMask<mask_t, mask_n>& vec_mask) {
    constexpr typename VecMask<mask_t, mask_n>::size_type size =
        VecMask<mask_t, mask_n>::size();
    static_assert(VectorizedN<data_t, data_n>::size() >= size);
    __at_align__ data_t data[size];
    __at_align__ mask_t mask[size];
    auto mask_ = VectorizedN<mask_t, mask_n>(vec_mask);
    mask_.store(mask);
    for (int i = 0; i < size; i++) {
      data[i] = mask[i] ? ptr[i] : static_cast<data_t>(0);
    }
    return VectorizedN<data_t, data_n>::loadu(data, size);
  }
};

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecMaskTo {
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    auto zeros = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(0));
    auto ones = VectorizedN<dst_t, dst_n>(static_cast<dst_t>(1));
    return VectorizedN<dst_t, dst_n>::blendv(
        zeros, ones, vec_mask.template cast<dst_t, dst_n>());
  }
};

template <typename dst_t, int dst_n, typename src_t, int src_n>
struct VecMaskCast {
  static inline VecMask<dst_t, dst_n> apply(
      const VecMask<src_t, src_n>& vec_mask) {
    return VecMask<dst_t, dst_n>::from(VectorizedN<src_t, src_n>(vec_mask));
  }
};

template <typename T, int N>
struct VecMaskCast<T, N, T, N> {
  static inline VecMask<T, N> apply(const VecMask<T, N>& vec_mask) {
    return vec_mask;
  }
};

template <typename T, int N>
class VecMask {
 public:
  using size_type = int;
  static constexpr size_type size() {
    return VectorizedN<T, N>::size();
  }

 private:
  VectorizedN<T, N> mask_;

 public:
  VecMask() : mask_(static_cast<T>(0)) {}
  VecMask(const VectorizedN<T, N>& mask) : mask_(mask) {}

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VecMask(const Vectorized<T>& mask) : mask_(mask) {}

  template <typename U, int L>
  static VecMask<T, N> from(const VectorizedN<U, L>& b_vec) {
    __at_align__ U b_buf[size()];
    if constexpr (size() >= VectorizedN<U, L>::size()) {
      b_vec.store(b_buf);
      for (int i = VectorizedN<U, L>::size(); i < size(); i++) {
        b_buf[i] = static_cast<U>(0);
      }
    } else {
      b_vec.store(b_buf, size());
    }
    return from(b_buf);
  }

  template <typename U>
  static VecMask<T, N> from(U b) {
    using int_t = int_same_size_t<T>;
    T mask = b ? c10::bit_cast<T>((int_t)(~(int_t)0)) : (T)0;
    return VectorizedN<T, N>(mask);
  }

  template <typename U>
  static VecMask<T, N> from(U* b) {
    using int_t = int_same_size_t<T>;
    __at_align__ T mask[size()];
#ifndef __msvc_cl__
#pragma unroll
#endif
    for (int i = 0; i < size(); i++) {
      *(int_t*)(mask + i) = b[i] ? ~(int_t)0 : (int_t)0;
    }
    return VectorizedN<T, N>(VectorizedN<T, N>::loadu(mask));
  }

  static VecMask<T, N> blendv(
    const VecMask<T, N>& c,
    const VecMask<T, N>& b,
    const VecMask<T, N>& a) {
    VectorizedN<T, N> result = VectorizedN<T, N>::blendv(
      VectorizedN<T, N>(c),
      VectorizedN<T, N>(b),
      VectorizedN<T, N>(a));
    return result;
  }

  static VecMask<T, N> set(
      const VecMask<T, N>& a,
      const VecMask<T, N>& b,
      int64_t count = size()) {
    VectorizedN<T, N> result = VectorizedN<T, N>::set(
      VectorizedN<T, N>(a),
      VectorizedN<T, N>(b),
      count);
    return result;
  }

  void store(bool* b, int count = size()) {
    constexpr int L = (VectorizedN<T, N>::size() + Vectorized<bool>::size() - 1)/ Vectorized<bool>::size();
    auto res = this->to<bool, L>();
    res.store(b, count);
    return;
  }

  template <typename U, int L, std::enable_if_t<L >= 2, int> = 0>
  inline VectorizedN<U, L> to() const {
    return VecMaskTo<U, L, T, N>::apply(*this);
  }

  template <typename U, int L, std::enable_if_t<L == 1, int> = 0>
  inline Vectorized<U> to() const {
    return VecMaskTo<U, L, T, N>::apply(*this);
  }

  template <typename U, int L>
  inline VecMask<U, L> cast() const {
    return VecMaskCast<U, L, T, N>::apply(*this);
  }

  inline bool all_zero() const {
    __at_align__ T mask[size()];
    mask_.store(mask);
    return std::all_of(
        mask, mask + size(), [](T m) { return m == static_cast<T>(0); });
  }

  inline bool all_masked() const {
    __at_align__ T mask[size()];
    mask_.store(mask);
    return std::all_of(
        mask, mask + size(), [](T m) { return m != static_cast<T>(0); });
  }

  inline bool is_masked(int i) const {
    __at_align__ T mask[size()];
    mask_.store(mask);
    return mask[i] != static_cast<T>(0);
  }

  inline operator VectorizedN<T, N>() const {
    return mask_;
  }

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  inline operator Vectorized<T>() const {
    return mask_[0];
  }

  inline Vectorized<T> operator[](int i) const {
    return mask_[i];
  }

  template <
      typename U,
      int L,
      std::enable_if_t<L >= 2 && VectorizedN<U, L>::size() >= size(), int> = 0>
  VectorizedN<U, L> loadu(const U* ptr) const {
    return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
  }

  template <
      typename U,
      int L,
      std::enable_if_t<L == 1 && Vectorized<U>::size() >= size(), int> = 0>
  Vectorized<U> loadu(const U* ptr) const {
    return VecMaskLoad<U, L, T, N>::apply(ptr, *this);
  }
};

#define VEC_MASK_DEFINE_UNARY_OP_GLOBAL(op)         \
  template <typename T, int N>                      \
  inline VecMask<T, N> op(const VecMask<T, N>& a) { \
    return op(VectorizedN<T, N>(a));                \
  }

#define VEC_MASK_DEFINE_BINARY_OP_GLOBAL(op)                                  \
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    return op(                                                                \
        VectorizedN<T, N>(a), VectorizedN<T, N>(b.template cast<T, N>()));    \
  }

#define VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(op, EXPR)                  \
  template <                                                                  \
      typename T,                                                             \
      int N,                                                                  \
      typename V,                                                             \
      int M,                                                                  \
      std::enable_if_t<VecMask<T, N>::size() == VecMask<V, M>::size(), int> = \
          0>                                                                  \
  inline VecMask<T, N> op(const VecMask<T, N>& a, const VecMask<V, M>& b) {   \
    return EXPR;                                                              \
  }

VEC_MASK_DEFINE_UNARY_OP_GLOBAL(operator~)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator&)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator|)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator^)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>, a & ~b)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<, ~a& b)
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator==, ~(a ^ b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator>=, (a == b) | (a > b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator<=, (a == b) | (a < b))
VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL(operator!=, (a ^ b))

#undef VEC_MASK_DEFINE_UNARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_WITH_EXPR_GLOBAL

} // namespace CPU_CAPABILITY
} // namespace at::vec
