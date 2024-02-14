#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_n.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <typename T, int N = 1>
class VecMask {
 public:
  using size_type = int;
  static constexpr size_type size() {
    return VectorizedN<T, N>::size();
  }

 private:
  VectorizedN<T, N> mask_;

  template <
      typename U,
      int L,
      std::enable_if_t<VectorizedN<U, L>::size() >= size(), int> = 0>
  VectorizedN<U, L> loadu_helper(const U* ptr) const {
    __at_align__ U data[size()];
    __at_align__ T mask[size()];
    mask_.store(mask);
    for (int i = 0; i < size(); i++) {
      data[i] = mask[i] ? ptr[i] : static_cast<U>(0);
    }
    return VectorizedN<U, L>::loadu(data, size());
  }

  template <typename U, int L>
  inline VectorizedN<U, L> to_helper() const {
    auto zeros = VectorizedN<U, L>(static_cast<U>(0));
    auto ones = VectorizedN<U, L>(static_cast<U>(1));
    return VectorizedN<U, L>::blendv(zeros, ones, cast<U, L>());
  }

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
    } else {
      b_vec.store(b_buf, size());
    }
    return from(b_buf);
  }

  template <typename U>
  static VecMask<T, N> from(U b) {
    using int_t = int_same_size_t<T>;
    T mask = b ? c10::bit_cast<T>(~(int_t)0) : (T)0;
    return VectorizedN<T, N>(mask);
  }

  template <typename U>
  static VecMask<T, N> from(U* b) {
    using int_t = int_same_size_t<T>;
    __at_align__ T mask[size()];
#pragma unroll
    for (int i = 0; i < size(); i++) {
      *(int_t*)(mask + i) = b[i] ? ~(int_t)0 : (int_t)0;
    }
    return VectorizedN<T, N>(VectorizedN<T, N>::loadu(mask));
  }

  template <typename U, int L, std::enable_if_t<L >= 2, int> = 0>
  inline VectorizedN<U, L> to() const {
    return to_helper<U, L>();
  }

  template <typename U, int L, std::enable_if_t<L == 1, int> = 0>
  inline Vectorized<U> to() const {
    return to_helper<U, L>();
  }

  // TODO: add specialization for T=U and L=N
  template <typename U, int L>
  inline VecMask<U, L> cast() const {
    return VecMask<U, L>::from(mask_);
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

  template <
      typename U,
      int L,
      std::enable_if_t<L >= 2 && VectorizedN<U, L>::size() >= size(), int> = 0>
  VectorizedN<U, L> loadu(const U* ptr) const {
    return loadu_helper<U, L>(ptr);
  }

  template <
      typename U,
      int L,
      std::enable_if_t<L == 1 && Vectorized<U>::size() >= size(), int> = 0>
  Vectorized<U> loadu(const U* ptr) const {
    return loadu_helper<U, L>(ptr);
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

VEC_MASK_DEFINE_UNARY_OP_GLOBAL(operator~)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator&)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator|)
VEC_MASK_DEFINE_BINARY_OP_GLOBAL(operator^)

#undef VEC_MASK_DEFINE_UNARY_OP_GLOBAL
#undef VEC_MASK_DEFINE_BINARY_OP_GLOBAL

} // namespace CPU_CAPABILITY
} // namespace at::vec
