#pragma once

#include <ATen/cpu/vec/vec_base.h>
#include <array>

namespace at::vec {
inline namespace CPU_CAPABILITY {

/**
 * @brief A class template representing a vectorized type with
 * `N * Vectorized<T>::size()` elements, aiming to support vectors of
 * arbitrary size. A specific use case of it is to represent vectors
 * converted from data types with different sizes but with the same
 * number of vector elements, e.g., `VectorizedN<float, 2>` can be
 * a vector converted from two `Vectorized<bfloat16>`, `VectorizedN<int64_t, 2>`
 * can be a vector converted from two `Vectorized<int32_t>` etc.
 *
 * It supports most of the operations of `Vectorized<T>`
 * and the implementation delegates to `Vectorized<T>` with loops over `N`.
 *
 * @tparam T The underlying type of the vectorized elements.
 * @tparam N The number of underlying `Vectorized<T>`.
 */
template <typename T, int N>
class VectorizedN {
 public:
  using value_type = T;
  using size_type = int;

  static constexpr size_type size_T = sizeof(T);
  static constexpr size_type size() {
    return Vectorized<T>::size() * N;
  }

 private:
  std::array<Vectorized<T>, N> values;

 public:
  // methods not implemented yet:
  // variadic constructor, operator T*, as_bytes, zero_mask

#define VECTORIZEDN_DEFINE_UNARY_OP(op)                             \
  VectorizedN<T, N> op() const {                                    \
    return unary_op([](const Vectorized<T>& a) { return a.op(); }); \
  }

#define VECTORIZEDN_DEFINE_BINARY_OP(op)                            \
  VectorizedN<T, N> op(const VectorizedN<T, N>& other) const {      \
    return binary_op(                                               \
        other, [](const Vectorized<T>& a, const Vectorized<T>& b) { \
          return a.op(b);                                           \
        });                                                         \
  }

  template <typename Op>
  inline VectorizedN<T, N> unary_op(Op op) const {
    VectorizedN<T, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i]);
    }
    return result;
  }

  template <typename Op>
  inline VectorizedN<T, N> binary_op(const VectorizedN<T, N>& other, Op op)
      const {
    VectorizedN<T, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i], other.values[i]);
    }
    return result;
  }

  template <typename Op>
  inline VectorizedN<T, N> ternary_op(
      const VectorizedN<T, N>& other,
      const VectorizedN<T, N>& other2,
      Op op) const {
    VectorizedN<T, N> result;
#ifndef _MSC_VER
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i], other.values[i], other2.values[i]);
    }
    return result;
  }

  VectorizedN() = default;

  explicit VectorizedN(T val) {
    for (int i = 0; i < N; ++i) {
      values[i] = Vectorized<T>(val);
    }
  }

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VectorizedN(const Vectorized<T>& val) : values({val}) {}

  template <int L = N, typename std::enable_if_t<L == 2, int> = 0>
  VectorizedN(const Vectorized<T>& val_0, const Vectorized<T>& val_1)
      : values({val_0, val_1}) {}

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  inline operator Vectorized<T>() const {
    return values[0];
  }

  inline const Vectorized<T>& operator[](int i) const {
    return values[i];
  }

  inline Vectorized<T>& operator[](int i) {
    return values[i];
  }

  template <int64_t mask>
  static VectorizedN<T, N> blend(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] =
          Vectorized<T>::template blend<mask>(a.values[i], b.values[i]);
    }
    return result;
  }

  static VectorizedN<T, N> blendv(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b,
      const VectorizedN<T, N>& mask) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] =
          Vectorized<T>::blendv(a.values[i], b.values[i], mask.values[i]);
    }
    return result;
  }

  template <typename step_t>
  static VectorizedN<T, N> arange(
      T base = static_cast<T>(0),
      step_t step = static_cast<step_t>(1)) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = Vectorized<T>::arange(base, step);
      base += step * Vectorized<T>::size();
    }
    return result;
  }

  static VectorizedN<T, N> set(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b,
      int64_t count = size()) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      if (count > 0) {
        result.values[i] = Vectorized<T>::set(
            a.values[i],
            b.values[i],
            std::min(count, (int64_t)Vectorized<T>::size()));
        count -= Vectorized<T>::size();
      } else {
        result.values[i] = a.values[i];
      }
    }
    return result;
  }

  static VectorizedN<T, N> loadu(const void* ptr) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = Vectorized<T>::loadu(ptr);
      ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
    }
    return result;
  }

  static VectorizedN<T, N> loadu(const void* ptr, int64_t count) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = Vectorized<T>::loadu(
          ptr, std::min(count, (int64_t)Vectorized<T>::size()));
      ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
      count -= Vectorized<T>::size();
      if (count <= 0) {
        break;
      }
    }
    return result;
  }

  void store(void* ptr) const {
    for (int i = 0; i < N; ++i) {
      values[i].store(ptr);
      ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
    }
  }

  void store(void* ptr, int count) const {
    for (int i = 0; i < N; ++i) {
      values[i].store(ptr, std::min(count, (int)Vectorized<T>::size()));
      ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
      count -= Vectorized<T>::size();
      if (count <= 0) {
        break;
      }
    }
  }

  bool has_inf_nan() const {
    for (int i = 0; i < N; ++i) {
      if (values[i].has_inf_nan()) {
        return true;
      }
    }
    return false;
  }

  VectorizedN<T, N> map(T (*const f)(T)) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = values[i].map(f);
    }
    return result;
  }

  VectorizedN<T, N> map(T (*const f)(const T&)) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = values[i].map(f);
    }
    return result;
  }

  VECTORIZEDN_DEFINE_UNARY_OP(isnan)
  VECTORIZEDN_DEFINE_UNARY_OP(abs)
  VECTORIZEDN_DEFINE_UNARY_OP(sgn)
  VECTORIZEDN_DEFINE_UNARY_OP(angle)
  VECTORIZEDN_DEFINE_UNARY_OP(real)
  VECTORIZEDN_DEFINE_UNARY_OP(imag)
  VECTORIZEDN_DEFINE_UNARY_OP(conj)
  VECTORIZEDN_DEFINE_UNARY_OP(acos)
  VECTORIZEDN_DEFINE_UNARY_OP(acosh)
  VECTORIZEDN_DEFINE_UNARY_OP(asin)
  VECTORIZEDN_DEFINE_UNARY_OP(asinh)
  VECTORIZEDN_DEFINE_UNARY_OP(atan)
  VECTORIZEDN_DEFINE_UNARY_OP(atanh)
  VECTORIZEDN_DEFINE_BINARY_OP(atan2)
  VECTORIZEDN_DEFINE_BINARY_OP(copysign)
  VECTORIZEDN_DEFINE_UNARY_OP(erf)
  VECTORIZEDN_DEFINE_UNARY_OP(erfc)
  VECTORIZEDN_DEFINE_UNARY_OP(erfinv)
  VECTORIZEDN_DEFINE_UNARY_OP(exp)
  VECTORIZEDN_DEFINE_UNARY_OP(exp2)
  VECTORIZEDN_DEFINE_UNARY_OP(expm1)
  VECTORIZEDN_DEFINE_UNARY_OP(exp_u20)
  VECTORIZEDN_DEFINE_UNARY_OP(frac)
  VECTORIZEDN_DEFINE_BINARY_OP(fmod)
  VECTORIZEDN_DEFINE_UNARY_OP(log)
  VECTORIZEDN_DEFINE_UNARY_OP(log10)
  VECTORIZEDN_DEFINE_UNARY_OP(log1p)
  VECTORIZEDN_DEFINE_UNARY_OP(log2)
  VECTORIZEDN_DEFINE_UNARY_OP(ceil)
  VECTORIZEDN_DEFINE_UNARY_OP(cos)
  VECTORIZEDN_DEFINE_UNARY_OP(cosh)
  VECTORIZEDN_DEFINE_UNARY_OP(floor)
  VECTORIZEDN_DEFINE_BINARY_OP(hypot)
  VECTORIZEDN_DEFINE_UNARY_OP(i0)
  VECTORIZEDN_DEFINE_UNARY_OP(i0e)
  VECTORIZEDN_DEFINE_UNARY_OP(digamma)
  VECTORIZEDN_DEFINE_BINARY_OP(igamma)
  VECTORIZEDN_DEFINE_BINARY_OP(igammac)
  VECTORIZEDN_DEFINE_UNARY_OP(neg)
  VECTORIZEDN_DEFINE_BINARY_OP(nextafter)
  VECTORIZEDN_DEFINE_UNARY_OP(round)
  VECTORIZEDN_DEFINE_UNARY_OP(sin)
  VECTORIZEDN_DEFINE_UNARY_OP(sinh)
  VECTORIZEDN_DEFINE_UNARY_OP(tan)
  VECTORIZEDN_DEFINE_UNARY_OP(tanh)
  VECTORIZEDN_DEFINE_UNARY_OP(trunc)
  VECTORIZEDN_DEFINE_UNARY_OP(lgamma)
  VECTORIZEDN_DEFINE_UNARY_OP(sqrt)
  VECTORIZEDN_DEFINE_UNARY_OP(reciprocal)
  VECTORIZEDN_DEFINE_UNARY_OP(rsqrt)
  VECTORIZEDN_DEFINE_BINARY_OP(pow)
  VECTORIZEDN_DEFINE_BINARY_OP(operator==)
  VECTORIZEDN_DEFINE_BINARY_OP(operator!=)
  VECTORIZEDN_DEFINE_BINARY_OP(operator>=)
  VECTORIZEDN_DEFINE_BINARY_OP(operator<=)
  VECTORIZEDN_DEFINE_BINARY_OP(operator>)
  VECTORIZEDN_DEFINE_BINARY_OP(operator<)
  VECTORIZEDN_DEFINE_BINARY_OP(eq)
  VECTORIZEDN_DEFINE_BINARY_OP(ne)
  VECTORIZEDN_DEFINE_BINARY_OP(gt)
  VECTORIZEDN_DEFINE_BINARY_OP(ge)
  VECTORIZEDN_DEFINE_BINARY_OP(lt)
  VECTORIZEDN_DEFINE_BINARY_OP(le)

#undef VECTORIZEDN_DEFINE_UNARY_OP
#undef VECTORIZEDN_DEFINE_BINARY_OP
};

#define VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL(op)                       \
  template <typename T, int N>                                       \
  inline VectorizedN<T, N> op(const VectorizedN<T, N>& a) {          \
    return a.unary_op([](const Vectorized<T>& a) { return op(a); }); \
  }

#define VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(op)                                \
  template <typename T, int N>                                                 \
  inline VectorizedN<T, N> op(                                                 \
      const VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {                \
    return a.binary_op(b, [](const Vectorized<T>& a, const Vectorized<T>& b) { \
      return op(a, b);                                                         \
    });                                                                        \
  }

#define VECTORIZEDN_DEFINE_TERNARY_OP_GLOBAL(op)             \
  template <typename T, int N>                               \
  inline VectorizedN<T, N> op(                               \
      const VectorizedN<T, N>& a,                            \
      const VectorizedN<T, N>& b,                            \
      const VectorizedN<T, N>& c) {                          \
    return a.ternary_op(                                     \
        b,                                                   \
        c,                                                   \
        [](const Vectorized<T>& a,                           \
           const Vectorized<T>& b,                           \
           const Vectorized<T>& c) { return op(a, b, c); }); \
  }

#define VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(op)                     \
  template <typename T, int N>                                              \
  inline VectorizedN<T, N>& op(                                             \
      VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {                   \
    a = a.binary_op(b, [](const Vectorized<T>& a, const Vectorized<T>& b) { \
      return op(a, b);                                                      \
    });                                                                     \
    return a;                                                               \
  }

VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator+)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator-)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator*)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator/)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator%)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator||)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator<<)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator>>)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(maximum)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(minimum)
VECTORIZEDN_DEFINE_TERNARY_OP_GLOBAL(fmadd)
VECTORIZEDN_DEFINE_TERNARY_OP_GLOBAL(fmsub)
VECTORIZEDN_DEFINE_TERNARY_OP_GLOBAL(clamp)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(clamp_max)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(clamp_min)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator&)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator|)
VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL(operator^)
VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL(operator~)

VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator+=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator-=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator*=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator/=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator%=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator<<=)
VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL(operator>>=)

#undef VECTORIZEDN_DEFINE_UNARY_OP_GLOBAL
#undef VECTORIZEDN_DEFINE_BINARY_OP_GLOBAL
#undef VECTORIZEDN_DEFINE_BINARY_OP_INPLACE_GLOBAL

template <typename T, int N, typename OpVec>
inline T vec_reduce_all(const OpVec& vec_fun, VectorizedN<T, N> acc_vec) {
  Vectorized<T> vec_result = acc_vec[0];
  for (int i = 1; i < N; i++) {
    vec_result = vec_fun(vec_result, acc_vec[i]);
  }
  return vec_reduce_all(vec_fun, vec_result);
}

template <typename T, int N>
std::ostream& operator<<(std::ostream& stream, const VectorizedN<T, N>& vec_n) {
  stream << "vec_n[";
  for (int i = 0; i < N; ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << vec_n[i];
  }
  stream << ']';
  return stream;
}
} // namespace CPU_CAPABILITY
} // namespace at::vec
