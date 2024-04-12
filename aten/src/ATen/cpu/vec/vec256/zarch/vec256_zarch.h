#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#if defined(__clang__)
#include <sleef.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#include <sleef.h>
#include <vecintrin.h>
#endif
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/complex.h>

#define SLEEF_MEMORY_WORKAROUND

namespace at {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

template <typename T>
constexpr bool is_zarch_implemented() {
  return (
      std::is_same<T, float>::value || std::is_same<T, double>::value ||
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
      std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value);
}

template <typename T>
constexpr bool is_zarch_implemented_quant() {
  return (
      std::is_same<T, c10::qint32>::value ||
      std::is_same<T, c10::qint8>::value ||
      std::is_same<T, c10::quint8>::value);
}

template <typename T>
constexpr bool is_zarch_implemented_complex() {
  return std::is_same<T, c10::complex<float>>::value ||
      std::is_same<T, c10::complex<double>>::value;
}

constexpr int offset0 = 0;
constexpr int offset16 = 16;

template <int N>
struct VecBinaryType {
  using type __attribute__((vector_size(16))) = uintmax_t;
};

template <>
struct VecBinaryType<8> {
  using type = __attribute__((vector_size(16))) unsigned long long;
};

template <>
struct VecBinaryType<4> {
  using type = __attribute__((vector_size(16))) unsigned int;
};

template <>
struct VecBinaryType<2> {
  using type = __attribute__((vector_size(16))) unsigned short;
};

template <>
struct VecBinaryType<1> {
  using type = __attribute__((vector_size(16))) unsigned char;
};

template <typename T>
struct VecInnerType {
  using Type __attribute__((vector_size(16))) = T;
  using BinaryType = typename VecBinaryType<sizeof(T)>::type;
  using ElementType = T;
  static constexpr int size = 16 / sizeof(T);
};

// define for int64_t properly for load
template <>
struct VecInnerType<int64_t> {
  using Type = __attribute__((vector_size(16))) signed long long;
  using ElementType = signed long long;
  using BinaryType = typename VecBinaryType<sizeof(signed long long)>::type;
  static constexpr int size = 16 / sizeof(signed long long);
};

template <typename T>
using ZSimdVect = typename VecInnerType<T>::Type;
template <typename T>
using ZSimdVectBinary = typename VecInnerType<T>::BinaryType;
template <typename T>
using ZSimdVectElement = typename VecInnerType<T>::ElementType;

constexpr int blendChoiceInner(
    const uint64_t mask,
    const uint64_t half1 = 0xF,
    const uint64_t half2 = 0xF0) {
  uint64_t none = 0;
  uint64_t both = half1 | half2;
  // clamp it between 0 and both
  auto res_mask = mask & both;
  // return  (a._vec0, a._vec1)
  if (res_mask == none)
    return 0;
  // return (b._vec0,b._vec1)
  else if (res_mask == both)
    return 1;
  // return  (b._vec0, a._vec1)
  else if (res_mask == half1)
    return 2;
  // return  (a._vec0,b._vec1)
  else if (res_mask == half2)
    return 3;
  // return  (*_vec0,a._vec1)
  else if (res_mask > 0 && res_mask < half1)
    return 4;
  // return  (*_vec0,b._vec1)
  else if ((res_mask & half2) == half2)
    return 5;
  // return (a._vec0,*_vec1)
  else if ((res_mask & half1) == 0 && res_mask > half1)
    return 6;
  // return (b._vec0,*_vec1)
  else if ((res_mask & half1) == half1 && res_mask > half1)
    return 7;
  // return (*_vec0,*_vec1)
  return 8;
}

// it can be used to emulate blend faster
template <int Z>
constexpr int blendChoice(const uint64_t mask) {
  static_assert(Z < 1 || Z > 8, "not implemented");
  return blendChoiceInner(mask);
}

template <>
constexpr int blendChoice<1>(const uint64_t mask) {
  return blendChoiceInner(mask, 0x0000FFFF, 0xFFFF0000);
}

template <>
constexpr int blendChoice<2>(const uint64_t mask) {
  return blendChoiceInner(mask, 0x00FF, 0xFF00);
}

template <>
constexpr int blendChoice<4>(const uint64_t mask) {
  return blendChoiceInner(mask, 0xF, 0xF0);
}

template <>
constexpr int blendChoice<8>(const uint64_t mask) {
  // clamp it 0 and 0xF
  return blendChoiceInner(mask, 0x3, 0xC);
}

template <int N>
constexpr auto GetMask1(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <int N>
constexpr auto GetMask2(const uint64_t mask) {
  return typename VecBinaryType<N>::type{};
}

template <>
constexpr auto GetMask1<1>(const uint64_t mask) {
  constexpr uint8_t t = (int)0xFF;
  uint8_t g0 = (mask & 1) * t;
  uint8_t g1 = ((mask & 2) >> 1) * t;
  uint8_t g2 = ((mask & 4) >> 2) * t;
  uint8_t g3 = ((mask & 8) >> 3) * t;
  uint8_t g4 = ((mask & 16) >> 4) * t;
  uint8_t g5 = ((mask & 32) >> 5) * t;
  uint8_t g6 = ((mask & 64) >> 6) * t;
  uint8_t g7 = ((mask & 128) >> 7) * t;
  uint8_t g8 = ((mask & 256) >> 8) * t;
  uint8_t g9 = ((mask & 512) >> 9) * t;
  uint8_t g10 = ((mask & 1024) >> 10) * t;
  uint8_t g11 = ((mask & 2048) >> 11) * t;
  uint8_t g12 = ((mask & 4096) >> 12) * t;
  uint8_t g13 = ((mask & 8192) >> 13) * t;
  uint8_t g14 = ((mask & 16384) >> 14) * t;
  uint8_t g15 = ((mask & 32768) >> 15) * t;
  return (typename VecBinaryType<1>::type){
      g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15};
}

template <>
constexpr auto GetMask2<1>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFFFFFFFF) >> 16;
  return GetMask1<1>(mask2);
}

template <>
constexpr auto GetMask1<2>(const uint64_t mask) {
  constexpr uint16_t t = (int)0xFFFF;
  uint16_t g0 = (mask & 1) * t;
  uint16_t g1 = ((mask & 2) >> 1) * t;
  uint16_t g2 = ((mask & 4) >> 2) * t;
  uint16_t g3 = ((mask & 8) >> 3) * t;
  uint16_t g4 = ((mask & 16) >> 4) * t;
  uint16_t g5 = ((mask & 32) >> 5) * t;
  uint16_t g6 = ((mask & 64) >> 6) * t;
  uint16_t g7 = ((mask & 128) >> 7) * t;
  return (typename VecBinaryType<2>::type){g0, g1, g2, g3, g4, g5, g6, g7};
}

template <>
constexpr auto GetMask2<2>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFFFF) >> 8;
  return GetMask1<2>(mask2);
}

template <>
constexpr auto GetMask1<4>(const uint64_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffff;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
  return (typename VecBinaryType<4>::type){g0, g1, g2, g3};
}

template <>
constexpr auto GetMask2<4>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xFF) >> 4;
  return GetMask1<4>(mask2);
}

template <>
constexpr auto GetMask1<8>(const uint64_t mask) {
  uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
  return (typename VecBinaryType<8>::type){g0, g1};
}

template <>
constexpr auto GetMask2<8>(const uint64_t mask) {
  uint64_t mask2 = (mask & 0xF) >> 2;
  return GetMask1<8>(mask2);
}

template <int Z>
constexpr int maskForComplex(uint32_t mask) {
  return 0;
}

template <>
constexpr int maskForComplex<8>(uint32_t mask) {
  mask = mask & 0xF;
  int complex_mask = 0;
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  if (mask & 4)
    complex_mask |= (3 << 4);
  if (mask & 8)
    complex_mask |= (3 << 6);
  return complex_mask;
}

template <>
constexpr int maskForComplex<16>(uint32_t mask) {
  mask = mask & 0x3;
  int complex_mask = 0;
  if (mask & 1)
    complex_mask |= 3;
  if (mask & 2)
    complex_mask |= (3 << 2);
  return complex_mask;
}

template <typename T = c10::complex<float>>
constexpr int blend_choice() {
  return 0xAA;
}

template <>
constexpr int blend_choice<c10::complex<double>>() {
  return 0x0A;
}

constexpr int64_t allbitset(int16_t x) {
  int64_t onex = 1;
  return (onex << x) - onex;
}

namespace { /* unnamed namespace */

ZSimdVect<float> vec_mergee(ZSimdVect<float> x, ZSimdVect<float> y) {
  constexpr ZSimdVectBinary<uint8_t> mergee_mask{
      0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
  return vec_perm(x, y, mergee_mask);
}

ZSimdVect<double> vec_mergee(ZSimdVect<double> x, ZSimdVect<double> y) {
  return vec_mergeh(x, y);
}

ZSimdVect<float> vec_mergeo(ZSimdVect<float> x, ZSimdVect<float> y) {
  constexpr ZSimdVectBinary<uint8_t> mergeo_mask{
      4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
  return vec_perm(x, y, mergeo_mask);
}

ZSimdVect<double> vec_mergeo(ZSimdVect<double> x, ZSimdVect<double> y) {
  return vec_mergel(x, y);
}

} /* unnamed namespace */

//
template <typename T>
constexpr auto GetBpermZeroMask() {
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      96,
      64,
      32,
      0};
}

template <>
constexpr auto GetBpermZeroMask<double>() {
  return ZSimdVectBinary<uint8_t>{
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      128,
      64,
      0};
}

constexpr auto GetSwapMaskFloat() {
  return ZSimdVectBinary<uint8_t>{
      4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
}

template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented<T>()>> {
 public:
  using value_type = T;
  using vtype = ZSimdVect<T>;
  using vmaskType = ZSimdVectBinary<T>;
  using size_type = int;
  // because of gcc inconsistency for int64_t we are obliged to use this, not
  // value_type
  using ElementType = ZSimdVectElement<T>;
  using vinner_data = std::pair<vtype, vtype>;

 private:
  vtype _vec0;
  vtype _vec1;

 public:
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(ElementType);
  }
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(vtype v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(const vinner_data &v) : _vec0{v.first}, _vec1{v.second} {}
  C10_ALWAYS_INLINE Vectorized(vtype v1, vtype v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(T s)
      : _vec0{vec_splats((ElementType)s)}, _vec1{vec_splats((ElementType)s)} {}

  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_xl(offset0, reinterpret_cast<const ElementType*>(ptr)),
          vec_xl(offset16, reinterpret_cast<const ElementType*>(ptr))};
    }

    __at_align__ ElementType tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(ElementType));

    return {
        vec_xl(offset0, reinterpret_cast<const ElementType*>(tmp_values)),
        vec_xl(offset16, reinterpret_cast<const ElementType*>(tmp_values))};
  }

  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu_one_fourth(const void* ptr) {
    // load only first 8 bytes
    // only intended to be used with uint8_t
    return loadu(ptr, 8 / sizeof(ElementType));
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_xst(_vec0, offset0, reinterpret_cast<ElementType*>(ptr));
      vec_xst(_vec1, offset16, reinterpret_cast<ElementType*>(ptr));
    } else if (count > 0) {
      __at_align__ ElementType tmp_values[size()];
      vec_xst(_vec0, offset0, reinterpret_cast<ElementType*>(tmp_values));
      vec_xst(_vec1, offset16, reinterpret_cast<ElementType*>(tmp_values));
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(ElementType));
    }
  }

  C10_ALWAYS_INLINE const vtype& vec0() const {
    return _vec0;
  }

  C10_ALWAYS_INLINE const vtype& vec1() const {
    return _vec1;
  }

  C10_ALWAYS_INLINE vinner_data data() const {
    return std::make_pair<>(_vec0, _vec1);
  }

  C10_ALWAYS_INLINE operator vinner_data() const {
    return data();
  }

  C10_ALWAYS_INLINE const vmaskType vecb0() const {
    return (vmaskType)_vec0;
  }
  C10_ALWAYS_INLINE const vmaskType vecb1() const {
    return (vmaskType)_vec1;
  }

  static Vectorized<T> C10_ALWAYS_INLINE blendv(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      const Vectorized<T>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask.vecb0()),
        vec_sel(a._vec1, b._vec1, mask.vecb1())};
  }

  template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
      : _vec0{s1, s2}, _vec1{s3, s4} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 4), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4, T s5, T s6, T s7, T s8)
      : _vec0{s1, s2, s3, s4}, _vec1{s5, s6, s7, s8} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 2), int> = 0>
  C10_ALWAYS_INLINE Vectorized(
      T s1,
      T s2,
      T s3,
      T s4,
      T s5,
      T s6,
      T s7,
      T s8,
      T s9,
      T s10,
      T s11,
      T s12,
      T s13,
      T s14,
      T s15,
      T s16)
      : _vec0{s1, s2, s3, s4, s5, s6, s7, s8},
        _vec1{s9, s10, s11, s12, s13, s14, s15, s16} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 1), int> = 0>
  C10_ALWAYS_INLINE Vectorized(
      T s1,
      T s2,
      T s3,
      T s4,
      T s5,
      T s6,
      T s7,
      T s8,
      T s9,
      T s10,
      T s11,
      T s12,
      T s13,
      T s14,
      T s15,
      T s16,
      T s17,
      T s18,
      T s19,
      T s20,
      T s21,
      T s22,
      T s23,
      T s24,
      T s25,
      T s26,
      T s27,
      T s28,
      T s29,
      T s30,
      T s31,
      T s32)
      : _vec0{s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16},
        _vec1{
            s17,
            s18,
            s19,
            s20,
            s21,
            s22,
            s23,
            s24,
            s25,
            s26,
            s27,
            s28,
            s29,
            s30,
            s31,
            s32} {}

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(base, base + step, base + 2 * step, base + 3 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 4, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 2, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 1, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step,
        base + 16 * step,
        base + 17 * step,
        base + 18 * step,
        base + 19 * step,
        base + 20 * step,
        base + 21 * step,
        base + 22 * step,
        base + 23 * step,
        base + 24 * step,
        base + 25 * step,
        base + 26 * step,
        base + 27 * step,
        base + 28 * step,
        base + 29 * step,
        base + 30 * step,
        base + 31 * step);
  }

  // blend section
  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 0, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 1, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 2, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 3, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 4, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 5, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    return {(vtype)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 6, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    // generated masks
    return {a._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 7, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    // generated masks
    return {b._vec0, (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice<sizeof(T)>(mask) == 8, Vectorized<T>>
      C10_ALWAYS_INLINE blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    const vmaskType mask_1st = GetMask1<sizeof(T)>(mask);
    const vmaskType mask_2nd = GetMask2<sizeof(T)>(mask);
    return {
        (vtype)vec_sel(a._vec0, b._vec0, mask_1st),
        (vtype)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z >= C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    return b;
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z < C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    if (count == Z)
      return blend<allbitset(Z)>(a, b);
    else
      return set_inner<Z + 1, C>(a, b, count);
  }

  static Vectorized<T> set(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count = size()) {
    if (count == 0)
      return a;
    return set_inner<1, size()>(a, b, count);
  }

  const ElementType& operator[](int idx) const = delete;
  ElementType& operator[](int idx) = delete;

  Vectorized<T> C10_ALWAYS_INLINE operator+(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec0 + other._vec0, _vec1 + other._vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator-(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec0 - other._vec0, _vec1 - other._vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator*(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec0 * other._vec0, _vec1 * other._vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator/(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec0 / other._vec0, _vec1 / other._vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator&(const Vectorized<T>& other) const {
    return Vectorized<T>{
        (vtype)(vecb0() & other.vecb0()), (vtype)(vecb1() & other.vecb1())};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator|(const Vectorized<T>& other) const {
    return Vectorized<T>{
        (vtype)(vecb0() | other.vecb0()), (vtype)(vecb1() | other.vecb1())};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator^(const Vectorized<T>& other) const {
    return Vectorized<T>{
        (vtype)(vecb0() ^ other.vecb0()), (vtype)(vecb1() ^ other.vecb1())};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator<<(const Vectorized<T> &other) const {
    constexpr ElementType max_shift = sizeof(ElementType) * CHAR_BIT;

    ElementType a_array[Vectorized<T>::size()];
    ElementType b_array[Vectorized<T>::size()];
    ElementType c_array[Vectorized<T>::size()];

    store(a_array);
    other.store(b_array);

    for (int i = 0; i != Vectorized<T>::size(); i++) {
      T shift = b_array[i];
      if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
        c_array[i] = 0;
      } else {
        c_array[i] = static_cast<std::make_unsigned_t<T>>(a_array[i]) << shift;
      }
   }

    return loadu(c_array);
  }

  Vectorized<T> C10_ALWAYS_INLINE operator>>(const Vectorized<T> &other) const {
    // right shift value to retain sign bit for signed and no bits for unsigned
    constexpr ElementType max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;

    ElementType a_array[Vectorized<T>::size()];
    ElementType b_array[Vectorized<T>::size()];
    ElementType c_array[Vectorized<T>::size()];

    store(a_array);
    other.store(b_array);

    for (int i = 0; i != Vectorized<T>::size(); i++) {
      T shift = b_array[i];
      if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
        c_array[i] = a_array[i] >> max_shift;
      } else {
        c_array[i] = a_array[i] >> shift;
      }
    }

    return loadu(c_array);
  }

  Vectorized<T> _not() const {
    return {(vtype)vec_nor(vecb0(), vecb0()), (vtype)vec_nor(vecb1(), vecb1())};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator==(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmpeq(_vec0, other._vec0), vec_cmpeq(_vec1, other._vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator!=(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmpeq(_vec0, other._vec0), vec_cmpeq(_vec1, other._vec1)}
        ._not();
  }
  Vectorized<T> C10_ALWAYS_INLINE operator>(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmpgt(_vec0, other._vec0), vec_cmpgt(_vec1, other._vec1)};
  }
  Vectorized<T> C10_ALWAYS_INLINE operator>=(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmpge(_vec0, other._vec0), vec_cmpge(_vec1, other._vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator<(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmplt(_vec0, other._vec0), vec_cmplt(_vec1, other._vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator<=(const Vectorized<T>& other) const {
    return Vectorized<T>{
        vec_cmple(_vec0, other._vec0), vec_cmple(_vec1, other._vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
    return (*this == other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
    return (*this != other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE gt(const Vectorized<T>& other) const {
    return (*this > other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE ge(const Vectorized<T>& other) const {
    return (*this >= other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE lt(const Vectorized<T>& other) const {
    return (*this < other) & Vectorized<T>((T)1.0);
  }
  Vectorized<T> C10_ALWAYS_INLINE le(const Vectorized<T>& other) const {
    return (*this <= other) & Vectorized<T>((T)1.0);
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_unsigned<U>::value, int> = 0>
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_unsigned<U>::value, int> = 0>
  Vectorized<U> C10_ALWAYS_INLINE abs() const {
    return {_vec0, _vec1};
  }

  Vectorized<T> C10_ALWAYS_INLINE neg() const {
    return {-_vec0, -_vec1};
  }

  Vectorized<T> isnan() const {
    auto x = *this;
    auto ret = (x == x);
    return ret._not();
  }

  bool has_inf_nan() const {
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec0[i]) || _isinf(_vec0[i])) {
        return true;
      }
    }
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec1[i]) || _isinf(_vec1[i])) {
        return true;
      }
    }
    return false;
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<U> angle() const {
    auto tmp = blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<U> angle() const {
    return blendv(
        Vectorized<U>(0), Vectorized<U>(c10::pi<U>), *this < Vectorized<U>(0));
  }

  Vectorized<T> real() const {
    return *this;
  }
  Vectorized<T> imag() const {
    return Vectorized<T>{0};
  }
  Vectorized<T> conj() const {
    return *this;
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  int zero_mask() const {
    auto cmp = (*this == Vectorized<U>(0));
    constexpr auto mask_zero_bits = GetBpermZeroMask<U>();
    ZSimdVectBinary<uint64_t> result0 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb0(), mask_zero_bits);
    ZSimdVectBinary<uint64_t> result1 =
        vec_bperm_u128((ZSimdVectBinary<uint8_t>)cmp.vecb1(), mask_zero_bits);
    return (result0[0] | (result1[0] << (size() / 2)));
  }

  Vectorized<T> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE rint() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<T> C10_ALWAYS_INLINE frac() const {
    return *this - trunc();
  }

  Vectorized<T> C10_ALWAYS_INLINE sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<T> C10_ALWAYS_INLINE reciprocal() const {
    return Vectorized<T>((T)1) / (*this);
  }
  Vectorized<T> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(float (*const f)(float)) const {
    float a00 = f(_vec0[0]);
    float a01 = f(_vec0[1]);
    float a02 = f(_vec0[2]);
    float a03 = f(_vec0[3]);
    float a10 = f(_vec1[0]);
    float a11 = f(_vec1[1]);
    float a12 = f(_vec1[2]);
    float a13 = f(_vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(double (*const f)(double)) const {
    return Vectorized<T>(f(_vec0[0]), f(_vec0[1]), f(_vec1[0]), f(_vec1[1]));
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      float (*const f)(float, float),
      const Vectorized<T>& b) const {
    float a00 = f(_vec0[0], b._vec0[0]);
    float a01 = f(_vec0[1], b._vec0[1]);
    float a02 = f(_vec0[2], b._vec0[2]);
    float a03 = f(_vec0[3], b._vec0[3]);
    float a10 = f(_vec1[0], b._vec1[0]);
    float a11 = f(_vec1[1], b._vec1[1]);
    float a12 = f(_vec1[2], b._vec1[2]);
    float a13 = f(_vec1[3], b._vec1[3]);
    return Vectorized<T>{a00, a01, a02, a03, a10, a11, a12, a13};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      double (*const f)(double, double),
      const Vectorized<T>& b) const {
    return Vectorized<T>(
        f(_vec0[0], b._vec0[0]),
        f(_vec0[1], b._vec0[1]),
        f(_vec1[0], b._vec1[0]),
        f(_vec1[1], b._vec1[1]));
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    vtype a0 = f(_vec0);
    vtype a1 = f(_vec1);
    return Vectorized<T>{a0, a1};
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d) const {
    return Vectorized<T>(d(_vec0), d(_vec1));
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d, const Vectorized<T>& b)
      const {
    vtype a0 = f(_vec0, b._vec0);
    vtype a1 = f(_vec1, b._vec1);
    return Vectorized<T>{a0, a1};
  }

  template <
      typename FloatOp,
      typename DoubleOp,
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  inline Vectorized<T> mapSleef(FloatOp f, DoubleOp d, const Vectorized<T>& b)
      const {
    return Vectorized<T>(d(_vec0, b._vec0), d(_vec1, b._vec1));
  }

  Vectorized<T> acos() const {
    return mapSleef(Sleef_acosf4_u10, Sleef_acosd2_u10);
  }
  Vectorized<T> asin() const {
    return mapSleef(Sleef_asinf4_u10, Sleef_asind2_u10);
  }
  Vectorized<T> atan() const {
    return mapSleef(Sleef_atanf4_u10, Sleef_atand2_u10);
  }
  Vectorized<T> atanh() const {
    return mapSleef(Sleef_atanhf4_u10, Sleef_atanhd2_u10);
  }

  Vectorized<T> erf() const {
    return mapSleef(Sleef_erff4_u10, Sleef_erfd2_u10);
  }
  Vectorized<T> erfc() const {
    return mapSleef(Sleef_erfcf4_u15, Sleef_erfcd2_u15);
  }

  Vectorized<T> exp() const {
    return mapSleef(Sleef_expf4_u10, Sleef_expd2_u10);
  }
  Vectorized<T> exp2() const {
    return mapSleef(Sleef_exp2f4_u10, Sleef_exp2d2_u10);
  }
  Vectorized<T> expm1() const {
    return mapSleef(Sleef_expm1f4_u10, Sleef_expm1d2_u10);
  }
  Vectorized<T> exp_u20() const {
    return exp();
  }

  Vectorized<T> log() const {
    return mapSleef(Sleef_logf4_u10, Sleef_logd2_u10);
  }
  Vectorized<T> log2() const {
    return mapSleef(Sleef_log2f4_u10, Sleef_log2d2_u10);
  }
  Vectorized<T> log10() const {
    return mapSleef(Sleef_log10f4_u10, Sleef_log10d2_u10);
  }
  Vectorized<T> log1p() const {
    return mapSleef(Sleef_log1pf4_u10, Sleef_log1pd2_u10);
  }

  Vectorized<T> sin() const {
#ifndef SLEEF_MEMORY_WORKAROUND
    return mapSleef(Sleef_sinf4_u10, Sleef_sind2_u10);
#else
    return mapOrdinary(std::sin);
#endif
  }
  Vectorized<T> sinh() const {
    return mapSleef(Sleef_sinhf4_u10, Sleef_sinhd2_u10);
  }
  Vectorized<T> cos() const {
#ifndef SLEEF_MEMORY_WORKAROUND
    return mapSleef(Sleef_cosf4_u10, Sleef_cosd2_u10);
#else
    return mapOrdinary(std::cos);
#endif
  }
  Vectorized<T> cosh() const {
    return mapSleef(Sleef_coshf4_u10, Sleef_coshd2_u10);
  }

  Vectorized<T> tan() const {
#ifndef SLEEF_MEMORY_WORKAROUND
    return mapSleef(Sleef_tanf4_u10, Sleef_tand2_u10);
#else
    return mapOrdinary(std::tan);
#endif
  }
  Vectorized<T> tanh() const {
    return mapSleef(Sleef_tanhf4_u10, Sleef_tanhd2_u10);
  }

  Vectorized<T> lgamma() const {
    return mapSleef(Sleef_lgammaf4_u10, Sleef_lgammad2_u10);
  }

  Vectorized<T> atan2(const Vectorized<T>& b) const {
    return mapSleef(Sleef_atan2f4_u10, Sleef_atan2d2_u10, b);
  }
  Vectorized<T> copysign(const Vectorized<T>& sign) const {
    return mapSleef(Sleef_copysignf4, Sleef_copysignd2, sign);
  }
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    return mapSleef(Sleef_fmodf4, Sleef_fmodd2, q);
  }

  Vectorized<T> hypot(const Vectorized<T>& b) const {
    return mapSleef(Sleef_hypotf4_u05, Sleef_hypotd2_u05, b);
  }

  Vectorized<T> pow(const Vectorized<T>& b) const {
    return mapSleef(Sleef_powf4_u10, Sleef_powd2_u10, b);
  }

  Vectorized<T> nextafter(const Vectorized<T>& b) const {
    return mapSleef(Sleef_nextafterf4, Sleef_nextafterd2, b);
  }

  Vectorized<T> erfinv() const {
    return mapOrdinary(calc_erfinv);
  }

  Vectorized<T> digamma() const {
    return mapOrdinary(calc_digamma);
  }

  Vectorized<T> igamma(const Vectorized<T>& x) const {
    return mapOrdinary(calc_igamma, x);
  }

  Vectorized<T> igammac(const Vectorized<T>& x) const {
    return mapOrdinary(calc_igammac, x);
  }

  Vectorized<T> i0() const {
    return mapOrdinary(calc_i0);
  }

  Vectorized<T> i0e() const {
    return mapOrdinary(calc_i0e);
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    return {vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> minimum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {vec_min(_vec0, other._vec0), vec_min(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    return {vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
  }

  /* Propagates NaN if either input is a NaN. */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> maximum(const Vectorized<T>& other) const {
    Vectorized<T> tmp = {vec_max(_vec0, other._vec0), vec_max(_vec1, other._vec1)};
    tmp = blendv(tmp, *this, isnan());
    return blendv(tmp, other, other.isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    return {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    Vectorized<T> tmp = {vec_max(_vec0, min._vec0), vec_max(_vec1, min._vec1)};
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<!std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    return {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
  }

  /* Keeps NaN if actual value is NaN */
  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    Vectorized<T> tmp = {vec_min(_vec0, max._vec0), vec_min(_vec1, max._vec1)};
    return blendv(tmp, *this, isnan());
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, float>::value, int> = 0>
  Vectorized<T> swapped() const {
    auto swap_mask = GetSwapMaskFloat();
    vtype v0 = vec_perm(_vec0, _vec0, swap_mask);
    vtype v1 = vec_perm(_vec1, _vec1, swap_mask);
    return {v0, v1};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, double>::value, int> = 0>
  Vectorized<T> swapped() const {
    vtype v0 = vec_permi(_vec0, _vec0, 2);
    vtype v1 = vec_permi(_vec1, _vec1, 2);
    return {v0, v1};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  static Vectorized<T> mergee(Vectorized<T>& first, Vectorized<T>& second) {
    return {
        vec_mergee(first._vec0, second._vec0),
        vec_mergee(first._vec1, second._vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  static Vectorized<T> mergeo(Vectorized<T>& first, Vectorized<T>& second) {
    return {
        vec_mergeo(first._vec0, second._vec0),
        vec_mergeo(first._vec1, second._vec1)};
  }

  static Vectorized<T> horizontal_add_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // summ
    auto first_ret = first + first_perm; // 2add
    auto second_ret = second + second_perm; // 2 add
    // now lets choose evens
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  static Vectorized<T> horizontal_sub_perm(
      Vectorized<T>& first,
      Vectorized<T>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.swapped(); // 2perm
    auto second_perm = second.swapped(); // 2perm
    // summ
    auto first_ret = first - first_perm; // 2sub
    auto second_ret = second - second_perm; // 2 sub
    // now lets choose evens
    return mergee(first_ret, second_ret); // 2 mergee's
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> mergee() const {
    return {vec_mergee(_vec0, _vec0), vec_mergee(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> mergeo() const {
    return {vec_mergeo(_vec0, _vec0), vec_mergeo(_vec1, _vec1)};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, uint8_t>::value, int> = 0>
  Vectorized<int32_t> to_vec_float_helper() const {
    int32_t values[8] = {
      _vec0[0],
      _vec0[1],
      _vec0[2],
      _vec0[3],
      _vec0[4],
      _vec0[5],
      _vec0[6],
      _vec0[7],
    };

    return Vectorized<int32_t>{
      values[0], values[1], values[2], values[3],
      values[4], values[5], values[6], values[7]
    };
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, int32_t>::value, int> = 0>
  Vectorized<uint8_t> to_vec_uint8_helper() const {
    // helper function for float to uint8_t conversion
    uint8_t values[8] = {
      static_cast<uint8_t>(_vec0[0]),
      static_cast<uint8_t>(_vec0[1]),
      static_cast<uint8_t>(_vec0[2]),
      static_cast<uint8_t>(_vec0[3]),
      static_cast<uint8_t>(_vec1[0]),
      static_cast<uint8_t>(_vec1[1]),
      static_cast<uint8_t>(_vec1[2]),
      static_cast<uint8_t>(_vec1[3]),
    };

    return Vectorized<uint8_t>{
      values[0], values[1], values[2], values[3],
      values[4], values[5], values[6], values[7],
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    };
  }
};

template <>
inline Vectorized<int64_t> operator~(const Vectorized<int64_t>& a) {
  return a._not();
}

template <>
inline Vectorized<int32_t> operator~(const Vectorized<int32_t>& a) {
  return a._not();
}

template <>
inline Vectorized<int16_t> operator~(const Vectorized<int16_t>& a) {
  return a._not();
}

template <>
inline Vectorized<int8_t> operator~(const Vectorized<int8_t>& a) {
  return a._not();
}

template <>
inline Vectorized<uint8_t> operator~(const Vectorized<uint8_t>& a) {
  return a._not();
}

#define DEFINE_MAXMIN_FUNCS(operand_type)                                     \
  template <>                                                                 \
  Vectorized<operand_type> inline maximum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.maximum(b);                                                      \
  }                                                                           \
  template <>                                                                 \
  Vectorized<operand_type> inline minimum(                                    \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& b) { \
    return a.minimum(b);                                                      \
  }

#define DEFINE_CLAMP_MAXMIN_FUNCS(typex)                          \
  DEFINE_MAXMIN_FUNCS(typex)                                      \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_min(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& min) { \
    return a.clamp_min(min);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp_max(                  \
      const Vectorized<typex>& a, const Vectorized<typex>& max) { \
    return a.clamp_max(max);                                      \
  }                                                               \
  template <>                                                     \
  Vectorized<typex> C10_ALWAYS_INLINE clamp(                      \
      const Vectorized<typex>& a,                                 \
      const Vectorized<typex>& min,                               \
      const Vectorized<typex>& max) {                             \
    return clamp_max(clamp_min(a, min), max);                     \
  }

DEFINE_CLAMP_MAXMIN_FUNCS(int8_t)
DEFINE_CLAMP_MAXMIN_FUNCS(uint8_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int16_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int32_t)
DEFINE_CLAMP_MAXMIN_FUNCS(int64_t)
DEFINE_CLAMP_MAXMIN_FUNCS(float)
DEFINE_CLAMP_MAXMIN_FUNCS(double)

namespace { /* unnamed namespace */

#if !defined(vec_float) || __ARCH__ < 13
#warning \
    "float->int and int->float conversion is simulated. compile for z15 for improved performance"
inline ZSimdVect<float> vec_int_flt(const ZSimdVect<int> x) {
  return ZSimdVect<float>{float(x[0]), float(x[1]), float(x[2]), float(x[3])};
}
inline ZSimdVect<int> vec_flt_int(const ZSimdVect<float> x) {
  return ZSimdVect<int>{int(x[0]), int(x[1]), int(x[2]), int(x[3])};
}
#else
#define vec_int_flt vec_float
#define vec_flt_int vec_signed
#endif

Vectorized<float> convert_to_float(const Vectorized<int32_t>& x) {
  return {vec_int_flt(x.vec0()), vec_int_flt(x.vec1())};
}

Vectorized<int32_t> convert_to_int(const Vectorized<float>& x) {
  return {vec_flt_int(x.vec0()), vec_flt_int(x.vec1())};
}

Vectorized<double> convert_to_float(const Vectorized<int64_t>& x) {
  return {vec_double(x.vec0()), vec_double(x.vec1())};
}

Vectorized<int64_t> convert_to_int(const Vectorized<double>& x) {
  return {vec_signed(x.vec0()), vec_signed(x.vec1())};
}

} /* unnamed namespace */

template <typename T, typename V>
Vectorized<V> cast_zvector(const Vectorized<T>& x) {
  using cast_type = typename Vectorized<V>::vtype;
  return Vectorized<V>{(cast_type)x.vec0(), (cast_type)x.vec1()};
}

template <>
Vectorized<float> C10_ALWAYS_INLINE fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return Vectorized<float>{
      __builtin_s390_vfmasb(a.vec0(), b.vec0(), c.vec0()),
      __builtin_s390_vfmasb(a.vec1(), b.vec1(), c.vec1())};
}
template <>
Vectorized<double> C10_ALWAYS_INLINE fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return Vectorized<double>{
      __builtin_s390_vfmadb(a.vec0(), b.vec0(), c.vec0()),
      __builtin_s390_vfmadb(a.vec1(), b.vec1(), c.vec1())};
}
template <>
Vectorized<int16_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b,
    const Vectorized<int16_t>& c) {
  return Vectorized<int16_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vectorized<int32_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b,
    const Vectorized<int32_t>& c) {
  return Vectorized<int32_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vectorized<int64_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b,
    const Vectorized<int64_t>& c) {
  return Vectorized<int64_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<double>(const Vectorized<double>& src) {
  return convert_to_int(src);
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<float>(const Vectorized<float>& src) {
  return convert_to_int(src);
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  // int32_t and float have same size
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    const int32_t* src_a = src + i;
    float* dst_a = dst + i;
    auto input_vec = Vectorized<int32_t>::loadu(src_a);
    auto output_vec = convert_to_float(input_vec);
    output_vec.store(dst_a);
  }

  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
inline void convert(const int64_t* src, double* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    const int64_t* src_a = src + i;
    double* dst_a = dst + i;
    auto input_vec = Vectorized<int64_t>::loadu(src_a);
    auto output_vec = convert_to_float(input_vec);
    output_vec.store(dst_a);
  }
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

#define DEFINE_REINTERPRET_CAST_FUNCS(Fst, Cst)     \
  template <>                                       \
  C10_ALWAYS_INLINE Vectorized<Cst> cast<Cst, Fst>( \
      const Vectorized<Fst>& src) {                 \
    return cast_zvector<Fst, Cst>(src);             \
  }

#define DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(Fst) \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, double)      \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, float)       \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int64_t)     \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int32_t)     \
  DEFINE_REINTERPRET_CAST_FUNCS(Fst, int16_t)

DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(float)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(double)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int64_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int32_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int16_t)

#undef DEFINE_REINTERPRET_CAST_FUNCS

template <typename T>
struct unpack_type {
  using type = T;
};
template <>
struct unpack_type<int8_t> {
  using type = int16_t;
};
template <>
struct unpack_type<uint8_t> {
  using type = int16_t;
};
template <>
struct unpack_type<int16_t> {
  using type = int32_t;
};

template <typename T>
struct pack_type {
  using type = T;
};
template <>
struct pack_type<int16_t> {
  using type = int8_t;
};
template <>
struct pack_type<int32_t> {
  using type = int16_t;
};

namespace { /* unnamed namespace */

template <typename T, typename V = typename unpack_type<T>::type>
std::pair<Vectorized<V>, Vectorized<V>> unpack(const Vectorized<T>& x) {
  auto vec0 = vec_unpackh(x.vec0());
  auto vec1 = vec_unpackl(x.vec0());
  auto vec2 = vec_unpackh(x.vec1());
  auto vec3 = vec_unpackl(x.vec1());
  return {Vectorized<V>{vec0, vec1}, Vectorized<V>{vec2, vec3}};
}

template <>
std::pair<Vectorized<int16_t>, Vectorized<int16_t>> unpack<uint8_t, int16_t>(
    const Vectorized<uint8_t>& x) {
  using typeX = typename Vectorized<uint16_t>::vtype;
  typeX vec0 = vec_unpackh(x.vec0());
  typeX vec1 = vec_unpackl(x.vec0());
  typeX vec2 = vec_unpackh(x.vec1());
  typeX vec3 = vec_unpackl(x.vec1());
  // auto mask = Vectorized<uint16_t>(0xFF);
  // vec0 = vec0 & mask;
  // vec1 = vec1 & mask;
  // vec2 = vec2 & mask;
  // vec3 = vec3 & mask;
  return {
      cast_zvector<uint16_t, int16_t>(Vectorized<uint16_t>{vec0, vec1}),
      cast_zvector<uint16_t, int16_t>(Vectorized<uint16_t>{vec2, vec3})};
}

template <typename T, typename V = typename pack_type<T>::type>
Vectorized<V> pack(const Vectorized<T>& first, const Vectorized<T>& second) {
  auto vec0 = vec_packs(first.vec0(), first.vec1());
  auto vec1 = vec_packs(second.vec0(), second.vec1());
  return Vectorized<V>{vec0, vec1};
}

template <>
Vectorized<uint8_t> pack(
    const Vectorized<int16_t>& first,
    const Vectorized<int16_t>& second) {
  auto vec0 = vec_packsu(first.vec0(), first.vec1());
  auto vec1 = vec_packsu(second.vec0(), second.vec1());
  return Vectorized<uint8_t>{vec0, vec1};
}

} /* unnamed namespace */

//////////////////////////////////QUANT///////////////////////////////////////////
template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented_quant<T>()>> {
 public:
  using value_type = typename T::underlying;
  using vtype = ZSimdVect<value_type>;
  using vmaskType = ZSimdVectBinary<value_type>;
  using vinner_type = Vectorized<value_type>;
  using size_type = int;

  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(value_type);
  }

  static constexpr size_t float_num_vecs() {
    return size() / Vectorized<float>::size();
  }
  static constexpr int int_num_vecs() {
    return float_num_vecs();
  }
  using float_vec_return_type = std::array<Vectorized<float>, float_num_vecs()>;
  using int_vec_return_type =
      std::array<Vectorized<c10::qint32>, int_num_vecs()>;

 private:
  vinner_type _vec;

 public:
  Vectorized() {}

  explicit C10_ALWAYS_INLINE Vectorized(vinner_type v) : _vec{v} {}
  Vectorized(const T& val) : _vec(val.val_) {}

  C10_ALWAYS_INLINE const vinner_type& vec() const {
    return _vec;
  }

  static Vectorized<T> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    return Vectorized<T>{vinner_type::loadu(ptr, count)};
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    _vec.store(ptr, count);
  }

  Vectorized<T> relu(Vectorized<T> zero_point) const {
    return Vectorized<T>{_vec.maximum(zero_point._vec)};
  }

  Vectorized<T> relu6(Vectorized<T> zero_point, Vectorized<T> q_six) const {
    auto ret_max = _vec.maximum(zero_point._vec);
    auto ret_min = ret_max.minimum(q_six._vec);
    return Vectorized<T>{ret_min};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  int_vec_return_type widening_subtract(Vectorized<T> b) const {
    return {*this - b};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    auto float_val = convert_to_float(_vec);
    return {fmadd(scale, float_val, scale_zp_premul)};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    auto float_val = convert_to_float(_vec);
    return {(float_val - zero_point) * scale};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 1, int> = 0>
  static Vectorized<T> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    Vectorized<float> vecf = rhs[0];
    vecf = vecf * Vectorized<float>(inverse_scale);
    vecf = vecf.rint() + Vectorized<float>((float)(zero_point));
    auto veci = convert_to_int(vecf);

    return Vectorized<T>{veci};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 1, int> = 0>
  static Vectorized<T> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    Vectorized<T> vi = inp[0];
    auto vecf = convert_to_float(vi.vec());
    vecf = vecf * Vectorized<float>(multiplier);
    vecf = vecf.rint();
    auto veci = convert_to_int(vecf) + Vectorized<int>(zero_point);

    return Vectorized<T>{veci};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 4, int> = 0>
  int_vec_return_type widening_subtract(Vectorized<U> b) const {
    auto ret16 = unpack(_vec);
    auto ret16B = unpack(b.vec());
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);
    auto ret32B_0 = unpack(ret16B.first);
    auto ret32B_1 = unpack(ret16B.second);

    return {
        Vectorized<c10::qint32>(ret32_0.first - ret32B_0.first),
        Vectorized<c10::qint32>(ret32_0.second - ret32B_0.second),
        Vectorized<c10::qint32>(ret32_1.first - ret32B_1.first),
        Vectorized<c10::qint32>(ret32_1.second - ret32B_1.second)};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // unpacking unsigned as signed
    auto ret16 = unpack(_vec);
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);

    auto vecf_0 = convert_to_float(ret32_0.first);
    auto vecf_1 = convert_to_float(ret32_0.second);

    auto vecf_2 = convert_to_float(ret32_1.first);
    auto vecf_3 = convert_to_float(ret32_1.second);
    return {
        fmadd(scale, vecf_0, scale_zp_premul),
        fmadd(scale, vecf_1, scale_zp_premul),
        fmadd(scale, vecf_2, scale_zp_premul),
        fmadd(scale, vecf_3, scale_zp_premul)};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // unpacking unsigned as signed
    auto ret16 = unpack(_vec);
    auto ret32_0 = unpack(ret16.first);
    auto ret32_1 = unpack(ret16.second);

    auto vecf_0 = convert_to_float(ret32_0.first);
    auto vecf_1 = convert_to_float(ret32_0.second);

    auto vecf_2 = convert_to_float(ret32_1.first);
    auto vecf_3 = convert_to_float(ret32_1.second);

    return {
        (vecf_0 - zero_point) * scale,
        (vecf_1 - zero_point) * scale,
        (vecf_2 - zero_point) * scale,
        (vecf_3 - zero_point) * scale };
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::float_num_vecs() == 4, int> = 0>
  static Vectorized<T> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto vec_inverse = Vectorized<float>(inverse_scale);
    auto vec_zero_point = Vectorized<float>((float)zero_point);

    auto vecf0 = rhs[0];
    auto vecf2 = rhs[1];
    auto vecf4 = rhs[2];
    auto vecf6 = rhs[3];

    vecf0 = vecf0 * vec_inverse;
    vecf2 = vecf2 * vec_inverse;
    vecf4 = vecf4 * vec_inverse;
    vecf6 = vecf6 * vec_inverse;

    vecf0 = vecf0.rint() + vec_zero_point;
    vecf2 = vecf2.rint() + vec_zero_point;
    vecf4 = vecf4.rint() + vec_zero_point;
    vecf6 = vecf6.rint() + vec_zero_point;

    auto veci0 = convert_to_int(vecf0);
    auto veci2 = convert_to_int(vecf2);
    auto veci4 = convert_to_int(vecf4);
    auto veci6 = convert_to_int(vecf6);

    auto vecshi0 = pack(veci0, veci2);
    auto vecshi2 = pack(veci4, veci6);
    auto ret = pack<int16_t, typename U::underlying>(vecshi0, vecshi2);

    return Vectorized<T>{ret};
  }

  template <
      typename U = T,
      std::enable_if_t<Vectorized<U>::int_num_vecs() == 4, int> = 0>
  static Vectorized<U> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    Vectorized<float> vec_multiplier = Vectorized<float>(multiplier);
    Vectorized<int32_t> vec_zero_point = Vectorized<int32_t>(zero_point);

    Vectorized<c10::qint32> vi0 = inp[0];
    Vectorized<c10::qint32> vi1 = inp[1];
    Vectorized<c10::qint32> vi2 = inp[2];
    Vectorized<c10::qint32> vi3 = inp[3];

    auto vecf0 = convert_to_float(vi0.vec());
    auto vecf2 = convert_to_float(vi1.vec());

    auto vecf4 = convert_to_float(vi2.vec());
    auto vecf6 = convert_to_float(vi3.vec());

    vecf0 = vecf0 * vec_multiplier;
    vecf2 = vecf2 * vec_multiplier;

    vecf4 = vecf4 * vec_multiplier;
    vecf6 = vecf6 * vec_multiplier;

    vecf0 = vecf0.rint();
    vecf2 = vecf2.rint();
    vecf4 = vecf4.rint();
    vecf6 = vecf6.rint();

    auto veci0 = convert_to_int(vecf0);
    auto veci2 = convert_to_int(vecf2);
    auto veci4 = convert_to_int(vecf4);
    auto veci6 = convert_to_int(vecf6);

    veci0 = veci0 + vec_zero_point;
    veci2 = veci2 + vec_zero_point;

    veci4 = veci4 + vec_zero_point;
    veci6 = veci6 + vec_zero_point;

    auto vecshi0 = pack<int32_t, int16_t>(veci0, veci2);
    auto vecshi2 = pack<int32_t, int16_t>(veci4, veci6);

    auto ret = pack<int16_t, typename U::underlying>(vecshi0, vecshi2);

    return Vectorized<U>{ret};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator+(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec + other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator-(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec - other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator*(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec * other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator/(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec / other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator&(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec & other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator|(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec | other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator^(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec ^ other._vec};
  }
  Vectorized<T> C10_ALWAYS_INLINE operator==(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec == other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator!=(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec != other._vec};
  }
  Vectorized<T> C10_ALWAYS_INLINE operator>(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec > other._vec};
  }
  Vectorized<T> C10_ALWAYS_INLINE operator>=(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec >= other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator<(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec < other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator<=(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec <= other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.eq(other._vec)};
  }
  Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.ne(other._vec)};
  }
  Vectorized<T> C10_ALWAYS_INLINE gt(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.gt(other._vec)};
  }
  Vectorized<T> C10_ALWAYS_INLINE ge(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.ge(other._vec)};
  }
  Vectorized<T> C10_ALWAYS_INLINE lt(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.lt(other._vec)};
  }
  Vectorized<T> C10_ALWAYS_INLINE le(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.le(other._vec)};
  }

  Vectorized<T> clamp_min(const Vectorized<T>& min) const {
    return Vectorized<T>{_vec.clamp_min(min._vec)};
  }

  Vectorized<T> clamp_max(const Vectorized<T>& max) const {
    return Vectorized<T>{_vec.clamp_max(max._vec)};
  }

  Vectorized<T> minimum(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.minimum(other._vec)};
  }

  Vectorized<T> maximum(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec.maximum(other._vec)};
  }
};

DEFINE_CLAMP_MAXMIN_FUNCS(c10::quint8)
DEFINE_CLAMP_MAXMIN_FUNCS(c10::qint8)
DEFINE_CLAMP_MAXMIN_FUNCS(c10::qint32)

template <typename U = float>
constexpr auto real_mask() {
  return (ZSimdVect<U>)ZSimdVectBinary<float>{0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
}

template <>
constexpr auto real_mask<double>() {
  return (ZSimdVect<double>)ZSimdVectBinary<double>{0xFFFFFFFFFFFFFFFF, 0};
}

template <typename U = float>
constexpr auto image_mask() {
  return (ZSimdVect<U>)ZSimdVectBinary<U>{0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
}

template <>
constexpr auto image_mask<double>() {
  return (ZSimdVect<double>)ZSimdVectBinary<double>{0, 0xFFFFFFFFFFFFFFFF};
}

template <typename U = float>
constexpr auto rsign_mask() {
  return ZSimdVect<U>{-0.f, 0.f, -0.f, 0.f};
}

template <>
constexpr auto rsign_mask<double>() {
  return ZSimdVect<double>{-0.0, 0.f};
}

template <typename U = float>
constexpr auto isign_mask() {
  return ZSimdVect<U>{0.0, -0.f, 0.0, -0.f};
}

template <>
constexpr auto isign_mask<double>() {
  return ZSimdVect<double>{0.0, -0.0};
}

template <typename U = float>
constexpr auto image_one() {
  return ZSimdVect<U>{0, 1.f, 0, 1.f};
}

template <>
constexpr auto image_one<double>() {
  return ZSimdVect<double>{0.0, 1.0};
}

template <typename U = float>
constexpr auto pi_half() {
  return ZSimdVect<U>{(float)(M_PI / 2.0), 0.f, (float)(M_PI / 2.0), 0.f};
}

template <>
constexpr auto pi_half<double>() {
  return ZSimdVect<double>{M_PI / 2.0, 0.0};
}

template <typename U = float>
constexpr auto image_half() {
  return ZSimdVect<U>{0, 0.5f, 0, 0.5f};
}

template <>
constexpr auto image_half<double>() {
  return ZSimdVect<double>{0.0, 0.5};
}

template <typename U>
constexpr U log2e_inv() {
  return static_cast<U>(1.4426950408889634);
}

template <typename U>
constexpr U log10e_inv() {
  return static_cast<U>(0.43429448190325176);
}

template <typename T>
struct Vectorized<T, std::enable_if_t<is_zarch_implemented_complex<T>()>> {
 public:
  using underline_type = decltype(std::declval<T>().imag());
  using value_type = T;
  using vtype = ZSimdVect<underline_type>;
  using vmaskType = ZSimdVectBinary<underline_type>;
  using vinner_type = Vectorized<underline_type>;
  using size_type = int;
  using vinner_data = typename Vectorized<underline_type>::vinner_data;

  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(value_type);
  }

 private:
  vinner_type _vec;

 public:
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(const vinner_data &v) : _vec{v.first, v.second} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 16), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2)
      : _vec{s1.real(), s1.imag(), s2.real(), s2.imag()} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s1, T s2, T s3, T s4)
      : _vec{
            s1.real(),
            s1.imag(),
            s2.real(),
            s2.imag(),
            s3.real(),
            s3.imag(),
            s4.real(),
            s4.imag()} {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 16), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s) : Vectorized<T>(s, s) {}

  template <typename U = T, std::enable_if_t<(sizeof(U) == 8), int> = 0>
  C10_ALWAYS_INLINE Vectorized(T s) : Vectorized<T>(s, s, s, s) {}

  C10_ALWAYS_INLINE operator vinner_type() const {
    return _vec;
  }

  C10_ALWAYS_INLINE const vinner_type& vec() const {
    return _vec;
  }

  C10_ALWAYS_INLINE operator vinner_data() const {
    return _vec.data();
  }

  C10_ALWAYS_INLINE vinner_data data() const {
    return _vec.data();
  }

  static Vectorized<T> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    return Vectorized<T>{vinner_type::loadu(ptr, 2 * count)};
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    return _vec.store(ptr, 2 * count);
  }

  static Vectorized<T> blendv(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      const Vectorized<T>& mask) {
    // convert std::complex<V> index mask to V index mask: xy -> xxyy
    vinner_type vmask = mask.vec();
    auto mask_complex = vinner_type(
        vec_mergeh(vmask.vec0(), vmask.vec0()),
        vec_mergeh(vmask.vec1(), vmask.vec1()));
    return Vectorized<T>{vinner_type::blendv(a.vec(), b.vec(), mask_complex)};
  }

  template <int64_t mask>
  static auto C10_ALWAYS_INLINE
  blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    constexpr int mask_complex = maskForComplex<sizeof(T)>(mask);
    return Vectorized<T>{
        vinner_type::template blend<mask_complex>(a.vec(), b.vec())};
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 16, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(base, base + step);
  }

  template <typename step_t, typename U = T>
  static std::enable_if_t<sizeof(U) == 8, Vectorized<T>> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + value_type(2) * step,
        base + value_type(3) * step);
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z >= C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    return b;
  }

  template <int16_t Z, int16_t C>
  static inline std::enable_if_t<(Z < C), Vectorized<T>> set_inner(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count) {
    if (count == Z)
      return blend<allbitset(Z)>(a, b);
    else
      return set_inner<Z + 1, C>(a, b, count);
  }

  static Vectorized<T> set(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      size_t count = size()) {
    if (count == 0)
      return a;
    return set_inner<1, size()>(a, b, count);
  }

  const T& operator[](int idx) const = delete;
  T& operator[](int idx) = delete;

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(const T&)) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{
        f(T(v0[0], v0[1])),
        f(T(v0[2], v0[3])),
        f(T(v1[0], v1[1])),
        f(T(v1[2], v1[3]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  Vectorized<U> mapOrdinary(T (*const f)(const T&)) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{f(T(v0[0], v0[1])), f(T(v1[0], v1[1]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(T)) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{
        f(T(v0[0], v0[1])),
        f(T(v0[2], v0[3])),
        f(T(v1[0], v1[1])),
        f(T(v1[2], v1[3]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  Vectorized<T> mapOrdinary(T (*const f)(T)) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    return Vectorized<T>{f(T(v0[0], v0[1])), f(T(v1[0], v1[1]))};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      T (*const f)(const T&, const T&),
      const Vectorized<T>& b) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    auto bvec = b.vec();
    auto b0 = bvec.vec0();
    auto b1 = bvec.vec1();
    T a00 = f(T(v0[0], v0[1]), T(b0[0], b0[1]));
    T a01 = f(T(v0[2], v0[3]), T(b0[2], b0[3]));
    T a02 = f(T(v1[0], v1[1]), T(b1[0], b1[1]));
    T a03 = f(T(v1[2], v1[3]), T(b1[2], b1[3]));
    return Vectorized<T>{a00, a01, a02, a03};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  inline Vectorized<T> mapOrdinary(
      T (*const f)(const T&, const T&),
      const Vectorized<T>& b) const {
    auto v0 = _vec.vec0();
    auto v1 = _vec.vec1();
    auto bvec = b.vec();
    auto b0 = bvec.vec0();
    auto b1 = bvec.vec1();
    U a00 = f(U(v0[0], v0[1]), U(b0[0], b0[1]));
    U a01 = f(U(v1[0], v1[1]), U(b1[0], b1[1]));
    return Vectorized<T>{a00, a01};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator+(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec + other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator-(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec - other._vec};
  }

  Vectorized<T> inline operator*(const Vectorized<T>& b) const {
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
    vinner_type bv = b.vec();
#if !defined(ZVECTOR_SIMULATE_X86_MULT)
    // this is more z arch friendly than simulating horizontal from x86
    vinner_type vi = bv.mergeo();
    vinner_type vr = bv.mergee();
    vi = vi ^ rsign_mask<underline_type>();
    vinner_type ret = _vec * vr;
    vinner_type vx_swapped = _vec.swapped();
    ret = fmadd(vx_swapped, vi, ret);
#else
    vinner_type ac_bd = _vec * b;
    vinner_type d_c = bv.swapped();
    d_c = d_c ^ isign_mask<underline_type>();
    vinner_type ad_bc = _vec * d_c;
    vinner_type ret = vinner_type::horizontal_sub_perm(ac_bd, ad_bc);
#endif
    return Vectorized<T>{ret};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<float>>::value, int> = 0>
  static typename Vectorized<T>::vinner_type real_neg(const typename Vectorized<T>::vinner_type &a)
  {
    const auto swap_mask = ZSimdVectBinary<uint8_t>{
      0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31};

    auto a_neg = a.neg();
    vtype v0 = vec_perm(a_neg.vec0(), a.vec0(), swap_mask);
    vtype v1 = vec_perm(a_neg.vec1(), a.vec1(), swap_mask);
    return {v0, v1};
  }

  template <
      typename U = T,
      std::enable_if_t<std::is_same<U, c10::complex<double>>::value, int> = 0>
  static typename Vectorized<T>::vinner_type real_neg(const typename Vectorized<T>::vinner_type &a)
  {
    auto a_neg = a.neg();
    auto v0 = vec_permi(a_neg.vec0(), a.vec0(), 1);
    auto v1 = vec_permi(a_neg.vec1(), a.vec1(), 1);
    return { v0, v1 };
  }

  Vectorized<T> inline operator/(const Vectorized<T>& b) const {
    // Unfortunately, this breaks some tests
    // Implement it like it's done for avx2
    auto fabs_cd = b.vec().abs();                               // |c|    |d|
    auto fabs_dc = fabs_cd.swapped();                           // |d|    |c|
    auto scale = vinner_type {1.0} / maximum(fabs_cd, fabs_dc); // 1/sc     1/sc
    auto a2 = vec() * scale;                                    // a/sc     b/sc
    auto b2 = b.vec() * scale;                                  // c/sc     d/sc
    auto acbd2 = a2 * b2;                                       // ac/sc^2  bd/sc^2

    auto dc2 = b2.swapped();                                    // d/sc         c/sc
    dc2 = Vectorized<T>::real_neg(dc2);                         // -d/|c,d|        c/sc
    auto adbc2 = a2 * dc2;                                      // -ad/sc^2      bc/sc^2
    auto sum1 = acbd2 + acbd2.swapped();                        // (ac+bd)/sc^2  (ac+bd)/sc^2
    auto sum2 = adbc2 + adbc2.swapped();                        // (bc-ad)/sc^2  (bc-ad)/sc^2
    auto res2 = vinner_type::mergee(sum1, sum2);                // (ac+bd)/sc^2  (bc-ad)/sc^2

    // get the denominator
    auto denom2 = Vectorized<T>{b2}.abs_2_();                   // (c^2+d^2)/sc^2   (c^2+d^2)/sc^2
    res2 = res2 / denom2;
    return Vectorized<T>{ res2 };
  }

  Vectorized<T> angle2_() const {
    auto b_a = _vec.swapped(); // b        a
    return Vectorized<T>{_vec.atan2(b_a).swapped()};
  }

  Vectorized<T> angle() const {
    return angle2_().real();
  }

  Vectorized<T> atan() const {
    // atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized<T>{vinner_type(image_one<underline_type>())};
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln *
        Vectorized<T>{vinner_type(image_half<underline_type>())}; // i/2*ln()
  }

  Vectorized<T> atanh() const {
    return mapOrdinary(std::atanh);
  }

  Vectorized<T> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
#if 1
    vinner_type cnj = conj().vec();
    vinner_type b_a = cnj.swapped();
    vinner_type ab = cnj * b_a;
    vinner_type im = ab + ab;
    vinner_type val_2 = _vec * _vec;
    vinner_type val_2_swapped = val_2.swapped();
    vinner_type re = vinner_type::horizontal_sub_perm(val_2, val_2_swapped);
    re = vinner_type(static_cast<underline_type>(1)) - re;
    constexpr int blend_mask =
        blend_choice<T>(); // 0x0A for complex<double> , 0xAA for complex<float>
    vinner_type blendx = vinner_type::template blend<blend_mask>(re, im);
    auto root = Vectorized<T>(blendx).sqrt();
    auto ln = Vectorized<T>(Vectorized<T>(b_a) + root).log();
    return Vectorized<T>(ln.vec().swapped()).conj();
#else
    return mapOrdinary(std::asin);
#endif
  }

  Vectorized<T> acos() const {
    // acos(x) = pi/2 - asin(x)
    return Vectorized<T>(vinner_type(pi_half<underline_type>())) - asin();
  }

  Vectorized<T> sin() const {
    return mapOrdinary(std::sin);
  }
  Vectorized<T> sinh() const {
    return mapOrdinary(std::sinh);
  }
  Vectorized<T> cos() const {
    return mapOrdinary(std::cos);
  }
  Vectorized<T> cosh() const {
    return mapOrdinary(std::cosh);
  }
  Vectorized<T> ceil() const {
    return Vectorized<T>{_vec.ceil()};
  }
  Vectorized<T> floor() const {
    return Vectorized<T>{_vec.floor()};
  }
  Vectorized<T> neg() const {
    return Vectorized<T>(_vec.neg());
  }
  Vectorized<T> round() const {
    return Vectorized<T>{_vec.round()};
  }
  Vectorized<T> tan() const {
    return mapOrdinary(std::tan);
  }
  Vectorized<T> tanh() const {
    return mapOrdinary(std::tanh);
  }
  Vectorized<T> trunc() const {
    return Vectorized<T>{_vec.trunc()};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator&(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec & other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator|(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec | other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator^(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec ^ other._vec};
  }
  Vectorized<T> C10_ALWAYS_INLINE operator==(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec == other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE operator!=(const Vectorized<T>& other) const {
    return Vectorized<T>{_vec != other._vec};
  }

  Vectorized<T> C10_ALWAYS_INLINE eq(const Vectorized<T>& other) const {
    auto eq = _vec.eq(other._vec);  // compares real and imag individually
    // If both real numbers and imag numbers are equal, then the complex numbers are equal
    auto real = eq & vinner_type(real_mask<underline_type>());
    auto imag = (eq & vinner_type(image_mask<underline_type>())).swapped();
    return Vectorized<T>{real & imag};
  }
  Vectorized<T> C10_ALWAYS_INLINE ne(const Vectorized<T>& other) const {
    auto ne = _vec.ne(other._vec);  // compares real and imag individually
    // If either real numbers or imag numbers are not equal, then the complex numbers are not equal
    auto real = ne & vinner_type(real_mask<underline_type>());
    auto imag = (ne & vinner_type(image_mask<underline_type>())).swapped();
    return Vectorized<T>{real | imag};
  }

  Vectorized<T> real() const {
    return Vectorized<T>(_vec & vinner_type(real_mask<underline_type>()));
  }
  Vectorized<T> imag_() const {
    return Vectorized<T>(_vec & vinner_type(image_mask<underline_type>()));
  }
  Vectorized<T> imag() const {
    return Vectorized<T>{
        (_vec & vinner_type(image_mask<underline_type>())).swapped()};
  }

  Vectorized<T> conj() const {
    return Vectorized<T>(_vec ^ vinner_type(isign_mask<underline_type>()));
  }

  vinner_data abs_2_() const {
    auto a = _vec * _vec;
    a = a + a.swapped();
    return a.mergee().data();
  }

  static T abs_helper(const T &value)
  {
    return T(std::abs(value));
  }

  Vectorized<T> abs() const {
    return mapOrdinary(abs_helper);
  }

  Vectorized<T> exp() const {
    return mapOrdinary(std::exp);
  }

  Vectorized<T> exp2() const {
    return mapOrdinary(exp2_impl);
  }

  Vectorized<T> expm1() const {
    return mapOrdinary(std::expm1);
  }

  Vectorized<T> log() const {
    return mapOrdinary(std::log);
  }

  Vectorized<T> log2() const {
    // log2eB_inv
    auto ret = log();
    return Vectorized<T>{ret._vec * vinner_type(log2e_inv<underline_type>())};
  }

  Vectorized<T> log10() const {
    auto ret = log();
    return Vectorized<T>{ret._vec * vinner_type(log10e_inv<underline_type>())};
  }

  Vectorized<T> log1p() const {
    return mapOrdinary(std::log1p);
  }

  Vectorized<T> sgn() const {
    return mapOrdinary(at::native::sgn_impl);
  }

  Vectorized<T> pow(const Vectorized<T>& exp) const {
    return mapOrdinary(std::pow, exp);
  }

  Vectorized<T> sqrt() const {
    return mapOrdinary(std::sqrt);
  }

  Vectorized<T> reciprocal() const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    vinner_type c_d = _vec ^ vinner_type(isign_mask<underline_type>());
    vinner_type abs = abs_2_();
    return Vectorized<T>{c_d / abs};
  }

  Vectorized<T> rsqrt() const {
    return sqrt().reciprocal();
  }

  Vectorized<T> operator<(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> operator<=(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> operator>(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> operator>=(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> lt(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> le(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> gt(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<T> ge(const Vectorized<T>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

template <typename T, std::enable_if_t<(sizeof(T) == 8), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_interleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a      = {a0, a1, a2, a3}
  //   b      = {b0, b1, b2, b3}
  using vtype = typename Vectorized<T>::vtype;
  vtype ab00 = vec_permi(a.vec0(), b.vec0(), 0);
  vtype ab11 = vec_permi(a.vec0(), b.vec0(), 3);
  vtype ab2_00 = vec_permi(a.vec1(), b.vec1(), 0);
  vtype ab2_11 = vec_permi(a.vec1(), b.vec1(), 3);
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(
      Vectorized<T>{ab00, ab11}, Vectorized<T>{ab2_00, ab2_11});
}

template <typename T, std::enable_if_t<(sizeof(T) == 8), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_deinterleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  using vtype = typename Vectorized<T>::vtype;
  vtype aa01 = vec_permi(a.vec0(), a.vec1(), 0);
  vtype aa23 = vec_permi(b.vec0(), b.vec1(), 0);

  vtype bb_01 = vec_permi(a.vec0(), a.vec1(), 3);
  vtype bb_23 = vec_permi(b.vec0(), b.vec1(), 3);

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(Vectorized<T>{aa01, aa23}, Vectorized<T>{bb_01, bb_23});
}

template <typename T, std::enable_if_t<(sizeof(T) == 4), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_interleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3,, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3,, b4, b5, b6, b7}
  using vtype = typename Vectorized<T>::vtype;
  vtype ab0011 = vec_mergeh(a.vec0(), b.vec0());
  vtype ab2233 = vec_mergel(a.vec0(), b.vec0());

  vtype ab2_0011 = vec_mergeh(a.vec1(), b.vec1());
  vtype ab2_2233 = vec_mergel(a.vec1(), b.vec1());
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1,, a2, b2, a3, b3}
  //          {a4, b4, a5, b5,, a6, b6, a7, b7}

  return std::make_pair(
      Vectorized<T>{ab0011, ab2233}, Vectorized<T>{ab2_0011, ab2_2233});
}

template <typename T, std::enable_if_t<(sizeof(T) == 4), int> = 0>
std::pair<Vectorized<T>, Vectorized<T>> inline inner_deinterleave2(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1,, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5,, a6, b6, a7, b7}
  using vtype = typename Vectorized<T>::vtype;
  // {a0,a2,b0,b2} {a1,a3,b1,b3}
  vtype a0a2b0b2 = vec_mergeh(a.vec0(), a.vec1());
  vtype a1a3b1b3 = vec_mergel(a.vec0(), a.vec1());

  vtype aa0123 = vec_mergeh(a0a2b0b2, a1a3b1b3);
  vtype bb0123 = vec_mergel(a0a2b0b2, a1a3b1b3);

  vtype a0a2b0b2_2 = vec_mergeh(b.vec0(), b.vec1());
  vtype a1a3b1b3_2 = vec_mergel(b.vec0(), b.vec1());

  vtype aa0123_2 = vec_mergeh(a0a2b0b2_2, a1a3b1b3_2);
  vtype bb0123_2 = vec_mergel(a0a2b0b2_2, a1a3b1b3_2);

  // it could be done with vec_perm ,too
  // swap lanes:
  //   return {a0, a1, a2, a3,, a4, a5, a6, a7}
  //          {b0, b1, b2, b3,, b4, b5, b6, b7}

  return std::make_pair(
      Vectorized<T>{aa0123, aa0123_2}, Vectorized<T>{bb0123, bb0123_2});
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return inner_interleave2<float>(a, b);
}

template <>
std::pair<Vectorized<int32_t>, Vectorized<int32_t>> inline interleave2<int32_t>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return inner_interleave2<int32_t>(a, b);
}

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return inner_interleave2<double>(a, b);
}

template <>
std::pair<Vectorized<int64_t>, Vectorized<int64_t>> inline interleave2<int64_t>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return inner_interleave2<int64_t>(a, b);
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return inner_deinterleave2<float>(a, b);
}

template <>
std::pair<Vectorized<int32_t>, Vectorized<int32_t>> inline deinterleave2<
    int32_t>(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return inner_deinterleave2<int32_t>(a, b);
}

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return inner_deinterleave2<double>(a, b);
}

template <>
std::pair<Vectorized<int64_t>, Vectorized<int64_t>> inline deinterleave2<
    int64_t>(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return inner_deinterleave2<int64_t>(a, b);
}

template <typename T>
typename std::enable_if<std::is_same<T, uint8_t>::value, at::vec::Vectorized<float>>::type
inline convert_int8_to_float(const Vectorized<T> &src) {
  // Note: this function only convert inputs number of elements equal to at::vec::Vectorized<float>.size()
  // Only handle first 64 bits
  auto vec_int = src.to_vec_float_helper();

  return convert_to_float(vec_int);
}

template <typename T>
typename std::enable_if<std::is_same<T, uint8_t>::value, at::vec::Vectorized<T>>::type
inline convert_float_to_int8(const Vectorized<float> &src) {
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();

  auto vec_int = clamp(convert_to_int(src), Vectorized<int32_t>(min_val), Vectorized<int32_t>(max_val));

  return vec_int.to_vec_uint8_helper();
}

#undef DEFINE_CLAMP_MAXMIN_FUNCS
#undef DEFINE_MAXMIN_FUNCS
} // namespace
} // namespace vec
} // namespace at
