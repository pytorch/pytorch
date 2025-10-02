#pragma once

#if defined(__aarch64__)
#include <ATen/cpu/vec/vec_common_aarch64.h>
#elif defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec128/vec128.h>
#include <ATen/cpu/vec/vec256/vec256.h>
#endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
  stream << val.val_;
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
  stream << static_cast<int>(val.val_);
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
  stream << static_cast<unsigned int>(val.val_);
  return stream;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vectorized<T>& vec) {
  T buf[Vectorized<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}

inline Vectorized<bool> convert_to_bool(Vectorized<int8_t> x) {
  __at_align__ bool buffer[x.size()];
  x.ne(Vectorized<int8_t>(0)).store(buffer);

  Vectorized<bool> ret;
  static_assert(x.size() == ret.size());
  std::memcpy(ret, buffer, ret.size() * sizeof(bool));
  return ret;
}

template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr) {
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr));
}

template <>
inline Vectorized<bool> Vectorized<bool>::loadu(
    const void* ptr,
    int64_t count) {
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr, count));
}

template <typename VT>
struct VecHoldType {
  using hold_type = typename VT::value_type;
};

template <>
struct VecHoldType<Vectorized<BFloat16>> {
  using hold_type = BFloat16;
};

template <>
struct VecHoldType<Vectorized<Half>> {
  using hold_type = Half;
};

template <typename VT>
using vechold_type = typename VecHoldType<VT>::hold_type;

} // namespace CPU_CAPABILITY
} // namespace at::vec
