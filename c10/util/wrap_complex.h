// wrap_complex:
// This helper converts c10::complex to std::complex
// This is helper that allows us to incrementally replace
// std::complex with c10::complex. When the switch from
// std::complex to c10::complex is completely done, we will not
// need this any more

#ifndef WRAP_COMPLEX
#define WRAP_COMPLEX

#include <c10/util/complex_type.h>
template <typename T>
struct wrap_complex_helper {
  using type = T;
  C10_HOST_DEVICE static inline type cast(T t) {
    return type(t);
  }
};

template <typename T>
struct wrap_complex_helper<c10::complex<T>> {
  using type = std::complex<T>;
  C10_HOST_DEVICE static inline type cast(T t) {
    return type(t);
  }
};

template <typename T>
using wrap_complex_t = typename wrap_complex_helper<T>::type;

template <typename T>
auto wrap_complex(T t) -> wrap_complex_t<T> {
  return wrap_complex_helper<T>::cast(t);
}

#endif
