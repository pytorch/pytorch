#ifndef WRAP_COMPLEX
#define WRAP_COMPLEX

#include <c10/util/complex_type.h>

template <typename T>
using wrap_complex_t = T;

template <typename T>
C10_HOST_DEVICE inline T wrap_complex(T t) {
  return t;
}

#endif
