#pragma once

#include "intrinsics.h"

#include "vec256_base.h"
#include "vec256_float.h"
#include "vec256_double.h"
#include "vec256_int.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace at {
namespace vec256 {
namespace {

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vec256<T>& vec) {
  T buf[Vec256<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != vec.size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}

}}}
