#pragma once

#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <iostream>


namespace c10 {
TORCH_API std::ostream& operator<<(std::ostream& out, Backend b);
}
namespace at {

TORCH_API std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t);
TORCH_API std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor,
    int64_t linesize);
static inline std::ostream& operator<<(std::ostream & out, const Tensor & t) {
  return print(out,t,80);
}
static inline void print(const Tensor & t, int64_t linesize=80) {
  print(std::cout,t,linesize);
}

static inline std::ostream& operator<<(std::ostream & out, Scalar s) {
  if (s.isFloatingPoint()) {
    return out << s.toDouble();
  }
  if (s.isComplex()) {
    return out << s.toComplexDouble();
  }
  if (s.isBoolean()) {
    return out << (s.toBool() ? "true" : "false");
  }
  if (s.isIntegral(false)) {
    return out << s.toLong();
  }
  throw std::logic_error("Unknown type in Scalar");
}

}
