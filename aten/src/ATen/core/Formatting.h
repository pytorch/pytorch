#pragma once

#include <ostream>
#include <string>

#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>

namespace c10 {
TORCH_API std::ostream& operator<<(std::ostream& out, Backend b);
TORCH_API std::ostream& operator<<(std::ostream & out, const Scalar& s);
TORCH_API std::string toString(const Scalar& s);
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
TORCH_API void print(const Tensor & t, int64_t linesize=80);
}
