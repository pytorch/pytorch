#pragma once

#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorMethods.h>
#include <ATen/core/Type.h>
#include <iostream>

namespace at {

CAFFE2_API std::ostream& operator<<(std::ostream& out, Backend b);
CAFFE2_API std::ostream& operator<<(std::ostream& out, const Type& t);
CAFFE2_API std::ostream& print(
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
  return out << (s.isFloatingPoint() ? s.toDouble() : s.toLong());
}

}
