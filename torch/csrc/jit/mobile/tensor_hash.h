#pragma once
#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

struct tensor_value_hash {
  std::size_t operator()(const at::Tensor& tensor) const {
    std::stringstream tensor_stream;
    tensor_stream << tensor;
    std::string tensor_str = tensor_stream.str();
    std::size_t h1 = std::hash<std::string>{}(tensor_str);
    return h1;
  }
};

struct tensor_value_equal {
  bool operator()(const at::Tensor& a, const at::Tensor& b) const {
    std::stringstream a_stream;
    a_stream << a;
    std::string a_str = a_stream.str();

    std::stringstream b_stream;
    b_stream << b;
    std::string b_str = b_stream.str();
    return a_str == b_str;
  }
};

} // namespace jit
} // namespace torch
