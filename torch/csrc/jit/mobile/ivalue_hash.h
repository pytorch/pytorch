#pragma once
#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

struct MyHash {
  std::size_t operator()(const c10::IValue& value) const {
    if (value.isTensor()) {
      std::stringstream tensor_stream;
      tensor_stream << value;
      std::string tensor_str = tensor_stream.str();
      std::size_t h1 = std::hash<std::string>{}(tensor_str);
      return h1;
    } else {
      return value.hash(value);
    }
  }
};

struct MyEqual {
  bool operator()(const c10::IValue& a, const c10::IValue& b) const {
    if (a.isTensor() && b.isTensor()) {
      std::stringstream a_stream;
      a_stream << a;
      std::string a_str = a_stream.str();

      std::stringstream b_stream;
      b_stream << b;
      std::string b_str = b_stream.str();
      return a_str == b_str;
    } else {
      return a == b;
    }
  }
};

} // namespace jit
} // namespace torch
