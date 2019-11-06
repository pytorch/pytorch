#pragma once

#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeId.h>
#include <torch/csrc/python_headers.h>

namespace torch {
namespace nested_tensor {

struct Pet {
  Pet(const std::string &name) : name(name) {}
  void setName(const std::string &name_) { name = name_; }
  const std::string &getName() const { return name; }

  std::string name;
};

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
