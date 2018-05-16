#pragma once

#include <torch/csrc/autograd/variable.h>

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace torch {
class ParameterList {
 public:
  ParameterList() = default;

  /* implicit */ ParameterList(
      std::initializer_list<std::pair<std::string, Variable>> list) {
    for (auto& pair : list) {
      push_back(std::move(pair.first), std::move(pair.second));
    }
  }

  void push_back(std::string name, Variable parameter) {
    names_.emplace_back(std::move(name));
    parameters_.emplace_back(std::move(parameter));
  }

  const std::string& name(size_t index) const noexcept {
    return names_[index];
  }

  Variable& operator[](size_t index) noexcept {
    return parameters_[index];
  }

  const Variable& operator[](size_t index) const noexcept {
    return parameters_[index];
  }

  size_t size() const noexcept {
    return parameters_.size();
  }

 private:
  std::vector<Variable> parameters_;
  std::vector<std::string> names_;
};
} // namespace torch
