#pragma once

#include <test/cpp/common/support.h>

#include <gtest/gtest.h>

#include <torch/nn/cloneable.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <string>
#include <utility>

namespace torch {
namespace test {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  void reset() override {}

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module_holder);
  }
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data<float>() == second.data<float>();
}

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
  }
};

} // namespace test
} // namespace torch
