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

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
  }
};

struct CerrRedirect {
  CerrRedirect(std::streambuf * new_buffer) : prev_buffer(std::cerr.rdbuf(new_buffer)) {}

  ~CerrRedirect( ) {
    std::cerr.rdbuf(prev_buffer);
  }

private:
  std::streambuf * prev_buffer;
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data_ptr<float>() == second.data_ptr<float>();
}

// yf225 TODO: clean these up!
inline void assert_equal(const at::Tensor& first, const at::Tensor& second) {
  ASSERT_TRUE(first.sizes() == second.sizes());
  if (first.dtype() == torch::kBool && second.dtype() == torch::kBool) {
    ASSERT_TRUE(first.equal(second));
  } else {
    ASSERT_TRUE(first.allclose(second));
  }
}

#define TENSOR(T, S) \
inline void assert_equal(const c10::ArrayRef<T>& first, const c10::ArrayRef<T>& second) { \
  ASSERT_TRUE(first.size() == second.size()); \
}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

inline void assert_not_equal(const at::Tensor& first, const at::Tensor& second) {
  ASSERT_FALSE(first.sizes() == second.sizes() && first.allclose(second));
}

template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return left.item<T>() == right;
}

template <typename T>
bool almost_equal(at::Tensor left, T right, T tolerance = 1e-4) {
  return std::abs(left.item<T>() - right) < tolerance;
}

inline int count_substr_occurrences(const std::string& str, const std::string& substr) {
  int count = 0;
  size_t pos = str.find(substr);

  while (pos != std::string::npos) {
    count++;
    pos = str.find(substr, pos + substr.size());
  }

  return count;
}

} // namespace test
} // namespace torch
