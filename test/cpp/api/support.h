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

inline int count_substr_occurrences(const std::string& str, const std::string& substr) {
  int count = 0;
  size_t pos = str.find(substr);

  while (pos != std::string::npos) {
    count++;
    pos = str.find(substr, pos + substr.size());
  }

  return count;
}

// A RAII, thread local (!) guard that changes default dtype upon
// construction, and sets it back to the original dtype upon destruction.
//
// Usage of this guard is synchronized across threads, so that at any given time,
// only one guard can take effect.
struct AutoDefaultDtypeMode {
  static std::mutex default_dtype_mutex;

  AutoDefaultDtypeMode(c10::ScalarType default_dtype) : prev_default_dtype(torch::typeMetaToScalarType(torch::get_default_dtype())) {
    default_dtype_mutex.lock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(default_dtype));
  }
  ~AutoDefaultDtypeMode() {
    default_dtype_mutex.unlock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(prev_default_dtype));
  }
  c10::ScalarType prev_default_dtype;
};

} // namespace test
} // namespace torch
