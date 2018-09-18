#pragma once

#include <torch/nn/cloneable.h>
#include <torch/tensor.h>

#include <cstdio>
#include <cstdlib>
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

inline std::string get_tempfile() {
#ifdef WIN32
  return std::tmpnam(nullptr);
#else
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  char filename[] = "/tmp/fileXXXXXX";
  mkstemp(filename);
  return std::string(filename);
#endif
}
} // namespace test
} // namespace torch
