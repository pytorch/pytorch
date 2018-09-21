#pragma once

#include <torch/nn/cloneable.h>
#include <torch/tensor.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#ifndef WIN32
#include <unistd.h>
#endif

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

#ifdef WIN32
struct TempFile {
  TempFile() : filename_(std::tmpnam(nullptr)) {}
  const std::string& str() const {
    return filename_;
  }
  std::string filename_;
};
#else
struct TempFile {
  TempFile() {
    // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
    char filename[] = "/tmp/fileXXXXXX";
    fd_ = mkstemp(filename);
    AT_CHECK(fd_ != -1, "Error creating tempfile");
    filename_.assign(filename);
  }

  ~TempFile() {
    close(fd_);
  }

  const std::string& str() const {
    return filename_;
  }

  std::string filename_;
  int fd_;
};
#endif
} // namespace test
} // namespace torch
