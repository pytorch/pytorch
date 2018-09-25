#pragma once

#include <gtest/gtest.h>

#include <torch/nn/cloneable.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
  }
};

#define ASSERT_THROWS_WITH(statement, prefix)                            \
  try {                                                                  \
    (void)statement;                                                     \
    FAIL() << "Expected statement `" #statement                          \
              "` to throw an exception, but it did not";                 \
  } catch (const std::exception& e) {                                    \
    std::string message = e.what();                                      \
    if (message.find(prefix) == std::string::npos) {                     \
      FAIL() << "Error message \"" << message                            \
             << "\" did not match expected prefix \"" << prefix << "\""; \
    }                                                                    \
  }

} // namespace test
} // namespace torch
