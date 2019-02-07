#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
namespace testing {

struct FileCheck;

struct TORCH_API FileCheck {
  static void checkFile(
      const std::string& check_file,
      const std::string& test_file);
};

} // namespace testing
} // namespace jit
} // namespace torch
