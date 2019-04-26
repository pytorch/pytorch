#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/utils/disallow_copy.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

struct DynamicLibrary {
  TH_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  TORCH_API DynamicLibrary(const char* name);

  TORCH_API void* sym(const char* name);

  TORCH_API ~DynamicLibrary();

 private:
  void* handle = nullptr;
};

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
