#if !(defined _WIN32)
#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

#include <string>

namespace torch { namespace jit { namespace cpufuser {

TORCH_API struct CPUFusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)