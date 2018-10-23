#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cpu {

struct CompilerConfig {
  CompilerConfig();
  ~CompilerConfig() = default;

  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

CompilerConfig& getConfig();

} // namespace cpu
} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
