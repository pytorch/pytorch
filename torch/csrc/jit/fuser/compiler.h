#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/arg_spec.h"
#include "torch/csrc/jit/fuser/common/fused_kernel.h"

#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fuser {

// Performs device-independent compilation of the given fusion_group
// Sets key to a key that can be used to run the fusion later
void registerFusion(int64_t& key, const Node* fusion_group);

std::shared_ptr<FusedKernel> compileKernel(
  const KernelSpec& spec
, const ArgSpec& arg_spec
, const std::vector<int64_t>& map_size
, const int device);

} // namespace fuser
} // namespace jit
} // namespace torch
