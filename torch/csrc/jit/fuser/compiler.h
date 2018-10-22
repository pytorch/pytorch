#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"

#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fuser {

// Performs device-independent compilation of the given fusion_group
// Sets key to a key that can be used to run the fusion later
void registerFusion(int64_t& key, const Node* fusion_group);

} // namespace fuser
} // namespace jit
} // namespace torch