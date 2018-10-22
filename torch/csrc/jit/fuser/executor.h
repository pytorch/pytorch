#pragma once

#include "torch/csrc/jit/stack.h"

#include <cstdint>

namespace torch { namespace jit { namespace fuser {

// Runs the fusion associated with the key (from registerFusion above)
// on the inputs taken from the given Stack.
void runFusion(
  const int64_t key
, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch