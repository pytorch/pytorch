#pragma once

#include "c10/util/Optional.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"

#include <cstdint> 

namespace torch { namespace jit { namespace fuser {

int64_t store(std::shared_ptr<Graph> graph);
at::optional<KernelSpec&> retrieve(const int64_t key);

} // namespace fuser
} // namespace jit
} // namespace torch