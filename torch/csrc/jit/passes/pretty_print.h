#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API std::ostream& PrettyPrint(std::ostream& out, const Graph& graph);

}}
