#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void StopInliningForONNX(Graph& graph);

}
} // namespace torch
