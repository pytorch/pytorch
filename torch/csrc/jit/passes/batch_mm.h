#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void BatchMM(std::shared_ptr<Graph>& graph);

}
} // namespace torch
