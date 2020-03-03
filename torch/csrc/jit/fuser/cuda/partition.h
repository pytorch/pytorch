#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_API bool isFusibleCudaFusionGroup(const Node* const node);

TORCH_API bool isFusibleCudaFusionGroup(
    const Node* const fusion,
    const Node* const node);

}}}}
