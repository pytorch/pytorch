#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool isFusibleCudaFusionGroup(const Node* const node);

bool isFusibleCudaFusionGroup(
    const Node* const fusion,
    const Node* const node);

}}}}
