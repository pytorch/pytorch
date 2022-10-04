#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// for more info on shared memory access see page 54-72 of:
// https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf

// Warning: The bank confliction checking utility here is not a replacement of
// nsight compute. This utility currently has the following assumptions and
// limitations:
//
//   1. This utility assumes that `blockDim.x` is large enough to hold one phase
//   2. This utility assumes that the address only depends on loop variables
//      (there can not be a thing like `T0.stride[0]`, `blockDim.x`)
//   3. This utility assumes that the data of the tensor is accessed by
//      `T0[index]`, where `index` is the one stored in the `TensorIndex`
//      object.
//   4. This utility only checks the first iteration, and the start of all
//      loop variables are assumed to be `0` (if we have something like
//      `T1_s[tidx, 5]`, then different iterations should have different
//      results, which this utility will not be able to handle all of them now)
//   5. This utility assumes that all tensors are independent, which means:
//      5.1 All shared memory tensors are allocated starting from a multiple of
//          4*32 bytes
//      5.2 The only source of bank confliction is from within a tensor.
//          There is no bank conflict between different tensors.
//
// Also note that this utility will not provide accurate estimation if the above
// assumptions are satisfied

std::unordered_map<const Expr*, std::pair<int, int>> getBankConflictInfo(
    kir::Kernel* kernel);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
