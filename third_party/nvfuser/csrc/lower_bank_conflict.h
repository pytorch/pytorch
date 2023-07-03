#pragma once

#include <dynamic_type.h>
#include <executor_launch_params.h>
#include <ir_base_nodes.h>
#include <kernel.h>

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
//   1. This utility assumes that the data of the tensor is accessed by
//      `T0[index]`, where `index` is the one stored in the `TensorIndex`
//      object.
//   2. This utility only checks the first iteration. If we have something like
//      `T1_s[tidx, 5]`, then different iterations should have different
//      conflictions, which will not be evaluated for all of them
//   3. This utility assumes that all tensors are independent, which means:
//      3.1 All shared memory tensors are allocated starting from a multiple of
//          4*32 bytes
//      3.2 The only source of bank confliction is from within a tensor.
//          There is no bank conflict between different tensors.
//
// Also note that this utility will not provide accurate estimation if the above
// assumptions are satisfied

std::unordered_map<const Expr*, std::pair<int, int>> getBankConflictInfo(
    kir::Kernel* kernel,
    c10::optional<LaunchParams> launch_params = c10::nullopt,
    const std::unordered_map<std::string, IntOrDouble>& known_values = {});

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
