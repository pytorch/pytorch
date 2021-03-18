#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/api/module.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace mobile {
using OperatorFunctor = std::function<void(Stack&)>;

/**
 * The function to check compatibility of an operator with name. It either
 * throws error with reason why the operator is not compatible, or returns an
 * operator functor that is compatible to this runtime.
 *
 * @param opname The operator name
 * @param op_version The operator version in the model file
 * @param model_version The bytecode version in the model file
 * @return The operator function pointer that is compatible to this runtime
 */
TORCH_API OperatorFunctor operator_resolver(
    const c10::OperatorName& opname,
    int64_t op_version,
    int64_t model_version);

/**
 * The function to query the supported operators and the supported versions in
 * current runtime.
 * @return a map with key: operator name, val: a pair of supported version range
 * [min, max]
 */
TORCH_API std::unordered_map<std::string, std::unordered_set<int64_t>>
get_op_version_table();
} // namespace mobile
} // namespace jit
} // namespace torch
