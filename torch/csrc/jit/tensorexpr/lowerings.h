// This file defines classes for registering standard lowerings from JIT to TE IR.
#pragma once

#include <c10/util/variant.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using ArgNone = c10::monostate;
using BufList = std::vector<tensorexpr::BufHandle>;
using IntList = std::vector<int64_t>;
using ArgValue = c10::variant<
    tensorexpr::BufHandle,
    tensorexpr::VarHandle,
    double,
    int64_t,
    bool,
    BufList,
    IntList,
    ArgNone>;

using NNCLoweringFunction = std::function<Tensor(
    const std::vector<ArgValue>&,
    const std::vector<ExprHandle>&,
    const c10::optional<ScalarType>&,
    at::Device)>;

std::unordered_map<std::string, NNCLoweringFunction>& getNNCLoweringRegistry();

struct RegisterNNCLoweringFunction {
  RegisterNNCLoweringFunction(const std::string& name, NNCLoweringFunction fn) {
    getNNCLoweringRegistry()[name] = fn;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
