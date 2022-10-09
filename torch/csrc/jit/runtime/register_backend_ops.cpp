// This file registers special JIT operators used to implement the PyTorch XPU
// API in TorchScript.
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators const reg({
#ifndef USE_CUDA
    Operator(
        "cuda::device_count() -> int",
        [](Stack& stack) { push(stack, 0); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_current_device() -> int",
        [](Stack& stack) {
          push(stack, -1);
        },
        aliasAnalysisFromSchema()),
#endif
    Operator(
        "xpu::device_count() -> int",
        [](Stack& stack) { push(stack, at::detail::getXPUHooks().device_count()); },
        aliasAnalysisFromSchema()),
    Operator(
        "xpu::current_device() -> int",
        [](Stack& stack) { push(stack, at::detail::getXPUHooks().current_device()); },
        aliasAnalysisFromSchema()),
});
} // namespace
} // namespace jit
} // namespace torch
