#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API Value* tryMatchInterfaceOps(
    Graph& graph,
    const SourceRange& loc,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    Symbol name,
    std::stringstream& failure_messages,
    bool allow_conversions);
}
} // namespace jit
} // namespace torch
