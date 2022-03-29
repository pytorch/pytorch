#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API c10::optional<std::shared_ptr<Graph>> DecompositionGraphForSchema(
    const FunctionSchema& schema);

TORCH_API void RunDecompositions(std::shared_ptr<Graph> g);

TORCH_API c10::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

} // namespace jit
} // namespace torch
