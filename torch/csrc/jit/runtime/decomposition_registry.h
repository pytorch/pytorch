#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API c10::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema);

TORCH_API void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g);

TORCH_API void RunDecompositions(std::shared_ptr<Graph> g);

TORCH_API c10::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

// To Embed in C++ Code, invoke as :
// GraphFunction * func
// std::once_flag get_func;
// std::call_once(get_func, [&]()) {
//    func = GetDecompositionExecutor("aten::var(Tensor self, bool unbiased=True) -> Tensor")
// });
TORCH_API GraphFunction* GetDecompositionExecutor(const char * schema_literal);

} // namespace jit
} // namespace torch
