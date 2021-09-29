#pragma once

#include <torch/csrc/jit/api/module.h>

#include <ATen/core/jit_type.h>

#include <functional>

namespace torch {
namespace jit {

using DebugHandleType = int64_t;

using NodeToDebugHandle = std::unordered_map<Node*, DebugHandleType>;

using BackendDebugHandleGenerator =
    std::function<NodeToDebugHandle(const std::shared_ptr<Graph>&)>;

namespace detail {

using BackendPreprocessFunction = std::function<c10::IValue(
    const Module&,
    const c10::Dict<IValue, IValue>&,
    const BackendDebugHandleGenerator& generate_debug_handles)>;

TORCH_API void registerBackendPreprocessFunction(
    const std::string& name,
    const BackendPreprocessFunction& preprocess);

bool hasBackendPreprocessFunction(const std::string& name);

BackendPreprocessFunction getBackendPreprocessFunction(const std::string& name);

TORCH_API Module codegen_backend_module(
    const std::string& backend_name,
    const Module& orig_module,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const c10::DictTypePtr& any_dict_ty);
} // namespace detail
} // namespace jit
} // namespace torch
