#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/ir.h>

// TODO: Ideally we would get rid of `schema_emitter` entirely and put
// the write logic in `ir_emitter_utils`

namespace torch {
namespace jit {

MatchedSchema matchSchemaAndPrepareGraph(
    const FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

std::pair<size_t, MatchedSchema> matchSchemasAndPrepareGraph(
    std::vector<const FunctionSchema*> schemas,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

Value* emitBuiltinNode(
    const MatchedSchema& matched_schema,
    const SourceRange& loc,
    Graph& graph,
    Symbol name);

TORCH_API Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

} // namespace jit
} // namespace torch
