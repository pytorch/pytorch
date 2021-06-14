#pragma once

#include <ATen/core/function_schema.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/named_value.h>

namespace torch {
namespace jit {

// try to match a list if inputs and keyword 'attributes' to this schema,
// if it works return the flat list of positional inputs to the call
// if it returns nullopt, then failure_messages contains a good error report

TORCH_API MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

TORCH_API std::pair<size_t, MatchedSchema> matchSchemas(
    const std::vector<const ::c10::FunctionSchema*>& schemas,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt,
    bool render_errors = false);

// Creates a list with the provided values if each value's type can be matched
// to an argument with type `elem_type`. If a type in `varargs` does not match
// `elem_type`, nullptr is returned. This is used for creating lists from
// varargs so that calls like torch.zeros(1, 2, 3) will be matched to
// aten::zeros(int[]).
TORCH_API bool convertibleToList(
    const TypePtr& type,
    const TypePtr& list_type_);

TORCH_API c10::optional<size_t> findInputWithName(
    const std::string& name,
    at::ArrayRef<NamedValue> kwargs);

// Applies implicit conversion from value trying to turn it into type
// concrete_type. It succeeds if `return_value->isSubtypeOf(concrete_type)`
TORCH_API Value* tryConvertToType(
    const SourceRange& loc,
    Graph& graph,
    std::shared_ptr<Graph> additions,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions);

} // namespace jit
} // namespace torch
