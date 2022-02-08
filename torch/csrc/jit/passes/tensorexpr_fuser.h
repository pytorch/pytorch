#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

// Run TensorExpressions-based fuser.
// If add_composed_op is true, creates a single operation that
// performs both the runtime check that types align
// and then the dispatch to the kernel/unoptimized graph
TORCH_API void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size = 2,
    bool add_composed_op = false,
    bool fuse_to_dynamic_shapes = false);

TORCH_API void setTensorExprFuserEnabled(bool val);
TORCH_API bool tensorExprFuserEnabled();
TORCH_API void setTensorExprDynamicShapeFusionEnabled(bool val);
TORCH_API bool tensorExprDynamicShapeFusionEnabled();
TORCH_API bool setTexprReductionsEnabled(bool value);
TORCH_API bool texprReductionsEnabled();

TORCH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);
TORCH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);
TORCH_API void removeTensorTypeSpecializations(Block* block);

using tensor_type_converter_t =
    c10::function_ref<TensorTypePtr(const TensorTypePtr& t)>;

// inserts a TypeCheck pattern
//
// around the guarded node that has a Subgraph attribute, this inserts a pattern
//
//   if TypeCheck(...):
//     guarded_node
//   else:
//     FallbackGraph(...)
//
// The TypeCheck includes the types of all Tensor inputs to the guarded_node,
// as processed by the type_converter, a lambda
// TensorTypePtr(const TensorTypePtr& t). This allows to erase irrelevant
// aspects of the type.
//
// The Fallback graph will have the same subgraph as the guarded node (with the
// expectation that the guarded_node's subgraph will then be optimized.
TORCH_API void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    c10::Symbol kind);

TORCH_API bool usedOnlyInSize(Value* v);
TORCH_API Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db);

namespace tensorexpr {
TORCH_API bool isSupported(Node* node);
} // namespace tensorexpr
} // namespace jit
} // namespace torch
