#pragma once
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;

// =========== helper functions for Value =========
// Check if a value is weight, since we need to use weight observer
// for weight
TORCH_API bool isWeight(Value* v);

// Check if a value is bias for conv and linear, which we do not
// quantize
TORCH_API bool isBiasOfConvOrLinear(Value* v);

// Check if the value may need observation or not
// one example for values that doesn't need observation is the
// scalar inputs for ops like add/mul
TORCH_API bool mayRequireObservation(Value* v);

// For a given value `v`, get the list of values that we need to check
// if they are observed/quantized or not, if so, we can say the
// `v` is also observed/quantized, since we can derive
// the quantization parameters for `v` given the list of values
TORCH_API std::vector<Value*> getPassThroughInputs(Value* v);

// =========== helper functions for Node =========
TORCH_API bool isSingleInputGeneralValueAtenFunction(Node* n);

TORCH_API bool isSingleInputGeneralCallFunction(Node* n);

TORCH_API bool isSingleInputGeneralAtenFunction(Node* n);

// We don't want to analyze the graph for some `builtin` CallFunctions
// like `linear` because we want to preserve the op boundary
TORCH_API bool userDefinedCallFunction(Node* n);

// Check if the node has scalar input
TORCH_API bool hasScalarInput(Node* n);

// Check if a node is quantizable
TORCH_API bool nodeQuantizable(Node* n, bool is_dynamic = false);

// Check if a use of the value is quantizable, this depends on
// both the use node and the offset
TORCH_API bool useQuantizable(const Use& use, bool is_dynamic);

// Given a CallFunction node, extract the graph of the called function
TORCH_API std::shared_ptr<Graph> getCallFunctionGraph(Node* n);

// =========== helper functions for Block =========
// checks if a block will always raise an Exception
TORCH_API bool alwaysRaisesException(Block* block);

// =========== helper functions for Graph ==========
// TODO: remove
TORCH_API std::vector<std::string> getModuleAccessPath(
    Value* instance,
    Value* self);
// TODO: remove
TORCH_API Module
findChildModule(const Module& module, const std::vector<std::string>& path);

// Given an CallMethod node, get the module instance corresponding
// to the instance Value
TORCH_API Module getInvokedModule(Module& module, Node* n, Value* self);
// =========== helper functions for Module  ==========

} // namespace jit
} // namespace torch
