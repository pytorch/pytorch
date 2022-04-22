#pragma once
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

#include <functional>
#include <regex>

namespace torch {
namespace jit {

using graph_rewrite_helper::getFuncName;

// Vector of a module and the name of its method
using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;
// Map of quantization parameter name and value
// for example _scale, _zero_point,
// _scalar_type and _axis(for per channel quantization)
using QParamVector = std::vector<std::pair<std::string, IValue>>;

// =========== helper functions for Value =========
// Check if a value is weight, since we need to use weight observer
// for weight
TORCH_API bool isWeight(Value* v);

// Check if a value is bias for conv and linear, which we do not
// quantize
TORCH_API bool isBiasOfConvOrLinear(Value* v);

TORCH_API bool isEmbeddingBagNonInput(Value* v);

// Get the use as scalar input of clamp ops for the input value
c10::optional<Use> getClampScalarInputUse(Value* v);

// For a given value `v`, get the list of values that we need to check
// if they are observed/quantized or not, if so, we can say the
// `v` is also observed/quantized, since we can derive
// the quantization parameters for `v` given the list of values
TORCH_API std::vector<Value*> getPassThroughInputs(Value* v);

// Check if a value in the graph is a Scalar value
TORCH_API bool isScalar(Value* v);

// Check if value is the input of the graph
TORCH_API bool hitGraphInput(Value* value);

// Converts a mangled name, such as
//   __torch__.torch.nn.quantized.modules.conv.___torch_mangle_7.Conv2d
// into an unmangled name, such as
//   __torch__.torch.nn.quantized.modules.conv.Conv2d
TORCH_API std::string removeTorchMangle(const std::string& orig_name);

// Return the module name that corresponds to the value.
TORCH_API c10::optional<std::string> getModuleName(Value* value);

// =========== helper functions for Node =========
TORCH_API bool isSingleInputGeneralShapeAtenFunction(Node* n);

TORCH_API bool isSingleInputGeneralValueAtenFunction(Node* n);

TORCH_API bool isSingleInputGeneralCallFunction(Node* n);

TORCH_API bool isSingleInputGeneralAtenFunction(Node* n);

TORCH_API bool isClamp(Node* n);

// Check if the node will produce the same result regardless of whether
// the input tensor is quantized or not, example: aten::size
TORCH_API bool isTensorInfoNode(Node* n);

// Check if this the the propaagate op that has single input, e.g. aten::cat
TORCH_API bool isPropagateQuantSingleInputOp(Node* n);

// Check if this is the propagate op that has two inputs, e.g. aten::add
TORCH_API bool isPropagateQuantBinaryOp(Node* n);

// Check if this is the node that we'll quantize or not quantize depending on
// whether the input of the node is quantized, example: aten::cat
TORCH_API bool isPropagateQuantOp(Node* n);

// Check if the node is a binary op like aten::add and aten::mul and
// if the input 1 is a scalar, these ops will be quantized to
// quantized::{op}_scalar
TORCH_API bool isBinaryOpWithScalarInput(Node* n);

TORCH_API c10::optional<std::tuple<c10::QScheme, QParamVector>> getFixedQParams(
    Node* n);

// We don't want to analyze the graph for some `builtin` CallFunctions
// like `linear` because we want to preserve the op boundary
TORCH_API bool userDefinedCallFunction(Node* n);

// Check if the node has scalar input
TORCH_API bool hasScalarInput(Node* n);

// Check if a node is quantizable
TORCH_API bool nodeQuantizable(
    Node* n,
    QuantType quant_type = QuantType::STATIC);

// Nodes which only require quantization of weight value, eg. embedding_bag
bool isWeightOnlyStaticQuantOp(Node* n);

// Check if a use of the value is quantizable, this depends on
// both the use node and the offset
TORCH_API bool useQuantizable(const Use& use, QuantType quant_type);

// Given a CallFunction node, extract the graph of the called function
TORCH_API std::shared_ptr<Graph> getCallFunctionGraph(Node* n);

// Check if `use` is a CallFunction of name `func_name` and if value
// `v` is the nth argument (if provided) of the function
bool matchCallFuncToUse(
    const Use& use,
    const std::string& func_name,
    c10::optional<int> nth_arg);

// Check if `use` is a AtenFunction of name `func_name` and if value
// `v` is the nth argument (if provided) of the function
bool matchAtenFuncToUse(
    const Use& use,
    const std::string& func_name,
    c10::optional<int> nth_arg);

// =========== helper functions for Block =========
// checks if a block will always raise an Exception
TORCH_API bool alwaysRaisesException(Block* block);

// =========== helper functions for Module  ==========
// TODO: remove
TORCH_API std::vector<std::string> getModuleAccessPath(
    Value* instance,
    Value* self);
// TODO: remove
TORCH_API Module
findChildModule(const Module& module, const std::vector<std::string>& path);

// Given an CallMethod node, get the module instance corresponding
// to the instance Value
// TODO: refactor all current uses of this function to the Opt one
TORCH_API Module getInvokedModule(Module& module, Node* n, Value* self);

// Given an CallMethod node, get the module instance corresponding
// to the instance Value if the instance is a module, otherwise return
// c10::nullopt
c10::optional<Module> getInvokedModuleOpt(
    const Module& module,
    Node* n,
    Value* self);

// ==================== filter functions for matches ==============
// filter to check Value `vname` is a constant of int value `value`
bool is_int_constant(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap,
    const std::string& vname,
    int value);

// filter to check if the %alpha argument of aten::add is constant 1
bool aten_add_alpha_is_one(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// filter to check if the functional in CallFunction is relu
bool is_functional_relu(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// filter to check if the module is torch.nn.ReLU
bool is_relu_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_linear_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// TODO: add a macro to declare the filters
bool is_conv1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_conv2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_conv3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_conv_transpose1d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_conv_transpose2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_batchnorm2d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

bool is_batchnorm3d_module(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

} // namespace jit
} // namespace torch
