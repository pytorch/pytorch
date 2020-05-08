#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

namespace torch {
namespace jit {

using graph_rewrite_helper::getFuncName;

struct FuncArg {
  std::string func_name;
  int arg_index;
};

using AtenFuncArgs = std::vector<FuncArg>;
using CallFuncArgs = std::vector<FuncArg>;

// White lists for quantizable operators
std::vector<std::string> _static_quantizable_call_funcs = {
    "conv2d",
    "linear",
    "batch_norm",
    "hardswish",
    "layer_norm",
};

std::vector<std::string> _static_quantizable_aten_funcs = {
    "conv2d",
    "conv3d",
    "linear",
    "addmm",
    "matmul",
    "add_",
    "add",
    "cat",
    "lstm",
    "mul",
    "mul_",
    "hardswish",
    "layer_norm",
};

std::vector<std::string> _dynamic_quantizable_call_funcs = {
    "linear",
};

std::vector<std::string> _dynamic_quantizable_aten_funcs = {
    "linear",
};

// These are the prim::CallFunctions that doesn't require observation and
// have a single input Tensor
// example: `prim::CallFunction(%dropout, %input_tensor, ...)
// so we propagate observed property from %input_tensor to the
// output of the `prim::CallFunction`
// Also these ops doesn't do computation on the value of Tensor, the
// operation only depends on the shape of the Tensor
std::vector<std::string> _single_input_general_shape_call_funcs = {
    "_max_pool1d",
    "_max_pool2d",
    "_max_pool3d",
    "dropout",
    "relu",
    "relu_",
    "sigmoid",
    "tanh",
    "hardsigmoid",
    "hardsigmoid_",
    "leaky_relu",
    "leaky_relu_",
};

// Similar to prim::CallFunctions, there are aten ops that doesn't
// require observation and have a single input Tensor
// Also these ops doesn't do computation on the value of Tensor, the
// operation only depends on the shape of the Tensor
// e.g. `aten::flatten(%input_tensor, ...)`
std::vector<std::string> _single_input_general_shape_aten_funcs = {
    "max_pool1d", "max_pool2d",  "max_pool3d",
    "flatten",    "max",         "min",
    "dropout",    "reshape",
    "resize_", // Non-inplace resize is deprecated
    "chunk",      "view",        "transpose",
    "contiguous", "permute",     "repeat_interleave",
    "relu",       "relu_",       "sigmoid",
    "tanh",       "hardsigmoid", "hardsigmoid_",
    "leaky_relu", "leaky_relu_",
};

// Theses are prim::CallFunctions for ops that doesn't require observation and
// have a single input Tensor
// Also these ops do computation on the value of Tensor
// TODO: [Need verify] looks like we can quantize simple functionals that just
// call into aten functions
std::vector<std::string> _single_input_general_value_call_funcs = {
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "interpolate",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
    "hardtanh",
    "elu",
};

// Theses are aten functions for ops that doesn't require observation and
// have a single input Tensor
// Also these ops do computation on the value of Tensor
// e.g. `aten::avg_pool2d(%input_tensor, ...)`
std::vector<std::string> _single_input_general_value_aten_funcs = {
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "mean",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_linear1d",
    "upsample_bilinear2d",
    "upsample_trilinear3d",
    "upsample_bicubic2d",
    "clamp",
    // "clamp_",  // Enable when quantized `clamp_` is ready
    "hardtanh",
    "hardtanh_",
    "elu",
    "elu_",
};

// Special checks for ops that do not require observers for all input tensors.
// For each operator in this list observers are inserted for the input based
// on the index specified.
AtenFuncArgs _observe_inputs_aten_func = {};
CallFuncArgs _observe_inputs_call_func = {{"batch_norm", 1}};

// Check if `use` is an aten function of name `func_name` and if value
// `v` is the nth argument (if provided) of the function.
bool matchAtenFuncToUse(
    const Use& use,
    const std::string& func_name,
    c10::optional<int> n) {
  Node* node = use.user;
  return node->kind() == Symbol::aten(func_name) &&
      (!n.has_value() || n.value() == use.offset);
}

// Check if `use` is a CallFunction of name `func_name` and if value
// `v` is the nth argument (if provided) of the function
bool matchCallFuncToUse(
    const Use& use,
    const std::string& func_name,
    c10::optional<int> n) {
  Node* node = use.user;
  return node->kind() == prim::CallFunction &&
      getFuncName(node->inputs()[0]) == func_name &&
      (!n.has_value() || n.value() == use.offset);
}

// Check any use of `v` matches the aten function call
// or CallFunction patterns
bool matchArgPattern(
    Value* v,
    const AtenFuncArgs& aten_func_args,
    const CallFuncArgs& call_func_args) {
  for (const Use& u : v->uses()) {
    for (const auto& func_arg : aten_func_args) {
      if (matchAtenFuncToUse(u, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }

    for (const auto& func_arg : call_func_args) {
      if (matchCallFuncToUse(u, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }
  }
  return false;
}

bool isWeight(Value* v) {
  bool result = matchArgPattern(
      v,
      AtenFuncArgs({{"conv2d", 1}, {"conv3d", 1}, {"linear", 1}, {"lstm", 2}}),
      CallFuncArgs({{"linear", 2}}));
  return result;
}

bool isBiasOfConvOrLinear(Value* v) {
  bool result = matchArgPattern(
      v,
      AtenFuncArgs({{"conv2d", 2}, {"conv3d", 2}, {"linear", 2}}),
      CallFuncArgs({{"linear", 3}}));
  return result;
}

bool mayRequireObservation(Value* v) {
  return !hasScalarInput(v->node());
}

std::vector<Value*> getPassThroughInputs(Value* v) {
  Node* n = v->node();
  if (isSingleInputGeneralCallFunction(n)) {
    return {n->input(1)};
  } else if (
      isSingleInputGeneralAtenFunction(n) ||
      (n->kind() == Symbol::aten("sort") && v->offset() == 0)) {
    return {n->input(0)};
  } else if (n->kind() == prim::If && n->outputs().size() == 1) {
    std::vector<Value*> inputs;
    for (Block* subblock : n->blocks()) {
      if (alwaysRaisesException(subblock)) {
        continue;
      }
      auto* output = subblock->outputs()[0];
      inputs.push_back(output);
    }
    return inputs;
  } else if (n->kind() == prim::ListUnpack) {
    return {n->input(0)};
  } else if (n->kind() == prim::ListConstruct) {
    std::vector<Value*> inputs;
    for (auto* v : n->inputs()) {
      inputs.push_back(v);
    }
    return inputs;
  }
  return {};
}

bool isAtenFunc(Node* n, const std::vector<std::string>& aten_funcs) {
  std::vector<Symbol> aten_func_symbols;
  std::transform(
      aten_funcs.begin(),
      aten_funcs.end(),
      std::back_inserter(aten_func_symbols),
      [](const std::string& s) { return Symbol::aten(s); });

  return std::find(
             aten_func_symbols.begin(), aten_func_symbols.end(), n->kind()) !=
      aten_func_symbols.end();
}

// TODO: factor out isCallFunc
bool isFunctionNode(
    Node* n,
    const std::vector<std::string>& call_funcs,
    const std::vector<std::string>& aten_funcs) {
  bool is_func_node = isAtenFunc(n, aten_funcs);
  if (n->kind() == prim::CallFunction) {
    auto func_name = getFuncName(n->inputs()[0]);
    is_func_node |=
        std::find(call_funcs.begin(), call_funcs.end(), func_name) !=
        call_funcs.end();
  }
  return is_func_node;
}

bool isSingleInputGeneralValueAtenFunction(Node* n) {
  return isFunctionNode(
      n,
      /* call_funcs = */ {},
      /* aten_funcs = */ _single_input_general_value_aten_funcs);
}

bool isSingleInputGeneralCallFunction(Node* n) {
  static std::vector<std::string> _single_input_general_call_funcs;
  std::copy(
      _single_input_general_shape_call_funcs.begin(),
      _single_input_general_shape_call_funcs.end(),
      std::back_inserter(_single_input_general_call_funcs));
  std::copy(
      _single_input_general_value_call_funcs.begin(),
      _single_input_general_value_call_funcs.end(),
      std::back_inserter(_single_input_general_call_funcs));
  return isFunctionNode(
      n,
      /* call_funcs = */ _single_input_general_shape_call_funcs,
      /* aten_funcs = */ {});
}

bool isSingleInputGeneralAtenFunction(Node* n) {
  static std::vector<std::string> _single_input_general_aten_funcs;
  std::copy(
      _single_input_general_shape_aten_funcs.begin(),
      _single_input_general_shape_aten_funcs.end(),
      std::back_inserter(_single_input_general_aten_funcs));
  std::copy(
      _single_input_general_value_aten_funcs.begin(),
      _single_input_general_value_aten_funcs.end(),
      std::back_inserter(_single_input_general_aten_funcs));
  return isFunctionNode(
      n,
      /* call_funcs = */ {},
      /* aten_funcs = */ _single_input_general_aten_funcs);
}

bool userDefinedCallFunction(Node* n) {
  return n->kind() == prim::CallFunction &&
      !isSingleInputGeneralCallFunction(n) &&
      !isFunctionNode(n, _static_quantizable_call_funcs, {});
}

bool hasScalarInput(Node* n) {
  std::vector<std::string> scalar_ops = {"add", "add_", "mul", "mul_"};

  return isAtenFunc(n, scalar_ops) &&
      n->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      n->input(1)->type()->isSubtypeOf(NumberType::get());
}

bool nodeQuantizable(Node* n, bool is_dynamic) {
  return isFunctionNode(
      n,
      /* call_funcs = */
      is_dynamic ? _dynamic_quantizable_call_funcs
                 : _static_quantizable_call_funcs,
      /* aten_funcs = */
      is_dynamic ? _dynamic_quantizable_aten_funcs
                 : _static_quantizable_aten_funcs);
}

bool useQuantizable(const Use& use, bool is_dynamic) {
  for (const auto& func_input : _observe_inputs_aten_func) {
    if (matchAtenFuncToUse(use, func_input.func_name, c10::nullopt)) {
      return use.offset == func_input.arg_index;
    }
  }

  for (const auto& func_input : _observe_inputs_call_func) {
    if (matchCallFuncToUse(use, func_input.func_name, c10::nullopt)) {
      return use.offset == func_input.arg_index;
    }
  }
  // Dynamic quantized ops that require special handling for inputs.
  if (is_dynamic && matchAtenFuncToUse(use, "lstm", c10::nullopt)) {
    return use.offset == 2;
  }

  return nodeQuantizable(use.user, is_dynamic);
}

std::shared_ptr<Graph> getCallFunctionGraph(Node* n) {
  auto* func_node = n->input(0)->node();
  auto func = func_node->output()->type()->expect<FunctionType>()->function();
  TORCH_CHECK(
      func->isGraphFunction(), "Quantization only works for graph function");
  return func->graph();
}

// Block helper functions
bool alwaysRaisesException(Block* block) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::RaiseException) {
      return true;
    }
    if (n->kind() == prim::If) {
      bool exception = true;
      for (Block* b : n->blocks()) {
        exception &= alwaysRaisesException(b);
      }
      if (exception) {
        return true;
      }
    }
  }
  return false;
}

// =================== Graph/Module analysis helper functions ============
// Check if value is the input of the graph
bool hitGraphInput(Value* value) {
  Graph* graph = value->owningGraph();
  const auto& inputs = graph->inputs();
  return std::find(inputs.begin(), inputs.end(), value) != inputs.end();
}

// Get the module access path for a Value representing a module instance
// by tracing back the GetAttr nodes and recording all the attribute
// names along the way.
// For example, the module access path will be ['sub', 'basic_block', 'conv1']
// for `self.sub.basic_block.conv1`
std::vector<std::string> getModuleAccessPath(Value* instance, Value* self) {
  std::vector<std::string> path;
  // Iterator to traverse back the GetAttr calls
  Value* iter = instance;
  // trace back the instance to recover the path of the submodule
  while (!hitGraphInput(iter) && iter->node()->kind() == prim::GetAttr) {
    Node* get_attr = iter->node();
    // record the name of GetAttr
    path.push_back(get_attr->s(attr::name));
    // trace back the chain of GetAttr
    iter = get_attr->inputs()[0];
  }
  TORCH_CHECK(
      iter == self,
      "Can't handle the access pattern of GetAttr "
      " in getModuleAccessPath, traced back to:",
      iter->debugName(),
      " which is not self:",
      self->debugName());
  std::reverse(path.begin(), path.end());
  return path;
}

Module findChildModule(
    const Module& module,
    const std::vector<std::string>& path) {
  Module m = module;
  for (const auto& p : path) {
    m = m.attr(p).toModule();
  }
  return m;
}

Module getInvokedModule(Module& module, Node* n, Value* self) {
  auto* instance = n->inputs()[0];
  auto path = getModuleAccessPath(instance, self);
  return findChildModule(module, path);
}

} // namespace jit
} // namespace torch
