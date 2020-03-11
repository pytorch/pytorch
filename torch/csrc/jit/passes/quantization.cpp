#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/quantization_patterns.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

#include <c10/core/QScheme.h>

#include <algorithm>
#include <stack>

namespace torch {
namespace jit {
namespace {

using OptionalModuleVector = std::vector<c10::optional<script::Module>>;
using ModuleMethodVector = std::vector<std::pair<script::Module, std::string>>;
using NameModuleVector = std::vector<std::pair<std::string, script::Module>>;
// Map of quantization parameter name and value
// for example _scale, _zero_point,
// _scalar_type and _axis(for per channel quantization)
using QParamVector = std::vector<std::pair<std::string, IValue>>;

// This struct contains a compiled IR pattens slated for use in the
// findPatternMatches function. The struct encapsulates the common
// information from parseIR that is used in conjunction with the
// pattern matching facility. A const instance of this struct can
// also be stored away to cache the compiled IR pattern and reduce
// runtime cost
struct PatternInfo {
  std::string pattern_string;
  std::unique_ptr<Graph> pattern_graph;
  std::unordered_map<std::string, Value*> vmap;

  static PatternInfo parse_from_str(std::string pattern_string) {
    PatternInfo rv{
        std::move(pattern_string), std::make_unique<Graph>(), decltype(vmap){}};
    script::parseIR(rv.pattern_string, rv.pattern_graph.get(), rv.vmap);
    return rv;
  }
};

struct PatternsAndModules {
  bool is_conv;
  bool is_per_channel;
  const PatternInfo& pattern;
  script::Module packed_params_module;
};

static Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return match_vmap.at(vmap.at(name));
}

static c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return toIValue(getValue(name, match_vmap, vmap));
}

void fillQConfigMap(
    const script::Module& module,
    const QConfigDict& qconfig_dict,
    ModuleQConfigMap& map,
    const std::string& key = "",
    const c10::optional<QConfig>& parent_qconfig = c10::nullopt) {
  c10::optional<QConfig> qconfig;
  if (qconfig_dict.find(key) != qconfig_dict.end()) {
    qconfig = qconfig_dict.at(key);
  } else {
    qconfig = parent_qconfig;
  }
  map[module._ivalue()] = qconfig;

  for (const script::NameModule& s : module.named_children()) {
    std::string child_key;
    if (key == "") {
      child_key = s.name;
    } else {
      child_key = key + "." + s.name;
    }
    fillQConfigMap(s.value._ivalue(), qconfig_dict, map, child_key, qconfig);
  }
}

std::string getFuncName(Value* func_value) {
  auto func_node = func_value->node();
  auto func = func_node->output()->type()->expect<FunctionType>()->function();
  const auto& qname = func->qualname();
  const auto& name = qname.qualifiedName();
  auto rdot_idx = name.rfind('.');
  if (rdot_idx != std::string::npos) {
    return name.substr(rdot_idx + 1, name.length());
  } else {
    return name;
  }
}

bool isFunctionNode(Node* n,
                    const std::vector<std::string>& call_funcs,
                    const std::vector<std::string>& aten_funcs) {
  std::vector<Symbol> aten_func_symbols;
  std::transform(
      aten_funcs.begin(),
      aten_funcs.end(),
      std::back_inserter(aten_func_symbols),
      [](const std::string& s) { return Symbol::aten(s); });

  bool is_quantizable =
      std::find(aten_func_symbols.begin(), aten_func_symbols.end(), n->kind()) !=
      aten_func_symbols.end();
  if (n->kind() == prim::CallFunction) {
    auto func_name = getFuncName(n->inputs()[0]);
    is_quantizable |=
        std::find(call_funcs.begin(), call_funcs.end(), func_name) !=
        call_funcs.end();
  }
  return is_quantizable;
}

// If the op doesn't require observation, return
// the the list of input indexes that we should check to see
// if they are observed/quantized, if so, we can say the output
// of this op is observed/quantized as well, since for these ops we can derive
// the quantization parameters for output given inputs
std::vector<size_t> getGeneralOpTensorInputIndexes(Node* n) {
  std::vector<std::string> single_input_aten_funcs = {
    "adaptive_avg_pool2d",
    "max_pool2d",
    "avg_pool2d",
    "flatten",
    "max",
    "min",
    "mean",
    // TODO: sort returns a tuple of Tensors, we have
    // to extend the API to support that
    // "sort",
    "__interpolate",
    "__upsample",
    "__upsample_bilinear",
    "__upsample_nearest",
    "dropout",
  };
  std::vector<std::string> single_input_call_funcs = {
    "adaptive_avg_pool2d",
    "_max_pool2d",
  };
  if (isFunctionNode(
          n,
      // We don't have call functions
      // after inline
      /* call_funcs = */ single_input_call_funcs,
      /* aten_funcs = */ {})) {
    return {1};
  } else if (isFunctionNode(
                 n,
                 // We don't have call functions
                 // after inline
                 /* call_funcs = */ {},
                 /* aten_funcs = */ single_input_aten_funcs)) {
    return {0};
  } else if (n->kind() == prim::ListConstruct) {
    std::vector<size_t> indexes;
    for (auto i = 0; i < n->inputs().size(); ++i) {
      indexes.push_back(i);
    }
    return indexes;
  }
  return {};
}

bool nodeQuantizable(Node* n) {
  return isFunctionNode(
      n,
      /* call_funcs = */ {
      "conv2d",
      "linear",
      "relu",
    }, /* aten_funcs = */ {
      "conv2d",
      "linear",
      "relu",
      "addmm",
      "matmul",
      "add_",
      "cat",
    });
}

script::Module findChildModule(
    const script::Module& module,
    const std::vector<std::string>& path) {
  script::Module m = module;
  for (const auto& p : path) {
    m = m.attr(p).toModule();
  }
  return m;
}

// Check if value is the input of the graph
bool hitGraphInput(Value* value) {
  Graph* graph = value->owningGraph();
  const auto& inputs = graph->inputs();
  return std::find(inputs.begin(), inputs.end(), value) != inputs.end();
}

// Get the module access path for a Value representing a module instance
// by tracing back the GetAttr nodes and recording all the attribute
// names along the way.
// For example, the module access path will be ['conv1', 'basic_block', 'sub']
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
  TORCH_CHECK(iter == self,
              "Can't handle the access pattern of GetAttr "
              " in getModuleAccessPath, traced back to:",
              iter->debugName(),
              " which is not self:",
              self->debugName());
  return path;
}

script::Module getInvokedModule(
    script::Module& module, Node* n, Value* self) {
  auto* instance = n->inputs()[0];
  auto path = getModuleAccessPath(instance, self);
  return findChildModule(module, path);
}

bool isPerChannel(at::QScheme qscheme) {
  return qscheme == c10::kPerChannelAffine ||
    qscheme == c10::kPerChannelSymmetric;
}

class ModuleCloneHelper {
 public:
  /** Clone according to module qconfig map, this is for handling the case
   *  where we have two module instances sharing the same ClassType
   *  but configured with different QConfig
   *  code is copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp
   */
  script::Module clone(
      const script::Module& module,
      const ModuleQConfigMap& module_qconfig_map) {
    std::unordered_map<TypePtr, QConfigTypePtrMap> type_remap;
    return clone_impl(module, module_qconfig_map, type_remap);
  }

 private:
  script::Module clone_impl(
      const script::Module& module,
      const ModuleQConfigMap& module_qconfig_map,
      std::unordered_map<TypePtr, QConfigTypePtrMap>& type_remap) {
    auto qconfig = module_qconfig_map.at(module._ivalue());
    auto type = module.type();
    // Create a new _ivalue in the same compilation unit.
    // Since now we have shared ClassType, we need to preserve the shared
    // ClassType during cloning, so we first use type and qconfig to check if
    // the type is already cloned, if so, we'll create a new module with the
    // cloned ClassType, if not, we'll create a new module and a new ClassType.
    bool type_already_cloned = type_remap.find(type) != type_remap.end() &&
        type_remap.at(type).find(qconfig) != type_remap.at(type).end();
    script::Module r;
    if (type_already_cloned) {
      // if we cloned the class type before, we'll reuse it
      script::Module new_module(
          module._ivalue()->compilation_unit(),
          type_remap.at(type).at(qconfig)->cast<ClassType>());
      r = new_module;
    } else {
      script::Module new_module(
          *type->name(), module._ivalue()->compilation_unit(), true);
      r = new_module;
      type_remap[type][module_qconfig_map.at(module._ivalue())] = r.type();
    }
    // Copy slots. If a slot is a module - recursively clone it.
    size_t N = type->numAttributes();
    for (size_t i = 0; i < N; ++i) {
      IValue s = module._ivalue()->getSlot(i);
      if (type->getAttribute(i)->is_module()) {
        const script::Module& orig = script::Module(s.toObject());
        script::Module cloned =
            clone_impl(orig, module_qconfig_map, type_remap);
        r.register_module(type->getAttributeName(i), cloned);
      } else {
        r.register_attribute(
            type->getAttributeName(i),
            type->getAttribute(i),
            s,
            type->is_parameter(i));
      }
    }

    // only clone the methods and constants if the ClassType is not cloned
    // before
    if (!type_already_cloned) {
      for (size_t i = 0; i < type->numConstants(); ++i) {
        r.type()->addConstant(type->getConstantName(i), type->getConstant(i));
      }
      // Clone methods remapping the types to the cloned ones.
      for (auto& fn : type->methods()) {
        clone_method(module, r, *fn, module_qconfig_map, type_remap);
      }
    }
    return r;
  }

  void remapTypes(
      Block* block,
      Value* self,
      const script::Module& source,
      script::Module& target,
      const ModuleQConfigMap& module_qconfig_map,
      const std::function<TypePtr(TypePtr, c10::optional<QConfig>)>&
          type_remap_fn) {
    // remap of %self will be done outside of the function
    // and we don't support the case when people pass in
    // module as argument of the method because in that case
    // we need to do more comprehensive analysis to decide the
    // QConfig for the module
    for (size_t i = 1; i < block->inputs().size(); ++i) {
      TORCH_CHECK(
          !block->inputs()[i]->type()->cast<ClassType>(),
          "We don't support quantizing methods that has Object as arguments");
    }
    for (Node* node : block->nodes()) {
      // remapping type for module instance
      if (node->kind() == prim::CallMethod) {
        Value* instance = node->inputs()[0];
        auto path = getModuleAccessPath(instance, self);
        auto child = findChildModule(source, path);
        auto qconfig = module_qconfig_map.at(child._ivalue());
        instance->setType(type_remap_fn(instance->type(), qconfig));
      }
      // We don't remap output and the remapping of module type
      // will be done in CallMethod, we don't support type remapping
      // for modules returned from methods or functions
      for (Block* sub_block : node->blocks()) {
        remapTypes(
            sub_block, self, source, target, module_qconfig_map, type_remap_fn);
      }
      for (Symbol name : node->attributeNames()) {
        if (node->kindOf(name) == AttributeKind::g) {
          remapTypes(
              node->g(name).get(),
              source,
              target,
              module_qconfig_map,
              type_remap_fn);
        } else if (node->kindOf(name) == AttributeKind::gs) {
          for (const auto& g : node->gs(name)) {
            remapTypes(
                g.get(), source, target, module_qconfig_map, type_remap_fn);
          }
        }
      }
    }
  }

  void remapTypes(
      Graph* graph,
      const script::Module& source,
      script::Module& target,
      const ModuleQConfigMap& module_qconfig_map,
      const std::function<TypePtr(TypePtr, c10::optional<QConfig>)>&
          type_remap_fn) {
    remapTypes(
        graph->block(),
        graph->inputs()[0],
        source,
        target,
        module_qconfig_map,
        type_remap_fn);
  }

  void clone_method(
      const script::Module& source,
      script::Module& target,
      const Function& method,
      const ModuleQConfigMap& module_qconfig_map,
      const std::unordered_map<TypePtr, QConfigTypePtrMap>& type_remap) {
    auto type_remap_fn = [&](TypePtr type_ptr,
                             const c10::optional<QConfig>& qconfig) {
      if (type_remap.find(type_ptr) != type_remap.end()) {
        const auto& qconfig_map = type_remap.at(type_ptr);
        if (qconfig_map.find(qconfig) != qconfig_map.end()) {
          return qconfig_map.at(qconfig);
        }
      }
      return type_ptr;
    };
    auto graph = method.graph()->copy();
    remapTypes(graph.get(), source, target, module_qconfig_map, type_remap_fn);
    // remap self
    graph->inputs()[0]->setType(target.type());
    const auto this_method_name =
        c10::QualifiedName(*target.type()->name(), method.name());
    auto copied = target._ivalue()->compilation_unit()->create_function(
        this_method_name, graph);
    target.type()->addMethod(copied);
    // we'll use default schema for cloned method
  }
};

class InsertObserversHelper {
 public:
  explicit InsertObserversHelper(const ModuleQConfigMap& map)
      : module_qconfig_map_(map) {}

  void preprocess(
    script::Module& module,
    const std::string& method_name);

  /**
   * Recursively insert observers for the method, also we'll process
   * the nodes in the graph in the order of execution of these nodes
   * since we need the context information to decide whether we want to
   * observe/quantize a value a not, we don't want to observe a value multiple
   * times.
   *
   * arguemnt: is_entry_point means whether the current method is the forward
   * method of the top level module.
   *
   *Since we want to insert observers in the call site instead of in the called
   * graph, we'll postpone inserting observer to caller as much as possible, if
   * we know the current method is the outer most method, then
   * we will insert all observers in the graph instead of postpone this to the
   * parent, note that this assumes we don't have recurisve method
   * calls
   *
   * returns a tuple of vectors of observer modules for input and output, these
   * are used for inserting observers for the input/output values
   * since we need to insert these values at call site.
   * And a vector of indexes of outputs that indicates whether the output value
   * is already observed or not, this is used for propagating the observed
   * property of a value through CallMethods, because we should skip inserting
   * observers for ops that don't require observation
   */
  std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
  insertObservers(script::Module& module,
                  const std::string& method_name,
                  bool is_entry_point = false,
                  std::unordered_set<Value*> graph_observed_values = std::unordered_set<Value*>());

  std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>> insertObserversFor(
      Block* block,
      script::Module& module,
      // this is a reference because when we insert observer for a value
      // in one block it is also observed in another block, we don't want to
      // insert multiple observers for the same value
      std::unordered_set<Value*>& block_observed_values,
      bool is_entry_point = false);

 private:
  ModuleMethodVector getInvokedMethods(
      script::Module& module,
      const std::string& method_name);

  bool valueNeedsToBeQuantized(Value* v);

  // Fill the map between the caller input/output to input/output
  // of called graph, this is used to navigate through the graph
  // to find the observer for a given value
  void fillBoundaryValueMap(
      script::Module& module, const std::string& method_name);

  // Fill the map from value to the corresponding observer module
  // this map is used in insertObservers to actually insert
  // observers to the module
  void fillValueObserverMap(script::Module& module, const std::string& method_name);


  // Clone observer module and add it to the original module,
  // and insert a call to observer forward function
  void insertObserverFor(
      Value* v,
      script::Module& module,
      const script::Module& observer_module,
      NameModuleVector& observer_name_and_modules);

  c10::optional<script::Module> getObserverFor(Value* v);

  void propagateObservedProperty(Value* output, std::unordered_set<Value*>& block_observed_values);

  void skipValuesInPattern(
      Graph& graph,
      const PatternInfo& pattern);

  void addIntermediateValuesToSkipObserver(
      const script::Module& module,
      const std::string& method_name);

  // Fill the map from values to the list of values that can pass the observed
  // property to it
  void fillPassThroughValueMap(const std::shared_ptr<Graph>& graph);

  const ModuleQConfigMap& module_qconfig_map_;
  // Values we want to skip observing, used to skip values in
  // the middle of the ops that are supposed to be fused, e.g.
  // the output value of conv in the conv - relu pattern
  std::unordered_set<Value*> values_to_skip_;
  std::unordered_set<Graph*> visited_graph_of_observer_map_;
  std::unordered_map<Value*, script::Module> observer_for_value_;
  std::unordered_map<Value*, Value*> caller_to_callee_;
  // Map from values from callsite into the values in the CallMethod graph
  std::unordered_map<Value*, std::unordered_set<Value*>> boundary_value_map_;
  std::unordered_set<Value*> observed_values_;
  // This is used for the observed values to pass through the ops like flatten,
  // so that output value of platten do not need to be observed
  // key of the map is the value from caller graph, and the value of the map
  // is the list of values in the callee graph (the graph
  // corresponding to the called method),
  // the reason it is a vector is that a value in the caller graph
  // can both correspond to the output of one callee graph and input of another
  // callee graph.
  std::unordered_map<Value*, std::vector<Value*>> pass_through_value_map_;
  // Unique id generator for observer module, used for generating
  // unique observer names when we insert observer module, we
  // record the current unique id used to avoid incrementing from 0
  // every time to find a unique id.
  int uid_ = 0;
  // Set of observer forward call nodes
  std::unordered_set<Node*> observer_nodes_;
  // Map from block to a vector of observer name and observer modules we
  // want to add to the module instance that has the block
  std::unordered_map<Block*, NameModuleVector> block_observer_map_;

  // These are the IR patterns we match to skip inserting observers.
  // They are compiled once on construction and used repeatedly within
  // the pass.
  const PatternInfo conv_functional_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %first_module = match::module[name="Conv2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )");
  const PatternInfo conv_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %first_module = match::module[name="Conv2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_module = match::module[name="ReLU"](%self)
    %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
    return (%second_output) )");
  const PatternInfo matmul_add = PatternInfo::parse_from_str(R"(
graph(%input, %weight, %bias, %4):
     %weight_t = aten::t(%weight)
     %first_output = aten::matmul(%input, %weight_t)
     %second_output = aten::add_(%first_output, %bias, %4)
     return (%second_output) )");
  const PatternInfo add_module_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %one = prim::Constant[value=1]()
     %first_output = aten::add_(%a, %b, %one)
     %second_module = match::module[name="ReLU"](%self)
     %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
     return (%second_output) )");

  const PatternInfo add_functional_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %inplace):
     %one = prim::Constant[value=1]()
     %first_output = aten::add_(%a, %b, %one)
     %relu = prim::Constant[name="relu"]()
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )");


  const std::vector<std::reference_wrapper<const PatternInfo>> skip_patterns = {
    conv_functional_relu,
    conv_relu,
    matmul_add,
    add_module_relu,
    add_functional_relu,
  };
};

// Check if `use` is an aten function of name `func_name` and if value
// `v` is the nth argument of the function
bool isAtenFuncNthArg(
    Value* v,
    Node* use,
    const std::string& func_name,
    int n) {
  return use->kind() == Symbol::aten(func_name) && v == use->inputs().at(n);
}

// Check if `use` is a CallFunction of name `func_name` and if value
// `v` is the nth argument of the function
bool isCallFunctionNthArg(
    Value* v,
    Node* use,
    const std::string& func_name,
    int n) {
  return use->kind() == prim::CallFunction &&
      getFuncName(use->inputs()[0]) == func_name && v == use->inputs().at(n);
}

struct FuncArg {
  std::string func_name;
  int arg_index;
};
using AtenFuncArgs = std::vector<FuncArg>;
using CallFuncArgs = std::vector<FuncArg>;

// Check any use of `v` matches the aten function call
// or CallFunction patterns
bool matchArgPattern(
    Value* v,
    const AtenFuncArgs& aten_func_args,
    const CallFuncArgs& call_func_args) {
  for (const Use& u : v->uses()) {
    for (const auto& func_arg : aten_func_args) {
      if (isAtenFuncNthArg(v, u.user, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }

    for (const auto& func_arg : call_func_args) {
      if (isCallFunctionNthArg(
              v, u.user, func_arg.func_name, func_arg.arg_index)) {
        return true;
      }
    }
  }
  return false;
}

bool isBiasOfConvOrLinear(Value* v) {
  bool result = matchArgPattern(
      v,
      AtenFuncArgs({{"conv2d", 2}, {"linear", 2}}),
      CallFuncArgs({{"linear", 3}}));
  if (result) {
    TORCH_CHECK(
        v->uses().size() == 1,
        "Graph mode quantization only supports conv/linear bias being used by"
        " one node.");
  }
  return result;
}

bool isWeightOfConvOrLinear(Value* v) {
  bool result = matchArgPattern(
      v,
      AtenFuncArgs({{"conv2d", 1}, {"linear", 1}}),
      CallFuncArgs({{"linear", 2}}));
  if (result) {
    TORCH_CHECK(
        v->uses().size() == 1,
        "Graph mode quantization only supports conv/linear weight being used by"
        " one node.");
  }
  return result;
}

script::Module getObserverModuleFor(Value* v, const QConfig& qconfig) {
  return
    isWeightOfConvOrLinear(v) ? std::get<1>(qconfig) : std::get<0>(qconfig);
}

void replaceConvolutionWithConv2d(std::shared_ptr<Graph>& graph) {
  std::string convolution = R"(
graph(%a, %w, %b, %stride, %padding, %dilation, %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation, %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%r) )";

  std::string conv2d = R"(
graph(%a, %w, %b, %stride, %padding, %dilation, %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // Filter the unsupported case
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto transposed_value =
        getIValue("transposed", match_vmap, vmap).value().toBool();
    auto benchmark_value =
        getIValue("benchmark", match_vmap, vmap).value().toBool();
    auto deterministic_value =
        getIValue("deterministic", match_vmap, vmap).value().toBool();
    auto cudnn_enabled_value =
        getIValue("cudnn_enabled", match_vmap, vmap).value().toBool();
    auto output_padding_value =
        getIValue("output_padding", match_vmap, vmap).value().toIntList();

    if (!transposed_value && !benchmark_value && !deterministic_value &&
        cudnn_enabled_value && (output_padding_value[0] == 0) &&
        (output_padding_value[1] == 0)) {
      return true;
    }
    return false;
  };

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(convolution, conv2d);
  rewriter.runOnGraph(graph, filter);
}

ModuleMethodVector InsertObserversHelper::getInvokedMethods(
    script::Module& module,
    const std::string& method_name) {
  ModuleMethodVector invoked_methods;
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip observer nodes
      if (observer_nodes_.count(n)) {
        continue;
      }
      if (n->kind() == prim::CallMethod) {
        invoked_methods.push_back(std::make_pair(getInvokedModule(module, n, graph->inputs()[0]), n->s(attr::name)));
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return invoked_methods;
}

void InsertObserversHelper::insertObserverFor(
    Value* v,
    script::Module& module,
    const script::Module& observer_module,
    NameModuleVector& observer_name_and_modules) {
  if (observed_values_.count(v)) {
    return;
  }
  script::Module observer = observer_module.clone_instance();
  std::string observer_name = "_observer_" + c10::to_string(uid_++);
  while (module.hasattr(observer_name)) {
    observer_name = "_observer_" + c10::to_string(uid_++);
  }
  module.register_module(observer_name, observer);
  observer_name_and_modules.push_back(std::make_pair(observer_name, observer));

  auto* g = v->owningGraph();
  // Get handle of observer module
  Node* observer_instance =
      g->createGetAttr(g->inputs()[0], observer_name)->insertAfter(v->node());
  observer_instance->output()->setDebugName(observer_name);

  {
    WithInsertPoint guard(observer_instance->next());
    // Match arguments to types of observer's arguments
    script::MatchedSchema forward_matched_schema = script::matchSchema(
        observer.get_method("forward").function().getSchema(),
        v->node()->sourceRange(),
        *g,
        {observer_instance->output(), v},
        {});
    // Insert call to observer's forward
    Node* call = g->insertMethodCall("forward", forward_matched_schema)->node();
    call->output()->copyMetadata(v);

    // Replace v with the output of observer
    v->replaceAllUsesWith(call->output());
    // The above also replaced the input to `call`, so switch it back to
    // the correct value
    call->replaceInput(1, v);
    observer_nodes_.emplace(call);
  }
  observed_values_.insert(v);
}

void InsertObserversHelper::skipValuesInPattern(
    Graph& graph,
    const PatternInfo& pattern) {
  const Graph& pattern_graph = *pattern.pattern_graph;
  const std::unordered_map<std::string, Value*>& vmap = pattern.vmap;

  const auto& matches = findPatternMatches(pattern_graph, graph);
  for (const auto& match : matches) {
    auto output_value = match.values_map.at(vmap.at("first_output"));
    GRAPH_DEBUG("Skipping value in function pattern:",
                output_value->debugName());
    values_to_skip_.insert(output_value);
  }
}

void InsertObserversHelper::addIntermediateValuesToSkipObserver(
    const script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

  for (const auto& pattern : skip_patterns) {
    skipValuesInPattern(*graph, pattern);
  }
}

void InsertObserversHelper::fillPassThroughValueMap(const std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      auto input_indexes = getGeneralOpTensorInputIndexes(n);
      for (auto i : input_indexes) {
        for (auto* output : n->outputs()) {
          pass_through_value_map_[output].push_back(n->input(i));
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

void InsertObserversHelper::fillBoundaryValueMap(
    script::Module& module, const std::string& method_name) {
  auto graph = module.get_method(method_name).graph();
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  auto* self = graph->inputs()[0];
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod) {
        auto m = getInvokedModule(module, n, self);
        auto g = m.get_method(n->s(attr::name)).graph();
        // add mapping from callsite value to value in called graph
        for (auto i = 0; i < g->outputs().size(); ++i) {
          auto* return_val = g->outputs()[i];
          boundary_value_map_[n->output(i)].insert(return_val);
        }
        for (auto i = 0; i < g->inputs().size(); ++i) {
          auto* input_val = g->inputs()[i];
          boundary_value_map_[n->input(i)].insert(input_val);
          caller_to_callee_[n->input(i)] = input_val;
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

void InsertObserversHelper::preprocess(
    script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();
  // TODO: remove constant prop, add separate graph
  // cleanup step before insert observers
  // To cleanup traced graph
  ConstantPooling(graph);
  ConstantPropagation(graph);
  // must do constant propagation first before replacement
  replaceConvolutionWithConv2d(graph);
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

  addIntermediateValuesToSkipObserver(module, method_name);
  fillValueObserverMap(module, method_name);
  fillBoundaryValueMap(module, method_name);
  fillPassThroughValueMap(graph);

  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    preprocess(invoked_module, invoked_method_name);
  }
}

bool InsertObserversHelper::valueNeedsToBeQuantized(Value* v) {
  if (!v->type()->isSubtypeOf(TensorType::get()) ||
      isBiasOfConvOrLinear(v)) {
    return false;
  }
  // Check whether producer is quantizable
  if (nodeQuantizable(v->node())) {
    return true;
  }
  // Check whether user is quantizable
  for (const auto& use : v->uses()) {
    if (nodeQuantizable(use.user)) {
      return true;
    }
  }
  return false;
}

void InsertObserversHelper::fillValueObserverMap(
    script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

  if (visited_graph_of_observer_map_.count(graph.get())) {
    return;
  }
  visited_graph_of_observer_map_.insert(graph.get());

  std::stack<Block*> blocks_to_visit;
  auto qconfig_opt = module_qconfig_map_.at(module._ivalue());
  if (!qconfig_opt) {
    return;
  }
  auto qconfig = *qconfig_opt;

  for (auto* v : graph->inputs()) {
    if (valueNeedsToBeQuantized(v)) {
      observer_for_value_[v] = getObserverModuleFor(v, qconfig);
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        if (valueNeedsToBeQuantized(v)) {
          observer_for_value_[v] = getObserverModuleFor(v, qconfig);
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

c10::optional<script::Module>
InsertObserversHelper::getObserverFor(Value* v) {
  if (observer_for_value_.count(v)) {
    auto observer = observer_for_value_.at(v);
    return observer;
  }
  c10::optional<script::Module> result;
  if (boundary_value_map_.count(v)) {
    for (Value* next : boundary_value_map_.at(v)) {
      auto observer_opt = getObserverFor(next);
      if (observer_opt) {
        // Need to make sure all values are
        // configured with same observer
        if (result) {
          TORCH_CHECK(
              *observer_opt == *result,
              "Expecting all values in the graph only configured with one observer");
        } else {
          result = observer_opt;
        }
      }
    }
  }
  return result;
}

std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>> InsertObserversHelper::insertObservers(
    script::Module& module,
    const std::string& method_name,
    bool is_entry_point,
    std::unordered_set<Value*> graph_observed_values) {
  auto graph = module.get_method(method_name).graph();
  return insertObserversFor(graph->block(), module, graph_observed_values, is_entry_point);
}

std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
InsertObserversHelper::insertObserversFor(
      Block* block,
      script::Module& module,
      std::unordered_set<Value*>& block_observed_values,
      bool is_entry_point) {
  // graph input/output values, used to skip inserting observers
  // for input and output of the graph, we have to insert the observers
  // at call site because the graph itself can be shared
  std::unordered_set<Value*> graph_inputs_outputs;
  // list of observer modules for input values
  std::vector<c10::optional<script::Module>> block_input_observers;
  // list of observer modules for output values
  std::vector<c10::optional<script::Module>> block_output_observers;

  // if the current block is the block for entry point graph(the forward graph
  // of the top level module), we can insert observers in the block directly
  if (!is_entry_point) {
    auto* graph = block->owningGraph();
    for (auto list : {graph->inputs(), graph->outputs()}) {
      for (auto* v : list) {
        graph_inputs_outputs.insert(v);
      }
    }

    for (auto* v : block->inputs()) {
      block_input_observers.push_back(getObserverFor(v));
    }

    for (auto* v : block->outputs()) {
      block_output_observers.push_back(getObserverFor(v));
    }
  }

  // This means the block is been processed before, we just
  // need to attach observer modules and construct the information
  // needed by call site here
  bool visited = block_observer_map_.count(block);
  if (visited) {
    // instance clone of observer module and setAttr
    for (const auto& observer_attrs : block_observer_map_.at(block)) {
      const auto& name = std::get<0>(observer_attrs);
      const auto& observer = std::get<1>(observer_attrs);
      module._ivalue()->setAttr(name, observer.clone_instance()._ivalue());
    }
  }
  // NB: Why do we need to process the graph even if it's visited?
  // Reason is `graph_observed_values` can
  // change depending on where the method is called, and
  // outputs that's been observed(third item of the returned result)
  // can change depending on that, so for each graph we'll need to go through
  // the whole process of inserting observers

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(block);
  auto* self = block->inputs()[0];
  // We first construct a map from value to the module, then
  // insert observers for them later, this is to avoid interference
  // of the inserted observers with the analysis to decide where
  // to insert observers, also we only insert observers for
  // "intermediate values" that is not the input/output of the
  // graph
  std::unordered_map<Value*, script::Module> values_to_observe;

  for (auto* v : block->inputs()) {
    if (!graph_inputs_outputs.count(v) && !values_to_observe.count(v)) {
      if (auto observer_opt = getObserverFor(v)) {
        values_to_observe[v] = *observer_opt;
      }
    }
  }
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (observer_nodes_.count(n)) {
        continue;
      }
      if (n->kind() == prim::CallMethod) {
        auto m = getInvokedModule(module, n, self);
        std::unordered_set<Value*> callee_observed_inputs;
        for (auto i = 0; i < n->inputs().size(); ++i) {
          if (block_observed_values.count(n->input(i))) {
            callee_observed_inputs.insert(caller_to_callee_[n->inputs()[i]]);
          }
        }
        auto* subblock = m.get_method(n->s(attr::name)).graph()->block();
        auto info_from_callee = insertObserversFor(subblock, m, callee_observed_inputs);
        auto input_observers = std::get<0>(info_from_callee);
        auto output_observers = std::get<1>(info_from_callee);
        auto callee_observed_outputs = std::get<2>(info_from_callee);
        for (auto idx : callee_observed_outputs) {
          block_observed_values.insert(n->outputs()[idx]);
        }
        for (auto i = 0; i < n->inputs().size(); ++i) {
          if (input_observers[i] && !graph_inputs_outputs.count(n->input(i))
              && !block_observed_values.count(n->input(i))) {
            values_to_observe[n->inputs()[i]] = *input_observers[i];
            block_observed_values.insert(n->input(i));
          }
        }
        for (auto i = 0; i < n->outputs().size(); ++i) {
          if (output_observers[i] && !graph_inputs_outputs.count(n->output(i))
              && !block_observed_values.count(n->output(i))) {
            values_to_observe[n->outputs()[i]] = *output_observers[i];
            block_observed_values.insert(n->output(i));
          }
        }
      } else {
        for (Value* v : n->outputs()) {
          propagateObservedProperty(v, block_observed_values);
          if (!graph_inputs_outputs.count(v) && !block_observed_values.count(v)) {
            if (auto observer_opt = getObserverFor(v)) {
              values_to_observe[v] = *observer_opt;
              block_observed_values.insert(v);
            }
          }
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  std::vector<size_t> output_idxs;
  for (auto i = 0; i < block->outputs().size(); ++i) {
    if (block_observed_values.count(block->outputs()[i])) {
      output_idxs.push_back(i);
    }
  }
  if (!visited) {
    NameModuleVector observer_name_and_modules;
    for (const auto& item : values_to_observe) {
      auto* v = item.first;
      auto observer = item.second;
      if (!values_to_skip_.count(v)) {
        insertObserverFor(v, module, observer, observer_name_and_modules);
      }
    }
    block_observer_map_[block] = observer_name_and_modules;
  }
  return std::make_tuple(block_input_observers, block_output_observers, output_idxs);
}

void InsertObserversHelper::propagateObservedProperty(
    Value* output, std::unordered_set<Value*>& block_observed_values) {
  if (pass_through_value_map_.count(output)) {
    // since the vector is always non-empty, we will
    // not return the initial value
    bool all_observed = true;
    for (Value* v : pass_through_value_map_.at(output)) {
      all_observed &= observed_values_.count(v) || block_observed_values.count(v);
    }
    if (all_observed) {
      // This is to propagate observed property through
      // all ops that doesn't require observation
      block_observed_values.insert(output);
    }
  }
}

void insertDeQuantCall(Graph* graph,
                       Value* quantized_val,
                       Value* original_val,
                       const std::vector<Use>& uses) {
  for (size_t i = 0; i < uses.size(); ++i) {
    auto* user = uses[i].user;
    // Insert dequantize node right before use node, because
    // we want to make sure use node and dequantize node reside
    // in the same block so that quant fusion can happen
    WithInsertPoint ins(user);
    Node* dequant =
      graph->create(Symbol::aten("dequantize"), {quantized_val});
    dequant->output()->setDebugName(
        original_val->debugName() + ".dequant." + c10::guts::to_string(i))
      ->setType(original_val->type());
    user->replaceInputWith(original_val, dequant->output());
    graph->insertNode(dequant);
  }
}

void insertQuantDeQuantCall(
    Value* self,
    Node* observer,
    bool is_per_channel,
    const std::vector<std::string>& qparam_names) {
  Graph* g = observer->owningGraph();
  // Original value that is observed
  Value* v = observer->input(1);

  // Inserting before insert point
  WithInsertPoint ins(v->node()->next());
  std::vector<Value*> inputs = {v};
  // Insert GetAttr nodes for quantization parameters
  for (const auto& qparam_name : qparam_names) {
    inputs.push_back(g->insertGetAttr(self, qparam_name));
  }
  std::string quantize_func;
  if (is_per_channel) {
    quantize_func = "quantize_per_channel";
  } else {
    quantize_func = "quantize_per_tensor";
  }
  Node* quant = g->create(at::Symbol::aten(quantize_func), inputs);
  quant->output()->setDebugName(v->debugName() + ".quant");
  g->insertNode(quant);

  // two passes to insert the dequant for every usage
  // in first pass, identify all the nodes using "v"
  std::vector<Use> uses;
  for (const auto& use : v->uses()) {
    // Skip quant node and observer node (we need to keep
    // observer nodes around since we need them to
    // find the quantization parameters)
    if (use.user != quant && use.user != observer) {
      uses.push_back(use);
    }
  }

  // in second pass, replace the input "v" with dequant output
  insertDeQuantCall(g, quant->output(), v, uses);
}

// find the observer for Value `v` and return the name of the observer
c10::optional<std::string> findObserverName(Value* v) {
  // Note that here we just check for the name of observer, but the ideally
  // we should be comparing the type of observer, this is a temporary
  // work around until data only clone of module.clone is supported.
  Node* n = v->node();
  if (n->kind() == prim::CallMethod && n->s(attr::name) == "forward") {
    auto module_instance = n->inputs().at(0);
    if (module_instance->node()->kind() == prim::GetAttr &&
        module_instance->node()->s(attr::name).find("_observer_") !=
            std::string::npos) {
      return module_instance->node()->s(attr::name);
    }
  }
  return c10::nullopt;
}

c10::QScheme toAffine(c10::QScheme qscheme) {
  switch (qscheme) {
    case c10::kPerTensorAffine:
    case c10::kPerTensorSymmetric:
      return c10::kPerTensorAffine;
    case c10::kPerChannelAffine:
    case c10::kPerChannelSymmetric:
      return c10::kPerChannelAffine;
    default:
      return qscheme;
  }
}

class InsertQuantDeQuantHelper {
 public:
  InsertQuantDeQuantHelper() {}
  void run(script::Module& module, const std::string& method_name);

  ModuleMethodVector getInvokedMethods(
      script::Module& module,
      const std::string& method_name);

  // Get quantization parameter map of the given Value in Graph
  // by searching for observer module of the value and extract the
  // quantization parameters from the observer module
  std::tuple<c10::QScheme, QParamVector> getQSchemeAndQParamVector(
      script::Module& module,
      Node* n);
  void checkQScheme(Graph* g, c10::QScheme qscheme) {
    if (qscheme_for_graph_.count(g)) {
      TORCH_CHECK(
          qscheme_for_graph_.at(g) == qscheme ||

              "Quantizing same graph with different types of "
              "QSchemes is not supported.\n",
          " Expecting:",
          c10::toString(qscheme_for_graph_.at(g)),
          " Got:",
          c10::toString(qscheme));
    } else {
      qscheme_for_graph_[g] = toAffine(qscheme);
    }
  }

  c10::optional<script::Module> findChildModuleToQuantize(
      script::Module& module,
      Value* child_instance);
  void collectObserverNodesAndValueToQuantize(script::Module& module, Value*);
  // Cleanup observer nodes from graph and observer modules
  // from module object and ClassType
  void cleanup(script::Module& module);
  void cleanup(script::Module& module, Graph* g);
  void quantizeTensors(script::Module& module, Graph* g, Value* self);

 private:
  std::unordered_map<Graph*, std::vector<std::string>>
      observer_modules_to_remove_;
  // We only remove observer module attributes from type in the
  // first encounter of the graph, after that since the attributes
  // is already removed from the ClassType, we'll use the list of slot index to
  // replay this removal
  std::unordered_map<Graph*, std::vector<int>> removed_observer_slots_;
  std::unordered_map<Graph*, std::vector<Node*>> nodes_to_destroy_;
  // Map from Graph to observer node, we can use observer node to
  // get the information of original value that's been observed and
  // the quantization parameters
  std::unordered_map<Graph*, std::vector<Node*>> observer_nodes_for_graph_;
  // A map from qparam name (e.g. _scale) to the attribute name in
  // the module(e.g. weight_scale_0)
  std::unordered_map<Node*, std::unordered_map<std::string, std::string>> qparam_name_map_for_node_;
  // Record qscheme for every graph, this is for checking
  // each graph is only quantized with one type of QScheme
  std::unordered_map<Graph*, c10::QScheme> qscheme_for_graph_;
};

void InsertQuantDeQuantHelper::collectObserverNodesAndValueToQuantize(
    script::Module& module,
    Value* v) {
  auto* g = v->owningGraph();
  auto observer_name = findObserverName(v);
  if (!observer_name) {
    return;
  }
  observer_modules_to_remove_[g].push_back(observer_name.value());

  Node* observer = v->node();
  TORCH_INTERNAL_ASSERT(
      observer->kind() == prim::CallMethod &&
      observer->s(attr::name) == "forward" &&
      observer->inputs()[0]->node()->kind() == prim::GetAttr &&
      observer->inputs()[0]->node()->s(attr::name) == observer_name);

  // Observer forward call node
  nodes_to_destroy_[g].push_back(observer);
  // GetAttr node for observer module
  nodes_to_destroy_[g].push_back(observer->inputs()[0]->node());
  Value* original_value = observer->input(1);
  v->replaceAllUsesWith(original_value);
  observer_nodes_for_graph_[g].push_back(observer);
}

void InsertQuantDeQuantHelper::cleanup(script::Module& module) {
  for (auto& method : module.get_methods()) {
    cleanup(module, method.graph().get());
  }
  for (script::Module m : module.children()) {
    cleanup(m);
  }
}

void InsertQuantDeQuantHelper::cleanup(script::Module& module, Graph* g) {
  GRAPH_DUMP("Before Remove Observers:", g);
  if (nodes_to_destroy_.count(g)) {
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->removeAllInputs();
    }
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->destroy();
    }
    nodes_to_destroy_.at(g).clear();
  }

  // 1. If we have seen this graph before, this means the observer
  // attributes has been removed from the type(see step 2) but the slot
  // index of these attributes are kept in the list, we'll replay the observer
  // slots removal using these slot indexes
  if (removed_observer_slots_.count(g)) {
    for (auto slot : removed_observer_slots_.at(g)) {
      module._ivalue()->unsafeRemoveSlot(slot);
    }
  }

  // 2. Remove observer modules from last one to first one in order to
  // reduce the time complexity, assuming all the observer modules
  // are added after the existing modules, we'll have complexity of
  // O(N) where N is number of observer modules with this optimization
  if (observer_modules_to_remove_.count(g)) {
    auto& observers = observer_modules_to_remove_.at(g);
    for (int64_t i = observers.size() - 1; i >= 0; --i) {
      auto observer_name = observers[i];
      GRAPH_DEBUG("Trying to remove: ", observer_name);
      if (module.type()->hasAttribute(observer_name)) {
        // We record the slot index here in order to replay the
        // slot removal in other objects that's sharing the ClassType
        // since we're going to remove attribute in the ClassType here
        removed_observer_slots_[g].push_back(
            module.type()->getAttributeSlot(observer_name));
        module._ivalue()->unsafeRemoveAttr(observer_name);
        module.type()->unsafeRemoveAttribute(observer_name);
      }
    }
    observers.clear();
  }
  GRAPH_DUMP("After remove observers :", g);
}

void InsertQuantDeQuantHelper::quantizeTensors(
    script::Module& module,
    Graph* g,
    Value* self) {
  if (!observer_nodes_for_graph_.count(g)) {
    return;
  }
  for (auto* n : observer_nodes_for_graph_.at(g)) {
    auto* original_value = n->input(1);
    auto tp = getQSchemeAndQParamVector(module, n);
    auto qscheme = std::get<0>(tp);
    auto qparam_map = std::get<1>(tp);
    checkQScheme(g, qscheme);
    std::vector<std::string> qparam_names;
    for (auto& pr : qparam_map) {
      const auto& name = pr.first;
      const auto& qparam = pr.second;
      size_t uid = 0;
      auto qparam_name = original_value->debugName() + name + "_" + c10::to_string(uid++);
      while (module.hasattr(qparam_name)) {
        qparam_name = original_value->debugName() + name + "_" + c10::to_string(uid++);
      }
      qparam_name_map_for_node_[n][name] = qparam_name;
      module.register_attribute(qparam_name, qparam.type(), qparam);
      qparam_names.push_back(qparam_name);
    }
    insertQuantDeQuantCall(self, n, isPerChannel(qscheme), qparam_names);
  }
}

void checkGetQParamsResult(const IValue& qparams) {
  TORCH_CHECK(
      qparams.isTuple(),
      "`get_qparams` function is expected to return a "
      "Tuple, but got:",
      qparams.tagKind());
  auto tp = qparams.toTuple();
  TORCH_CHECK(
      tp->elements().size() == 2 || tp->elements().size() == 3,
      "`get_qparams` function is expected to return a "
      "Tuple of size 2 or 3, got Tuple of size ",
      tp->elements().size());
  // Expect first two elements of the tuple to be Tensor
  for (size_t i = 0; i < 2; ++i) {
    TORCH_CHECK(
        tp->elements()[i].isTensor(),
        "Element of Tuple is expected to be Tensor, but element ",
        i,
        " has type: ",
        tp->elements()[i].tagKind());
  }
  // Expect the third elements of the tuple to be int
  if (tp->elements().size() == 3) {
    TORCH_CHECK(
        tp->elements()[2].isInt(),
        "Element of Tuple is expected to be int, but element ",
        2,
        " has type: ",
        tp->elements()[2].tagKind());
  }
}

std::tuple<c10::QScheme, QParamVector> InsertQuantDeQuantHelper::
    getQSchemeAndQParamVector(script::Module& module, Node* n) {
  // TODO: refactor findObserverName to take Node* as input
  Value* v = n->output();
  TORCH_INTERNAL_ASSERT(
      v->type()->isSubtypeOf(TensorType::get()),
      "Expected output of observer node to be Tensor");
  auto observer_name = findObserverName(v);
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQSchemeAndParamMap expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  auto observer_module = module.attr(observer_name.value()).toModule();
  auto get_qparams = observer_module.get_method("get_qparams");
  IValue result = get_qparams(std::vector<IValue>());
  checkGetQParamsResult(result);
  auto scalar_type = observer_module.attr("dtype");
  TORCH_CHECK(
      scalar_type.toScalarType() != at::ScalarType::Undefined,
      "dtype of observer can't be undefined");
  auto tp = result.toTuple();
  at::Tensor scale = tp->elements()[0].toTensor().to(at::kFloat);
  at::Tensor zero_point = tp->elements()[1].toTensor().to(at::kInt);
  // quantization parameters should appear in the same order as
  // the argument for quantize_per_tensor/quantize_per_channel function
  QParamVector qparams;
  auto qscheme = observer_module.attr("qscheme").toQScheme();
  if (isPerChannel(qscheme)) {
    qparams.push_back(std::make_pair("_scale", scale));
    qparams.push_back(std::make_pair("_zero_point", zero_point));
    qparams.push_back(std::make_pair("_axis", tp->elements()[2].toInt()));
  } else {
    qparams.push_back(std::make_pair("_scale", scale.item<double>()));
    qparams.push_back(std::make_pair("_zero_point", zero_point.item<int64_t>()));
  }
  qparams.push_back(std::make_pair("_scalar_type", scalar_type));
  return std::make_tuple(qscheme, qparams);
}

c10::optional<script::Module> InsertQuantDeQuantHelper::
    findChildModuleToQuantize(script::Module& module, Value* child_instance) {
  TORCH_INTERNAL_ASSERT(
      child_instance->node()->kind() == prim::GetAttr,
      "Child instance should come from GetAttr.");
  auto child_module_name = child_instance->node()->s(attr::name);
  if (child_module_name.find("_observer_") == std::string::npos) {
    return module.attr(child_module_name).toModule();
  }
  return c10::nullopt;
}

ModuleMethodVector InsertQuantDeQuantHelper::getInvokedMethods(
    script::Module& module,
    const std::string& method_name) {
  auto graph = module.get_method(method_name).graph();

  ModuleMethodVector invoked_methods;
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod) {
        auto module_instance = n->inputs()[0];
        auto module_method_name = n->s(attr::name);
        c10::optional<script::Module> m;
        // calling method on self
        if (module_instance == graph->inputs()[0]) {
          m = module;
        } else {
          m = findChildModuleToQuantize(module, module_instance);
        }
        if (m) {
          invoked_methods.push_back({*m, module_method_name});
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return invoked_methods;
}

void InsertQuantDeQuantHelper::run(
    script::Module& module,
    const std::string& method_name) {
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    run(invoked_module, invoked_method_name);
  }

  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

  // We only need to register new parameters if the graph has
  // been quantized before
  // TODO: dedup this part with code in quantizeTensors
  if (observer_nodes_for_graph_.count(graph.get())) {
    for (auto* n : observer_nodes_for_graph_.at(graph.get())) {
      auto tp = getQSchemeAndQParamVector(module, n);
      checkQScheme(graph.get(), std::get<0>(tp));
      auto qparam_map = std::get<1>(tp);
      TORCH_INTERNAL_ASSERT(qparam_name_map_for_node_.count(n),
                            "Expected to have a qparam_name_map for node:",
                           *n);
      auto qparam_name_map = qparam_name_map_for_node_.at(n);
      for (auto& pr : qparam_map) {
        const auto& name = pr.first;
        const auto& qparam = pr.second;
        module._ivalue()->setAttr(qparam_name_map.at(name), qparam);
      }
    }
    return;
  }

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  std::vector<Value*> input_values;
  for (size_t idx = 1; idx < method.num_inputs(); ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      input_values.push_back(v);
    }
  }

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end;) {
      Node* n = *it++;
      for (Value* v : n->outputs()) {
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          continue;
        }
        collectObserverNodesAndValueToQuantize(module, v);
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  for (Value* v : input_values) {
    collectObserverNodesAndValueToQuantize(module, v);
  }
  GRAPH_DUMP("Before Quantize Tensors:", graph);
  Value* self = graph->inputs()[0];
  quantizeTensors(module, graph.get(), self);
  GRAPH_DUMP("After Quantize Tensors:", graph);
}

void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  std::string linear_with_quant = R"(
graph(%a_dequant, %w_quant, %b):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  std::string linear_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b):
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::linear(%a_dequant, %w_dequant, %b_unpacked)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(linear_with_quant, linear_with_quant_prepack);
  rewriter.runOnGraph(graph);
}

void insertPrepackUnpackForConv2d(std::shared_ptr<Graph>& graph) {
  std::string conv_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(conv_with_quant, conv_with_quant_prepack);
  rewriter.runOnGraph(graph);
}

c10::optional<IValue> toTwoElementIntList(Value* v) {
  auto* n = v->node();
  if (n->kind() == prim::Constant) {
    auto iv = toIValue(v);
    if (iv && iv.value().isIntList() && iv.value().toIntList().size() == 2) {
      return iv;
    }
  }

  if (n->kind() == prim::ListConstruct && n->inputs().size() == 2) {
    auto e0 = toIValue(n->inputs()[0]);
    auto e1 = toIValue(n->inputs()[1]);
    if (!e0 || !e1 || !e0.value().isInt() || !e1.value().isInt()) {
      return c10::nullopt;
    }
    return IValue(c10::List<int64_t>({e0.value().toInt(), e1.value().toInt()}));
  }
  return c10::nullopt;
}

// A helper class to make uses of module unique
class ModuleUseDeduper {
 public:
  ModuleUseDeduper(script::Module& module) : module_(module) {}
  void dedup() {
    for (auto& method : module_.get_methods()) {
      const auto& graph = method.graph();
      findModuleUses(graph.get());
    }
    dedupModuleUses();
  }

 private:
  // Analyze the code to record information represents
  // uses of the module, which we'll use later to actually perform the dedup
  // operation Please see the comments of member variables of the class for more
  // information
  void findModuleUses(Graph* graph) {
    GRAPH_DUMP("Finding module uses for ", graph);

    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(graph->block());
    Value* self = graph->inputs()[0];
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (Node* n : b->nodes()) {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
        if (n->kind() != prim::CallMethod) {
          continue;
        }
        Value* instance = n->inputs()[0];
        // boundary_val is the value we get when we trace back
        // the GetAttr access chain until we hit the input of graph
        // or a node that is not prim::GetAttr
        auto path = getModuleAccessPath(instance, self);

        // path.size() == 0 means we're calling a method
        // on self, we don't need to dedup uses of self
        if (path.size() == 0) {
          continue;
        }
        value_to_path_map_[instance] = path;
        auto m = findChildModule(module_, path);
        // If we fail to insert the module to the unique_modules_ set,
        // which means there are uses of this module before this point,
        // we'll have to rewrite the use
        if (!unique_modules_.insert(m._ivalue()).second) {
          uses_to_rewrite_.push_back(instance);
          GRAPH_DEBUG("Found use to rewrite: ", instance->debugName());
        }
      }
    }
  }

  // Deduplicate module uses given the information we recorded before
  void dedupModuleUses() {
    for (Value* v : uses_to_rewrite_) {
      const auto& path = value_to_path_map_.at(v);
      const auto& m = findChildModule(module_, path);
      // add a clone of the child module to the parent of the duplicated module
      const auto& child_name = addChildModule(module_, m, path);
      TORCH_INTERNAL_ASSERT(v->node()->kind() == prim::GetAttr);
      // change the name in GetAttr call
      auto original_name = v->node()->s(attr::name);
      v->node()->s_(attr::name, child_name);
      GRAPH_UPDATE(
          "Module use dedup: changing use of original module ",
          original_name,
          " to ",
          child_name);
    }
  }

  std::string addChildModule(
      script::Module& module,
      const script::Module& child_module,
      const std::vector<std::string>& path) {
    TORCH_INTERNAL_ASSERT(
        path.size() > 0, "path must have at least one element.");
    // Parent module of the leaf child module corresponding to
    // the path
    auto parent_of_leaf = findChildModule(
        module, std::vector<std::string>(path.begin(), path.end() - 1));

    // Original name of the child module
    std::string original_name = path[path.size() - 1];
    int uid = 0;
    std::string child_name = original_name + "_" + c10::to_string(uid++);
    while (parent_of_leaf.hasattr(child_name)) {
      child_name = original_name + "_" + c10::to_string(uid++);
    }
    parent_of_leaf.register_module(child_name, child_module.clone_instance());
    return child_name;
  }

  script::Module module_;
  // Map from value of module instance to the list of names of submodules
  // starting from the top level module, e.g. ["sub1", "sub2", "relu"]
  // Also this is a cache of calling `getModuleAccessPath` of the value
  std::unordered_map<Value*, std::vector<std::string>> value_to_path_map_;
  // Set of unique modules that are used in the graphs
  std::unordered_set<script::ModulePtr> unique_modules_;
  // Values that represent the module instance(the use of the module)
  // that we'll need to rewrite as a use of a cloned module
  // instance
  std::vector<Value*> uses_to_rewrite_;
};

struct ConvBNParameters {
  at::Tensor conv_w;
  at::Tensor conv_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

static bool hastensor(script::Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTensor();
}

class FoldConvBatchNorm2dHelper {
 public:
  /**
   * In this step we find all Conv2d - BatchNorm2d patterns in the graph
   * and extract the corresponding parameters for these two modules,
   * and record informations for the modifications of the graph without
   * actually performing these modifications.
   */
  void analyze(script::Module& module);
  /**
   * In this step we perform all the modifications including
   * setting the attributes for conv module, rewriting values
   * and deleting nodes in the graph
   */
  void transform();

 private:
  bool tryExtractingConvBNParameters(
      script::Module& conv,
      script::Module& bn,
      ConvBNParameters& r);

  /**
   * Given the current weight and bias tensors of a Conv2d module and parameters
   * of the BatchNorm2d module we're folding with, compute the updated values
   * for the weight and bias.
   *
   * The function is basically copied from torch/nn/utils/fusion.py
   */
  std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
      const ConvBNParameters& p);

  std::unordered_map<script::ModulePtr,
                     std::tuple<at::Tensor, at::Tensor>> conv_module_and_params_;
  std::unordered_map<Graph*, std::vector<std::tuple<std::string, std::string>>> conv_bn_names_;
  std::unordered_map<Value*, Value*> rewrite_map_;
  std::vector<Value*> values_to_rewrite_;
  std::unordered_set<Node*> nodes_to_delete_;
};

std::tuple<at::Tensor, at::Tensor> FoldConvBatchNorm2dHelper::
    computeUpdatedConvWeightAndBias(const ConvBNParameters& p) {
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape({-1, 1, 1, 1});
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  return std::make_tuple(new_w, new_b);
}

bool FoldConvBatchNorm2dHelper::tryExtractingConvBNParameters(
    script::Module& conv,
    script::Module& bn,
    ConvBNParameters& r) {
  if (!hastensor(conv, "weight") || !conv.hasattr("bias") ||
      !hastensor(bn, "weight") || !hastensor(bn, "bias") ||
      !hastensor(bn, "running_mean") || !hastensor(bn, "running_var") ||
      !bn.hasattr("eps")) {
    return false;
  }

  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();
  r.bn_eps = bn.attr("eps").toDouble();
  r.bn_w = bn.attr("weight").toTensor();
  r.bn_b = bn.attr("bias").toTensor();

  r.conv_w = conv.attr("weight").toTensor();
  r.conv_b = at::zeros_like(r.bn_rm);
  auto bias_opt = conv.attr("bias").toOptional<at::Tensor>();
  if (bias_opt) {
    r.conv_b = *bias_opt;
  }

  return true;
}

void FoldConvBatchNorm2dHelper::analyze(script::Module& module) {
  const PatternInfo pattern = PatternInfo::parse_from_str(R"IR(
graph(%self, %x):
    %conv_submodule = match::module[name="Conv2d"](%self)
    %conv_out = prim::CallMethod[name="forward"](%conv_submodule, %x)
    %bn_submodule = match::module[name="BatchNorm2d"](%self)
    %bn_out = prim::CallMethod[name="forward"](%bn_submodule, %conv_out)
    return (%bn_out))IR");

  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;
  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_bn_out = vmap.at("bn_out");
  Value* pattern_conv_submodule = vmap.at("conv_submodule");
  Value* pattern_bn_submodule = vmap.at("bn_submodule");
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  // We will put submodules into this worklist and keep processing items from it
  // one by one. We start by just putting the top module there.
  std::stack<script::Module> worklist({module});
  while (!worklist.empty()) {
    script::Module current = worklist.top();
    worklist.pop();

    // Queue submodules for processing
    for (const script::Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    // Process all method of the current module
    for (auto& method : current.get_methods()) {
      GRAPH_DUMP(
          current.type()->name()->name() + "::" + method.name() +
          "() before Conv2d-BatchNorm2d folding",
          method.graph());
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());

      GRAPH_DEBUG("number of Conv2d-BatchNorm2d matches: ", matches.size());
      Graph* g = method.graph().get();
      if (!conv_bn_names_.count(g)) {
        // This is to make sure we don't visit one graph multiple times
        conv_bn_names_[g] = {};
        for (const Match& match : matches) {
          GRAPH_DEBUG("Checking next match...");
          Node* matched_conv = match.nodes_map.at(pattern_conv);
          Node* matched_bn = match.nodes_map.at(pattern_bn);
          Node* matched_conv_submodule =
            match.values_map.at(pattern_conv_submodule)->node();
          Node* matched_bn_submodule =
            match.values_map.at(pattern_bn_submodule)->node();

          TORCH_INTERNAL_ASSERT(matched_conv_submodule->kind() == prim::GetAttr);
          TORCH_INTERNAL_ASSERT(matched_bn_submodule->kind() == prim::GetAttr);

          const auto& conv_module_name = matched_conv_submodule->s(Symbol::attr("name"));
          const auto& bn_module_name = matched_bn_submodule->s(Symbol::attr("name"));

          script::Module conv_submodule =
            current.attr(conv_module_name).toModule();
          script::Module bn_submodule =
            current.attr(bn_module_name).toModule();

          ConvBNParameters params;
          if (!tryExtractingConvBNParameters(
                  conv_submodule, bn_submodule, params)) {
            GRAPH_DEBUG(
                "Conv and BN modules didn't have all required parameters or attributes...");
            continue;
          }
          conv_bn_names_[g].push_back(
              std::make_tuple(conv_module_name, bn_module_name));
          // We are using a separate vector for saving Values we want to rewrite to
          // make sure that the order in which we perform these transformations is
          // deterministic. Iterating through keys of rewrite_map would result in
          // non-determinism that might not manifest as a bug now, but can bite us
          // later.
          values_to_rewrite_.push_back(matched_bn->output());
          rewrite_map_[matched_bn->output()] = matched_conv->output();
          GRAPH_UPDATE(
              "Rewriting %",
              matched_bn->output()->debugName(),
              " with %",
              matched_conv->output()->debugName());

          nodes_to_delete_.insert(matched_bn);
          nodes_to_delete_.insert(matched_bn_submodule);
          GRAPH_UPDATE("Deleting ", *matched_bn);
          GRAPH_UPDATE("Deleting ", *matched_bn_submodule);

          auto slot = conv_submodule.type()->getAttributeSlot("bias");
          TORCH_CHECK(conv_submodule.type()->is_parameter(slot),
                      "Expected conv module to have a bias parameter");
        } // matches
      }

      for (const auto& conv_bn : conv_bn_names_.at(g)) {
        script::Module conv_submodule =
          current.attr(std::get<0>(conv_bn))
          .toModule();
        script::Module bn_submodule =
          current.attr(std::get<1>(conv_bn))
          .toModule();

        ConvBNParameters params;
        TORCH_INTERNAL_ASSERT(tryExtractingConvBNParameters(
                                  conv_submodule, bn_submodule, params));
        auto new_w_b = computeUpdatedConvWeightAndBias(params);
        conv_module_and_params_[conv_submodule._ivalue()] = new_w_b;
      } // conv_bn module
    } // methods
  } // while
}

void FoldConvBatchNorm2dHelper::transform() {
  for (const auto& item : conv_module_and_params_) {
    script::Module conv(item.first);
    auto w_b = item.second;
    conv.setattr("weight", std::get<0>(w_b));
    conv.setattr("bias", std::get<1>(w_b));
  }

  // Perform planned rewritings
  for (auto v : values_to_rewrite_) {
    v->replaceAllUsesWith(rewrite_map_.at(v));
  }

  // Perform planned deletions
  for (auto n : nodes_to_delete_) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    n->destroy();
  }
}

} // namespace

TORCH_API script::Module InsertObservers(
    script::Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace) {
  ModuleQConfigMap map_before_clone;
  fillQConfigMap(input_module, qconfig_dict, map_before_clone);
  ModuleCloneHelper mh;
  script::Module module =
      inplace ? input_module : mh.clone(input_module, map_before_clone);
  ModuleQConfigMap module_qconfig_map;
  // Since the types are changed after clone, we need to fill
  // the qconfig map again
  fillQConfigMap(module, qconfig_dict, module_qconfig_map);
  InsertObserversHelper helper(module_qconfig_map);
  helper.preprocess(module, method_name);
  helper.insertObservers(module, method_name, true);
  return module;
}

script::Module InsertQuantDeQuant(
    script::Module& input_module,
    const std::string& method_name,
    bool inplace) {
  script::Module module = inplace ? input_module : input_module.clone();
  InsertQuantDeQuantHelper h;
  h.run(module, method_name);
  h.cleanup(module);
  return module;
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void SwapFunctionalLinear(script::Module& module) {
  for (auto& method : module.get_methods()) {
    std::shared_ptr<Graph> g = method.graph();
    SwapFunctionalLinear(g);
  }
  for (script::Module m : module.children()) {
    SwapFunctionalLinear(m);
  }
}

void SwapFunctionalLinear(std::shared_ptr<Graph>& graph) {
  std::string functional_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = prim::CallFunction(%linear, %input, %weight, %bias)
  return (%r) )";
  std::string aten_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = aten::linear(%input, %weight, %bias)
  return (%r) )";
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear = getValue("linear", match_vmap, vmap);
    auto func_name = getFuncName(linear);
    return func_name == "linear";
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(functional_linear, aten_linear);
  // TODO: runOnGraph takes const ref?
  rewriter.runOnGraph(graph, filter);
}

void ReplicateDeQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::vector<Node*> dequant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == Symbol::aten("dequantize") &&
          n->output()->uses().size() > 1) {
        dequant_nodes_to_rewrite.push_back(n);
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (Node* n : dequant_nodes_to_rewrite) {
    auto* quantized_val = n->inputs()[0];
    auto* dequantized_val = n->output();
    // copy uses to vector since value->uses() is a reference
    // and changing the graph will also change the uses() list
    std::vector<Use> uses = dequantized_val->uses();
    insertDeQuantCall(graph.get(), quantized_val, dequantized_val, uses);
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->removeAllInputs();
  }
  for (Node* n : dequant_nodes_to_rewrite) {
    n->destroy();
  }
}

// This is the pass to handle ops that does not require observation
// for example: flatten, average_pool, upsample
// This is called after inline and before graph execution
void SwapDeQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      auto input_indexes = getGeneralOpTensorInputIndexes(n);
      if (input_indexes.size() > 0) {
        bool is_dequantized = true;
        for (auto i : input_indexes) {
          is_dequantized &= n->inputs()[i]->node()->kind() == Symbol::aten("dequantize");
        }
        if (!is_dequantized) {
          continue;
        }
        // Delete dequantize node, we have one dequantize
        // for each use of the value
        for (auto i : input_indexes) {
          auto* dequantized_val = n->inputs()[i];
          auto* dequantize_node = dequantized_val->node();
          TORCH_INTERNAL_ASSERT(dequantized_val->uses().size() == 1,
                                "Expect to have one dequantize node for each use");
          // Replace useses of dequantized_val with the input of
          // dequantize node
          dequantized_val->replaceAllUsesWith(dequantize_node->inputs()[0]);
          dequantize_node->removeAllInputs();
          dequantize_node->destroy();
        }
        TORCH_CHECK(n->outputs().size() == 1, "We only support dequantize swapping for ops"
                    " with one output right now");
        auto* output = n->output();
        std::vector<Use> uses = output->uses();
        // Insert new dequantize node for each use of the output
        insertDeQuantCall(graph.get(), output, output, uses);
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

void QuantFusion(std::shared_ptr<Graph>& graph) {
  for (const auto& item : quant_fusion_pattern_and_replacements()) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(item.first, item.second);
    rewriter.runOnGraph(graph);
  }
}

script::Module FoldConvBatchNorm2d(const script::Module& module) {
  FoldConvBatchNorm2dHelper h;
  script::Module m = module.clone();
  h.analyze(m);
  h.transform();
  return m;
}

void FoldQuantizeCallIntoBuffer(
    script::Module& module,
    const std::string& method_name) {
  const PatternInfo& pattern = PatternInfo::parse_from_str(R"(
graph(%self, %scale, %zero_point, %dtype):
   %weight = prim::GetAttr[name="weight"](%self)
   %weight_quant = aten::quantize_per_tensor(%weight, %scale, %zero_point, %dtype)
   return (%weight_quant) )");
  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;

  auto method = module.get_method(method_name);
  auto graph = method.graph();
  const auto& matches = findPatternMatches(pattern_graph, *graph);
  // Extra filter on scale/zero_point/dtype to make sure they are Constant
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto scale_node = match_vmap.at(vmap.at("scale"))->node();
    auto zero_point_node = match_vmap.at(vmap.at("zero_point"))->node();
    auto dtype_node = match_vmap.at(vmap.at("dtype"))->node();
    return scale_node->kind() == prim::Constant &&
        zero_point_node->kind() == prim::Constant &&
        dtype_node->kind() == prim::Constant;
  };
  std::unordered_set<Node*> nodes_to_delete;
  for (const auto& match : matches) {
    if (!filter(match, vmap)) {
      continue;
    }
    auto match_vmap = match.values_map;
    auto float_weight = module.attr("weight").toTensor().data();
    auto scale = toIValue(match_vmap.at(vmap.at("scale"))).value().toDouble();
    auto zero_point =
        toIValue(match_vmap.at(vmap.at("zero_point"))).value().toInt();
    auto dtype =
        toIValue(match_vmap.at(vmap.at("dtype"))).value().toScalarType();
    module.register_buffer(
        "_quantized_weight",
        at::quantize_per_tensor(float_weight, scale, zero_point, dtype));

    // Replace the GetAttr[weight]->quantize_per_tensor sequence
    // with a simple GetAttr[_quantized_weight] node.
    Value* orig_weight = match_vmap.at(vmap.at("weight"));
    Value* orig_weight_quant = match_vmap.at(vmap.at("weight_quant"));

    orig_weight->node()->s_(attr::name, "_quantized_weight");
    orig_weight_quant->replaceAllUsesWith(orig_weight);
    nodes_to_delete.insert(orig_weight_quant->node());
  }

  for (Node* n : nodes_to_delete) {
    n->destroy();
  }
}

void InsertPrepackUnpack(std::shared_ptr<Graph>& graph) {
  insertPrepackUnpackForLinear(graph);
  insertPrepackUnpackForConv2d(graph);
}

void InsertPrepackUnpack(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    InsertPrepackUnpack(graph);
  }
  for (script::Module m : module.children()) {
    InsertPrepackUnpack(m);
  }
}

struct FoldPrepackedWeightIntoModuleHelper {
  void run(
      script::Module& module,
      const std::string& method_name,
      const script::Module& linear_params_module,
      const script::Module& conv_params_module) {
    auto method = module.get_method(method_name);
    auto graph = method.graph();
    GRAPH_DUMP("Before FoldPrepackWeightIntoModule: ", graph);

    // (is_conv, is_per_channel, pattern, packed_params_module)
    std::vector<PatternsAndModules> pattern_and_modules = {
        {false, false, linear_prepack_per_tensor, linear_params_module},
        {false, true, linear_prepack_per_channel, linear_params_module},
        {true, false, conv2d_prepack, conv_params_module},
        {true, true, conv2d_prepack_per_channel, conv_params_module}};
    for (const auto& pm : pattern_and_modules) {
      const Graph& pattern_graph = *pm.pattern.pattern_graph;
      const auto& vmap = pm.pattern.vmap;
      const auto& matches = findPatternMatches(pattern_graph, *graph);
      TORCH_INTERNAL_ASSERT(
          matches.size() <= 1, "We only support at most one match right now");
      for (const auto& match : matches) {
        const auto& match_vmap = match.values_map;
        auto w_dtype_opt = getIValue("w_dtype", match_vmap, vmap);
        auto w_scale_opt = getIValue("w_scale", match_vmap, vmap);
        auto w_zero_point_opt = getIValue("w_zero_point", match_vmap, vmap);
        if (!w_dtype_opt || !w_scale_opt || !w_zero_point_opt) {
          GRAPH_DEBUG(
              "dtype, scale or zero_point for weight(",
              getValue("w_dtype", match_vmap, vmap)->debugName(),
              ", ",
              getValue("w_scale", match_vmap, vmap)->debugName(),
              ", ",
              getValue("w_zero_point", match_vmap, vmap)->debugName(),
              ") is not constant, skipping the match.");
          continue;
        }
        auto w_dtype = w_dtype_opt.value().toScalarType();
        auto w = module.attr("weight").toTensor().data();
        at::Tensor w_quant;
        if (pm.is_per_channel) {
          auto w_axis_opt = getIValue("w_axis", match_vmap, vmap);
          if (!w_axis_opt) {
            GRAPH_DEBUG(
                "axis for weight ",
                getValue("w_axis", match_vmap, vmap)->debugName(),
                " is non-constant, skipping the match");
            continue;
          }
          auto w_scale = w_scale_opt.value().toTensor().to(at::kFloat);
          auto w_zero_point = w_zero_point_opt.value().toTensor().to(at::kInt);
          int w_axis = w_axis_opt.value().toInt();
          TORCH_CHECK(
              w_scale.sizes() == w_zero_point.sizes(),
              "scale and zero_point must have the same size");
          w_quant = at::quantize_per_channel(
              w, w_scale, w_zero_point, w_axis, w_dtype);
        } else {
          auto w_scale = w_scale_opt.value().toDouble();
          auto w_zero_point = w_zero_point_opt.value().toInt();
          w_quant = at::quantize_per_tensor(w, w_scale, w_zero_point, w_dtype);
        }
        c10::optional<at::Tensor> b = c10::nullopt;
        if (hastensor(module, "bias")) {
          b = module.attr("bias").toTensor().data();
        }
        script::Module wrapper_module = pm.packed_params_module.clone();
        auto set_weight_bias = wrapper_module.get_method("set_weight_bias");
        std::string module_name_prefix;
        if (pm.is_conv) {
          module_name_prefix = "_conv_packed_params_module_for_";
          auto stride_opt =
              toTwoElementIntList(getValue("stride", match_vmap, vmap));
          auto padding_opt =
              toTwoElementIntList(getValue("padding", match_vmap, vmap));
          auto dilation_opt =
              toTwoElementIntList(getValue("dilation", match_vmap, vmap));
          auto groups_opt = getIValue("groups", match_vmap, vmap);
          auto set_conv_params = wrapper_module.get_method("set_conv_params");
          if (!stride_opt || !padding_opt || !dilation_opt) {
            GRAPH_DEBUG(
                "Failed to extract two element IntList for stride/padding/dilation, (",
                getValue("stride", match_vmap, vmap)->debugName(),
                ", ",
                getValue("padding", match_vmap, vmap)->debugName(),
                ", ",
                getValue("dilation", match_vmap, vmap)->debugName(),
                ") skipping the match");
            continue;
          }
          set_conv_params(std::vector<IValue>{stride_opt.value(),
                                              padding_opt.value(),
                                              dilation_opt.value(),
                                              groups_opt.value()});
        } else {
          module_name_prefix = "_linear_packed_params_module_for_";
        }
        set_weight_bias(std::vector<IValue>{IValue(w_quant), IValue(b)});
        auto w_quant_val = getValue("w_quant", match_vmap, vmap);
        // unique name for the module based on %w_quant
        int uid = 0;
        auto module_name = module_name_prefix + c10::to_string(uid++);
        while (module.hasattr(module_name)) {
          module_name_prefix + c10::to_string(uid++);
        }
        GRAPH_UPDATE("Adding new module: ", module_name);
        module.register_module(module_name, wrapper_module);

        // Add GetAttr of the packed module
        auto packed_params_val = getValue("packed_params", match_vmap, vmap);
        WithInsertPoint ins(packed_params_val->node());
        // wrapper_module =
        // self.{_conv,_linear}_packed_params_module_for_{unique_id}
        Value* packed_params_module =
            graph->insertGetAttr(graph->inputs()[0], module_name)
                ->setType(wrapper_module.type());
        GRAPH_UPDATE("Adding GetAttr node for the wrapper module");

        // packed_params = wrapper_module._packed_params
        Value* packed_params_from_attr =
            graph->insertGetAttr(packed_params_module, "_packed_params");
        GRAPH_UPDATE(
            "Adding GetAttr node for _packed_params: ",
            packed_params_from_attr->debugName());
        packed_params_val->replaceAllUsesWith(packed_params_from_attr);

        // Delete nodes
        std::vector<Node*> nodes_to_delete = {w_quant_val->node(),
                                              packed_params_val->node()};
        for (auto n : nodes_to_delete) {
          n->removeAllInputs();
        }
        for (auto n : nodes_to_delete) {
          GRAPH_UPDATE("Deleting node: ", n);
          n->destroy();
        }
      }
    }
  }

  void run(
      script::Module& module,
      const script::Module& linear_params_module,
      const script::Module& conv_params_module) {
    for (auto& method : module.get_methods()) {
      run(module, method.name(), linear_params_module, conv_params_module);
    }
    for (script::Module m : module.children()) {
      run(m, linear_params_module, conv_params_module);
    }
  }

  const PatternInfo linear_prepack_per_tensor = PatternInfo::parse_from_str(R"(
graph(%a_dequant, %w, %b, %w_scale, %w_zero_point, %w_dtype):
        %w_quant = aten::quantize_per_tensor(%w, %w_scale, %w_zero_point, %w_dtype)
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        return (%packed_params) )");

  const PatternInfo linear_prepack_per_channel = PatternInfo::parse_from_str(R"(
graph(%a_dequant, %w, %b, %w_scale, %w_zero_point, %w_axis, %w_dtype):
        %w_quant = aten::quantize_per_channel(%w, %w_scale, %w_zero_point, %w_axis, %w_dtype)
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        return (%packed_params) )");

  const PatternInfo conv2d_prepack = PatternInfo::parse_from_str(R"(
graph(%a_dequant, %w, %b, %w_scale, %w_zero_point, %w_dtype, %stride, %padding, %dilation, %groups):
        %w_quant = aten::quantize_per_tensor(%w, %w_scale, %w_zero_point, %w_dtype)
        %packed_params = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        return (%packed_params) )");

  const PatternInfo conv2d_prepack_per_channel = PatternInfo::parse_from_str(R"(
graph(%a_dequant, %w, %b, %w_scale, %w_zero_point, %w_axis, %w_dtype, %stride, %padding, %dilation, %groups):
        %w_quant = aten::quantize_per_channel(%w, %w_scale, %w_zero_point, %w_axis, %w_dtype)
        %packed_params = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        return (%packed_params) )");
};

void FoldPrepackedWeightIntoModule(
    script::Module& module,
    const script::Module& linear_params_module,
    const script::Module& conv_params_module) {
  FoldPrepackedWeightIntoModuleHelper h;
  h.run(module, linear_params_module, conv_params_module);
}

void DedupModuleUses(script::Module& module) {
  ModuleUseDeduper d(module);
  d.dedup();
}

script::Module Finalize(script::Module& module) {
  SwapFunctionalLinear(module);
  auto graph = module.get_method("forward").graph();
  Inline(*graph);
  ReplicateDeQuant(graph);
  SwapDeQuant(graph);
  InsertPrepackUnpack(graph);
  ConstantPropagation(graph);
  QuantFusion(graph);
  return freeze_module(module);
}

} // namespace jit
} // namespace torch
