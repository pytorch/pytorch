#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/quantization_patterns.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/subgraph_matcher.h>

#include <algorithm>
#include <stack>

namespace torch {
namespace jit {
namespace {

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
    PatternInfo rv{std::move(pattern_string),
                   at::guts::make_unique<Graph>(),
                   decltype(vmap){}};
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

bool nodeQuantizable(Node* n) {
  static std::vector<std::string> call_funcs = {
      "conv2d",
      "linear",
      "relu",
  };
  std::vector<Symbol> aten_funcs = {
      Symbol::aten("addmm"), Symbol::aten("matmul"), Symbol::aten("add_")};
  std::transform(
      call_funcs.begin(),
      call_funcs.end(),
      std::back_inserter(aten_funcs),
      [](const std::string& s) { return Symbol::aten(s); });
  bool is_quantizable =
      std::find(aten_funcs.begin(), aten_funcs.end(), n->kind()) !=
      aten_funcs.end();
  if (n->kind() == prim::CallFunction) {
    auto func_name = getFuncName(n->inputs()[0]);
    is_quantizable |=
        std::find(call_funcs.begin(), call_funcs.end(), func_name) !=
        call_funcs.end();
  }
  return is_quantizable;
}

bool valueNeedsToBeQuantized(Value* v) {
  if (!v->type()->isSubtypeOf(TensorType::get())) {
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

class InsertObserversHelper {
 public:
  explicit InsertObserversHelper(const ModuleQConfigMap& map)
      : module_qconfig_map_(map) {}
  void insertObservers(script::Module& module, const std::string& method_name);

 private:
  Node* insertObserverFor(
      Value* v,
      Graph* g,
      script::Module& module,
      const QConfig& qconfig);

  void findIntermediateValuesInPattern(
      Graph& graph,
      const PatternInfo& pattern);

  void addIntermediateValuesToSkipObserver(
      const script::Module& module,
      const std::string& method_name);

  const ModuleQConfigMap& module_qconfig_map_;
  // Values we want to skip observing, used to skip values in
  // the middle of the ops that are supposed to be fused, e.g.
  // the output value of conv in the conv - relu pattern
  std::unordered_set<Value*> values_to_skip_;
  // Unique id generator for observer module, used for generating
  // unique observer names when we insert observer module, we
  // record the current unique id used to avoid incrementing from 0
  // every time to find a unique id.
  int uid_ = 0;

  // These are the IR patterns we match to skip inserting observers.
  // They are compiled once on construction and used repeatedly within
  // the pass.
  const PatternInfo conv_functional_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %conv = match::module[name="Conv2d"](%self)
    %intermediate_val = prim::CallMethod[name="forward"](%conv, %input)
    %r = prim::CallFunction(%relu, %intermediate_val, %inplace)
    return (%r) )");
  const PatternInfo conv_relu_module = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %conv = match::module[name="Conv2d"](%self)
    %intermediate_val = prim::CallMethod[name="forward"](%conv, %input)
    %relu = match::module[name="ReLU"](%self)
    %r = prim::CallMethod[name="forward"](%relu, %intermediate_val)
    return (%r) )");
  const PatternInfo matmul_add = PatternInfo::parse_from_str(R"(
graph(%input, %weight, %bias, %4):
     %weight_t = aten::t(%weight)
     %intermediate_val = aten::matmul(%input, %weight_t)
     %res = aten::add_(%intermediate_val, %bias, %4)
     return (%res) )");
  const std::vector<std::reference_wrapper<const PatternInfo>> patterns = {
      conv_functional_relu,
      conv_relu_module,
      matmul_add};
};

bool isBiasOfConvOrLinear(Value* v) {
  for (const Use& u : v->uses()) {
    if (u.user->kind() == Symbol::aten("conv2d")) {
      if (v == u.user->inputs().at(2)) {
        return true;
      }
    } else if (u.user->kind() == prim::CallFunction) {
      auto func_name = getFuncName(u.user->inputs()[0]);
      if (func_name == "linear" && v == u.user->inputs().at(3)) {
        return true;
      }
    }
  }
  return false;
}

bool isWeightOfConvOrLinear(Value* v) {
  for (const Use& u : v->uses()) {
    if (u.user->kind() == Symbol::aten("conv2d") &&
        v == u.user->inputs().at(1)) {
      return true;
    } else if (
        u.user->kind() == prim::CallFunction &&
        getFuncName(u.user->inputs()[0]) == "linear" &&
        v == u.user->inputs().at(2)) {
      return true;
    }
  }
  return false;
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

// Clone observer module and add it to the original module,
// and insert a call to observer forward function
Node* InsertObserversHelper::insertObserverFor(
    Value* v,
    Graph* g,
    script::Module& module,
    const QConfig& qconfig) {
  // Skip observing bias
  if (isBiasOfConvOrLinear(v)) {
    return nullptr;
  }

  script::Module observer_module;
  if (isWeightOfConvOrLinear(v)) {
    TORCH_CHECK(
        v->uses().size() == 1,
        "We only support weight being used by one node.");
    observer_module = std::get<1>(qconfig);
  } else {
    observer_module = std::get<0>(qconfig);
  }

  script::Module observer = observer_module.clone();
  std::string observer_name = "_observer_" + c10::to_string(uid_++);
  while (module.hasattr(observer_name)) {
    observer_name = "_observer_" + c10::to_string(uid_++);
  }
  module.register_module(observer_name, observer);

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
    return call;
  }
}

void InsertObserversHelper::findIntermediateValuesInPattern(
    Graph& graph,
    const PatternInfo& pattern) {
  const Graph& pattern_graph = *pattern.pattern_graph;
  const std::unordered_map<std::string, Value*>& vmap = pattern.vmap;

  const auto& matches = findPatternMatches(pattern_graph, graph);
  for (const auto& match : matches) {
    auto output_value = vmap.at("intermediate_val");
    TORCH_INTERNAL_ASSERT(
        match.values_map.find(output_value) != match.values_map.end(),
        "Didn't find Value output in match result.");
    values_to_skip_.emplace(match.values_map.at(output_value));
  }
}

void InsertObserversHelper::addIntermediateValuesToSkipObserver(
    const script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

  for (const auto& pattern : patterns) {
    findIntermediateValuesInPattern(*graph, pattern);
  }
}

void InsertObserversHelper::insertObservers(
    script::Module& module,
    const std::string& method_name) {
  if (!module_qconfig_map_.count(module._ivalue())) {
    // the module is added by us, e.g.: observer module
    return;
  }

  script::Method method = module.get_method(method_name);
  auto graph = method.graph();
  ConstantPropagation(graph);
  // must do constant propagation first before replacement
  replaceConvolutionWithConv2d(graph);
  addIntermediateValuesToSkipObserver(module, method_name);
  // For storing all values that need to be instrumented with an observer call.
  std::vector<Value*> values_to_observe;

  // For traversing all blocks in the graph including subblocks.
  std::stack<Block*> blocks_to_visit;

  // Mark observer nodes for inputs so we dont add observers
  // for observers while traversing graph
  std::unordered_set<Node*> observer_for_input;

  // Add observer for external input nodes excluding parameters
  // These are treated as activation as they vary across batches
  // and need to be observed.

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  for (size_t idx = 1; idx < method.num_inputs(); ++idx) {
    auto& v = graph->inputs()[idx];
    if (!values_to_skip_.count(v) && valueNeedsToBeQuantized(v)) {
      auto qconfig = module_qconfig_map_.at(module._ivalue());
      if (qconfig) {
        auto observer_node =
            insertObserverFor(v, v->owningGraph(), module, qconfig.value());
        if (observer_node) {
          observer_for_input.emplace(observer_node);
        }
      }
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip observer nodes
      if (observer_for_input.count(n) != 0) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        if (!values_to_skip_.count(v) && valueNeedsToBeQuantized(v)) {
          values_to_observe.push_back(v);
        }
      }

      if (n->kind() == prim::CallMethod) {
        // If we find a call to a method of a child module,
        // we'll recursively insert observers for the forward function to
        // the child module.
        auto module_instance = n->inputs()[0];
        auto module_method_name = n->s(attr::name);
        script::Module callee_module;
        if (module_instance->node()->kind() == prim::GetAttr) {
          auto child_module_name = module_instance->node()->s(attr::name);
          callee_module = module.attr(child_module_name).toModule();
        } else {
          TORCH_INTERNAL_ASSERT(
              module_instance == graph->inputs()[0],
              "We only support call method either on %self"
              "or child instance in insert_observers_pass right now");
          callee_module = module;
        }
        auto method_graph =
            callee_module.get_method(module_method_name).graph();
        // Recursively insert observer for the forward function of child
        // module
        insertObservers(callee_module, module_method_name);
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // Actually add observer nodes.
  for (Value* v : values_to_observe) {
    auto qconfig = module_qconfig_map_.at(module._ivalue());
    // Skip inserting observer if no qconfig is specified
    if (qconfig) {
      insertObserverFor(v, v->owningGraph(), module, qconfig.value());
    }
  }
}

Node* insertQuantDeQuantCall(
    Value* v,
    const IValue& qparams,
    const IValue& scalar_type) {
  Graph* g = v->node()->owningGraph();
  auto tp = qparams.toTuple();
  at::Tensor scale = tp->elements()[0].toTensor().to(at::kFloat);
  at::Tensor zero_point = tp->elements()[1].toTensor().to(at::kInt);

  bool is_per_channel = scale.numel() > 1;
  std::string quantize_func;
  std::vector<Value*> inputs = {v};

  // Inserting before insert point
  WithInsertPoint ins(v->node()->next());
  if (is_per_channel) {
    quantize_func = "quantize_per_channel";
    inputs.push_back(g->insertConstant(scale));
    inputs.push_back(g->insertConstant(zero_point));
    inputs.push_back(g->insertConstant(tp->elements()[2].toInt()));
  } else {
    quantize_func = "quantize_per_tensor";
    inputs.push_back(g->insertConstant(scale.item<double>()));
    inputs.push_back(g->insertConstant(zero_point.item<int64_t>()));
  }
  Value* scalar_type_val = g->insertConstant(IValue(scalar_type));
  inputs.push_back(scalar_type_val);

  Node* quant = g->create(at::Symbol::aten(quantize_func), inputs);
  quant->output()->setDebugName(v->debugName() + ".quant");

  Node* dequant = g->create(at::Symbol::aten("dequantize"), {quant->output()});
  dequant->output()->setDebugName(v->debugName() + ".dequant");

  g->insertNode(quant);
  g->insertNode(dequant);

  return dequant;
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

class QuantizeHelper {
 public:
  QuantizeHelper(script::Module& m) : module_(m) {}
  // quantization parameters and scalar type
  std::tuple<IValue, IValue> getQParams(Value* v);
  c10::optional<script::Module> findChildModuleToQuantize(
      Value* child_instance);
  void collectObserverNodesAndValueToQuantize(Value*);
  void removeObservers();
  void quantizeTensors();

 private:
  script::Module& module_;
  std::vector<std::string> observer_modules_to_remove_;
  std::vector<Node*> nodes_to_destroy_;
  std::vector<Value*> values_to_quantize_;
  std::unordered_map<Value*, std::tuple<IValue, IValue> > values_to_qparams_;
};


void QuantizeHelper::collectObserverNodesAndValueToQuantize(Value* v) {
  auto observer_name = findObserverName(v);
  if (!observer_name) {
    return;
  }
  observer_modules_to_remove_.push_back(observer_name.value());

  Node* observer = v->node();
  TORCH_INTERNAL_ASSERT(
      observer->kind() == prim::CallMethod &&
      observer->s(attr::name) == "forward" &&
      observer->inputs()[0]->node()->kind() == prim::GetAttr &&
      observer->inputs()[0]->node()->s(attr::name) == observer_name);

  // Observer forward call node
  nodes_to_destroy_.push_back(observer);
  // GetAttr node for observer module
  nodes_to_destroy_.push_back(observer->inputs()[0]->node());
  Value* new_value = observer->input(1);
  v->replaceAllUsesWith(new_value);
  values_to_quantize_.push_back(new_value);
  values_to_qparams_.insert({new_value, getQParams(v)});
}

void QuantizeHelper::removeObservers() {
  for (auto& n : nodes_to_destroy_) {
    n->removeAllInputs();
  }
  for (auto& n : nodes_to_destroy_) {
    n->destroy();
  }
  // Remove observer modules from last one to first one in order to
  // reduce the time complexity, assuming all the observer modules
  // are added after the existing modules, we'll have complexity of
  // O(N) where N is number of observer moduels with this optimization
  for (int64_t i = observer_modules_to_remove_.size() - 1; i >= 0; --i) {
    auto observer_name = observer_modules_to_remove_[i];
    module_._ivalue()->unsafeRemoveAttr(observer_name);
    module_.type()->unsafeRemoveAttribute(observer_name);
  }
}

void QuantizeHelper::quantizeTensors() {
  for (auto& v : values_to_quantize_) {
    TORCH_INTERNAL_ASSERT(values_to_qparams_.count(v));
    auto tp = values_to_qparams_[v];
    auto qparams = std::get<0>(tp);
    auto scalar_type = std::get<1>(tp);
    // NB: v is updated here, since removeObserver replaces
    // v with the input to the observer call
    Node* dequant;
    dequant = insertQuantDeQuantCall(v, qparams, scalar_type);
    v->replaceAllUsesWith(dequant->output());
    Node* q = dequant->input(0)->node();
    // replaceAllUsesWith rewrote all uses of V, but we want to keep one: the one
    // used in quant node. Restore it here:
    q->replaceInputWith(dequant->output(), v);
  }
  // no need to clear the vector or map
}

void checkCalculateQParamsResult(const IValue& qparams) {
  TORCH_CHECK(
      qparams.isTuple(),
      "`calculate_qparams` function is expected to return a "
      "Tuple, but got:",
      qparams.tagKind());
  auto tp = qparams.toTuple();
  TORCH_CHECK(
      tp->elements().size() == 2 || tp->elements().size() == 3,
      "`calculate_qparams` function is expected to reutrn a "
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

std::tuple<IValue, IValue> QuantizeHelper::getQParams(Value* v) {
  TORCH_INTERNAL_ASSERT(v->type()->isSubtypeOf(TensorType::get()));
  auto observer_name = findObserverName(v);
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQParams expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  auto om = module_.attr(observer_name.value()).toModule();
  auto calculate_qparams = om.get_method("calculate_qparams");
  IValue qparams = calculate_qparams(std::vector<IValue>());
  checkCalculateQParamsResult(qparams);
  auto scalar_type = om.attr("dtype");
  TORCH_CHECK(
      scalar_type.toScalarType() != at::ScalarType::Undefined,
      "dtype of observer can't be undefined");
  return std::make_tuple(qparams, scalar_type);
}

c10::optional<script::Module> QuantizeHelper::findChildModuleToQuantize(
    Value* child_instance) {
  TORCH_INTERNAL_ASSERT(
      child_instance->node()->kind() == prim::GetAttr,
      "Child instance should come from GetAttr.");
  auto child_module_name = child_instance->node()->s(attr::name);
  if (child_module_name.find("_observer_") == std::string::npos) {
    return module_.attr(child_module_name).toModule();
  }
  return c10::nullopt;
}

void InsertQuantDeQuantImpl(
    script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();

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

  QuantizeHelper qh(module);
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
        if (v->node()->kind() == prim::CallMethod) {
          auto module_instance = v->node()->inputs()[0];
          auto module_method_name = v->node()->s(attr::name);
          c10::optional<script::Module> m;
          // calling method on self
          if (module_instance == graph->inputs()[0]) {
            m = module;
          } else {
            m = qh.findChildModuleToQuantize(module_instance);
          }
          if (m) {
            InsertQuantDeQuantImpl(m.value(), module_method_name);
          }
        }
        qh.collectObserverNodesAndValueToQuantize(v);
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  for (Value* v : input_values) {
    qh.collectObserverNodesAndValueToQuantize(v);
  }
  qh.removeObservers();
  qh.quantizeTensors();
}

void insertPrepackUnpackForLinear(std::shared_ptr<Graph>& graph) {
  std::string linear_with_quant = R"(
graph(%linear, %a_dequant, %w_quant, %b):
        %w_dequant = aten::dequantize(%w_quant)
        %r = prim::CallFunction(%linear, %a_dequant, %w_dequant, %b)
        return (%r) )";

  std::string linear_with_quant_prepack = R"(
graph(%linear, %a_dequant, %w_quant, %b):
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = prim::CallFunction(%linear, %a_dequant, %w_dequant, %b)
        return (%r) )";

  // Filter to match linear CallFunction
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear_value = match_vmap.at(vmap.at("linear"));
    auto func_name = getFuncName(linear_value);
    if (func_name == "linear") {
      return true;
    }
    return false;
  };

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(linear_with_quant, linear_with_quant_prepack);
  rewriter.runOnGraph(graph, filter);
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
} // namespace

TORCH_API script::Module InsertObservers(
    script::Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace) {
  script::Module module = inplace ? input_module : input_module.clone();
  ModuleQConfigMap module_qconfig_map;
  fillQConfigMap(module, qconfig_dict, module_qconfig_map);
  InsertObserversHelper helper(module_qconfig_map);
  helper.insertObservers(module, method_name);
  return module;
}

script::Module InsertQuantDeQuant(
    script::Module& input_module,
    const std::string& method_name,
    bool inplace) {
  script::Module module = inplace ? input_module : input_module.clone();
  InsertQuantDeQuantImpl(module, method_name);

  // NOTE: Remove observer module does not work right now, we'll return
  // the module with observer modules as a temporary workaround
  // TODO: remove observer modules after we have a remove_module API
  return module;
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void QuantFusion(std::shared_ptr<Graph>& graph) {
  for (const auto& item : quant_fusion_pattern_and_replacements()) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(item.first, item.second);
    rewriter.runOnGraph(graph);
  }
}

struct ConvBNParameters {
  at::Tensor conv_w;
  at::Tensor conv_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Conv2d module and parameters
 * of the BatchNorm2d module we're folding with, compute the updated values for
 * the weight and bias.
 *
 * The function is basically copied from torch/nn/utils/fusion.py
 */
static std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p) {
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape({-1, 1, 1, 1});
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  return std::make_tuple(new_w, new_b);
}

static bool hastensor(script::Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTensor();
}

static bool tryExtractingConvBNParameters(
    script::Module& conv,
    script::Module& bn,
    ConvBNParameters& r) {
  if (!hastensor(conv, "weight") || !hastensor(bn, "weight") ||
      !hastensor(bn, "bias") || !hastensor(bn, "running_mean") ||
      !hastensor(bn, "running_var")) {
    return false;
  }

  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();
  r.bn_eps = 1e-5; // TODO: allow access to the actual value. NOLINT
                   // Now we cannot do it because we inline all fields that are
                   // in __constants__ and lose all tracks of them.
  r.bn_w = bn.attr("weight").toTensor();
  r.bn_b = bn.attr("bias").toTensor();

  r.conv_w = conv.attr("weight").toTensor();
  if (conv.hasattr("bias")) {
    r.conv_b = conv.attr("bias").toTensor();
  } else {
    r.conv_b = at::zeros_like(r.bn_rm);
  }

  return true;
}

void FoldConvBatchNorm2d(const script::Module& module) {
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

    // Process forward method of the current module
    std::unordered_map<Value*, Value*> rewrite_map;
    std::vector<Value*> values_to_rewrite;
    std::unordered_set<Node*> nodes_to_delete;

    script::Method method = current.get_method("forward");
    GRAPH_DUMP(
        current.type()->name()->name() +
            "::forward() before Conv2d-BatchNorm2d folding",
        method.graph());
    const auto& matches = findPatternMatches(pattern_graph, *method.graph());

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

      script::Module conv_submodule =
          current.attr(matched_conv_submodule->s(Symbol::attr("name")))
              .toModule();
      script::Module bn_submodule =
          current.attr(matched_bn_submodule->s(Symbol::attr("name")))
              .toModule();

      ConvBNParameters params;
      if (!tryExtractingConvBNParameters(
              conv_submodule, bn_submodule, params)) {
        GRAPH_DEBUG(
            "Conv and BN modules didn't have all required parameters or attributes...");
        continue;
      }

      // We are using a separate vector for saving Values we want to rewrite to
      // make sure that the order in which we perform these transformations is
      // deterministic. Iterating through keys of rewrite_map would result in
      // non-determinism that might not manifest as a bug now, but can bite us
      // later.
      values_to_rewrite.push_back(matched_bn->output());
      rewrite_map[matched_bn->output()] = matched_conv->output();
      GRAPH_UPDATE(
          "Rewriting %",
          matched_bn->output()->debugName(),
          " with %",
          matched_conv->output()->debugName());

      nodes_to_delete.insert(matched_bn);
      GRAPH_UPDATE("Deleting ", *matched_bn);

      auto new_w_b = computeUpdatedConvWeightAndBias(params);
      conv_submodule.setattr("weight", std::get<0>(new_w_b));
      if (conv_submodule.hasattr("bias")) {
        conv_submodule.setattr("bias", std::get<1>(new_w_b));
      } else {
        conv_submodule.register_parameter("bias", std::get<1>(new_w_b), false);
      }
    }

    // Perform planned rewritings
    for (auto v : values_to_rewrite) {
      v->replaceAllUsesWith(rewrite_map.at(v));
    }

    // Perform planned deletions
    for (auto n : nodes_to_delete) {
      n->removeAllInputs();
    }
    for (auto n : nodes_to_delete) {
      n->destroy();
    }
  }
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
        return (%packed_params))");

  const PatternInfo conv2d_prepack_per_channel = PatternInfo::parse_from_str(R"(
graph(%a_dequant, %w, %b, %w_scale, %w_zero_point, %w_axis, %w_dtype, %stride, %padding, %dilation, %groups):
        %w_quant = aten::quantize_per_channel(%w, %w_scale, %w_zero_point, %w_axis, %w_dtype)
        %packed_params = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        return (%packed_params))");
};

void FoldPrepackedWeightIntoModule(
    script::Module& module,
    const script::Module& linear_params_module,
    const script::Module& conv_params_module) {
  FoldPrepackedWeightIntoModuleHelper h;
  h.run(module, linear_params_module, conv_params_module);
}
} // namespace jit
} // namespace torch
