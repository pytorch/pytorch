#include <torch/csrc/jit/passes/quantization/insert_observers.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>

namespace torch {
namespace jit {

using ModuleQConfigMap = std::unordered_map<ModulePtr, c10::optional<QConfig>>;

namespace {

struct OptionalQConfigHash {
  inline size_t operator()(const c10::optional<QConfig>& qconfig_opt) const {
    if (qconfig_opt.has_value()) {
      const auto& m1 = std::get<0>(*qconfig_opt);
      const auto& m2 = std::get<1>(*qconfig_opt);
      constexpr int CONST = 7;
      return std::hash<Module>()(m1) + CONST * std::hash<Module>()(m2);
    }
    return 0;
  }
};
using QConfigTypePtrMap =
    std::unordered_map<c10::optional<QConfig>, TypePtr, OptionalQConfigHash>;
using NameModuleVector = std::vector<std::pair<std::string, Module>>;
using OptionalModuleVector = std::vector<c10::optional<Module>>;
using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;
using graph_rewrite_helper::PatternInfo;
using graph_rewrite_helper::replaceConvolutionWithAtenConv;

// helper functions
void fillQConfigMap(
    const Module& module,
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

  for (const NameModule& s : module.named_children()) {
    std::string child_key;
    if (key == "") {
      child_key = s.name;
    } else {
      child_key = key + "." + s.name;
    }
    fillQConfigMap(s.value._ivalue(), qconfig_dict, map, child_key, qconfig);
  }
}

Module getObserverModuleFor(Value* v, const QConfig& qconfig) {
  return isWeight(v) ? std::get<1>(qconfig) : std::get<0>(qconfig);
}

// helper classes
class ModuleCloneHelper {
 public:
  /** Clone according to module qconfig map, this is for handling the case
   *  where we have two module instances sharing the same ClassType
   *  but configured with different QConfig
   *  code is copied and modified from
   * https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp
   */
  Module clone(
      const Module& module,
      const ModuleQConfigMap& module_qconfig_map) {
    std::unordered_map<TypePtr, QConfigTypePtrMap> type_remap;
    return clone_impl(module, module_qconfig_map, type_remap);
  }

 private:
  Module clone_impl(
      const Module& module,
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
    Module r;
    if (type_already_cloned) {
      // if we cloned the class type before, we'll reuse it
      Module new_module(
          module._ivalue()->compilation_unit(),
          type_remap.at(type).at(qconfig)->cast<ClassType>());
      r = new_module;
    } else {
      Module new_module(
          *type->name(), module._ivalue()->compilation_unit(), true);
      r = new_module;
      type_remap[type][module_qconfig_map.at(module._ivalue())] = r.type();
    }
    // Copy slots. If a slot is a module - recursively clone it.
    size_t N = type->numAttributes();
    for (size_t i = 0; i < N; ++i) {
      IValue s = module._ivalue()->getSlot(i);
      if (type->getAttribute(i)->is_module()) {
        const Module& orig = Module(s.toObject());
        Module cloned = clone_impl(orig, module_qconfig_map, type_remap);
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
      const Module& source,
      Module& target,
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
      const Module& source,
      Module& target,
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
      const Module& source,
      Module& target,
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

  void preprocess(Module& module, const std::string& method_name);

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
   * Since we want to insert observers in the call site instead of in the called
   * graph, we'll postpone inserting observer to caller as much as possible, if
   * we know the current method is the outer most method, then
   * we will insert all observers in the graph instead of postpone this to the
   * parent, note that this assumes we don't have recursive method
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
  insertObservers(
      Module& module,
      const std::string& method_name,
      bool is_entry_point = false,
      std::unordered_set<Value*> graph_observed_values =
          std::unordered_set<Value*>());

  void setDynamicFlag(bool is_dynamic) {
    is_dynamic_ = is_dynamic;
  }

 private:
  std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
  insertObserversFor(
      Block* block,
      script::Module& module,
      // this is a reference because when we insert observer for a value
      // in one block it is also observed in another block, we don't want to
      // insert multiple observers for the same value
      std::unordered_set<Value*>& block_observed_values,
      bool is_entry_point = false,
      bool is_user_defined_function = false);

  void recordObserved(
      Value* v,
      const Module& observer_module,
      std::unordered_map<Value*, Module>& values_to_observe,
      std::unordered_set<Value*>& block_observed_values);

  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  bool valueNeedsToBeQuantized(Value* v);

  bool isObserved(
      Value* v,
      const std::unordered_set<Value*>& block_observed_values) {
    return block_observed_values.count(v) || observed_values_.count(v);
  }

  // Fill the map between the caller input/output to input/output
  // of called graph, this is used to navigate through the graph
  // to find the observer for a given value
  void fillBoundaryValueMap(Module& module, const std::string& method_name);

  // Fill the map from value to the corresponding observer module
  // this map is used in insertObservers to actually insert
  // observers to the module
  void fillValueObserverMap(Module& module, const std::string& method_name);

  // Clone observer module and add it to the original module,
  // and insert a call to observer forward function
  void insertObserverFor(
      Value* v,
      Module& module,
      const Module& observer_module,
      NameModuleVector& observer_name_and_modules);

  c10::optional<Module> getObserverFor(Value* v);

  void propagateObservedProperty(
      Value* output,
      std::unordered_set<Value*>& block_observed_values);

  bool shouldPropagateQuant(
      Node* n, const std::unordered_set<Value*>& block_observed_values) {
    return isObserved(n->input(0), block_observed_values);
  }


  void delayObservingValuesInPattern(Graph& graph, const PatternInfo& pattern);

  void addValuesToDelayObservation(
      const Module& module,
      const std::string& method_name);

  // Fill the map from values to the list of values that can pass the observed
  // property to it
  void fillPassThroughValueMap(const std::shared_ptr<Graph>& graph);

  const ModuleQConfigMap& module_qconfig_map_;
  // Values we want to delay observation, used to delay the observation for
  // values in the middle of the ops that are supposed to be fused, e.g.
  // the output value of conv in the conv - relu pattern
  // the key is the intermediate output, e.g. output of conv
  // the value is the value we want to observe, e.g. output of relu
  std::unordered_map<Value*, Value*> delay_observation_map_;
  std::unordered_set<Graph*> visited_graph_of_observer_map_;
  std::unordered_map<Value*, Module> observer_for_value_;
  // Map from values from callsite into the values in the CallMethod graph
  // key of the map is the value from caller graph, and the value of the map
  // is the list of values in the callee graph (the graph
  // corresponding to the called method),
  // the reason it is a set is that a value in the caller graph
  // can both correspond to the output of one callee graph and input of another
  // callee graph.
  std::unordered_map<Value*, std::unordered_set<Value*>> boundary_value_map_;
  std::unordered_set<Value*> observed_values_;
  // This is used for the observed values to pass through the ops like flatten,
  // so that output value of flatten does not need to be observed
  // key is the output of the op, value is a vector of values that need
  // to be observed in order to pass the observed property to the output
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

  // Is dynamic quantization enabled for the observer pass.
  bool is_dynamic_ = false;
  // These are the IR patterns we match to skip inserting observers.
  // They are compiled once on construction and used repeatedly within
  // the pass.
  const PatternInfo nn_conv1d_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %first_module = match::module[name="Conv1d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )");
  const PatternInfo nn_conv1d_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %first_module = match::module[name="Conv1d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_module = match::module[name="ReLU"](%self)
    %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
    return (%second_output) )");
  const PatternInfo nn_conv2d_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %first_module = match::module[name="Conv2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )");
  const PatternInfo nn_conv2d_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %first_module = match::module[name="Conv2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_module = match::module[name="ReLU"](%self)
    %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
    return (%second_output) )");
  const PatternInfo nn_conv3d_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %first_module = match::module[name="Conv3d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )");
  const PatternInfo nn_conv3d_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %first_module = match::module[name="Conv3d"](%self)
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
  const PatternInfo aten_add_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %one = prim::Constant[value=1]()
     %first_output = aten::add_(%a, %b, %one)
     %second_module = match::module[name="ReLU"](%self)
     %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
     return (%second_output) )");

  const PatternInfo aten_add_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %inplace):
     %one = prim::Constant[value=1]()
     %first_output = aten::add_(%a, %b, %one)
     %relu = prim::Constant[name="relu"]()
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )");

  const PatternInfo nn_bn_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input):
    %first_module = match::module[name="BatchNorm2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_module = match::module[name="ReLU"](%self)
    %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
    return (%second_output) )");

  // TODO: Make fusion support BN+Functional Relu with inplace.
  const PatternInfo nn_bn_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %input, %inplace):
    %relu = prim::Constant[name="relu"]()
    %first_module = match::module[name="BatchNorm2d"](%self)
    %first_output = prim::CallMethod[name="forward"](%first_module, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )");

  const PatternInfo aten_mul_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul(%a, %b)
     %second_module = match::module[name="ReLU"](%self)
     %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
     return (%second_output) )");

  const PatternInfo inplace_mul_nn_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul_(%a, %b)
     %second_module = match::module[name="ReLU"](%self)
     %second_output = prim::CallMethod[name="forward"](%second_module, %first_output)
     return (%second_output) )");

  const PatternInfo aten_mul_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %inplace):
     %first_output = aten::mul(%a, %b)
     %relu = prim::Constant[name="relu"]()
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )");

  const PatternInfo inplace_mul_f_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %inplace):
     %first_output = aten::mul_(%a, %b)
     %relu = prim::Constant[name="relu"]()
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )");

  const std::vector<std::reference_wrapper<const PatternInfo>> delay_patterns =
      {
          nn_conv1d_f_relu,
          nn_conv1d_nn_relu,
          nn_conv2d_f_relu,
          nn_conv2d_nn_relu,
          nn_conv3d_f_relu,
          nn_conv3d_nn_relu,
          matmul_add,
          aten_add_nn_relu,
          aten_add_f_relu,
          nn_bn_nn_relu,
          nn_bn_f_relu,
          aten_mul_nn_relu,
          inplace_mul_nn_relu,
          aten_mul_f_relu,
          inplace_mul_f_relu,
  };
};

ModuleMethodVector InsertObserversHelper::getInvokedMethods(
    Module& module,
    const std::string& method_name) {
  ModuleMethodVector invoked_methods;
  Method method = module.get_method(method_name);
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
        invoked_methods.push_back(std::make_pair(
            getInvokedModule(module, n, graph->inputs()[0]), n->s(attr::name)));
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
    Module& module,
    const Module& observer_module,
    NameModuleVector& observer_name_and_modules) {
  if (observed_values_.count(v)) {
    return;
  }
  Module observer = observer_module.deepcopy();
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
    MatchedSchema forward_matched_schema = matchSchema(
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
    observed_values_.insert(call->output());
  }
}

void InsertObserversHelper::delayObservingValuesInPattern(
    Graph& graph,
    const PatternInfo& pattern) {
  const Graph& pattern_graph = *pattern.pattern_graph;
  const std::unordered_map<std::string, Value*>& vmap = pattern.vmap;

  const auto& matches = findPatternMatches(pattern_graph, graph);
  for (const auto& match : matches) {
    auto first_output = match.values_map.at(vmap.at("first_output"));
    auto second_output = match.values_map.at(vmap.at("second_output"));
    GRAPH_DEBUG(
        "Delay observation for value in function pattern:",
        first_output->debugName(),
        " to ",
        second_output->debugName());
    delay_observation_map_[first_output] = second_output;
  }
}

void InsertObserversHelper::addValuesToDelayObservation(
    const Module& module,
    const std::string& method_name) {
  Method method = module.get_method(method_name);
  auto graph = method.graph();

  for (const auto& pattern : delay_patterns) {
    delayObservingValuesInPattern(*graph, pattern);
  }
}

void InsertObserversHelper::fillPassThroughValueMap(
    const std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (userDefinedCallFunction(n)) {
        auto g = getCallFunctionGraph(n);
        blocks_to_visit.push(g->block());
      }
      for (auto* output : n->outputs()) {
        for (auto* input : getPassThroughInputs(output)) {
          pass_through_value_map_[output].push_back(input);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

void InsertObserversHelper::fillBoundaryValueMap(
    Module& module,
    const std::string& method_name) {
  auto graph = module.get_method(method_name).graph();
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  auto* self = graph->inputs()[0];
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod || userDefinedCallFunction(n)) {
        std::shared_ptr<Graph> g;
        // offset of input for the caller node, since the first
        // input of CallFunction is the function node and the graph
        // for CallFunction start with actual input
        size_t input_offset;
        if (n->kind() == prim::CallMethod) {
          auto m = getInvokedModule(module, n, self);
          g = m.get_method(n->s(attr::name)).graph();
          input_offset = 0;
        } else {
          g = getCallFunctionGraph(n);
          input_offset = 1;
        }
        // add mapping from callsite value to value in called graph
        for (auto i = 0U; i < g->outputs().size(); ++i) {
          auto* return_val = g->outputs()[i];
          boundary_value_map_[n->output(i)].insert(return_val);
        }
        for (auto i = 0U; i < g->inputs().size(); ++i) {
          auto caller_input_index = i + input_offset;
          auto* caller_input = n->input(caller_input_index);
          auto* input_val = g->inputs()[i];
          boundary_value_map_[caller_input].insert(input_val);
        }
      } else if (n->kind() == prim::If) {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
          for (Value* v : n->outputs()) {
            boundary_value_map_[v].insert(subblock->outputs()[v->offset()]);
          }
        }
      } else {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }
  }
}

void InsertObserversHelper::preprocess(
    Module& module,
    const std::string& method_name) {
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // TODO: remove constant prop, add separate graph
  // cleanup step before insert observers
  // To cleanup traced graph
  ConstantPooling(graph);
  ConstantPropagation(graph);
  // must do constant propagation first before replacement
  replaceConvolutionWithAtenConv(graph);
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);

  addValuesToDelayObservation(module, method_name);
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
  if (isBiasOfConvOrLinear(v) ||
      !(v->type()->isSubtypeOf(TensorType::get()) ||
        v->type()->isSubtypeOf(ListType::ofTensors()))) {
    return false;
  }
  // For dynamic quantization we only insert observers at the input
  // of the quantizable function.
  if (!is_dynamic_) {
    // Check whether producer is quantizable
    if ((mayRequireObservation(v) && nodeQuantizable(v->node())) ||
        isPropagateQuantNode(v->node())) {
      return true;
    }
  }
  // Check whether node input value is quantizable
  for (const auto& use : v->uses()) {
    if (useQuantizable(use, is_dynamic_)) {
      return true;
    }
  }
  return false;
}

void InsertObserversHelper::fillValueObserverMap(
    Module& module,
    const std::string& method_name) {
  Method method = module.get_method(method_name);
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

c10::optional<Module> InsertObserversHelper::getObserverFor(Value* v) {
  if (observer_for_value_.count(v)) {
    auto observer = observer_for_value_.at(v);
    return observer;
  }
  c10::optional<Module> result;
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

std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
InsertObserversHelper::insertObservers(
    Module& module,
    const std::string& method_name,
    bool is_entry_point,
    std::unordered_set<Value*> graph_observed_values) {
  auto graph = module.get_method(method_name).graph();
  return insertObserversFor(
      graph->block(), module, graph_observed_values, is_entry_point);
}

void InsertObserversHelper::recordObserved(
    Value* v,
    const Module& observer_module,
    std::unordered_map<Value*, Module>& values_to_observe,
    std::unordered_set<Value*>& block_observed_values) {
  Value* to_observe = v;
  if (delay_observation_map_.count(v)) {
    to_observe = delay_observation_map_.at(v);
  }
  values_to_observe[to_observe] = observer_module;
  block_observed_values.insert(to_observe);
}

std::tuple<OptionalModuleVector, OptionalModuleVector, std::vector<size_t>>
InsertObserversHelper::insertObserversFor(
    Block* block,
    script::Module& module,
    std::unordered_set<Value*>& block_observed_values,
    bool is_entry_point,
    bool is_user_defined_function) {
  // input/output values, used to skip inserting observers
  // for input and output of the block and the owning graph,
  // we have to insert the observers at call site because
  // the graph itself can be shared
  std::unordered_set<Value*> inputs_outputs;
  // list of observer modules for input values
  std::vector<c10::optional<Module>> block_input_observers;
  // list of observer modules for output values
  std::vector<c10::optional<Module>> block_output_observers;

  // if the current block is the block for entry point graph(the forward graph
  // of the top level module), we can insert observers in the block directly
  if (!is_entry_point) {
    auto* graph = block->owningGraph();
    // graph inputs/outputs
    for (auto list : {graph->inputs(), graph->outputs()}) {
      for (auto* v : list) {
        inputs_outputs.insert(v);
      }
    }
    // block outputs
    for (auto* v : block->outputs()) {
      inputs_outputs.insert(v);
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
      module._ivalue()->setAttr(name, observer.deepcopy()._ivalue());
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
  auto* self = block->owningGraph()->inputs()[0];
  // We first construct a map from value to the module, then
  // insert observers for them later, this is to avoid interference
  // of the inserted observers with the analysis to decide where
  // to insert observers, also we only insert observers for
  // "intermediate values" that is not the input/output of the
  // graph
  std::unordered_map<Value*, Module> values_to_observe;

  for (auto* v : block->inputs()) {
    if (!inputs_outputs.count(v) && !values_to_observe.count(v)) {
      if (auto observer_opt = getObserverFor(v)) {
        recordObserved(
            v, *observer_opt, values_to_observe, block_observed_values);
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
      if (n->kind() == prim::CallMethod || userDefinedCallFunction(n)) {
        script::Module m;
        std::shared_ptr<Graph> g;
        size_t input_offset;
        bool is_udf_for_subblock = is_user_defined_function;
        if (n->kind() == prim::CallMethod) {
          m = getInvokedModule(module, n, self);
          g = m.get_method(n->s(attr::name)).graph();
          input_offset = 0;
        } else { // CallFunction
          m = module;
          g = getCallFunctionGraph(n);
          input_offset = 1;
          is_udf_for_subblock = true;
        }

        std::unordered_set<Value*> callee_observed_inputs;
        for (auto i = 0U; i < g->inputs().size(); ++i) {
          auto* node_input = n->input(i + input_offset);
          if (isObserved(node_input, block_observed_values)) {
            callee_observed_inputs.insert(g->inputs()[i]);
          }
        }
        auto* subblock = g->block();
        auto info_from_callee = insertObserversFor(
            subblock, m, callee_observed_inputs, false, is_udf_for_subblock);
        auto input_observers = std::get<0>(info_from_callee);
        auto output_observers = std::get<1>(info_from_callee);
        auto callee_observed_outputs = std::get<2>(info_from_callee);
        for (auto idx : callee_observed_outputs) {
          block_observed_values.insert(n->outputs()[idx]);
        }
        for (auto i = 0U; i < g->inputs().size(); ++i) {
          auto* node_input = n->input(i + input_offset);
          if (input_observers[i] && !inputs_outputs.count(node_input) &&
              !isObserved(node_input, block_observed_values)) {
            recordObserved(
                node_input,
                *input_observers[i],
                values_to_observe,
                block_observed_values);
          }
        }
        for (auto i = 0U; i < n->outputs().size(); ++i) {
          if (output_observers[i] && !inputs_outputs.count(n->output(i)) &&
              !isObserved(n->output(i), block_observed_values)) {
            recordObserved(
                n->output(i),
                *output_observers[i],
                values_to_observe,
                block_observed_values);
          }
        }
      } else if (n->kind() == prim::If) {
        std::vector<size_t> aggregated_observed_outputs;
        std::vector<c10::optional<script::Module>> aggregated_output_observers;
        for (Block* subblock : n->blocks()) {
          if (alwaysRaisesException(subblock)) {
            continue;
          }
          // subblock has access to all the values in the scope of prim::If,
          // so subblock_observed_values == block_observed_values
          auto info_from_subblock =
              insertObserversFor(subblock, module, block_observed_values);
          auto output_observers = std::get<1>(info_from_subblock);
          auto subblock_observed_outputs = std::get<2>(info_from_subblock);
          // subblock for prim::If doesn't have inputs
          if (aggregated_observed_outputs.size() > 0) {
            TORCH_CHECK(
                aggregated_observed_outputs == subblock_observed_outputs,
                "quantization doesn't work for the case where branches "
                "of `if` doesn't both return quantized/non-quantized "
                "values");
          } else {
            for (auto idx : subblock_observed_outputs) {
              block_observed_values.insert(n->output(idx));
            }
            aggregated_observed_outputs = subblock_observed_outputs;
          }
          if (aggregated_output_observers.size() > 0) {
            TORCH_CHECK(
                aggregated_output_observers == output_observers,
                "quantization doesn't work for the case where branches "
                "of `if` doesn't both return values quantized the same "
                "way");
          } else {
            for (auto i = 0; i < n->outputs().size(); ++i) {
              if (output_observers[i] && !inputs_outputs.count(n->output(i)) &&
                  !block_observed_values.count(n->output(i)) &&
                  !observed_values_.count(n->output(i))) {
                recordObserved(
                    n->output(i),
                    *output_observers[i],
                    values_to_observe,
                    block_observed_values);
              }
            }
            aggregated_output_observers = output_observers;
          }
        }
      } else {
        for (Value* v : n->outputs()) {
          propagateObservedProperty(v, block_observed_values);
          if (!inputs_outputs.count(v) &&
              !isObserved(v, block_observed_values)) {
            auto observer_opt = getObserverFor(v);
            // If the node is one of the propagate quant node, e.g.
            // aten::cat, we should observe its output only
            // if the input of the node is observed
            if (observer_opt &&
                (!isPropagateQuantNode(n) ||
                 shouldPropagateQuant(n, block_observed_values))) {
              recordObserved(
                  v, *observer_opt, values_to_observe, block_observed_values);
            }
          }
        }
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }
  }
  std::vector<size_t> output_idxs;
  for (auto i = 0U; i < block->outputs().size(); ++i) {
    if (isObserved(block->outputs()[i], block_observed_values)) {
      output_idxs.push_back(i);
    }
  }
  if (!visited) {
    NameModuleVector observer_name_and_modules;
    for (const auto& item : values_to_observe) {
      auto* v = item.first;
      auto observer = item.second;
      TORCH_CHECK(
          !is_user_defined_function,
          "Inserting observers for user defined functions is not "
          "supported right now");
      insertObserverFor(v, module, observer, observer_name_and_modules);
    }
    block_observer_map_[block] = observer_name_and_modules;
  }
  return std::make_tuple(
      block_input_observers, block_output_observers, output_idxs);
}

void InsertObserversHelper::propagateObservedProperty(
    Value* output,
    std::unordered_set<Value*>& block_observed_values) {
  if (pass_through_value_map_.count(output)) {
    // since the vector is always non-empty, we will
    // not return the initial value
    bool all_observed = true;
    for (Value* v : pass_through_value_map_.at(output)) {
      all_observed &=
          observed_values_.count(v) || block_observed_values.count(v);
    }
    if (all_observed) {
      // This is to propagate observed property through
      // all ops that doesn't require observation
      block_observed_values.insert(output);
    }
  }
}

} // namespace

Module InsertObservers(
    Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    bool is_dynamic) {
  ModuleQConfigMap map_before_clone;
  fillQConfigMap(input_module, qconfig_dict, map_before_clone);
  ModuleCloneHelper mh;
  Module module =
      inplace ? input_module : mh.clone(input_module, map_before_clone);
  ModuleQConfigMap module_qconfig_map;
  // Since the types are changed after clone, we need to fill
  // the qconfig map again
  fillQConfigMap(module, qconfig_dict, module_qconfig_map);
  InsertObserversHelper helper(module_qconfig_map);
  helper.setDynamicFlag(is_dynamic);
  helper.preprocess(module, method_name);
  helper.insertObservers(module, method_name, true);
  return module;
}
} // namespace jit
} // namespace torch
