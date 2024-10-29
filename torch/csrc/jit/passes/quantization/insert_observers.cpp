#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/quantization/insert_observers.h>

#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

#include <memory>
#include <stack>
#include <string>
#include <utility>

namespace torch {
namespace jit {

using ModuleQConfigMap = std::unordered_map<ModulePtr, std::optional<QConfig>>;

namespace {

struct OptionalQConfigHash {
  inline size_t operator()(const std::optional<QConfig>& qconfig_opt) const {
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
    std::unordered_map<std::optional<QConfig>, TypePtr, OptionalQConfigHash>;
using NameModuleVector = std::vector<std::pair<std::string, Module>>;
using OptionalModuleVector = std::vector<std::optional<Module>>;
using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;
using graph_rewrite_helper::PatternInfo;
using graph_rewrite_helper::replaceConvolutionWithAtenConv;

// helper functions
void fillQConfigMap(
    const Module& module,
    const QConfigDict& qconfig_dict,
    ModuleQConfigMap& map,
    const std::string& key = "",
    const std::optional<QConfig>& parent_qconfig = std::nullopt) {
  std::optional<QConfig> qconfig;
  if (qconfig_dict.find(key) != qconfig_dict.end()) {
    GRAPH_DEBUG("Got module config for key:", key);
    qconfig = qconfig_dict.at(key);
  } else {
    GRAPH_DEBUG("Inheriting qconfig from parent module:", key);
    qconfig = parent_qconfig;
  }
  map[module._ivalue()] = qconfig;

  for (const NameModule& s : module.named_children()) {
    std::string child_key;
    if (key.empty()) {
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
   * inplace option means if the copy of the Tensor is deepcopy or not
   * if inplace is true, the cloned module will share the tensors with
   * original model instead of deepcopy them
   */
  Module clone(
      const Module& module,
      const ModuleQConfigMap& module_qconfig_map,
      bool inplace = false) {
    std::unordered_map<TypePtr, QConfigTypePtrMap> type_remap;
    IValue::HashIdentityIValueMap memo;
    return clone_impl(
        module, module_qconfig_map, type_remap, inplace, std::move(memo));
  }

 private:
  Module clone_impl(
      const Module& module,
      const ModuleQConfigMap& module_qconfig_map,
      std::unordered_map<TypePtr, QConfigTypePtrMap>& type_remap,
      bool inplace,
      IValue::HashIdentityIValueMap memo) {
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
    for (const auto i : c10::irange(N)) {
      IValue s = module._ivalue()->getSlot(i);
      std::string attr_name = type->getAttributeName(i);
      TypePtr attr_type = type->getAttribute(i);
      if (attr_type->is_module()) {
        const Module& orig = Module(s.toObject());
        Module cloned =
            clone_impl(orig, module_qconfig_map, type_remap, inplace, memo);

        // NOTE: why do we need to manually setattr on object instead of using
        // register_module here? because the attr can be a module interface
        // type and hold a Module object still. register_module will not let us
        // correctly set up the type for this attr, so we had to do this
        // manually. In the case it's an interface type, the type will be shared
        // by the new cloned instance in the same compilation unit bc it only
        // contains a list of functionSchema
        r.type()->addOrCheckAttribute(
            attr_name,
            attr_type->cast<ClassType>() ? cloned.type() : attr_type);
        r._ivalue()->setAttr(attr_name, cloned._ivalue());
      } else {
        // we'll deepcopy the IValue in non inplace option
        r.register_attribute(
            type->getAttributeName(i),
            type->getAttribute(i),
            inplace ? s : s.deepcopy(memo),
            type->is_parameter(i),
            type->is_buffer(i));
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
      // Execute __setstate__(__getstate__()) to initialize custom class
      // members.
      if (auto setstate_method = r.find_method("__setstate__")) {
        auto getstate_method = r.find_method("__getstate__");
        TORCH_INTERNAL_ASSERT(getstate_method, "expect __getstate__");
        auto state = (*getstate_method)(Stack{});
        (*setstate_method)(Stack{std::move(state)});
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
      const std::function<TypePtr(TypePtr, std::optional<QConfig>)>&
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
      if (node->kind() == prim::CallMethod || node->kind() == prim::GetAttr) {
        Value* instance = node->inputs()[0];
        auto child_opt = getInvokedModuleOpt(source, node, self);
        if (child_opt.has_value()) {
          auto qconfig = module_qconfig_map.at(child_opt->_ivalue());
          instance->setType(type_remap_fn(instance->type(), qconfig));
        }
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
      const std::function<TypePtr(TypePtr, std::optional<QConfig>)>&
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
                             const std::optional<QConfig>& qconfig) {
      if (type_remap.find(type_ptr) != type_remap.end()) {
        const auto& qconfig_map = type_remap.at(type_ptr);
        if (qconfig_map.find(qconfig) != qconfig_map.end()) {
          return qconfig_map.at(qconfig);
        }
      }
      return type_ptr;
    };
    auto graph = toGraphFunction(method).graph()->copy();
    remapTypes(graph.get(), source, target, module_qconfig_map, type_remap_fn);
    // remap self
    graph->inputs()[0]->setType(target.type());
    // we only support %self being Module in the arguments of function
    auto schema_type_remap_fn = [&](TypePtr type_ptr) {
      return type_remap_fn(
          std::move(type_ptr), module_qconfig_map.at(source._ivalue()));
    };
    auto schema =
        method.getSchema().cloneWithRemappedTypes(schema_type_remap_fn);
    const auto this_method_name =
        c10::QualifiedName(*target.type()->name(), method.name());
    auto copied = target._ivalue()->compilation_unit()->create_function(
        this_method_name, std::move(graph));
    target.type()->addMethod(copied);
    copied->setSchema(std::move(schema));
  }
};

class InsertObserversHelper {
 public:
  explicit InsertObserversHelper(
      const ModuleQConfigMap& map,
      QuantType quant_type)
      : module_qconfig_map_(map), quant_type_(quant_type) {}

  // TODO: replace (module, method_name) with graph?
  // preprocess to clean up the graph from tracing
  void preprocess(Module& module, const std::string& method_name);

  // Fill the map between the caller input/output to input/output
  // of called graph, this is used to navigate through the graph
  // to find the observer for a given value
  void fillBoundaryValueMap(Module& module, const std::string& method_name);

  // analyze the graph and record necessary information that can
  // be used in insert observers
  void analyze(Module& module, const std::string& method_name);

  void removeActivationObservers();

  /**
   * Recursively insert observers for the method, also we'll process
   * the nodes in the graph in the order of execution of these nodes
   * since we need the context information to decide whether we want to
   * observe/quantize a value a not, we don't want to observe a value multiple
   * times.
   *
   * argument: is_entry_point means whether the current method is the forward
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

  void setInsertResetObserverMethod(
      bool insert_reset_observer_method,
      const std::string& method_name) {
    insert_reset_observer_method_ = insert_reset_observer_method;
    reset_observer_method_name_ = "reset_observers_" + method_name;
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

  // Record v as "ready for observation" by storing it in values_to_observe.
  // If v is a part of a delayed observation pattern, record v's descendant
  // (per delay rules) instead. The observers are inserted at a later stage
  // by reading the state created by this function.
  void recordObserved(
      Value* v,
      const Module& observer_module,
      std::unordered_map<Value*, Module>& values_to_observe,
      std::unordered_set<Value*>& block_observed_values);

  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  bool valueNeedsToBeQuantized(Value* v, const QConfig& qconfig);

  bool isObserved(
      Value* v,
      const std::unordered_set<Value*>& block_observed_values) {
    return block_observed_values.count(v) || observed_values_.count(v);
  }

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

  void insertObserverResetMinMax(
      Module& module,
      const NameModuleVector& observer_name_and_modules);

  // Uses the state created by fillBoundaryValueMap and fillValueObserverMap
  // to return an observer configured for a value, if it is needed.
  std::optional<Module> getObserverFor(Value* v);

  // Uses the state created by fillPassThroughValueMap to propagage observed
  // property which should pass through from inputs to outputs.
  void propagateObservedProperty(
      Value* output,
      std::unordered_set<Value*>& block_observed_values);

  // for cat/add/mul we will only observe their output if their input
  // are observed
  bool shouldObserve(
      Node* n,
      const std::unordered_set<Value*>& block_observed_values,
      QuantType quant_type) {
    // Check whether node output uses can be quantized, eg cat followed by
    // linear op
    for (Value* v : n->outputs()) {
      for (const auto& use : v->uses()) {
        if (useQuantizable(use, quant_type)) {
          return true;
        }
      }
    }
    if (isPropagateQuantSingleInputOp(n)) {
      return isObserved(n->input(0), block_observed_values);
    } else if (isPropagateQuantBinaryOp(n)) {
      // This checks both of the input should be tensor and observed.
      // There is one check that we didn't do here, which is
      // !isScalar(isObserved(n->input(1), block_observed_values)
      // to make sure input 1 is not a scalar, because scalar tensor input
      // for add/mul won't be observed with current rule, we can omit
      // this check here
      return isObserved(n->input(0), block_observed_values) &&
          isObserved(n->input(1), block_observed_values);
    }
    return true;
  }

  void delayObservingValuesInPattern(Graph& graph, const PatternInfo& pattern);

  // Find and mark known patterns such as conv-relu (and others) where
  // we should not insert observers in the middle of the pattern.
  void addValuesToDelayObservation(
      const Module& module,
      const std::string& method_name);

  // Fill the map from values to the list of values that can pass the observed
  // property to it
  void fillPassThroughValueMap(const std::shared_ptr<Graph>& graph);

  bool insertResetObserverMethod() {
    return insert_reset_observer_method_;
  }

  const ModuleQConfigMap& module_qconfig_map_;

  // Values we want to delay observation, used to delay the observation for
  // values in the middle of the ops that are supposed to be fused, e.g.
  // the output value of conv in the conv - relu pattern
  // the key is the intermediate output, e.g. output of conv
  // the value is the value we want to observe, e.g. output of relu
  //
  // example, assuming we want to delay conv-relu:
  //   %x1 = conv(%x0)
  //   %x2 = relu(%x1)
  //
  // delay_observation_map_ = {
  //   %x1: %x2,
  // }
  std::unordered_map<Value*, Value*> delay_observation_map_;

  std::unordered_set<Graph*> visited_graph_of_observer_map_;

  // Map of value to observer module configured for that value.
  std::unordered_map<Value*, Module> observer_for_value_;

  // Map from values from callsite into the values in the CallMethod graph
  // key of the map is the value from caller graph, and the value of the map
  // is the list of values in the callee graph (the graph
  // corresponding to the called method),
  // the reason it is a set is that a value in the caller graph
  // can both correspond to the output of one callee graph and input of another
  // callee graph.
  //
  // example:
  //   // top level module
  //   %x1 = conv(%x0)
  //   %x2 = prim::CallFunction(%foo, %x1)
  //
  //   // graph of %foo
  //   %y2 = conv(%y1)
  //   return %y2
  //
  // boundary_value_map = {
  //   // current module's output values to corresponding return values from
  //   subgraph %x2: %y2,
  //   // current module's input values to corresponding input value to subgraph
  //   %x1: %y1,
  // }
  std::unordered_map<Value*, std::unordered_set<Value*>> boundary_value_map_;

  std::unordered_set<Value*> observed_values_;

  // This is used for the observed values to pass through the ops like flatten,
  // so that output value of flatten does not need to be observed
  // key is the output of the op, value is a vector of values that need
  // to be observed in order to pass the observed property to the output
  //
  // example:
  //   %x1 = flatten(%x0) // pass_through
  //   %x2 = conv(%x1) // not pass_through
  //
  // pass_through_value_map_ = {
  //   %x1: [%x0],
  // }
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

  // Type of quantization for this pass.
  QuantType quant_type_ = QuantType::STATIC;
  // These are the IR patterns we match to skip inserting observers.
  // They are compiled once on construction and used repeatedly within
  // the pass.

  // nn.Linear + nn.ReLU
  const PatternInfo nn_linear_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_linear_module, is_relu_module});

  // nn.Linear + F.relu
  const PatternInfo nn_linear_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %linear, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_linear_module, is_functional_relu});

  // nn.Linear + aten::relu
  const PatternInfo nn_linear_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_linear_module});

  // nn.Linear + aten::relu_
  const PatternInfo nn_linear_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%input, %linear, %relu):
    %first_output = prim::CallMethod[name="forward"](%linear, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_linear_module});

  // aten::linear + nn.ReLU
  const PatternInfo aten_linear_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %weight, %bias, %relu):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_relu_module});

  // aten::linear + F.relu
  const PatternInfo aten_linear_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %weight, %bias, %relu, %inplace):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_functional_relu});

  // aten::linear + aten::relu
  const PatternInfo aten_linear_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%input, %weight, %bias):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = aten::relu(%first_output)
    return (%second_output) )");

  // aten::linear + aten::relu_
  const PatternInfo aten_linear_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%input, %weight, %bias):
    %first_output = aten::linear(%input, %weight, %bias)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )");

  const PatternInfo nn_conv1d_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_conv1d_module, is_functional_relu});

  const PatternInfo nn_conv1d_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_conv1d_module, is_relu_module});

  const PatternInfo nn_conv1d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_conv1d_module});

  const PatternInfo nn_conv1d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_conv1d_module});

  const PatternInfo nn_conv2d_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_conv2d_module, is_functional_relu});

  const PatternInfo nn_conv2d_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_conv2d_module, is_relu_module});

  const PatternInfo nn_conv2d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_conv2d_module});

  const PatternInfo nn_conv2d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_conv2d_module});

  const PatternInfo nn_conv3d_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_conv3d_module, is_functional_relu});

  const PatternInfo nn_conv3d_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %relu):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_conv3d_module, is_relu_module});

  const PatternInfo nn_conv3d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %conv, %input):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_conv3d_module});

  const PatternInfo nn_conv3d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv):
    %first_output = prim::CallMethod[name="forward"](%conv, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_conv3d_module});

  const PatternInfo add_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %alpha, %relu):
     %first_output = aten::add(%a, %b, %alpha)
     %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
     return (%second_output) )",
      {aten_add_alpha_is_one, is_relu_module});

  const PatternInfo add_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %alpha, %relu, %inplace):
     %first_output = aten::add(%a, %b, %alpha)
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )",
      {aten_add_alpha_is_one, is_functional_relu});

  const PatternInfo inplace_add_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %alpha, %relu):
     %first_output = aten::add_(%a, %b, %alpha)
     %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
     return (%second_output) )",
      {aten_add_alpha_is_one, is_relu_module});

  const PatternInfo inplace_add_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %alpha, %relu, %inplace):
     %first_output = aten::add_(%a, %b, %alpha)
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )",
      {aten_add_alpha_is_one, is_functional_relu});

  const PatternInfo add_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
     %first_output = aten::add(%a, %b, %alpha)
     %second_output = aten::relu(%first_output)
     return (%second_output) )");

  const PatternInfo add_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
     %first_output = aten::add(%a, %b, %alpha)
     %second_output = aten::relu_(%first_output)
     return (%second_output) )");

  const PatternInfo inplace_add_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
     %first_output = aten::add_(%a, %b, %alpha)
     %second_output = aten::relu(%first_output)
     return (%second_output) )");

  const PatternInfo inplace_add_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b, %alpha):
     %first_output = aten::add_(%a, %b, %alpha)
     %second_output = aten::relu_(%first_output)
     return (%second_output) )");

  const PatternInfo nn_bn2d_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm, %relu):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_batchnorm2d_module, is_relu_module});

  const PatternInfo nn_bn2d_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_batchnorm2d_module, is_functional_relu});

  const PatternInfo nn_bn2d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_batchnorm2d_module});

  const PatternInfo nn_bn2d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_batchnorm2d_module});

  const PatternInfo nn_bn3d_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm, %relu):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallMethod[name="forward\\d*"](%relu, %first_output)
    return (%second_output) )",
      {is_batchnorm3d_module, is_relu_module});

  const PatternInfo nn_bn3d_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm, %relu, %inplace):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = prim::CallFunction(%relu, %first_output, %inplace)
    return (%second_output) )",
      {is_batchnorm3d_module, is_functional_relu});

  const PatternInfo nn_bn3d_aten_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu(%first_output)
    return (%second_output) )",
      {is_batchnorm3d_module});

  const PatternInfo nn_bn3d_aten_relu_ = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %batchnorm):
    %first_output = prim::CallMethod[name="forward"](%batchnorm, %input)
    %second_output = aten::relu_(%first_output)
    return (%second_output) )",
      {is_batchnorm3d_module});

  const PatternInfo mul_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %relu):
     %first_output = aten::mul(%a, %b)
     %second_output = prim::CallMethod[name="forward"](%relu, %first_output)
     return (%second_output) )",
      {is_relu_module});

  const PatternInfo mul_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %relu, %inplace):
     %first_output = aten::mul(%a, %b)
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )",
      {is_functional_relu});

  const PatternInfo inplace_mul_nn_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %relu):
     %first_output = aten::mul_(%a, %b)
     %second_output = prim::CallMethod[name="forward"](%relu, %first_output)
     return (%second_output) )",
      {is_relu_module});

  const PatternInfo inplace_mul_f_relu = PatternInfo::parse_from_str(
      R"(
graph(%self, %a, %b, %relu, %inplace):
     %first_output = aten::mul_(%a, %b)
     %second_output = prim::CallFunction(%relu, %first_output, %inplace)
     return (%second_output) )",
      {is_functional_relu});

  const PatternInfo mul_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul(%a, %b)
     %second_output = aten::relu(%first_output)
     return (%second_output) )");

  const PatternInfo mul_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul(%a, %b)
     %second_output = aten::relu_(%first_output)
     return (%second_output) )");

  const PatternInfo inplace_mul_aten_relu = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul_(%a, %b)
     %second_output = aten::relu(%first_output)
     return (%second_output) )");

  const PatternInfo inplace_mul_aten_relu_ = PatternInfo::parse_from_str(R"(
graph(%self, %a, %b):
     %first_output = aten::mul_(%a, %b)
     %second_output = aten::relu_(%first_output)
     return (%second_output) )");

  const std::vector<std::reference_wrapper<const PatternInfo>> delay_patterns =
      {
          nn_linear_f_relu,      nn_linear_nn_relu,
          nn_linear_aten_relu,   nn_linear_aten_relu_,
          aten_linear_f_relu,    aten_linear_nn_relu,
          aten_linear_aten_relu, aten_linear_aten_relu_,

          nn_conv1d_f_relu,      nn_conv1d_nn_relu,
          nn_conv1d_aten_relu,   nn_conv1d_aten_relu_,
          nn_conv2d_f_relu,      nn_conv2d_nn_relu,
          nn_conv2d_aten_relu,   nn_conv2d_aten_relu_,
          nn_conv3d_f_relu,      nn_conv3d_nn_relu,
          nn_conv3d_aten_relu,   nn_conv3d_aten_relu_,

          add_nn_relu,           add_f_relu,
          inplace_add_nn_relu,   inplace_add_f_relu,
          add_aten_relu,         add_aten_relu_,
          inplace_add_aten_relu, inplace_add_aten_relu_,

          nn_bn2d_nn_relu,       nn_bn2d_f_relu,
          nn_bn2d_aten_relu,     nn_bn2d_aten_relu_,
          nn_bn3d_nn_relu,       nn_bn3d_f_relu,
          nn_bn3d_aten_relu,     nn_bn3d_aten_relu_,

          mul_nn_relu,           mul_f_relu,
          inplace_mul_nn_relu,   inplace_mul_f_relu,
          mul_aten_relu,         mul_aten_relu_,
          inplace_mul_aten_relu, inplace_mul_aten_relu_,
  };

  bool insert_reset_observer_method_{false};
  std::string reset_observer_method_name_;
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
        auto m_opt = getInvokedModuleOpt(module, n, graph->inputs()[0]);
        if (m_opt.has_value()) {
          invoked_methods.emplace_back(*m_opt, n->s(attr::name));
        }
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
  GRAPH_DEBUG("Inserting observer for:", v->debugName());
  Module observer = observer_module.deepcopy();
  std::string observer_name = "_observer_" + std::to_string(uid_++);
  while (module.hasattr(observer_name)) {
    observer_name = "_observer_" + std::to_string(uid_++);
  }
  module.register_module(observer_name, observer);
  observer_name_and_modules.emplace_back(observer_name, observer);

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

void InsertObserversHelper::insertObserverResetMinMax(
    Module& module,
    const NameModuleVector& observer_name_and_modules) {
  if (observer_name_and_modules.empty()) {
    return;
  }
  auto reset_min_max_opt = module.find_method(reset_observer_method_name_);
  if (!reset_min_max_opt.has_value()) {
    std::shared_ptr<Graph> reset_observer_graph = std::make_shared<Graph>();
    Value* module_value = reset_observer_graph->addInput("self");
    Node* output_node = reset_observer_graph->createNone();
    reset_observer_graph->insertNode(output_node);
    reset_observer_graph->registerOutput(output_node->output());
    module_value->setType(module._ivalue()->type());
    const auto method_name = c10::QualifiedName(
        *(module.type()->name()), reset_observer_method_name_);
    auto reset_observer_fn =
        module._ivalue()->compilation_unit()->create_function(
            method_name, std::move(reset_observer_graph));
    auto self_arg = c10::Argument("self", module.type());
    auto output_arg = c10::Argument("none", output_node->output()->type());
    auto schema = c10::FunctionSchema(
        reset_observer_method_name_,
        "",
        {std::move(self_arg)},
        {std::move(output_arg)});
    reset_observer_fn->setSchema(std::move(schema));
    module.type()->addMethod(reset_observer_fn);
  }
  auto reset_min_max_graph =
      module.get_method(reset_observer_method_name_).graph();
  Value* self = reset_min_max_graph->inputs()[0];

  for (const auto& pair : observer_name_and_modules) {
    const auto& observer_name = pair.first;
    const auto& observer = pair.second;
    Value* observer_value =
        reset_min_max_graph->insertGetAttr(self, observer_name);
    MatchedSchema reset_minmax_schema = matchSchema(
        observer.get_method("reset_min_max_vals").function().getSchema(),
        observer_value->node()->sourceRange(),
        *reset_min_max_graph,
        {observer_value},
        {});
    reset_min_max_graph->insertMethodCall(
        "reset_min_max_vals", reset_minmax_schema);
  }
}

void InsertObserversHelper::delayObservingValuesInPattern(
    Graph& graph,
    const PatternInfo& pattern) {
  const Graph& pattern_graph = *pattern.pattern_graph;
  const std::unordered_map<std::string, Value*>& vmap = pattern.vmap;

  const auto& matches = findPatternMatches(pattern_graph, graph);
  for (const auto& match : matches) {
    if (!std::all_of(
            pattern.filters.begin(),
            pattern.filters.end(),
            [&](const MatchFilter& f) { return f(match, vmap); })) {
      continue;
    }
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
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    fillBoundaryValueMap(invoked_module, invoked_method_name);
  }

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
        size_t input_offset = 0;
        if (n->kind() == prim::CallMethod) {
          auto m_opt = getInvokedModuleOpt(module, n, self);
          if (!m_opt.has_value()) {
            continue;
          }
          auto m = *m_opt;
          g = m.get_method(n->s(attr::name)).graph();
          input_offset = 0;
        } else {
          g = getCallFunctionGraph(n);
          input_offset = 1;
        }
        // add mapping from callsite value to value in called graph
        for (auto i = 0U; i < g->outputs().size(); ++i) {
          auto* return_val = g->outputs()[i];
          GRAPH_DEBUG(
              "Boundary Map[return]:",
              n->output(i)->debugName(),
              " -> ",
              return_val->debugName());
          boundary_value_map_[n->output(i)].insert(return_val);
        }
        for (auto i = 0U; i < g->inputs().size(); ++i) {
          auto caller_input_index = i + input_offset;
          auto* caller_input = n->input(caller_input_index);
          auto* input_val = g->inputs()[i];
          GRAPH_DEBUG(
              "Boundary Map[input]:",
              caller_input->debugName(),
              " -> ",
              input_val->debugName());
          boundary_value_map_[caller_input].insert(input_val);
        }
      } else if (n->kind() == prim::If) {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
          for (Value* v : n->outputs()) {
            Value* subblock_output = subblock->outputs()[v->offset()];
            GRAPH_DEBUG(
                "Boundary Map[if_output]:",
                v->debugName(),
                " -> ",
                subblock_output->debugName());
            boundary_value_map_[v].insert(subblock_output);
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
  // run preprocess for child module before parent, since preprocess
  // mutates the graph and it might affect passes like fillBoundaryValueMap
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    preprocess(invoked_module, invoked_method_name);
  }

  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // Inline fork-wait calls
  InlineForkWait(graph);
  // fuse decomposed linear into aten::linear
  FuseLinear(graph);
  replaceConvolutionWithAtenConv(graph);
  RemoveListMutation(graph);
}

void InsertObserversHelper::analyze(
    Module& module,
    const std::string& method_name) {
  for (auto& invoked_method : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_method);
    const auto& invoked_method_name = std::get<1>(invoked_method);
    analyze(invoked_module, invoked_method_name);
  }

  // fill out various internal state which will be later used in
  // insertObservers to insert the correct observer
  addValuesToDelayObservation(module, method_name);
  fillValueObserverMap(module, method_name);
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  fillPassThroughValueMap(graph);
}

bool InsertObserversHelper::valueNeedsToBeQuantized(
    Value* v,
    const QConfig& qconfig) {
  if (isBiasOfConvOrLinear(v) ||
      !(v->type()->isSubtypeOf(*TensorType::get()) ||
        v->type()->isSubtypeOf(*ListType::ofTensors())) ||
      isEmbeddingBagNonInput(v)) {
    return false;
  }
  // For dynamic quantization we only insert observers at the input
  // of the quantizable function.
  if (quant_type_ == QuantType::STATIC) {
    // Check whether producer is quantizable
    if (!isWeightOnlyStaticQuantOp(v->node()) &&
        (nodeQuantizable(v->node()) || isPropagateQuantOp(v->node()))) {
      return true;
    }
  }
  if (quant_type_ == QuantType::DYNAMIC) {
    // Check the dtype of the observer module.
    Module observer_module = getObserverModuleFor(v, qconfig);
    auto scalar_type = observer_module.attr("dtype");
    // For inputs with Fp16 type that are not-weights we don't observer them for
    // dynamic quantization.
    if (scalar_type == at::ScalarType::Half && !isWeight(v)) {
      return false;
    }
  }
  // Check whether node input value is quantizable
  for (const auto& use : v->uses()) {
    if (useQuantizable(use, quant_type_)) {
      return true;
    }
  }
  return false;
}

void InsertObserversHelper::removeActivationObservers() {
  std::vector<std::unordered_map<Value*, Module>::iterator>
      values_to_be_removed;
  for (auto it = observer_for_value_.begin(); it != observer_for_value_.end();
       it++) {
    if (!isWeight(it->first)) {
      values_to_be_removed.push_back(it);
    }
  }
  for (auto it : values_to_be_removed) {
    observer_for_value_.erase(it);
  }
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
    if (valueNeedsToBeQuantized(v, qconfig)) {
      GRAPH_DEBUG("Recording observer for ", v->debugName());
      GRAPH_DUMP("In graph:", v->owningGraph());
      observer_for_value_[v] = getObserverModuleFor(v, qconfig);
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        if (valueNeedsToBeQuantized(v, qconfig)) {
          GRAPH_DEBUG("Recording observer for ", v->debugName());
          GRAPH_DUMP("In graph:", v->owningGraph());
          observer_for_value_[v] = getObserverModuleFor(v, qconfig);
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
}

std::optional<Module> InsertObserversHelper::getObserverFor(Value* v) {
  if (observer_for_value_.count(v)) {
    auto observer = observer_for_value_.at(v);
    GRAPH_DEBUG("Got observer module config for:", v->debugName());
    return observer;
  }
  std::optional<Module> result;
  if (boundary_value_map_.count(v)) {
    for (Value* next : boundary_value_map_.at(v)) {
      GRAPH_DEBUG(
          "Going through boundary map:",
          v->debugName(),
          " --> ",
          next->debugName());
      GRAPH_DUMP("From graph:", v->owningGraph());
      GRAPH_DUMP("To graph:", next->owningGraph());
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
  GRAPH_DEBUG(
      "Observer module config for ", v->debugName(), ":", result.has_value());
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
  std::vector<std::optional<Module>> block_input_observers;
  // list of observer modules for output values
  std::vector<std::optional<Module>> block_output_observers;

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
      block_input_observers.emplace_back(getObserverFor(v));
    }

    for (auto* v : block->outputs()) {
      // we need explicitly skip the values that are already observed
      // this might happen in subblocks for `if` since
      // these subblock has access to all values before the `if` node
      if (!isObserved(v, block_observed_values)) {
        block_output_observers.emplace_back(getObserverFor(v));
      } else {
        block_output_observers.emplace_back(std::nullopt);
      }
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
  // Reason is `block_observed_values` can
  // change depending on where the method is called, and
  // outputs that's been observed(third item of the returned result)
  // can change depending on that, so for each graph we'll need to go through
  // the whole process of inserting observers, the observers inserted in this
  // block won't change, but the information we return to the caller will change
  // based on `block_observed_values`

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
        size_t input_offset = 0;
        bool is_udf_for_subblock = is_user_defined_function;
        if (n->kind() == prim::CallMethod) {
          auto m_opt = getInvokedModuleOpt(module, n, self);
          if (!m_opt.has_value()) {
            continue;
          }
          m = *m_opt;
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
        // a vector recoding whether each output is observed or not
        std::vector<bool> aggregated_output_observe_state;
        for (Block* subblock : n->blocks()) {
          if (alwaysRaisesException(subblock)) {
            continue;
          }
          // subblock has access to all the values in the scope of prim::If,
          // so subblock_observed_values == block_observed_values
          auto info_from_subblock =
              insertObserversFor(subblock, module, block_observed_values);
          // subblock for prim::If doesn't have inputs
          auto output_observers = std::get<1>(info_from_subblock);
          auto subblock_observed_outputs = std::get<2>(info_from_subblock);

          // We'll insert output observer for each subblock, and in the end
          // we will check if output of subblocks are quantized consistently
          for (size_t i = 0; i < subblock->outputs().size(); ++i) {
            Value* output = subblock->outputs()[i];
            if (output_observers[i] && !inputs_outputs.count(output) &&
                !isObserved(output, block_observed_values)) {
              recordObserved(
                  output,
                  *output_observers[i],
                  values_to_observe,
                  block_observed_values);
            }
          }
          for (auto idx : subblock_observed_outputs) {
            block_observed_values.insert(subblock->outputs()[idx]);
          }
          std::vector<bool> subblock_output_observe_state;
          for (size_t i = 0; i < subblock->outputs().size(); ++i) {
            Value* output = subblock->outputs()[i];
            subblock_output_observe_state.push_back(
                isObserved(output, block_observed_values));
          }
          if (!aggregated_output_observe_state.empty()) {
            TORCH_CHECK(
                aggregated_output_observe_state ==
                    subblock_output_observe_state,
                "branches for `if` should return values that are observed "
                "consistently, if node:",
                *n);
          } else {
            aggregated_output_observe_state = subblock_output_observe_state;
          }
        }
        // mark the output of if as observed
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          if (aggregated_output_observe_state[i]) {
            block_observed_values.insert(n->output(i));
          }
        }
      } else if (n->kind() == prim::Loop) {
        TORCH_WARN_ONCE(
            "prim::Loop is not yet supported in quantization, "
            "please make sure nothing needs to be quantized in the "
            "loop");
      }
      for (Value* v : n->outputs()) {
        propagateObservedProperty(v, block_observed_values);
        if (!inputs_outputs.count(v) && !isObserved(v, block_observed_values)) {
          auto observer_opt = getObserverFor(v);
          // If the node is one of the propagate quant node, e.g.
          // aten::cat, we should observe its output only
          // if the input of the node is observed
          if (observer_opt &&
              shouldObserve(n, block_observed_values, quant_type_)) {
            recordObserved(
                v, *observer_opt, values_to_observe, block_observed_values);
          }
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
    if (insertResetObserverMethod()) {
      insertObserverResetMinMax(module, observer_name_and_modules);
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
      GRAPH_DEBUG("Pass through observed property in node:", *output->node());
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
    QuantType quant_type) {
  ModuleQConfigMap map_before_clone;
  fillQConfigMap(input_module, qconfig_dict, map_before_clone);
  ModuleCloneHelper mh;
  Module module = mh.clone(input_module, map_before_clone, inplace);
  SwapFunctionalLinear(module);
  ModuleQConfigMap module_qconfig_map;
  // Since the types are changed after clone, we need to fill
  // the qconfig map again
  fillQConfigMap(module, qconfig_dict, module_qconfig_map);
  GRAPH_DEBUG("Quant type:", quant_type);
  InsertObserversHelper helper(module_qconfig_map, quant_type);
  helper.preprocess(module, method_name);
  helper.fillBoundaryValueMap(module, method_name);
  // analyze needs to run after fillBoundaryValueMap
  // since we need to know the boundary value mapping to trace
  // through the calls
  helper.analyze(module, method_name);
  helper.insertObservers(module, method_name, /* is_entry_point */ true);
  return module;
}

Module InsertObserversForOnDevicePTQ(
    Module& input_module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    bool inplace,
    QuantType quant_type) {
  ModuleQConfigMap map_before_clone;
  fillQConfigMap(input_module, qconfig_dict, map_before_clone);
  ModuleCloneHelper mh;
  Module cloned_module = mh.clone(input_module, map_before_clone, inplace);
  std::shared_ptr<Graph> g = cloned_module.get_method(method_name).graph();
  SwapFunctionalLinear(g);
  std::string observer_method_name = "observe_" + method_name;
  cloneMethod(cloned_module, method_name, observer_method_name);
  ModuleQConfigMap module_qconfig_map;
  // Since the types are changed after clone, we need to fill
  // the qconfig map again
  fillQConfigMap(cloned_module, qconfig_dict, module_qconfig_map);
  GRAPH_DEBUG("Quant type:", quant_type);
  InsertObserversHelper helper(module_qconfig_map, quant_type);
  // Removes list mutation part is not clear. Is it needed
  helper.preprocess(cloned_module, observer_method_name);
  // Since we expect the graph to be inlined this should not have any use
  // However, this function does handle if blocks
  // Although as far as I understood If blocks are not really handled
  // in JIT quantization. Should we just protect against this. That is if we
  // find observable value inside If block? Also side effect of inlining is that
  // you will have multiple getattrs for the same attribute and thus potentially
  // multiple observers observing the same value. This will also lead to
  // increased size of the packed param struct. I dont expect this to be a
  // common pattern but something to be aware fo Note that current quant
  // workflow does not prevent this anyway since during inset quant dequant
  // things are inlined anyway
  helper.fillBoundaryValueMap(cloned_module, observer_method_name);
  // analyze needs to run after fillBoundaryValueMap
  // since we need to know the boundary value mapping to trace
  // through the calls
  helper.analyze(cloned_module, observer_method_name);
  // Remove activation observer if quant_type is dynamic
  if (quant_type == QuantType::DYNAMIC) {
    helper.removeActivationObservers();
  }
  helper.setInsertResetObserverMethod(true, method_name);
  helper.insertObservers(
      cloned_module, observer_method_name, /* is_entry_point */ true);
  return cloned_module;
}
} // namespace jit
} // namespace torch
