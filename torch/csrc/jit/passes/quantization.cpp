#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>


#include <stack>

namespace torch {
namespace jit {
namespace {
// QuantizerUtils
struct ParamInfo {
  Value* v;
  Use consumer;
  at::Tensor value;
};

bool nodeQuantizable(Node* n) {
  TORCH_INTERNAL_ASSERT(n != nullptr);
  // This is map for quantizable nodes. It will be expanded in future to
  // support more ops and patterns.
  static const OperatorSet quantnodeLookup = {
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] \
stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::relu(Tensor self) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] \
stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, \
int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor"};
  return quantnodeLookup.find(n) != nullptr;
}

Value* getScaleValue(Node* n) {
  if (n->kind().toQualString() != std::string("aten::_dequantize_linear")) {
    return nullptr;
  }
  TORCH_CHECK(n->inputs().size() == 4);
  // Fetch scale from the dequant node
  return n->inputs()[1];
}

Node* traverseToQuantNode(Node* dq) {
  TORCH_INTERNAL_ASSERT(dq != nullptr);
  TORCH_INTERNAL_ASSERT(dq->inputs().size() != 0);
  Node* intrepr = dq->inputs()[0]->node();
  TORCH_INTERNAL_ASSERT(intrepr != nullptr);
  TORCH_INTERNAL_ASSERT(intrepr->inputs().size() != 0);
  return intrepr->inputs()[0]->node();
}

struct ParamValue {
  Value* definition;
  IValue value;
};

static void gatherParams(
    const script::Module& module,
    Value* module_value,
    std::vector<ParamValue>& params) {
  for (const Use& u : module_value->uses()) {
    if (u.user->kind() != prim::GetAttr) {
      continue;
    }
    const std::string& field = u.user->s(attr::name);
    if (const auto& sub = module.find_module(field)) {
      gatherParams(*sub, u.user->output(), params);
    } else if (auto slot = module.find_parameter(field)) {
      params.emplace_back(ParamValue{u.user->output(), slot->value()});
    }
  }
}

std::vector<ParamInfo> getQuantizableParamsofName(
    script::Method& method,
    const std::string& param_name) {
  std::vector<ParamValue> params;
  gatherParams(method.owner(), method.graph()->inputs().at(0), params);
  std::vector<ParamInfo> params_to_insert_qdq;
  for(const ParamValue& pv : params) {
    if (!pv.definition->type()->isSubtypeOf(TensorType::get())) {
      continue;
    }
    for(const Use& u : pv.definition->uses()) {
      if (!nodeQuantizable(u.user) ||
          u.user->schema().arguments().at(u.offset).name() != param_name) {
        continue;
      }
      params_to_insert_qdq.emplace_back(
          ParamInfo{pv.definition, u, pv.value.toTensor().detach()});
    }
  }

  return params_to_insert_qdq;
}

std::vector<Value*> insertQuantParamNodes(
    Node* quant,
    const std::tuple<std::string, float, int>& qparam) {
  std::vector<Value*> qparam_vals;
  auto& scale = std::get<1>(qparam);
  auto& zero_point = std::get<2>(qparam);

  // All params inserted before quant node which is
  // beginning of the quant-dequant pattern
  WithInsertPoint ins(quant);
  Value* scale_v = quant->owningGraph()->insertConstant(scale);
  qparam_vals.emplace_back(scale_v);
  Value* zeropoint_v = quant->owningGraph()->insertConstant(zero_point);
  qparam_vals.emplace_back(zeropoint_v);
  return qparam_vals;
}

Value* insertScalarType(Node* ins_node, at::ScalarType t) {
  TORCH_INTERNAL_ASSERT(t != at::ScalarType::Undefined);
  WithInsertPoint ins(ins_node);
  // ScalarType inserted before ins_node node which is
  // beginning of the quant-dequant pattern
  Value* scalartype_v =
      ins_node->owningGraph()->insertConstant(IValue(static_cast<int>(t)));
  return scalartype_v;
}

// Create Quant Node
Node* createQuantNode(Value* v, Graph* g) {
  Node* quant = g->create(at::Symbol::fromQualString("aten::quantize_linear"));
  TORCH_INTERNAL_ASSERT(quant != nullptr, "Failed to create quant node");
  quant->output()->setDebugName(v->debugName() + ".quant");
  return quant;
}

// Create Dequant node
Node* createDeQuantNode(Value* v, Graph* g) {
  Node* dequant =
      g->create(at::Symbol::fromQualString("aten::_dequantize_linear"));
  TORCH_INTERNAL_ASSERT(dequant != nullptr, "Failed to create dequant node");
  dequant->output()->setDebugName(v->debugName() + ".dequant");
  return dequant;
}

// Create IntTensor Node
Node* createIntReprNode(Value* v, Graph* g) {
  Node* intrepr = g->create(at::Symbol::fromQualString("aten::int_repr"));
  TORCH_INTERNAL_ASSERT(intrepr != nullptr, "Failed to create inttensor node");
  intrepr->output()->setDebugName(v->debugName() + ".intrepr");
  return intrepr;
}

script::Module clone_module(const script::Module& module) {
  script::Module result(c10::QualifiedName(module.name().qualifiedName()), module.class_compilation_unit(), true);
  for (const auto& param: module.get_parameters()) {
    result.register_parameter(param.name(), module.get_parameter(param.name()), false);
  }

  for (const auto& attr: module.get_attributes()) {
    result.register_attribute(attr.name(), attr.type(), module.get_attribute(attr.name()));
  }

  for (const auto& mod: module.get_module_slots()) {
    result.register_module(mod.name(), module.get_module(mod.name()));
  }
  return result;
}

// Create observer.forward Node and insert a call to observer forward function
Node* insertObserverForwardCall(Value* v, Graph* g, script::Module module, script::Module observer_module) {
  std::string observer_name = "observer_for_" + v->debugName();
  module.register_module(observer_name, observer_module.clone());
  WithInsertPoint ins(v->node());
  // Get handle of observer module
  Node* observer_instance = g->create(c10::prim::GetAttr);
  // self.observer_for_v
  observer_instance->addInput(g->inputs()[0]);
  observer_instance->s_(c10::attr::name, observer_name);
  observer_instance->output()->setDebugName(observer_name);
  observer_instance->output()->setType(observer_module.type());
  observer_instance->insertAfter(v->node());

  // Create forward method call
  Node* call = observer_instance->owningGraph()->create(c10::prim::CallMethod);
  TORCH_INTERNAL_ASSERT(call != nullptr, "Failed to create forward call node");
  call->s_(c10::attr::name, "forward");
  call->addInput(observer_instance->output());
  call->addInput(v);
  call->output()->setType(v->type());
  call->output()->setDebugName(v->debugName() + ".observed");
  call->insertAfter(observer_instance);
  return call;
}

// Insert Quant-Dequant node pattern for quantizable node outputs
Node* addQuantDeQuantNodesFor(
    Value* v,
    Node* insert_point,
    const std::tuple<std::string, float, int>& qparam,
    at::ScalarType t) {
  TORCH_INTERNAL_ASSERT(v != nullptr);
  WithCurrentScope scope_guard(
      *insert_point->owningGraph(), insert_point->scope());
  Node* quant = createQuantNode(v, insert_point->owningGraph());
  Node* intrepr = createIntReprNode(v, insert_point->owningGraph());
  Node* dequant = createDeQuantNode(v, insert_point->owningGraph());

  // Add quant-intrepr-dequant nodes and replace for all uses of Value
  quant->insertAfter(insert_point);
  intrepr->insertAfter(quant);
  dequant->insertAfter(intrepr);

  // Attach inputs to quantization pattern nodes
  quant->addInput(v);
  intrepr->addInput(quant->output());
  dequant->addInput(intrepr->output());
  // Insert qparam nodes
  auto qparam_values = insertQuantParamNodes(quant, qparam);
  for (Value* qparam_value : qparam_values) {
    quant->addInput(qparam_value);
    dequant->addInput(qparam_value);
  }
  // Add ScalarType Node for q-dq
  Value* scalartype_v = insertScalarType(quant, t);
  TORCH_INTERNAL_ASSERT(scalartype_v != nullptr);
  quant->addInput(scalartype_v);
  dequant->addInput(scalartype_v);
  return dequant;
}

template <typename... ArgT>
bool matchQParamDictKeytoObserver(
    Node* n,
    const std::unordered_map<std::string, std::tuple<ArgT...>>& qparam_dict,
    std::tuple<ArgT...>& qparam_value) {
  // Observer nodes have two inputs
  if (n->kind() != prim::PythonOp || n->inputs().size() != 2) {
    return false;
  }
  // For observer node, qparam dict key matches the
  // second input name for observer node
  Value* vname = n->inputs()[1];
  TORCH_INTERNAL_ASSERT(toIValue(vname).has_value());
  IValue valuekey = toIValue(vname).value();
  if (!valuekey.isString()) {
    return false;
  }
  auto it = qparam_dict.find(valuekey.toStringRef());
  if (it == qparam_dict.end()) {
    return false;
  }
  // Extract the qparam_dict value
  qparam_value = it->second;
  return true;
}

} // namespace

// PyBind APIs
void PropagateQuantInfo(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

static Node* addObserverFor(
    Value* v,
    Node* original_observer_node,
    Node* insert_point) {
  TORCH_INTERNAL_ASSERT(insert_point != nullptr);
  WithInsertPoint ins(insert_point);

  // We need to pass the value name to observer function - create a constant
  // holding this name.
  Value* vname = insert_point->owningGraph()->insertConstant(v->debugName());

  // Create a new observer node. We just need to clone the original one.
  Node* observerNode = insert_point->owningGraph()->createClone(
      &*original_observer_node, [&](Value* v) { return v; }, false);

  // Set the type and the name of the output of the new observer node. It will
  // be used instead of the original value v.
  Value* observedValue = observerNode->addOutput();
  observedValue->setType(v->type());
  observedValue->setDebugName(v->debugName() + ".observed");

  // Now we can add the inputs.
  observerNode->addInput(v);
  observerNode->addInput(vname);
  return observerNode;
}

static bool outputsNeedToBeObserved(Node* n) {
  return n->kind() != prim::Constant;
}

void InsertObserverNodes(
    const std::shared_ptr<Graph>& graph,
    Node* observer_node,
    size_t num_activation_inputs) {
  TORCH_CHECK(graph != nullptr);
  // num_activation_inputs is the number of activations or external data
  // excluding the parameters
  TORCH_CHECK(num_activation_inputs <= graph->inputs().size());
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
  Node* insert_node = *graph->nodes().begin();
  for (size_t idx = 0; idx < num_activation_inputs; ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      Node* new_observer_node = addObserverFor(v, observer_node, insert_node);
      new_observer_node->insertBefore(insert_node);
      observer_for_input.emplace(new_observer_node);
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip nodes that we don't need to observe, e.g. 'prim::Constant' or
      // observer nodes
      if (!outputsNeedToBeObserved(n) || observer_for_input.count(n) != 0) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        values_to_observe.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // Actually add observer nodes.
  for (Value* v : values_to_observe) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      Node* clone_observer_node = addObserverFor(v, observer_node, v->node());
      clone_observer_node->insertAfter(v->node());
    }
  }
}

void InsertObserverNodes(
    const script::Module& moduleObj,
    const std::string& method_name,
    Node* observer_node) {
  script::Method method = moduleObj.get_method(method_name);
  InsertObserverNodes(method.graph(), observer_node, method.num_inputs());
}

void InsertObserverNodes(
    Function* function_var,
    Node* observer_node) {
  InsertObserverNodes(
      function_var->graph(), observer_node, function_var->num_inputs());
}

void InsertQuantDequantNodes(
    const std::shared_ptr<Graph>& graph,
    const std::unordered_map<std::string, std::tuple<std::string, float, int>>&
        qparam_dict) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // For storing quantizable values - node pairs that are external
  // or intermediate inputs to quantizable nodes
  std::vector<Use> quantInputs;
  // For storing quantizable values that are output of quantizable nodes
  // Since same value can go to multiple nodes, we use set so that
  // we insert quant-dequant node pairs for value only once
  std::vector<Value*> quantOutputs;
  std::unordered_set<Value*> valueLookup;

  // Observer nodes to remove from graph
  std::vector<Node*> nodes_to_remove;

  // Create value:qparam dict. Once qparam dict key is matched
  // to the observer node, we create value node to qparam for lookup.
  std::unordered_map<Value*, std::tuple<std::string, float, int>>
      qparam_value_dict;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    for (Node* n : b->nodes()) {
      // Schedule the sub blocks
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }

      std::tuple<std::string, float, int> qparam_data;
      if (matchQParamDictKeytoObserver<std::string, float, int>(
              n, qparam_dict, qparam_data)) {
        // This is the observer node. Mark it and the second input
        // constant node for deletion.
        Value* qparam_value = n->inputs()[0];
        qparam_value_dict.emplace(qparam_value, qparam_data);
        nodes_to_remove.emplace_back(n);
        nodes_to_remove.emplace_back(n->inputs()[1]->node());
        continue;
      }

      // We iterate over node inputs to identify which Values
      // need to be quantized depending on node type
      for (size_t i = 0; i <  n->inputs().size(); ++i) {
        Value* v = n->inputs().at(i);
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          // Skip quantization for non tensors
          continue;
        }

        if (nodeQuantizable(v->node())) {
          // Goal of this iteration is to identify the parent node for V
          // that is quantizable and replace all uses of Value with
          // quant-dequant output. Usage of set helps adding single
          // q-dq nodes for all V->users
          // Example N1 -> (V1 -> (N2), V2 -> (N3))
          //         N1 is quantizable node. So we insert quant-dequant
          //         nodes for all outputs of N1 (V1, V2) once
          if (!valueLookup.count(v)) {
            valueLookup.emplace(v);
            quantOutputs.emplace_back(v);
          }
        } else if (nodeQuantizable(n)) {
          // Goal of this iteration is to identify nodes that are
          // quantizable but input value originate from non quantizable
          // node. This requires selectively inserting q-dq nodes for
          // inputs into node N(V, N pair) because parent node might
          // also have input into other non quantizable nodes
          // Example : N1(prim::Param) -> (V1 -> (N4, N5), V2 -> (N6, N7), V3)
          //           N1 is not quantizable node but N4 and N7 are
          //           quantizable nodes. So we add the (V1, N4) and
          //           (V2, N7) as insertion points for quant-dequant nodes
          quantInputs.emplace_back(Use(n, i));
        }
      }
    } // End Loop for nodes within block

    // Since we are iterating node inputs only above, we need to iterate
    // over block outputs values and if they originate from quantizable
    // node, we push to quantOutputs set
    auto outputVals = b->outputs();
    for (auto& v : outputVals) {
      if (nodeQuantizable(v->node()) &&
          v->type()->isSubtypeOf(TensorType::get())) {
        quantOutputs.emplace_back(v);
      }
    } // end for
  } // end Block traversal

  // Destory Observer Nodes
  for (auto& n : nodes_to_remove) {
    n->destroy();
  }

  // Insert the quant-dequant pair for values output from quantizable nodes
  for (auto& v_to_quant : quantOutputs) {
    if (qparam_value_dict.count(v_to_quant) != 0) {
      Node* dq = addQuantDeQuantNodesFor(
          v_to_quant,
          v_to_quant->node(),
          qparam_value_dict[v_to_quant],
          at::ScalarType::QUInt8);
      TORCH_INTERNAL_ASSERT(dq != nullptr);
      v_to_quant->replaceAllUsesWith(dq->output());
      // Above step replaces v->quant with vdq->quant. We need to restore link.
      // Below chain traverse up from dq to q node.
      Node* q = traverseToQuantNode(dq);
      TORCH_INTERNAL_ASSERT(q != nullptr);
      q->replaceInputWith(dq->output(), v_to_quant);
    }
  }

  // Insert the quant-dequant pair for values inputs to quantizable nodes
  for (const Use& u : quantInputs) {
    Value* v = u.user->inputs().at(u.offset);
    if (qparam_value_dict.count(v) != 0) {
      Node* dq = addQuantDeQuantNodesFor(
          v,
          v->node(),
          qparam_value_dict[v],
          at::ScalarType::QUInt8);
      TORCH_INTERNAL_ASSERT(dq != nullptr);
      u.user->replaceInput(u.offset, dq->output());
    }
  }
}

void InsertQuantDequantNodes(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::unordered_map<std::string, std::tuple<std::string, float, int>>&
        qparam_dict) {
  script::Method method = moduleObj.get_method(method_name);
  InsertQuantDequantNodes(method.graph(), qparam_dict);
}

void QuantLinting(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void InsertQuantDequantNodesForParam(
    script::Method& method,
    const std::string& param_name,
    const std::function<std::tuple<std::string, float, int>(at::Tensor)>&
        getQParamFunc,
    at::ScalarType t) {
  TORCH_CHECK(getQParamFunc != nullptr);
  auto params_to_insert_qdq = getQuantizableParamsofName(method, param_name);

  for (auto& param_info : params_to_insert_qdq) {
    auto qparam = getQParamFunc(param_info.value);
    Node* dq = addQuantDeQuantNodesFor(
        param_info.v, param_info.v->node()->next(), qparam, t);
    TORCH_INTERNAL_ASSERT(dq != nullptr);
    param_info.consumer.user->replaceInput(param_info.consumer.offset, dq->output());
  }
}

void InsertQuantDequantNodesForParam(
    script::Method& method,
    const std::string& param_name,
    const std::function<std::tuple<std::string, float, int>(float, float)>&
        getQParamFunc,
    at::ScalarType t) {
  TORCH_CHECK(getQParamFunc != nullptr);
  auto params_to_insert_qdq = getQuantizableParamsofName(method, param_name);

  for (const ParamInfo& param_info : params_to_insert_qdq) {
    // This getQParamFunc requires scale for weight and activation because for
    // quantized ops that involve matmul with weight and bias(WX+b), input scale
    // for bias is computed from input activation and weight. if weight attr
    // not present we skip inserting q-dq node.
    Node* n = param_info.consumer.user;
    auto param_index = n->schema().argumentIndexWithName("weight");
    if (!param_index) {
      continue;
    }
    std::vector<size_t> node_inputs_idx{0, (size_t)*param_index};
    std::array<float, 2> scale_factors = {0, 0};
    bool skip_node = false;
    for (size_t idx = 0; idx < node_inputs_idx.size(); idx++) {
      size_t input_index = node_inputs_idx[idx];
      Value* input_value = n->inputs()[input_index];
      Node* n_input_value = input_value->node();
      Value* scale_value = getScaleValue(n_input_value);
      if (!scale_value) {
        // Dequant node pattern for input is missing
        skip_node = true;
        break;
      }
      c10::IValue scale_ivalue = toIValue(scale_value).value();
      float input_scale = static_cast<float>(scale_ivalue.toDouble());
      TORCH_CHECK(input_scale != 0.0);
      scale_factors[idx] = input_scale;
    }
    if (skip_node) {
      continue;
    }
    auto bias_qparam = getQParamFunc(scale_factors[0], scale_factors[1]);
    Node* dq = addQuantDeQuantNodesFor(
        param_info.v, param_info.v->node()->next(), bias_qparam, t);
    TORCH_INTERNAL_ASSERT(dq != nullptr);
    param_info.consumer.user->replaceInput(param_info.consumer.offset, dq->output());
  }
}

// Exposing the template api helps reuse the same interface for different
// qparamfunc for different qschemes and params.
template <typename Fn>
void InsertQuantDequantNodesForParam(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const Fn& getQParamFunc,
    at::ScalarType t) {
  script::Method method = moduleObj.get_method(method_name);
  InsertQuantDequantNodesForParam(method, param_name, getQParamFunc, t);
}

// Explicit Supported Template specialization for getQParamFunc.
template TORCH_API void InsertQuantDequantNodesForParam(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const std::function<std::tuple<std::string, float, int>(at::Tensor)>&
        getQParamFunc,
    at::ScalarType t);

template TORCH_API void InsertQuantDequantNodesForParam(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const std::function<std::tuple<std::string, float, int>(float, float)>&
        getQParamFunc,
    at::ScalarType t);


static Node* prepQuantAddObserverFor(
    Value* v,
    Node* original_observer_node,
    Node* insert_point) {
  TORCH_INTERNAL_ASSERT(insert_point != nullptr);
  WithInsertPoint ins(insert_point);

  // Create a new observer node. We just need to clone the original one.
  Node* observerNode = insert_point->owningGraph()->createClone(
      &*original_observer_node, [&](Value* v) { return v; }, false);

  // Set the type and the name of the output of the new observer node. It will
  // be used instead of the original value v.
  Value* observedValue = observerNode->addOutput();
  observedValue->setType(v->type());
  observedValue->setDebugName(v->debugName() + ".observed");

  // Now we can add the inputs.
  observerNode->addInput(v);
  return observerNode;
}

TORCH_API void PrepareQuant(
    const script::Module& module,
    const std::string& method_name,
    const script::Module& observer_module) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();
  auto num_activation_inputs = method.num_inputs();
  TORCH_CHECK(graph != nullptr);
  // num_activation_inputs is the number of activations or external data
  // excluding the parameters
  TORCH_CHECK(num_activation_inputs <= graph->inputs().size());
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
  Node* insert_node = *graph->nodes().begin();
  for (size_t idx = 0; idx < num_activation_inputs; ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      Node* observer_node = insertObserverForwardCall(v, v->owningGraph(), module, observer_module);
      // Node* new_observer_node = addObserverFor(v, observer_node, insert_node);
      // new_observer_node->insertBefore(insert_node);
      observer_for_input.emplace(observer_node);
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip nodes that we don't need to observe, e.g. 'prim::Constant' or
      // observer nodes
      if (!outputsNeedToBeObserved(n) || observer_for_input.count(n) != 0) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        values_to_observe.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // Actually add observer nodes.
  for (Value* v : values_to_observe) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      // Skip inserting observer for bias
      if (v->node()->kind() == prim::GetAttr && v->node()->s(c10::attr::name) == "bias") {
        continue;
      }
      insertObserverForwardCall(v, v->owningGraph(), module, observer_module);
    }
  }
}

Node* insertQuantDeQuantCall(Value* v, IValue qparams, at::ScalarType t, bool after=true) {
  Graph* g = v->node()->owningGraph();
  Node* quant = createQuantNode(v, g);
  Node* intrepr = createIntReprNode(v, g);
  Node* dequant = createDeQuantNode(v, g);
  Node* insert_point = after ? v->node() : *g->nodes().begin();
  WithInsertPoint ins(insert_point);

  // Add quant-intrepr-dequant nodes and replace for all uses of Value
  std::cout << "insertQuantDeQuantCall for " << v->debugName() << std::endl;
  // Create qparam constant nodes
  TORCH_INTERNAL_ASSERT(qparams.isTuple(), "qparams must be tuple");
  auto tp = qparams.toTuple();
  IValue scale = tp->elements()[0];
  IValue zero_point = tp->elements()[1];
  Value* scale_val = g->insertConstant(scale);
  Value* zero_point_val = g->insertConstant(zero_point);

  // Insert quant/int_repr/dequant nodes
  if (after) {
    quant->insertAfter(insert_point);
  } else {
    quant->insertBefore(insert_point);
  }

  intrepr->insertAfter(quant);
  dequant->insertAfter(intrepr);

  // Attach inputs to quantization pattern nodes
  quant->addInput(v);
  intrepr->addInput(quant->output());
  dequant->addInput(intrepr->output());

  quant->addInput(scale_val);
  quant->addInput(zero_point_val);
  dequant->addInput(scale_val);
  dequant->addInput(zero_point_val);

  Value* scalar_type_val = insertScalarType(quant, t);
  TORCH_INTERNAL_ASSERT(scalar_type_val != nullptr);
  quant->addInput(scalar_type_val);
  dequant->addInput(scalar_type_val);
  return dequant;
}

IValue getQParam(const script::Module& module, Value* v) {
    TORCH_INTERNAL_ASSERT(v->type()->isSubtypeOf(TensorType::get()));
    auto observer_module_name = "observer_for_" + v->debugName();
    auto observer_module = module.find_module(observer_module_name);
    std::cout << "trying to get observer: " << observer_module_name << std::endl;
    TORCH_INTERNAL_ASSERT(observer_module);
    auto calculate_qparams = (*observer_module).get_method("calculate_qparams");
    IValue qparams = calculate_qparams(std::vector<IValue>());
    return qparams;
}

void quantizeBias(const script::Module& module, Value* v) {
  // Traverse to the place where this is used
  Node* aten_call_node = nullptr;
  std::vector<std::string> ops_with_bias = {"aten::conv2d", "aten::_convolution"};
  for (const auto& use: v->uses()) {
    if (std::find(ops_with_bias.begin(), ops_with_bias.end(), use.user->kind().toQualString()) != ops_with_bias.end()) {
      Value* activation = use.user->inputs()[0];
      Value* weight = use.user->inputs()[1];
      // Get qparam from activation
      IValue act_qparam = getQParam(module, activation);
      // Get qparam from weight
      IValue weight_qparam = getQParam(module, weight);
      IValue bias_scale =  1.0 / act_qparam.toTuple()->elements()[0].toDouble() / weight_qparam.toTuple()->elements()[0].toDouble();
      IValue bias_qparam = c10::ivalue::Tuple::create(std::vector<IValue>({bias_scale, IValue(0)}), act_qparam.toTuple()->type);
      Node* dequant = insertQuantDeQuantCall(v, bias_qparam, at::kQInt32);
      v->replaceAllUsesWith(dequant->output());
      Node* q = traverseToQuantNode(dequant);
      TORCH_INTERNAL_ASSERT(q != nullptr);
      q->replaceInputWith(dequant->output(), v);
      break;
    }
  }
}

script::Module InsertQuantDeQuant(
    const script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();
  TORCH_CHECK(graph != nullptr);
  auto num_activation_inputs = method.num_inputs();
  std::vector<Value*> values_to_observe;
  std::vector<Value*> input_values;

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
  for (size_t idx = 0; idx < num_activation_inputs; ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      input_values.push_back(v);
    }
  }

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip nodes that we don't need to observe, e.g. 'prim::Constant' or
      // observer nodes
      if (!outputsNeedToBeObserved(n)) { // || observer_for_input.count(n) != 0) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        values_to_observe.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  std::vector<std::string> observer_modules_to_remove;
  std::vector<Node*> nodes_to_destroy;

  // quantize bias first
  for (Value* v : values_to_observe) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto observer_module_name = "observer_for_" + v->debugName();
      auto observer_module = module.find_module(observer_module_name);
      if (!observer_module) {
        if (v->node()->kind() == prim::GetAttr && v->node()->s(c10::attr::name) == "bias") {
          quantizeBias(module, v);
        }
      }
    }
  }

  for (Value* v : values_to_observe) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto observer_module_name = "observer_for_" + v->debugName();
      auto observer_module = module.find_module(observer_module_name);
      if (observer_module) {
        std::cout << "found observer module for " << observer_module_name << std::endl;
        // remove observer_module
        observer_modules_to_remove.push_back(observer_module_name);
        // remove observer forward call
        for (const Use& u: v->uses()) {
          Node* user = u.user;
          if (user->kind() == prim::CallMethod && user->s(c10::attr::name) == "forward" && user->inputs()[0]->debugName() == observer_module_name) {
            // Observer forward call node
            nodes_to_destroy.push_back(user);
            // GetAttr node for observer module
            nodes_to_destroy.push_back(user->inputs()[0]->node());
          }
        }
        auto calculate_qparams = (*observer_module).get_method("calculate_qparams");
        IValue qparams = calculate_qparams(std::vector<IValue>());
        Node* dequant = insertQuantDeQuantCall(v, qparams, at::kQInt8);
        v->replaceAllUsesWith(dequant->output());
        Node* q = traverseToQuantNode(dequant);
        TORCH_INTERNAL_ASSERT(q);
        q->replaceInputWith(dequant->output(), v);
      }
    }
  }

  // TODO: Remove
  for (Value* v : input_values) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto observer_module_name = "observer_for_" + v->debugName();
      auto observer_module = module.find_module(observer_module_name);
      if (observer_module) {
        std::cout << "found observer module for " << observer_module_name << std::endl;
        // remove observer_module
        observer_modules_to_remove.push_back(observer_module_name);
        // remove observer forward call
        for (const Use& u: v->uses()) {
          Node* user = u.user;
          if (user->kind() == prim::CallMethod && user->s(c10::attr::name) == "forward" && user->inputs()[0]->debugName() == observer_module_name) {
            // Observer forward call node
            nodes_to_destroy.push_back(user);
            // GetAttr node for observer module
            nodes_to_destroy.push_back(user->inputs()[0]->node());
          }
        }
        auto calculate_qparams = (*observer_module).get_method("calculate_qparams");
        IValue qparams = calculate_qparams(std::vector<IValue>());
        Node* dequant = insertQuantDeQuantCall(v, qparams, at::kQInt8, false);
        v->replaceAllUsesWith(dequant->output());
        Node* q = traverseToQuantNode(dequant);
        TORCH_INTERNAL_ASSERT(q);
        q->replaceInputWith(dequant->output(), v);
      }
    }
  }
  // Destroy observer forward calls
  for (auto& n: nodes_to_destroy) {
    n->destroy();
  }

  // Remove observer module, moudle API doesn't support
  // remove, we'll create a new module but skip registering
  // modules in `observer_modules_to_remove`
  script::Module result(c10::QualifiedName(module.name().qualifiedName()), module.class_compilation_unit(), true);
  for (const auto& param: module.get_parameters()) {
    result.register_parameter(param.name(), module.get_parameter(param.name()), false);
  }

  for (const auto& attr: module.get_attributes()) {
    result.register_attribute(attr.name(), attr.type(), module.get_attribute(attr.name()));
  }

  for (const auto& mod: module.get_module_slots()) {
    if (std::find(observer_modules_to_remove.begin(), observer_modules_to_remove.end(), mod.name()) == observer_modules_to_remove.end()) {
      result.register_module(mod.name(), module.get_module(mod.name()));
    }
  }

  return result;
}

void QuantFusion(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;
  std::string pattern = R"(
graph(%a, %w, %b, %a_scale, %a_zero_point, %a_dtype, %w_scale, %w_zero_point, %w_dtype, %b_scale, %b_zero_point, %b_dtype, %r_scale, %r_zero_point, %r_dtype, %c, %d, %e, %f):
        %a_quant = aten::quantize_linear(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_intrepr = aten::int_repr(%a_quant)
        %a_dequant = aten::_dequantize_linear(%a_intrepr, %a_scale, %a_zero_point, %a_dtype)
        %w_quant = aten::quantize_linear(%w, %w_scale, %w_zero_point, %w_dtype)
        %w_intrepr = aten::int_repr(%w_quant)
        %w_dequant = aten::_dequantize_linear(%w_intrepr, %w_scale, %w_zero_point, %w_dtype)
        %b_quant = aten::quantize_linear(%b, %b_scale, %b_zero_point, %b_dtype)
        %b_intrepr = aten::int_repr(%b_quant)
        %b_dequant = aten::_dequantize_linear(%b_intrepr, %b_scale, %b_zero_point, %b_dtype)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_dequant, %c, %d, %e, %f)
        %r_quant = aten::quantize_linear(%r, %r_scale, %r_zero_point, %r_dtype)
        %r_intrepr = aten::int_repr(%r_quant)
        %r_dequant = aten::_dequantize_linear(%r_intrepr, %r_scale, %r_zero_point, %r_dtype)
        return (%r_dequant))";

      // aten function for fbgemm_conv is not ready yet, so the following
      // code is not runnable
  std::string replacement = R"(
graph(%a, %w, %b, %a_scale, %a_zero_point, %a_dtype, %w_scale, %w_zero_point, %w_dtype, %b_scale, %b_zero_point, %b_dtype, %r_scale, %r_zero_point, %r_dtype, %c, %d, %e, %f):
        %a_quant = aten::quantize_linear(%a, %a_scale, %a_zero_point, %a_dtype)
        %w_quant = aten::quantize_linear(%w, %w_scale, %w_zero_point, %w_dtype)
        %b_quant = aten::quantize_linear(%b, %b_scale, %b_zero_point, %b_dtype)
        %r = aten::conv2d(%a_quant, %w_quant, %b_quant, %c, %d, %e, %f)
        %r_intrepr = aten::int_repr(%r)
        %r_dequant = aten::_dequantize_linear(%r_intrepr, %r_scale, %r_zero_point, %r_dtype)
        return (%r_dequant))";
  rewriter.RegisterRewritePattern(pattern, replacement);
  rewriter.runOnGraph(graph);
}

} // namespace jit
} // namespace torch
