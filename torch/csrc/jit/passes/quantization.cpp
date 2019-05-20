#include <torch/csrc/jit/passes/quantization.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>

#include <stack>

namespace torch {
namespace jit {
namespace {
// QuantizerUtils
struct param_info_t {
  Value* v;
  Node* n; // Value consumer
  size_t idx; // Index in input param vector
};

Operator* checkIfNodeQuantizable(Node* n) {
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
  return quantnodeLookup.find(n);
}

// Look for index of particular param in op schema
size_t getParamIndexinOpArgs(Node* n, const std::string& param_name) {
  TORCH_INTERNAL_ASSERT(n != nullptr);
  Operator* optr = checkIfNodeQuantizable(n);
  if (optr == nullptr) {
    return static_cast<size_t>(-1);
  }
  auto& opargs = optr->schema().arguments();
  for (size_t idx = 0; idx < opargs.size(); idx++) {
    if (opargs[idx].name() == param_name) {
      return idx;
    }
  }
  return static_cast<size_t>(-1);
}

std::vector<param_info_t> getQuantizableParamsofName(
    script::Method& method,
    const std::string& param_name) {
  std::vector<param_info_t> params_to_insert_qdq;
  auto graph = method.graph();
  // Parameters input to this method
  size_t param_input_len = method.initial_ivalues().size();
  // External inputs to this method
  size_t ext_input_len = method.num_inputs();
  std::unordered_map<Node*, size_t> node_paramidx_map;

  for (size_t idx = 0; idx < param_input_len; idx++) {
    auto& v = graph->inputs()[idx + ext_input_len];
    if (!v->type()->isSubtypeOf(TensorType::get()) || !v->hasUses()) {
      continue;
    }
    Node* n = v->uses()[0].user;

    // For every param name, we match it against the quantizable op schema and
    // find its position. if the param is present we store it in vector so
    // later we can insert quant-dequant nodes. Caching the param index helps
    // faster lookup for same kind of node visited multiple timees.
    size_t param_idx;
    auto it = node_paramidx_map.find(n);
    if (it != node_paramidx_map.end()) {
      param_idx = it->second;
    } else {
      param_idx = getParamIndexinOpArgs(n, param_name);
      node_paramidx_map.emplace(n, param_idx);
    }
    if (param_idx >= n->inputs().size() || n->inputs()[param_idx] != v) {
      // Either node does not contain param or this Value
      // is not of type param_name
      continue;
    }
    params_to_insert_qdq.emplace_back(param_info_t{v, n, idx});
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
Node* createQuantNode(Value* v, Node* n) {
  Node* quant = n->owningGraph()->create(
      at::Symbol::fromQualString("aten::quantize_linear"));
  TORCH_INTERNAL_ASSERT(quant != nullptr, "Failed to create quant node");
  quant->output()->setUniqueName(v->uniqueName() + ".quant");
  quant->setScope(n->scope());
  return quant;
}

// Create Dequant node
Node* createDeQuantNode(Value* v, Node* n) {
  Node* dequant = n->owningGraph()->create(
      at::Symbol::fromQualString("aten::dequantize_linear"));
  TORCH_INTERNAL_ASSERT(dequant != nullptr, "Failed to create dequant node");
  dequant->output()->setUniqueName(v->uniqueName() + ".dequant");
  dequant->setScope(n->scope());
  return dequant;
}

// Create IntTensor Node
Node* createIntReprNode(Value* v, Node* n) {
  Node* intrepr =
      n->owningGraph()->create(at::Symbol::fromQualString("aten::int_repr"));
  TORCH_INTERNAL_ASSERT(intrepr != nullptr, "Failed to create inttensor node");
  intrepr->output()->setUniqueName(v->uniqueName() + ".intrepr");
  intrepr->setScope(n->scope());
  return intrepr;
}

// Insert Quant-Dequant node pattern for quantizable node outputs
void addQuantDeQuantNodes(
    Value* v,
    const std::tuple<std::string, float, int>& qparam,
    at::ScalarType t = at::ScalarType::Undefined) {
  TORCH_INTERNAL_ASSERT(v != nullptr);
  Node* n = v->node();
  Node* quant = createQuantNode(v, n);
  Node* intrepr = createIntReprNode(v, n);
  Node* dequant = createDeQuantNode(v, n);

  // Add quant-intrepr-dequant nodes and replace for all uses of Value
  quant->insertAfter(n);
  intrepr->insertAfter(quant);
  dequant->insertAfter(intrepr);
  v->replaceAllUsesWith(dequant->output());

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
  // optional argument required only for quantization
  // of specific attributes eg: bias.
  if (t != at::ScalarType::Undefined) {
    Value* scalartype_v = insertScalarType(quant, t);
    TORCH_INTERNAL_ASSERT(scalartype_v != nullptr);
    quant->addInput(scalartype_v);
    dequant->addInput(scalartype_v);
  }
}

// Insert Quant-Dequant node pattern for specific input to node n
void addQuantDeQuantNodesForInput(
    Value* v,
    Node* n,
    const std::tuple<std::string, float, int>& qparam,
    at::ScalarType t = at::ScalarType::Undefined) {
  TORCH_INTERNAL_ASSERT(v != nullptr);
  TORCH_INTERNAL_ASSERT(n != nullptr);
  Node* quant = createQuantNode(v, n);
  Node* intrepr = createIntReprNode(v, n);
  Node* dequant = createDeQuantNode(v, n);

  // Insert the quant-intrepr-dequant node for the V->N
  // pair which is identified as quantizable during
  // graph iteration
  dequant->insertBefore(n);
  intrepr->insertBefore(dequant);
  quant->insertBefore(intrepr);
  n->replaceInputWith(v, dequant->output());

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
  if (t != at::ScalarType::Undefined) {
    Value* scalartype_v = insertScalarType(quant, t);
    TORCH_INTERNAL_ASSERT(scalartype_v != nullptr);
    quant->addInput(scalartype_v);
    dequant->addInput(scalartype_v);
  }
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
  Value* vname = insert_point->owningGraph()->insertConstant(v->uniqueName());

  // Create a new observer node. We just need to clone the original one.
  Node* observerNode = insert_point->owningGraph()->createClone(
      &*original_observer_node, [&](Value* v) { return v; }, false);

  // Set the type and the name of the output of the new observer node. It will
  // be used instead of the original value v.
  Value* observedValue = observerNode->addOutput();
  observedValue->setType(v->type());
  observedValue->setUniqueName(v->uniqueName() + ".observed");

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
    std::shared_ptr<script::Module>& moduleObj,
    const std::string& method_name,
    Node* observer_node) {
  const auto& method = moduleObj->get_method(method_name);
  InsertObserverNodes(method.graph(), observer_node, method.num_inputs());
}

void InsertObserverNodes(
    std::shared_ptr<script::Function>& function_var,
    Node* observer_node) {
  InsertObserverNodes(
      function_var->graph(), observer_node, function_var->num_inputs());
}

void InsertQuantDequantNodes(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<std::string, std::tuple<std::string, float, int>>&
        qparam_dict) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // For storing quantizable values - node pairs that are external
  // or intermediate inputs to quantizable nodes
  std::vector<std::pair<Value*, Node*>> quantInputs;
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
      for (auto& v : n->inputs()) {
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          // Skip quantization for non tensors
          continue;
        }

        if (checkIfNodeQuantizable(v->node())) {
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
        } else if (checkIfNodeQuantizable(n)) {
          // Goal of this iteration is to identify nodes that are
          // quantizable but input value originate from non quantizable
          // node. This requires selectively inserting q-dq nodes for
          // inputs into node N(V, N pair) because parent node might
          // also have input into other non quantizable nodes
          // Example : N1(prim::Param) -> (V1 -> (N4, N5), V2 -> (N6, N7), V3)
          //           N1 is not quantizable node but N4 and N7 are
          //           quantizable nodes. So we add the (V1, N4) and
          //           (V2, N7) as insertion points for quant-dequant nodes
          quantInputs.emplace_back(v, n);
        }
      }
    } // End Loop for nodes within block

    // Since we are iterating node inputs only above, we need to iterate
    // over block outputs values and if they originate from quantizable
    // node, we push to quantOutputs set
    auto outputVals = b->outputs();
    for (auto& v : outputVals) {
      if (checkIfNodeQuantizable(v->node()) &&
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
  for (auto& ele : quantOutputs) {
    if (qparam_value_dict.count(ele) != 0) {
      addQuantDeQuantNodes(ele, qparam_value_dict[ele]);
    }
  }

  // Insert the quant-dequant pair for values inputs to quantizable nodes
  for (auto& ele : quantInputs) {
    if (qparam_value_dict.count(ele.first) != 0) {
      addQuantDeQuantNodesForInput(
          ele.first, ele.second, qparam_value_dict[ele.first]);
    }
  }
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
    auto& param_slot = method.initial_ivalues()[param_info.idx];
    const auto& itensor = param_slot.value();
    at::Tensor tensor_var = itensor.toTensor().detach();
    auto qparam = getQParamFunc(tensor_var);
    addQuantDeQuantNodesForInput(param_info.v, param_info.n, qparam, t);
  }
}

// Exposing the template api helps reuse the same interface for different
// qparamfunc for different qschemes.
template <typename Fn>
void InsertQuantDequantNodesForParam(
    std::shared_ptr<script::Module>& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const Fn& getQParamFunc,
    at::ScalarType t) {
  auto& method = moduleObj->get_method(method_name);
  InsertQuantDequantNodesForParam(method, param_name, getQParamFunc, t);
}

// Explicit Supported Template specialization for getQParamFunc.
template TORCH_API void InsertQuantDequantNodesForParam(
    std::shared_ptr<script::Module>& moduleObj,
    const std::string& method_name,
    const std::string& param_name,
    const std::function<std::tuple<std::string, float, int>(at::Tensor)>&
        getQParamFunc,
    at::ScalarType t);

} // namespace jit
} // namespace torch
