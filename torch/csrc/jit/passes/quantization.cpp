#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>

#include <stack>

namespace torch {
namespace jit {
namespace {

Node* traverseToQuantNode(Node* dq) {
  TORCH_INTERNAL_ASSERT(dq != nullptr);
  TORCH_INTERNAL_ASSERT(dq->inputs().size() != 0);
  Node* intrepr = dq->inputs()[0]->node();
  TORCH_INTERNAL_ASSERT(intrepr != nullptr);
  TORCH_INTERNAL_ASSERT(intrepr->inputs().size() != 0);
  return intrepr->inputs()[0]->node();
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

// c10::optional<QConfig> getQConfig(Value* v, script::Module module, const QConfigDict& qconfig_dict) {
//   TORCH_INTERNAL_ASSERT(v->node()->kind() == prim::GetAttr,
//                         "We can only get qconfig for output of GetAttr node");
//   // module that owns value
//   Value* vi = v->node()->inputs()[0];
//   // name for the module
//   std::string key = vi->node()->s(c10::attr::name);
//   // move up the module hierarchy
//   vi = vi->node()->inputs()[0];
//   std::cout << "module: " << key << std::endl;
//   while (vi->node()->kind() == prim::GetAttr) {
//     key = vi->node()->s(c10::attr::name) + "." + key;
//     std::cout << "module: " << key << std::endl;
//     vi = vi->node()->inputs()[0];
//   }
//   // now we constructed the "absolute path" for the current scope
//   // for example: sub1.sub2.conv

//   // Find qconfig
//   while(std::count(key.begin(), key.end(), '.') > 0) {
//     if (qconfig_dict.find(key) != qconfig_dict.end()) {
//       std::cout << "returning: " << key;
//       return qconfig_dict.at(key);
//     }
//     // move up hierarchy by removing the last part after "."
//     auto pos = key.rfind('.');
//     key = key.substr(0, pos);
//   }

//   if (qconfig_dict.find(key) != qconfig_dict.end()) {
//     std::cout << "returning: " << key;
//     return qconfig_dict.at(key);
//   } else {
//     return c10::nullopt;
//   }
// }

c10::optional<QConfig> getQConfig(std::string key, c10::optional<QConfig> parent_qconfig, const QConfigDict& qconfig_dict) {
  if (qconfig_dict.find(key) != qconfig_dict.end()) {
    return qconfig_dict.at(key);
  }
  return parent_qconfig;
}

// Clone observer module and add it to the original module,
// and insert a call to observer forward function
Node* insertObserver(Value* v, Graph* g,
                     script::Module module,
                     const QConfigDict& qconfig_dict,
                     const std::string& key,
                     c10::optional<QConfig> parent_qconfig) {
  if (v->node()->kind() == prim::CallMethod && v->node()->s(attr::name) == "forward") {
    auto child_instance = v->node()->inputs()[0];
    TORCH_INTERNAL_ASSERT(child_instance->node()->kind() == prim::GetAttr, "Child instance should come from GetAttr.");
    auto child_module_name = child_instance->node()->s(attr::name);
    auto child_module = module.find_module(child_module_name);
    TORCH_INTERNAL_ASSERT(child_module, "Child module " + child_module_name + " does not exist");
    std::string child_key = key;
    if (child_key == "") {
      child_key = child_module_name;
    } else {
      child_key = key + "." + child_module_name;
    }
    auto m = InsertObservers(child_module.value(), "forward", qconfig_dict, child_key, parent_qconfig);
  }
  script::Module observer_module;
  auto qconfig = getQConfig(key, parent_qconfig, qconfig_dict);
  // Skip observer if no qconfig is found
  if (!qconfig) {
    return nullptr;
  }
  if (v->node()->kind() == prim::GetAttr && v->node()->s(attr::name) == "weight") {
    std::tie(std::ignore, observer_module) = qconfig.value();
  } else {
    std::tie(observer_module, std::ignore) = qconfig.value();
  }
  std::string observer_name = "observer_for_" + v->debugName();
  script::Module observer = observer_module.clone();
  module.register_module(observer_name, observer);
  // Get handle of observer module
  Node* observer_instance = g->create(c10::prim::GetAttr);
  // self.observer_for_v
  observer_instance->addInput(g->inputs()[0]);
  observer_instance->s_(c10::attr::name, observer_name);
  observer_instance->output()->setDebugName(observer_name);
  observer_instance->output()->setType(observer.type());
  observer_instance->insertAfter(v->node());

  // Create forward method call
  Node* call = g->create(c10::prim::CallMethod);
  TORCH_INTERNAL_ASSERT(call != nullptr, "Failed to create forward call node");
  call->s_(c10::attr::name, "forward");
  call->addInput(observer_instance->output());
  call->addInput(v);
  call->output()->setType(v->type());
  call->output()->setDebugName(v->debugName() + ".observed");
  call->insertAfter(observer_instance);
  return call;
}

} // namespace

// PyBind APIs
void PropagateQuantInfo(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

static bool outputsNeedToBeObserved(Node* n) {
  return n->kind() != prim::Constant;
}

void QuantLinting(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

TORCH_API script::Module InsertObservers(
    const script::Module& module,
    const std::string& method_name,
    const QConfigDict& qconfig_dict,
    const std::string& key,
    c10::optional<QConfig> parent_qconfig) {
  script::Module input_module = module;
  script::Method method = input_module.get_method(method_name);
  auto graph = method.graph();
  TORCH_CHECK(graph != nullptr);
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
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto observer_node = insertObserver(v, v->owningGraph(), input_module, qconfig_dict, key, parent_qconfig);
      if (observer_node) {
        observer_for_input.emplace(observer_node);
      }
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
      } else {
        insertObserver(v, v->owningGraph(), input_module, qconfig_dict, key, parent_qconfig);
      }
    }
  }
  return input_module;
}

Node* insertQuantDeQuantCall(Value* v, const IValue& qparams, at::ScalarType t, bool insert_after=true) {
  Graph* g = v->node()->owningGraph();
  Node* quant = createQuantNode(v, g);
  Node* intrepr = createIntReprNode(v, g);
  Node* dequant = createDeQuantNode(v, g);
  Node* insert_point = insert_after ? v->node() : *g->nodes().begin();
  WithCurrentScope scope_guard(
      *insert_point->owningGraph(), insert_point->scope());

  // Add quant-intrepr-dequant nodes and replace for all uses of Value
  // Create qparam constant nodes
  TORCH_INTERNAL_ASSERT(qparams.isTuple(), "qparams must be tuple");
  auto tp = qparams.toTuple();
  IValue scale = tp->elements()[0].toTensor().item().toFloat();
  IValue zero_point = tp->elements()[1].toTensor().item().toInt();
  Value* scale_val = g->insertConstant(scale);
  Value* zero_point_val = g->insertConstant(zero_point);

  // Insert quant/int_repr/dequant nodes
  if (insert_after) {
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

// find the observer for Value `v` and return the name of the observer
c10::optional<std::string> findObserverName(Value* v) {
  for (const Use& u: v->uses()) {
    // Note that here we just check for the name of observer, but the ideally
    // we should be comparing the type of observer, this is a temporary
    // work around until data only clone of module.clone is supported.
    if (u.user->kind() == prim::CallMethod && u.user->s(attr::name) == "forward") {
      auto module_instance = u.user->inputs().at(0);
      if (module_instance->node()->kind() == prim::GetAttr &&
          module_instance->node()->s(attr::name).find("observer_for_") != std::string::npos) {
        return module_instance->node()->s(attr::name);
      }
    }
  }
  return c10::nullopt;
}

class QuantizeHelper {
 public:
  QuantizeHelper(const script::Module& m) : module_(m) {}
  IValue getQParams(Value* v);
  void quantizeBias(Value* v);
  void quantizeTensor(Value* v, bool insert_after=true);
  void removeObserver(Value* v, const std::string& observer_name);
  void destroyNodes() {
    // Destroy observer forward calls
    for (auto& n: nodes_to_destroy_) {
      n->destroy();
    }
  }

 private:
  const script::Module& module_;
  std::vector<std::string> observer_modules_to_remove_;
  std::vector<Node*> nodes_to_destroy_;
};

void QuantizeHelper::removeObserver(Value* v, const std::string& observer_name) {
  // remove observer_module
  observer_modules_to_remove_.push_back(observer_name);
  // remove observer forward call
  for (const Use& u: v->uses()) {
    Node* user = u.user;
    if (user->kind() == prim::CallMethod &&
        user->s(attr::name) == "forward" &&
        user->inputs()[0]->node()->kind() == prim::GetAttr &&
        user->inputs()[0]->node()->s(attr::name) == observer_name) {
      // Observer forward call node
      nodes_to_destroy_.push_back(user);
      // GetAttr node for observer module
      nodes_to_destroy_.push_back(user->inputs()[0]->node());
    }
  }
}

IValue QuantizeHelper::getQParams(Value* v) {
    TORCH_INTERNAL_ASSERT(v->type()->isSubtypeOf(TensorType::get()));
    auto observer_name = findObserverName(v);
    TORCH_INTERNAL_ASSERT(observer_name,
                          "getQParams expects the corresponding observer for ",
                          v->debugName(),
                          " exists.");
    auto observer_module = module_.find_module(observer_name.value());
    TORCH_INTERNAL_ASSERT(observer_module,
                          "getQParams expects the corresponding observer for ",
                          v->debugName(),
                          " exists.");
    auto calculate_qparams = observer_module.value().get_method("calculate_qparams");
    IValue qparams = calculate_qparams(std::vector<IValue>());
    return qparams;
}

double getScale(const IValue& qparam) {
  return qparam.toTuple()->elements()[0].toTensor().item().toDouble();
}

void QuantizeHelper::quantizeBias(Value* v) {
  // Traverse to the place where this is used
  std::vector<Symbol> ops_with_bias = {Symbol::aten("conv2d"), Symbol::aten("_convolution")};
  for (const auto& use: v->uses()) {
    if (std::find(ops_with_bias.begin(), ops_with_bias.end(),
                  use.user->kind()) != ops_with_bias.end()) {
      // Make sure there is no observer module for bias
      auto observer_name = findObserverName(v);
      TORCH_INTERNAL_ASSERT(!observer_name,
                            "bias should not be observed!");
      Value* activation = use.user->inputs()[0];
      Value* weight = use.user->inputs()[1];
      // Get qparam from activation
      IValue act_qparam = getQParams(activation);
      // Get qparam from weight
      IValue weight_qparam = getQParams(weight);
      IValue bias_scale =  at::scalar_tensor(
          c10::Scalar(getScale(act_qparam) * getScale(weight_qparam)),
          at::kDouble);
      IValue bias_qparam = c10::ivalue::Tuple::create(
          std::vector<IValue>({bias_scale,
                               at::scalar_tensor(c10::Scalar(0))}), act_qparam.toTuple()->type);
      Node* dequant = insertQuantDeQuantCall(v, bias_qparam, at::kQInt32);
      v->replaceAllUsesWith(dequant->output());
      Node* q = traverseToQuantNode(dequant);
      TORCH_INTERNAL_ASSERT(q != nullptr);
      q->replaceInputWith(dequant->output(), v);
    }
  }
}

void QuantizeHelper::quantizeTensor(Value* v,
                                    bool insert_after) {
  auto observer_name = findObserverName(v);
  if (!observer_name) {
    return;
  }
  IValue qparams = getQParams(v);
  removeObserver(v, observer_name.value());
  Node* dequant;
  if (v->node()->kind() == prim::GetAttr && v->node()->s(c10::attr::name) == "weight") {
    dequant = insertQuantDeQuantCall(v, qparams, at::kQInt8);
  } else {
    dequant = insertQuantDeQuantCall(v, qparams, at::kQUInt8, insert_after);
  }
  v->replaceAllUsesWith(dequant->output());
  Node* q = traverseToQuantNode(dequant);
  TORCH_INTERNAL_ASSERT(q);
  q->replaceInputWith(dequant->output(), v);
}

void InsertQuantDeQuant(
    script::Module& module,
    const std::string& method_name) {
  script::Method method = module.get_method(method_name);
  auto graph = method.graph();
  std::vector<Value*> values_to_quantize;
  std::vector<Value*> input_values;

  // For traversing all blocks in the graph including subblocks.
  std::stack<Block*> blocks_to_visit;

  // Add observer for external input nodes excluding parameters
  // These are treated as activation as they vary across batches
  // and need to be observed.

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  for (size_t idx = 1; idx < method.num_inputs(); ++idx) {
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
      if (!outputsNeedToBeObserved(n)) {
        continue;
      }

      for (Value* v : n->outputs()) {
        values_to_quantize.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  QuantizeHelper qh(module);

  for (Value* v : values_to_quantize) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      if (v->node()->kind() == prim::GetAttr && v->node()->s(c10::attr::name) == "bias") {
        qh.quantizeBias(v);
      } else {
        qh.quantizeTensor(v);
      }
    }
  }

  for (Value* v : input_values) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      qh.quantizeTensor(v, false);
    }
  }

  qh.destroyNodes();

  // NOTE: Remove observer module does not work right now, we'll return
  // the module with observer modules as a temporary workaround
  // TODO: remove observer modules after we have a remove_module API
}

void QuantFusion(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;
  std::string pattern = R"(
graph(%a_quant, %w_quant, %b_quant, %a_scale, %a_zero_point, %a_dtype, %w_scale, %w_zero_point, %w_dtype, %b_scale, %b_zero_point, %b_dtype, %r_scale, %r_zero_point, %r_dtype, %c, %d, %e, %f):
        %a_intrepr = aten::int_repr(%a_quant)
        %a_dequant = aten::_dequantize_linear(%a_intrepr, %a_scale, %a_zero_point, %a_dtype)
        %w_intrepr = aten::int_repr(%w_quant)
        %w_dequant = aten::_dequantize_linear(%w_intrepr, %w_scale, %w_zero_point, %w_dtype)
        %b_intrepr = aten::int_repr(%b_quant)
        %b_dequant = aten::_dequantize_linear(%b_intrepr, %b_scale, %b_zero_point, %b_dtype)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_dequant, %c, %d, %e, %f)
        %r_quant = aten::quantize_linear(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

  std::string replacement = R"(
graph(%a_quant, %w_quant, %b_quant, %a_scale, %a_zero_point, %a_dtype, %w_scale, %w_zero_point, %w_dtype, %b_scale, %b_zero_point, %b_dtype, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %3 : int = prim::Constant[value=3]()
        %in_param : int[] = prim::ListConstruct(%0, %2, %3, %1)
        %a_perm : Tensor = aten::permute(%a_quant, %in_param)
        %w_perm : Tensor = aten::permute(%w_quant, %in_param)
        %w_packed = quantized::fbgemm_conv_prepack(%w_perm, %stride, %padding, %dilation, %groups)
        %r = quantized::fbgemm_conv2d(%a_perm, %w_packed, %b_quant, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
        %out_param : int[] = prim::ListConstruct(%0, %3, %1, %2)
        %r_perm = aten::permute(%r, %out_param)
        return (%r_perm))";
  rewriter.RegisterRewritePattern(pattern, replacement);
  rewriter.runOnGraph(graph);
}

} // namespace jit
} // namespace torch
