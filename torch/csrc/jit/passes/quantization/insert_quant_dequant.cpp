#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>

#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <stack>

namespace torch {
namespace jit {

namespace {
using graph_rewrite_helper::PatternInfo;

// dynamic quantization ops for activation: choose_qparams, quant, dequant
using DynamicQuantOps = std::tuple<Node*, Node*, Node*>;

std::string kScalarType = "_scalar_type";

struct QuantOpParams {
  c10::QScheme qscheme{c10::kPerTensorAffine};
  std::vector<Value*> qparams;
  // This is only so that insertQuantizationOps can be templatized
  // and subsequntly significant portion of that code can be reused.
  std::string back() const {
    return "AttributeDoesNotExist";
  }
};

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

bool isPerChannel(at::QScheme qscheme) {
  return qscheme == c10::kPerChannelAffine ||
      qscheme == c10::kPerChannelSymmetric;
}

// Go through the CallMethod graph to check if the value is Weight.
bool isWeight(Module& module, Value* v) {
  if (isWeight(v)) {
    return true;
  }
  c10::optional<bool> result;
  auto* self = v->owningGraph()->inputs()[0];
  for (const Use& u : v->uses()) {
    Node* n = u.user;
    if (n->kind() == prim::CallMethod) {
      auto m_opt = getInvokedModuleOpt(module, n, self);
      if (!m_opt.has_value()) {
        return false;
      }
      auto m = *m_opt;
      auto g = m.get_method(n->s(attr::name)).graph();
      auto call_method_result = isWeight(m, g->inputs()[u.offset]);
      if (result.has_value()) {
        // Check to make sure all the CallMethods in the graph produce the same
        // output.
        TORCH_CHECK(
            call_method_result == result.value(),
            "Expected all CallMethods to use either weight "
            "or non-weight value.",
            v->debugName());
      } else {
        result = call_method_result;
      }
    }
  }
  return result.has_value() ? result.value() : false;
}

Node* insertChooseQParams(Graph* graph, Value* original_val) {
  std::string choose_qparams_func = "_choose_qparams_per_tensor";
  // Set the reduce range to default to true, since qnnpack backend ignores this
  // argument.
  bool reduce_range_param = true;
  auto reduce_range = graph->insertConstant(reduce_range_param);
  // choose_qparams_per_tensor has 2 outputs, (scale, zero_point).
  Node* choose_qparams = graph->create(
      at::Symbol::aten(choose_qparams_func),
      {original_val, reduce_range},
      /* num_outputs = */ 2);
  choose_qparams->output(0)->setDebugName(original_val->debugName() + ".scale");
  choose_qparams->output(0)->setType(FloatType::get());
  choose_qparams->output(1)->setDebugName(
      original_val->debugName() + ".zero_point");
  choose_qparams->output(1)->setType(IntType::get());
  graph->insertNode(choose_qparams);
  return choose_qparams;
}

Node* insertQuant(
    Graph* graph,
    const std::vector<Value*>& inputs,
    NodeKind quant_kind,
    const std::string& debugName) {
  Node* quant = graph->create(quant_kind, inputs);
  quant->output()->setDebugName(debugName);
  graph->insertNode(quant);
  return quant;
}

Node* insertDeQuant(
    Graph* graph,
    Value* quantized_val,
    Value* original_val,
    size_t id = 0) {
  Node* dequant = graph->create(Symbol::aten("dequantize"), {quantized_val});
  dequant->output()
      ->setDebugName(
          original_val->debugName() + ".dequant." + c10::guts::to_string(id))
      ->setType(original_val->type());
  graph->insertNode(dequant);
  return dequant;
}

std::vector<Value*> insertDeQuantForAllUse(
    Graph* graph,
    Value* quantized_val,
    Value* original_val) {
  // copy uses to vector since value->uses() is a reference
  // and changing the graph will also change the uses() list
  const std::vector<Use> uses = original_val->uses();
  std::vector<Value*> outputs;
  for (const auto i : c10::irange(uses.size())) {
    auto* user = uses[i].user;
    // Insert dequantize node right before use node, because
    // we want to make sure use node and dequantize node reside
    // in the same block so that quant fusion can happen
    WithInsertPoint ins(user);
    Node* dequant = insertDeQuant(graph, quantized_val, original_val, i);
    user->replaceInput(uses[i].offset, dequant->output());
    outputs.push_back(dequant->output());
  }
  return outputs;
}

Node* insertQParam(
    Graph* graph,
    Value* quantized_input,
    NodeKind node_kind,
    const TypePtr& output_type,
    const std::string& param_name) {
  Node* qparam = graph->create(node_kind, {quantized_input});
  qparam->output()
      ->setDebugName(quantized_input->debugName() + "." + param_name)
      ->setType(output_type);
  graph->insertNode(qparam);
  return qparam;
}

Node* insertScalarToTensor(Graph* graph, Value* scalar_value) {
  Node* n = scalar_value->node();
  WithInsertPoint ins(n->next());
  Value* float_scalar_type = graph->insertConstant(IValue(c10::kFloat));
  Value* none = graph->insertConstant(IValue());
  Node* tensor_node = graph->create(
      Symbol::aten("scalar_tensor"),
      {scalar_value, float_scalar_type, none, none, none});
  Value* tensor_output = tensor_node->output();
  tensor_output->setDebugName(scalar_value->debugName() + ".tensor");
  graph->insertNode(tensor_node);
  // replace original_output with tensor
  scalar_value->replaceAllUsesAfterNodeWith(tensor_node, tensor_output);
  return tensor_node;
}

Node* insertItem(Graph* graph, Value* tensor, const TypePtr& output_type) {
  WithInsertPoint ins(tensor->node()->next());
  Node* n = graph->create(Symbol::aten("item"), {tensor});
  Value* scalar = n->output();
  scalar->setDebugName(tensor->debugName() + ".scalar")->setType(output_type);
  graph->insertNode(n);
  return n;
}

DynamicQuantOps insertChooseQParamQuantDequant(
    Graph* graph,
    Value* original_val,
    Value* dtype,
    NodeKind quant_kind) {
  Node* choose_qparams = insertChooseQParams(graph, original_val);
  std::vector<Value*> quant_inputs = {original_val};
  for (auto& out : choose_qparams->outputs()) {
    quant_inputs.push_back(out);
  }
  quant_inputs.push_back(dtype);
  Node* quant = insertQuant(
      graph, quant_inputs, quant_kind, original_val->debugName() + ".quant");
  Node* dequant = insertDeQuant(graph, quant->output(), original_val);
  return std::make_tuple(choose_qparams, quant, dequant);
}

Node* insertFP16CastOps(Graph* graph, Value* observer_out) {
  // If the weight value is outside of the range for FP16 range, i.e. [5.96e-8,
  // 65504], we saturate the values to the min/max of this range.
  Node* saturated_weight =
      graph->create(Symbol::aten("_saturate_weight_to_fp16"), {observer_out});
  graph->insertNode(saturated_weight);
  graph->lint();

  return saturated_weight;
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

bool isPlaceholderObserver(Value* observer) {
  if (getModuleName(observer).has_value()) {
    auto name = getModuleName(observer).value();
    // if PlaceholderObserver is (anywhere) in name
    if (name.find("PlaceholderObserver") != std::string::npos) {
      return true;
    }
  }
  return false;
}

at::ScalarType getObserverDtype(Module& module, Value* v) {
  auto observer_name = findObserverName(v);
  if (observer_name.has_value()) {
    auto observer_module = module.attr(observer_name.value()).toModule();
    at::ScalarType scalar_type = observer_module.attr("dtype").toScalarType();
    return scalar_type;
  }
  return at::ScalarType::Undefined;
}

at::ScalarType getObserverComputeDtype(Module& module, Value* v) {
  auto observer_name = findObserverName(v);
  if (observer_name.has_value()) {
    auto observer_module = module.attr(observer_name.value()).toModule();
    if (observer_module.hasattr("compute_dtype")) {
      at::ScalarType scalar_type =
          observer_module.attr("compute_dtype").toScalarType();
      return scalar_type;
    }
  }
  return at::ScalarType::Undefined;
}

c10::optional<std::string> getEmbeddingBagObsName(
    script::Module& module,
    Node* n) {
  Value* v = n->output();
  auto observer = n->input(0);
  auto observer_module = module.attr(findObserverName(v).value()).toModule();
  if (observer_module.hasattr("custom_op")) {
    auto op_name = observer_module.attr("custom_op").toStringRef();
    return isPlaceholderObserver(observer) ? op_name : "";
  }
  return c10::nullopt;
}

bool isEmbeddingBagOp(
    Node* observer,
    c10::optional<std::string> embedding_bag_name) {
  return embedding_bag_name &&
      embedding_bag_name.value().find("embedding_bag_") != std::string::npos;
}

template <typename T>
Node* insertQuantDequantNodes(
    Value* self,
    Node* observer,
    T& qparams,
    const std::string& quantize_func);

// Insert quant and dequant nodes into the graph for both static and dynamic
// quant.
template <>
Node* insertQuantDequantNodes<std::vector<std::string>>(
    Value* self,
    Node* observer,
    std::vector<std::string>& qparam_names,
    const std::string& quantize_func) {
  Graph* g = observer->owningGraph();
  Value* observer_out = observer->output();
  Value* original_val = observer->input(1);
  std::vector<Value*> inputs = {observer_out};
  // Insert GetAttr nodes for quantization parameters
  for (const auto& qparam_name : qparam_names) {
    inputs.push_back(g->insertGetAttr(self, qparam_name));
  }
  Node* quant = insertQuant(
      g,
      inputs,
      at::Symbol::aten(quantize_func),
      original_val->debugName() + ".quant");
  Node* dequant = insertDeQuant(g, quant->output(), original_val);
  return dequant;
}

Node* insertEmbeddingBagOps(Node* observer, const std::string& op_name) {
  Graph* g = observer->owningGraph();
  auto observer_out = observer->output();

  std::string prepack_fn, quant_fn;
  std::vector<Value*> prepack_inputs = {observer_out};
  if (op_name == "embedding_bag_4bit") {
    bool optimized_qparams = false;
    constexpr int NBINS = 200;
    constexpr float RATIO = 0.16;
    Value* optimized_qparams_false = g->insertConstant(optimized_qparams);
    Value* nbins_200 = g->insertConstant(NBINS);
    Value* ratio_0_16 = g->insertConstant(RATIO);
    prepack_fn = "quantized::embedding_bag_4bit_prepack";
    quant_fn = "quantized::embedding_bag_4bit_rowwise_offsets";
    prepack_inputs.push_back(optimized_qparams_false);
    prepack_inputs.push_back(nbins_200);
    prepack_inputs.push_back(ratio_0_16);
  } else if (op_name == "embedding_bag_byte") {
    prepack_fn = "quantized::embedding_bag_byte_prepack";
    quant_fn = "quantized::embedding_bag_byte_rowwise_offsets";
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Graph Mode Quantization currently supports 4-bit and 8-bit embedding bag quantization.");
  }

  std::vector<Use> uses = observer_out->uses();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Node* embedding_bag_float_op;
  // We expect that the output of the weight observer will be consumed by the
  // embedding_bag operator.
  for (const Use& use : uses) {
    if (matchCallFuncToUse(use, "embedding_bag", 2) ||
        matchAtenFuncToUse(use, "embedding_bag", 0)) {
      embedding_bag_float_op = use.user;
    }
  }

  // Insert prepack op
  Node* prepack = g->create(Symbol::fromQualString(prepack_fn), prepack_inputs);
  g->insertNode(prepack);

  std::vector<Value*> embedding_bag_inputs =
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
      embedding_bag_float_op->inputs().vec();
  std::vector<Value*> qembedding_bag_inputs = {prepack->output()};
  const auto inputs_size = embedding_bag_float_op->inputs().size();
  const bool is_aten_op =
      embedding_bag_float_op->kind() == Symbol::aten("embedding_bag");
  // Create and insert quantized embedding op.
  Value* none = g->insertConstant(IValue());
  Value* zero = g->insertConstant(IValue(0));
  bool pruned_wt = false;
  auto pruned_const = g->insertConstant(pruned_wt);

  if (is_aten_op) {
    TORCH_CHECK(
        inputs_size == 9,
        "Expecting FP aten::embedding_bag operator to have 9 inputs");
    // input 0 is the output of prepack op.
    // Last input is added after we account for extra input in 4-bit case.
    for (unsigned long i = 1; i < inputs_size - 2; ++i) {
      qembedding_bag_inputs.push_back(embedding_bag_inputs[i]);
    }
    // The sparse field in the float operator denotes sparse gradients.
    // For inference this stands for pruned weights. We currently don't support
    // pruning in graph mode API so we set the field to 0 for inference.
    qembedding_bag_inputs[5] = pruned_const;
  } else {
    TORCH_CHECK(
        inputs_size == 12,
        "Expecting F.embedding_bag operator to have 12 inputs");
    qembedding_bag_inputs.push_back(embedding_bag_inputs[1]); // indices
    qembedding_bag_inputs.push_back(embedding_bag_inputs[3]); // offsets
    qembedding_bag_inputs.push_back(
        embedding_bag_inputs[6]); // scale_grad_by_freq
    qembedding_bag_inputs.push_back(zero); // mode
    qembedding_bag_inputs.push_back(pruned_const); // pruned_weights
    qembedding_bag_inputs.push_back(
        embedding_bag_inputs[9]); // per_sample_weights
  }

  qembedding_bag_inputs.push_back(none); // compressed_indices_mapping
  qembedding_bag_inputs.push_back(embedding_bag_inputs[inputs_size - 2]);

  TORCH_CHECK(
      embedding_bag_inputs[inputs_size - 1]->mustBeNone(),
      "Expected aten::embedding_bag padding_idx input to be None");

  Node* qembedding_bag =
      g->create(Symbol::fromQualString(quant_fn), qembedding_bag_inputs);
  if (is_aten_op) {
    WithInsertPoint ins(embedding_bag_float_op);
    g->insertNode(qembedding_bag);
    // Verify that the outputs (apart from index 0) have no uses in the graph.
    for (const auto i :
         c10::irange(1, embedding_bag_float_op->outputs().size())) {
      TORCH_CHECK(
          !embedding_bag_float_op->output(i)->hasUses(),
          "Expected aten::embedding_bag to only have use for its first output.");
    }
  } else {
    g->insertNode(qembedding_bag);
  }
  embedding_bag_float_op->output(0)->replaceAllUsesWith(
      qembedding_bag->output());
  embedding_bag_float_op->removeAllInputs();
  embedding_bag_float_op->destroy();
  g->lint();
  return qembedding_bag;
}

template <typename T>
void insertQuantizationOps(
    Module& module,
    Value* self,
    Node* observer,
    bool is_per_channel,
    T& qparams,
    QuantType quant_type = QuantType::STATIC) {
  Graph* g = observer->owningGraph();
  // Observer output
  Value* observer_out = observer->output();
  // Inserting before insert point
  WithInsertPoint ins(observer_out->node()->next());

  std::string quantize_func;
  if (is_per_channel) {
    quantize_func = "quantize_per_channel";
  } else {
    quantize_func = "quantize_per_tensor";
  }
  Value* original_val = observer->input(1);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Node *quant, *choose_qparams, *dequant;
  // Temporary solution to quantize embedding_bag operators. Will be re-written
  // once we support quantization of embedding_bag weights.
  auto embedding_bag_name = getEmbeddingBagObsName(module, observer);
  if (isEmbeddingBagOp(observer, embedding_bag_name)) {
    if (isWeight(module, observer_out)) {
      auto op_name = embedding_bag_name.value();
      Node* dequant = insertEmbeddingBagOps(observer, op_name);
      observer_out->replaceAllUsesWith(original_val);
      original_val->replaceAllUsesAfterNodeWith(dequant, dequant->output());
    } else {
      // Special case for embedding bag operators indices input - we don't
      // quantize the input but we still need to insert observers for it because
      // the order of input and weight can be changed in the module code.
      observer_out->replaceAllUsesWith(original_val);
    }
    return;
  }
  if (quant_type == QuantType::DYNAMIC) {
    if (getObserverDtype(module, observer_out) == at::ScalarType::Half) {
      dequant = insertFP16CastOps(g, observer_out);
    } else if (!isWeight(module, observer_out)) {
      auto observer_dtype = getObserverDtype(module, observer_out);
      auto observer_compute_dtype =
          getObserverComputeDtype(module, observer_out);
      if (observer_dtype == at::ScalarType::QUInt8 ||
          observer_dtype == at::ScalarType::QInt8 ||
          observer_compute_dtype == at::ScalarType::QUInt8 ||
          observer_compute_dtype == at::ScalarType::QInt8) {
        // For activation tensors we insert choose_qparams, quant, dequant ops.
        Value* dtype = g->insertGetAttr(self, qparams.back());
        std::tie(choose_qparams, quant, dequant) =
            insertChooseQParamQuantDequant(
                g, observer_out, dtype, at::Symbol::aten(quantize_func));
      } else {
        // dtype does not require quantization, e.g. float32
        // will just remove the observer call
        observer_out->replaceAllUsesWith(original_val);
        return;
      }
    } else {
      // For weight tensors we insert quant-dequant ops.
      dequant = insertQuantDequantNodes(self, observer, qparams, quantize_func);
    }
  } else { // Static quant
    dequant = insertQuantDequantNodes(self, observer, qparams, quantize_func);
  }
  observer_out->replaceAllUsesWith(original_val);

  original_val->replaceAllUsesAfterNodeWith(dequant, dequant->output());
  GRAPH_DUMP("insert nodes:", original_val->owningGraph());
}

void ReplicateChooseQParamsQuantDequant(std::shared_ptr<Graph>& graph) {
  const PatternInfo& dynamic_quant_pattern = PatternInfo::parse_from_str(R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )");
  const Graph& dynamic_quant_graph = *dynamic_quant_pattern.pattern_graph;

  const auto& matches = findPatternMatches(dynamic_quant_graph, *graph);
  if (matches.size() == 0) {
    return;
  }

  const auto& vmap = dynamic_quant_pattern.vmap;
  Value* dequant_val = vmap.at("a_dequant");
  Node* pattern_dequant = dequant_val->node();
  Value* quant_val = vmap.at("a_quant");
  Node* pattern_quant = quant_val->node();
  Value* choose_qparam_val = vmap.at("a_scale");
  Node* pattern_choose_qparam = choose_qparam_val->node();

  std::vector<DynamicQuantOps> nodes_to_rewrite;
  std::vector<Node*> choose_qparam_nodes_to_rewrite;
  for (const Match& match : matches) {
    Node* matched_dequantize = match.nodes_map.at(pattern_dequant);
    Node* matched_quantize = match.nodes_map.at(pattern_quant);
    Node* matched_choose_qparam = match.nodes_map.at(pattern_choose_qparam);
    if (matched_dequantize->output()->uses().size() > 1) {
      nodes_to_rewrite.emplace_back(std::make_tuple(
          matched_choose_qparam, matched_quantize, matched_dequantize));
    }
  }
  for (const auto& nodes : nodes_to_rewrite) {
    auto quant_node = std::get<1>(nodes);
    auto dequant_node = std::get<2>(nodes);
    // get input of quantize call.
    Value* original_val = quant_node->inputs()[0];
    Value* dequant_out = dequant_node->output();
    Value* dtype = quant_node->inputs()[3];
    std::vector<Use> uses = dequant_out->uses();
    for (const Use& use : uses) {
      auto* user = use.user;
      WithInsertPoint ins(user);
      auto quant_ops = insertChooseQParamQuantDequant(
          graph.get(), original_val, dtype, quant_node->kind());
      user->replaceInputWith(dequant_out, std::get<2>(quant_ops)->output());
    }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Node *choose_qparams, *quant, *dequant;
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->removeAllInputs();
    quant->removeAllInputs();
    choose_qparams->removeAllInputs();
  }
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->destroy();
    quant->destroy();
    choose_qparams->destroy();
  }
}

void RemoveRedundantDequantize(std::shared_ptr<Graph>& graph) {
  const std::string dequantize = R"(
    graph(%a_quant):
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  const std::string dequantize_replacement = R"(
    graph(%a):
        return (%a) )";
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    // Values can be used multiple times in a single node
    if (dequant_out->uses().size() != 1) {
      return false;
    }
    Node* user = dequant_out->uses()[0].user;
    return isTensorInfoNode(user);
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(dequantize, dequantize_replacement);
  rewriter.runOnGraph(graph, filter);
}

void RemoveRedundantQuantizationOps(std::shared_ptr<Graph>& graph) {
  const std::string dynamic_quant_ops = R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  const std::string dynamic_quant_replacement = R"(
    graph(%a, %reduce_range, %a_dtype):
        return (%a) )";
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    // Values can be used multiple times in a single node
    if (dequant_out->uses().size() != 1) {
      return false;
    }
    Node* user = dequant_out->uses()[0].user;
    return !nodeQuantizable(user, QuantType::DYNAMIC);
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(dynamic_quant_ops, dynamic_quant_replacement);
  rewriter.runOnGraph(graph, filter);
}

void ReplicateClampScalarArgs(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::unordered_set<Node*> scalar_nodes_to_rewrite;
  ;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      for (Value* output : n->outputs()) {
        if (getClampScalarInputUse(output) && output->uses().size() > 1) {
          scalar_nodes_to_rewrite.insert(n);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  for (Node* n : scalar_nodes_to_rewrite) {
    const std::vector<Use> uses = n->output()->uses();
    for (const auto& use : uses) {
      Node* user = use.user;
      WithInsertPoint ins(user);
      Node* cloned_node = graph->createClone(n, [](Value* v) { return v; });
      graph->insertNode(cloned_node);
      user->replaceInput(use.offset, cloned_node->output());
    }
  }

  for (Node* n : scalar_nodes_to_rewrite) {
    n->removeAllInputs();
  }

  for (Node* n : scalar_nodes_to_rewrite) {
    n->destroy();
  }
}

void checkCalculateQParamsResult(const IValue& qparams) {
  TORCH_CHECK(
      qparams.isTuple(),
      "`calculate_qparams` function is expected to return a "
      "Tuple, but got:",
      qparams.tagKind());
  auto tp = qparams.toTuple();
  TORCH_CHECK(
      tp->elements().size() == 2,
      "`calculate_qparams` function is expected to return a "
      "Tuple of size 2, got Tuple of size ",
      tp->elements().size());
  // Expect first two elements of the tuple to be Tensor
  for (const auto i : c10::irange(2)) {
    TORCH_CHECK(
        tp->elements()[i].isTensor(),
        "Element of Tuple is expected to be Tensor, but element ",
        i,
        " has type: ",
        tp->elements()[i].tagKind());
  }
}

class SubGraphCloneHelper {
 public:
  // Given a list of nodes, build a graph corresponding to these nodes.
  // User should make sure to run this graph with expected input.
  std::unique_ptr<GraphFunction> buildGraphFromNodes(
      const std::vector<Node*>& nodes,
      const std::string& name);

  // Given a list of nodes in src, produce a Graph with these nodes.
  void buildObserverSubgraph(
      const std::vector<Node*>& src,
      std::shared_ptr<Graph> dest);

 private:
  // Clone node in the destination Graph g.
  void cloneNodeInGraph(
      Node* node,
      std::shared_ptr<Graph>& g,
      std::unordered_map<Value*, Value*>& remap_values);
};

class InsertQuantDeQuantHelper {
 public:
  InsertQuantDeQuantHelper(QuantType quant_type, bool debug)
      : quant_type_(quant_type), debug_(debug) {}

  void run(Module& module, const std::string& method_name);

  void runForOnDevicePTQ(Module& module, const std::string& method_name);

  // Cleanup observer nodes from graph and observer modules
  // from module object and ClassType
  void cleanup(Module& module);

  // Cleanup observer nodes only but not modules
  // This is for ondevice PTQ
  void removeObserverNodes(Module& m);

  // In order to propagate quantization ops through the ops that doesn't
  // require observation, we'll first inline the graph, and call the
  // PropgateQuantizationOps pass
  void propagateQuantizationOps(Module& module);

  // Used for dynamic quantization to selectively run the weight observers.
  // It extracts the subgraph corresponding to the weight and runs it with
  // the module instance.
  void runWeightObserver(Module& module, const std::string& method_name);

 private:
  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  // Get quantization parameter map of the given Value in Graph
  // by searching for observer module of the value and extract the
  // quantization parameters from the observer module
  std::tuple<c10::QScheme, QParamVector> getQSchemeAndQParamVector(
      script::Module& module,
      Node* n);
  QuantOpParams insertCalculateQParams(
      script::Module& module,
      Graph* g,
      Node* n);

  void checkQScheme(Graph* g, c10::QScheme qscheme) {
    if (qscheme_for_graph_.count(g)) {
      // FIXME[T110786721]: This check was broken before nevery failing.
      // Once fixed, this check triggers and fails tests.
      // Fix the tests that enabling this check produce!
      /*
      TORCH_CHECK(
          qscheme_for_graph_.at(g) == qscheme,
          "Quantizing same graph with different types of "
          "QSchemes is not supported.\n",
          " Expecting:",
          c10::toString(qscheme_for_graph_.at(g)),
          " Got:",
          c10::toString(qscheme));
      */
    } else {
      qscheme_for_graph_[g] = toAffine(qscheme);
    }
  }

  void collectObserverNodesAndValueToQuantize(Module& module, Value*);
  void cleanup(Module& module, Graph* g);
  void removeObserverNodes(Graph* g);
  void quantizeTensors(Module& module, Graph* g, Value* self);
  void insertCalculateQParamsAndQuantizationOps(
      Module& module,
      Graph* g,
      Value* self);

  // Function that extracts and runs the weight observer in a separate
  // subgraph.
  void extractAndRunWeightObserver(
      Module& module,
      Value* self,
      Value* weight_value);

  // Recursively find the nodes that produce the value and add to subgraph.
  void findSubgraph(Value* self, Value* v, std::vector<Node*>& weight_subgraph);

  // Quantizes two types of general ops(ops that works both for floating point
  // and quantized Tensors) in this pass
  // for ops that only manipulates shape, e.g. flatten, quantization
  // is done by swapping with previous dequantize op
  // for ops that manipulates values of Tensor, e.g. average pool, quantization
  // is done by inserting quant/dequant ops after the op
  // also has a special handling of clamp/hardtanh
  void propagateQuantizationOps(Block* block);

  // Propagate quantization parameters from other quantized tensors
  void propagateQParams(
      Value* original_output,
      const std::vector<Value*>& inputs,
      bool is_scalar = false,
      const c10::optional<std::tuple<c10::QScheme, QParamVector>>& qparams_opt =
          c10::nullopt);

  bool isQuantized(Value* v) {
    return quantized_values_.count(v) != 0;
  }

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
  std::unordered_map<Node*, std::unordered_map<std::string, std::string>>
      qparam_name_map_for_node_;
  // Record qscheme for every graph, this is for checking
  // each graph is only quantized with one type of QScheme
  std::unordered_map<Graph*, c10::QScheme> qscheme_for_graph_;

  // Set of quantized values, so that we quantize each value only
  // once
  std::unordered_set<Value*> quantized_values_;

  // Map from original weight value to GraphFunction corresponding to the
  // subgraph that includes the weight observer and dependent nodes.
  std::unordered_map<Value*, std::unique_ptr<GraphFunction>>
      weight_to_graph_fn_;

  QuantType quant_type_ = QuantType::STATIC;
  bool debug_ = false;
};

void InsertQuantDeQuantHelper::collectObserverNodesAndValueToQuantize(
    Module& module,
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
  observer_nodes_for_graph_[g].push_back(observer);
}

void InsertQuantDeQuantHelper::removeObserverNodes(Module& module) {
  for (auto& method : module.get_methods()) {
    removeObserverNodes(method.graph().get());
  }
  for (Module m : module.children()) {
    removeObserverNodes(m);
  }
}

void InsertQuantDeQuantHelper::removeObserverNodes(Graph* g) {
  if (nodes_to_destroy_.count(g)) {
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->removeAllInputs();
    }
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->destroy();
    }
    nodes_to_destroy_.at(g).clear();
  }
}

void InsertQuantDeQuantHelper::cleanup(Module& module) {
  for (auto& method : module.get_methods()) {
    cleanup(module, method.graph().get());
  }
  for (Module m : module.children()) {
    cleanup(m);
  }
}

void InsertQuantDeQuantHelper::cleanup(Module& module, Graph* g) {
  GRAPH_DUMP("Before Remove Observers:", g);
  removeObserverNodes(g);

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

void SubGraphCloneHelper::cloneNodeInGraph(
    Node* node,
    std::shared_ptr<Graph>& g,
    std::unordered_map<Value*, Value*>& remap_old_to_new) {
  auto* block = g->block();
  auto value_fn = [&](Value* v) {
    if (remap_old_to_new.count(v) == 0) {
      auto new_value = g->block()->addInput();
      remap_old_to_new[v] = new_value;
      new_value->copyMetadata(v);
      return new_value;
    } else {
      return remap_old_to_new[v];
    }
  };

  auto new_node = block->appendNode(g->createClone(node, value_fn));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    remap_old_to_new[oo] = no;
  }
}

void SubGraphCloneHelper::buildObserverSubgraph(
    const std::vector<Node*>& weight_subgraph,
    std::shared_ptr<Graph> dest_graph) {
  std::unordered_map<Value*, Value*> remap_old_to_new;
  // Build weight subgraph
  for (auto n : weight_subgraph) {
    cloneNodeInGraph(n, dest_graph, remap_old_to_new);
  }
  LintGraph(dest_graph);

  // Add last node output value as subgraph output.
  for (auto out : weight_subgraph.back()->outputs()) {
    dest_graph->registerOutput(remap_old_to_new[out]);
  }
  GRAPH_DUMP("New weight observer subgraph: ", dest_graph);
}

std::unique_ptr<GraphFunction> SubGraphCloneHelper::buildGraphFromNodes(
    const std::vector<Node*>& nodes,
    const std::string& name) {
  auto observer_subgraph = std::make_shared<Graph>();
  auto build_observer_graph = [&](GraphFunction& func) {
    buildObserverSubgraph(nodes, func.graph());
  };
  return torch::make_unique<GraphFunction>(
      name, observer_subgraph, build_observer_graph);
}

void InsertQuantDeQuantHelper::findSubgraph(
    Value* self,
    Value* input_val,
    std::vector<Node*>& weight_subgraph) {
  Node* node = input_val->node();
  weight_subgraph.push_back(node);
  const auto& inputs = node->inputs().vec();
  for (auto v : inputs) {
    if (!hitGraphInput(v)) {
      findSubgraph(self, v, weight_subgraph);
    } else {
      TORCH_CHECK(
          v == self,
          "Unexpected value found when handling weight value "
          " in findSubgraph, traced back to:",
          v->debugName(),
          " which is not self:",
          self->debugName());
    }
  }
}

void InsertQuantDeQuantHelper::extractAndRunWeightObserver(
    Module& module,
    Value* self,
    Value* weight_value) {
  std::vector<Node*> weight_subgraph;
  // If the graph was already visited, return the GraphFunction directly.
  // Multiple module instances can share the same graph code, so we don't need
  // to re-run the extraction process.
  if (weight_to_graph_fn_.count(weight_value) == 0) {
    // Extract the subgraph nodes.
    findSubgraph(self, weight_value, weight_subgraph);

    // Reverse to traverse subgraph in correct direction
    std::reverse(weight_subgraph.begin(), weight_subgraph.end());

    // Build the graph using the nodes found from the weight observer.
    SubGraphCloneHelper o;
    std::unique_ptr<GraphFunction> func =
        o.buildGraphFromNodes(weight_subgraph, "observer_subgraph");
    weight_to_graph_fn_[weight_value] = std::move(func);
  }
  Stack module_inp = {module._ivalue()};
  // Run the graph with the module input.
  weight_to_graph_fn_[weight_value]->run(module_inp);
}

void InsertQuantDeQuantHelper::quantizeTensors(
    Module& module,
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
      auto qparam_name =
          original_value->debugName() + name + "_" + c10::to_string(uid++);
      while (module.hasattr(qparam_name)) {
        qparam_name =
            original_value->debugName() + name + "_" + c10::to_string(uid++);
      }
      qparam_name_map_for_node_[n][name] = qparam_name;
      module.register_attribute(qparam_name, qparam.type(), qparam);
      qparam_names.push_back(qparam_name);
    }
    insertQuantizationOps(
        module, self, n, isPerChannel(qscheme), qparam_names, quant_type_);
  }
}

std::tuple<c10::QScheme, QParamVector> InsertQuantDeQuantHelper::
    getQSchemeAndQParamVector(script::Module& module, Node* n) {
  // TODO: refactor findObserverName to take Node* as input
  Value* v = n->output();
  TORCH_INTERNAL_ASSERT(
      v->type()->isSubtypeOf(*TensorType::get()),
      "Expected output of observer node to be Tensor");
  auto observer_name = findObserverName(v);
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQSchemeAndParamMap expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  QParamVector qparams;
  c10::QScheme qscheme = c10::kPerTensorAffine;

  auto observer_module = module.attr(observer_name.value()).toModule();
  auto scalar_type = observer_module.attr("dtype");
  if (isPlaceholderObserver(n->input(0))) {
    // get compute_dtype for dynamic quantization
    if (observer_module.hasattr("compute_dtype")) {
      qparams.push_back(
          std::make_pair(kScalarType, observer_module.attr("compute_dtype")));
    }
    return std::make_tuple(qscheme, qparams);
  } else if (scalar_type == at::ScalarType::Half) {
    return std::make_tuple(qscheme, qparams);
  }
  auto calculate_qparams = observer_module.get_method("calculate_qparams");
  IValue result = calculate_qparams(std::vector<IValue>());
  checkCalculateQParamsResult(result);
  TORCH_CHECK(
      scalar_type.toScalarType() != at::ScalarType::Undefined,
      "dtype of observer can't be undefined");
  auto tp = result.toTuple();
  at::Tensor scale = tp->elements()[0].toTensor().to(at::kFloat);
  at::Tensor zero_point = tp->elements()[1].toTensor().to(at::kInt);
  // quantization parameters should appear in the same order as
  // the argument for quantize_per_tensor/quantize_per_channel function

  qscheme = observer_module.attr("qscheme").toQScheme();
  if (isPerChannel(qscheme)) {
    auto axis = observer_module.attr("ch_axis");
    qparams.push_back(std::make_pair("_scale", scale));
    qparams.push_back(std::make_pair("_zero_point", zero_point));
    qparams.push_back(std::make_pair("_axis", axis.toInt()));
  } else {
    qparams.push_back(std::make_pair("_scale", scale.item<double>()));
    qparams.push_back(
        std::make_pair("_zero_point", zero_point.item<int64_t>()));
  }
  qparams.push_back(std::make_pair(kScalarType, scalar_type));
  return std::make_tuple(qscheme, qparams);
}

ModuleMethodVector InsertQuantDeQuantHelper::getInvokedMethods(
    Module& module,
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
        c10::optional<Module> m;
        // calling method on self
        if (module_instance == graph->inputs()[0]) {
          m = module;
        } else if (
            module_instance->node()->kind() == prim::GetAttr &&
            module_instance->node()->s(attr::name).find("_observer_") ==
                std::string::npos) {
          m = getInvokedModuleOpt(module, n, graph->inputs()[0]);
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

void InsertQuantDeQuantHelper::propagateQParams(
    Value* original_output,
    const std::vector<Value*>& inputs,
    bool is_scalar,
    const c10::optional<std::tuple<c10::QScheme, QParamVector>>& qparams_opt) {
  Node* n = original_output->node();
  Graph* graph = n->owningGraph();
  if (is_scalar) {
    // convert Scalar to Tensor
    n = insertScalarToTensor(graph, original_output);
    original_output = n->output();
  }
  // for ops like average pool, we'll insert quant dequant after the op
  // We'll assume the tensor is a PerTensorAffine quantized Tensor for
  // now, and may generalize later if this becomes an issue
  TORCH_INTERNAL_ASSERT(
      inputs.size() == 1, "Expecting single input for the aten function");
  // input of the dequantize node
  Value* quantized_input = inputs[0]->node()->input(0);
  // insert ops after the general op
  Node* quantized_input_node = quantized_input->node();
  // Insert after the node that is later in topological order
  WithInsertPoint ins(
      quantized_input_node->isAfter(n) ? quantized_input_node->next()
                                       : n->next());
  std::vector<Value*> quant_inputs;
  auto quant_kind = Symbol::aten("quantize_per_tensor");
  if (qparams_opt.has_value()) {
    quant_inputs = {original_output};
    auto qscheme = std::get<0>(*qparams_opt);
    auto qparams = std::get<1>(*qparams_opt);
    if (isPerChannel(qscheme)) {
      quant_kind = Symbol::aten("quantize_per_channel");
    }
    for (const auto& qparam : qparams) {
      Value* qparam_val = graph->insertConstant(qparam.second);
      qparam_val->setDebugName(quantized_input->debugName() + qparam.first);
      quant_inputs.push_back(qparam_val);
    }
  } else {
    // Only per tensor affine quantized tensor is supported in this case
    // get quantization parameters from previous quantized op
    Node* scale = insertQParam(
        graph,
        quantized_input,
        at::Symbol::aten("q_scale"),
        FloatType::get(),
        "q_scale");
    Node* zero_point = insertQParam(
        graph,
        quantized_input,
        at::Symbol::aten("q_zero_point"),
        IntType::get(),
        "q_zero_point");
    Node* dtype = insertQParam(
        graph, quantized_input, prim::dtype, IntType::get(), "dtype");
    quant_inputs = {
        original_output,
        scale->output(),
        zero_point->output(),
        dtype->output()};
  }
  Node* quant = insertQuant(
      graph, quant_inputs, quant_kind, original_output->debugName() + ".quant");
  Value* quantized_output = quant->output();
  // replace uses of original output of the general op with quantized
  // output
  original_output->replaceAllUsesAfterNodeWith(quant, quantized_output);
  const auto& outputs =
      insertDeQuantForAllUse(graph, quantized_output, quantized_output);
  for (auto* output : outputs) {
    if (is_scalar) {
      // Convert the dequantized Tensor back to Scalar
      Node* item = insertItem(graph, output, FloatType::get());
      Value* scalar = item->output();
      output->replaceAllUsesAfterNodeWith(item, scalar);
      output = scalar;
    }
    quantized_values_.insert(output);
  }
}

void removeDequantizeFromInputs(const std::unordered_set<Value*>& inputs) {
  // Delete dequantize node, we have one dequantize
  // for each use of the value
  for (auto* dequantized_val : inputs) {
    auto* dequantize_node = dequantized_val->node();
    TORCH_INTERNAL_ASSERT(
        dequantized_val->uses().size() == 1,
        "Expect to have one dequantize node for each use");
    // Replace useses of dequantized_val with the input of
    // dequantize node
    dequantized_val->replaceAllUsesWith(dequantize_node->inputs()[0]);
    dequantize_node->removeAllInputs();
    dequantize_node->destroy();
  }
}

// Check if we need to propagate the quantization ops from input to
// output
c10::optional<std::vector<Value*>> getDequantizedInputs(Value* output) {
  auto inputs = getPassThroughInputs(output);
  if (inputs.size() > 0) {
    // note that we don't need to recursively check for prim::If
    // here because if all inputs of a prim::If is dequantized
    // the dequantize will be factored out before we get to this
    // point
    bool is_dequantized = true;
    for (auto* input : inputs) {
      GRAPH_DEBUG(
          "checking if input:",
          input->debugName(),
          " in node:",
          *input->node(),
          "is quantized");
      is_dequantized &= input->node()->kind() == Symbol::aten("dequantize");
    }
    if (is_dequantized) {
      return inputs;
    }
  }
  return c10::nullopt;
}

void InsertQuantDeQuantHelper::propagateQuantizationOps(Block* block) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::If) {
      for (Block* subblock : n->blocks()) {
        propagateQuantizationOps(subblock);
      }
      if (n->outputs().size() == 0) {
        continue;
      }
      if (n->outputs().size() > 1) {
        // Factoring out dequantize for if blocks with multiple outputs
        // is not supported right now
        continue;
      }
    }
    if (isSingleInputGeneralValueAtenFunction(n)) {
      for (auto* output : n->outputs()) {
        if (isQuantized(output)) {
          continue;
        }
        if (auto inputs = getDequantizedInputs(output)) {
          propagateQParams(output, *inputs);
          if (isClamp(n)) {
            for (size_t i = 1; i <= 2; ++i) {
              // propagate qparams for min and max scalar arguments
              // for aten::clamp/aten::hardtanh
              propagateQParams(n->input(i), *inputs, /* is_scalar */ true);
            }
          }
        }
      }
    } else if (auto qparams_opt = getFixedQParams(n)) {
      for (auto* output : n->outputs()) {
        if (isQuantized(output)) {
          continue;
        }
        if (auto inputs = getDequantizedInputs(output)) {
          propagateQParams(output, *inputs, /* is_scalar */ false, qparams_opt);
        }
      }
    } else {
      // For ops that are quantized by propagating dequantize ops,
      // e.g. flatten we need to
      // 1. check if we need to propagate dequantize op
      // 2. remove the dequantize ops from inputs
      // 3. insert dequantize for all outputs
      // to make sure it works for ops with multiple outputs
      // since removing dequantize from inputs is mutating the graph
      // and it will affect future checks for whether all the inputs
      // has been quantized or not(since currently we just check if
      // the value is produced by dequantize op to decide if the value
      // is quantized or not
      // list of dequantized input values
      std::unordered_set<Value*> dequantized_inputs;
      std::vector<Value*> outputs_to_dequantize;
      // 1. collect dequantized inputs and outputs we need to dequantize
      for (auto* output : n->outputs()) {
        if (isQuantized(output)) {
          continue;
        }
        if (auto inputs = getDequantizedInputs(output)) {
          std::copy(
              inputs->begin(),
              inputs->end(),
              std::inserter(dequantized_inputs, dequantized_inputs.end()));
          outputs_to_dequantize.push_back(output);
        }
      }
      // 2. remove the dequantize ops from inputs
      removeDequantizeFromInputs(dequantized_inputs);
      // 3. insert dequantize op for outpus
      for (auto* output : outputs_to_dequantize) {
        insertDeQuantForAllUse(output->owningGraph(), output, output);
      }
    }

    if (isBinaryOpWithScalarInput(n)) {
      // Print warning for add_scalar/mul_scalar when debug is enabled
      // since the quantization parameter for these ops depends on
      // input and it's too complicated to encode the equations in
      // the IR:
      // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/BinaryOps.cpp#L64-L74
      if (debug_) {
        TORCH_WARN_ONCE(
            "debug option for add_scalar and mul_scalar is not supported, "
            "please don't use debug option for models that uses these ops.");
      }
    }
  }
}

void InsertQuantDeQuantHelper::runWeightObserver(
    Module& module,
    const std::string& method_name) {
  if (quant_type_ != QuantType::DYNAMIC) {
    return;
  }

  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    runWeightObserver(invoked_module, invoked_method_name);
  }
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  Value* self = graph->inputs()[0];

  std::vector<Value*> weight_values;
  // Visit all blocks in the current graph to find weight values.
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto n : b->nodes()) {
      for (Value* v : n->outputs()) {
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
          continue;
        }
        auto observer_name = findObserverName(v);
        if (observer_name && isWeight(module, v)) {
          weight_values.push_back(v);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  // For all the observed weight values, find the corresponding subgraph that
  // contributes to the weight tensor, and run that subgraph to observe the
  // weight.
  for (const auto& v : weight_values) {
    extractAndRunWeightObserver(module, self, v);
  }
}

void InsertQuantDeQuantHelper::run(
    Module& module,
    const std::string& method_name) {
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    run(invoked_module, invoked_method_name);
  }

  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // We only need to register new parameters if the graph has
  // been quantized before
  // TODO: dedup this part with code in quantizeTensors
  if (observer_nodes_for_graph_.count(graph.get())) {
    for (auto* n : observer_nodes_for_graph_.at(graph.get())) {
      auto tp = getQSchemeAndQParamVector(module, n);
      checkQScheme(graph.get(), std::get<0>(tp));
      auto qparam_map = std::get<1>(tp);
      // We check the size here because for some observers (like
      // PlaceholderObserver) the qparams might be empty.
      if (qparam_map.size() > 0) {
        TORCH_INTERNAL_ASSERT(
            qparam_name_map_for_node_.count(n),
            "Expected to have a qparam_name_map for node:",
            *n);
        auto qparam_name_map = qparam_name_map_for_node_.at(n);
        for (auto& pr : qparam_map) {
          const auto& name = pr.first;
          const auto& qparam = pr.second;
          module._ivalue()->setAttr(qparam_name_map.at(name), qparam);
        }
      }
    }
    return;
  }

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  std::vector<Value*> input_values;
  for (const auto idx : c10::irange(1, method.num_inputs())) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(*TensorType::get())) {
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
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
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

void InsertQuantDeQuantHelper::propagateQuantizationOps(Module& module) {
  SwapFunctionalLinear(module);
  auto graph = module.get_method("forward").graph();
  Inline(*graph);
  ConstantPropagation(graph);
  ReplicateChooseQParamsQuantDequant(graph);
  RemoveRedundantQuantizationOps(graph);
  ReplicateQuant(graph);
  ReplicateDeQuant(graph);
  // TODO: add filter to the clamp patterns and remove this pass
  ReplicateClampScalarArgs(graph);
  propagateQuantizationOps(graph->block());
  RemoveRedundantDequantize(graph);
}

// Insert quant and dequant nodes into the graph for both static and dynamic
// quant.
template <>
Node* insertQuantDequantNodes<QuantOpParams>(
    Value* self,
    Node* observer,
    QuantOpParams& qparams,
    const std::string& quantize_func) {
  Graph* g = observer->owningGraph();
  Value* observer_out = observer->output();
  Value* original_val = observer->input(1);
  std::vector<Value*> inputs;
  // + 1 for tensor to be quantized
  inputs.reserve(qparams.qparams.size() + 1);
  inputs.push_back({observer_out});
  for (const auto& qparam_values : qparams.qparams) {
    inputs.push_back(qparam_values);
  }
  Node* quant = insertQuant(
      g,
      inputs,
      at::Symbol::aten(quantize_func),
      original_val->debugName() + ".quant");
  // Have to make sure that quant node appears after the values it depends on.
  for (Value* v : inputs) {
    quant->moveAfter(v->node());
  }
  Node* dequant = insertDeQuant(g, quant->output(), original_val);
  dequant->moveAfter(quant);
  return dequant;
}

void checkCalculateQParamsResultTypes(const Node* out) {
  TORCH_CHECK(
      out->outputs().size() == 2,
      "cacluate_qparams should produce output of size 2 (scale, zero_point).");
  Value* scale = out->output(0);
  Value* zp = out->output(1);
  TORCH_CHECK(
      scale->type()->expect<TensorType>(),
      "Scale value should be of Tensor type.");
  TORCH_CHECK(
      zp->type()->expect<TensorType>(), "Scale value should be of float type.");
}

QuantOpParams InsertQuantDeQuantHelper::insertCalculateQParams(
    script::Module& module,
    Graph* g,
    Node* n) {
  // TODO: refactor findObserverName to take Node* as input
  Value* self = g->inputs()[0];
  Value* v = n->output();
  TORCH_INTERNAL_ASSERT(
      v->type()->isSubtypeOf(*TensorType::get()),
      "Expected output of observer node to be Tensor");
  auto observer_name = findObserverName(v);
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQSchemeAndParamMap expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  std::vector<Value*> qparams_graph_values;
  QuantOpParams quant_op_params;

  TORCH_CHECK(
      !isPlaceholderObserver(n->input(0)),
      "Placeholder observers are not supported in ondevice PTQ.");
  auto observer_module = module.attr(observer_name.value()).toModule();
  Value* observer_module_value = g->insertGetAttr(self, observer_name.value());
  auto scalar_type = observer_module.attr("dtype");
  TORCH_CHECK(
      scalar_type.toScalarType() != at::ScalarType::Undefined,
      "dtype of observer can't be undefined");
  // Not sure if we need to support this for on device PTQ.
  if (scalar_type == at::ScalarType::Half) {
    return quant_op_params;
  }
  auto calculate_qparams = observer_module.get_method("calculate_qparams");
  auto calculate_qparams_schema = calculate_qparams.function().getSchema();
  MatchedSchema matched_schema = matchSchema(
      calculate_qparams_schema,
      v->node()->sourceRange(),
      *g,
      {observer_module_value},
      {});
  Node* call = g->insertMethodCall("calculate_qparams", matched_schema)->node();
  Node* scale_zp_node = g->insertNode(g->createTupleUnpack(call->output(0)));
  checkCalculateQParamsResultTypes(scale_zp_node);
  auto qscheme = observer_module.attr("qscheme").toQScheme();
  quant_op_params.qscheme = qscheme;
  quant_op_params.qparams.push_back(scale_zp_node->output(0)); // scale Value*
  quant_op_params.qparams.push_back(
      scale_zp_node->output(1)); // zero_point Value*
  if (isPerChannel(qscheme)) {
    Value* ch_axis_value = g->insertGetAttr(observer_module_value, "ch_axis");
    quant_op_params.qparams.push_back(ch_axis_value);
  }
  Value* scalar_type_value = g->insertGetAttr(observer_module_value, "dtype");
  quant_op_params.qparams.push_back(scalar_type_value);
  return quant_op_params;
}

void InsertQuantDeQuantHelper::insertCalculateQParamsAndQuantizationOps(
    Module& module,
    Graph* g,
    Value* self) {
  if (!observer_nodes_for_graph_.count(g)) {
    return;
  }
  for (auto* n : observer_nodes_for_graph_.at(g)) {
    Graph* g = n->owningGraph();
    // Observer output
    Value* observer_out = n->output();
    // Inserting before insert point
    WithInsertPoint insert_qparams_calc(observer_out->node()->next());
    auto quant_op_params = insertCalculateQParams(module, g, n);
    insertQuantizationOps(
        module,
        self,
        n,
        isPerChannel(quant_op_params.qscheme),
        quant_op_params,
        quant_type_);
  }
}

void InsertQuantDeQuantHelper::runForOnDevicePTQ(
    Module& module,
    const std::string& method_name) {
  // In all likelihood this really wont do anything because we expect that
  // the input method for quantization's prepare step will be inlined. Thus
  // only call methods we will see will belong to observer's forward calls.
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    runForOnDevicePTQ(invoked_module, invoked_method_name);
  }

  Method method = module.get_method(method_name);
  auto graph = method.graph();
  // Unliked the run method we dont need to extract new qparam values for the
  // the same graph used in different call site.
  // Reason is that for on device PTQ we dont:
  // 1. Run calculate_qparams
  // 2. Get the scale and zero point
  // 3. get axis and dtype
  // 4. register values from 2 and 3 as attributes on the parent module.
  // Instead we insert call to calculate_qparams (1) via insertCalculateQParams
  // in the graph itself. Then instead of 2 and 3, we get the output Value*
  // and for 3, we insert GetAttr for axis and dtype and use those Value*
  // with insterQuantizationOps

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  std::vector<Value*> input_values;
  for (const auto idx : c10::irange(1, method.num_inputs())) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(*TensorType::get())) {
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
        if (!v->type()->isSubtypeOf(*TensorType::get())) {
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

  GRAPH_DUMP("Before insertCalculateQparamsAndQuantizationOps:", graph);
  Value* self = graph->inputs()[0];
  insertCalculateQParamsAndQuantizationOps(module, graph.get(), self);
  GRAPH_DUMP("After insertCalculateQparamsAndQuantizationOps:", graph);
}

} // namespace

void ReplicateQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::vector<Node*> quant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // find quantize node that quantizes the output of if
      if ((n->kind() == Symbol::aten("quantize_per_tensor") ||
           n->kind() == Symbol::aten("quantize_per_channel")) &&
          n->input(0)->node()->kind() == prim::If) {
        quant_nodes_to_rewrite.push_back(n);
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (Node* n : quant_nodes_to_rewrite) {
    Node* if_node = n->input(0)->node();
    // move the nodes that produces the quantization parameters before
    // prim::If
    for (const auto i : c10::irange(1, n->inputs().size())) {
      n->input(i)->node()->moveBefore(if_node);
    }
    // replace all uses of the quantized node with the output of if node
    n->output()->replaceAllUsesWith(if_node->output());
    // add quantize nodes to the end of all blocks
    for (Block* if_block : if_node->blocks()) {
      TORCH_CHECK(
          if_block->outputs().size() == 1,
          "replicate quantize only works for `if` node with one output right now");
      // the original return value of the block
      Value* ret_val = if_block->outputs()[0];
      std::vector<Value*> quantize_inputs = n->inputs().vec();
      quantize_inputs[0] = ret_val;
      WithInsertPoint ins(if_block->return_node());
      Node* quant = graph->create(n->kind(), quantize_inputs);
      if_block->replaceOutput(0, quant->output());
      quant->output()->copyMetadata(ret_val);
      graph->insertNode(quant);
    }
  }

  for (Node* n : quant_nodes_to_rewrite) {
    n->removeAllInputs();
  }
  for (Node* n : quant_nodes_to_rewrite) {
    n->destroy();
  }
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
    auto* quantized_val = n->input(0);
    auto* dequantized_val = n->output();
    insertDeQuantForAllUse(graph.get(), quantized_val, dequantized_val);
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->removeAllInputs();
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->destroy();
  }
}

Module InsertQuantDeQuant(
    Module& input_module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type) {
  Module module = input_module.clone(inplace);
  InsertQuantDeQuantHelper h(quant_type, debug);
  h.runWeightObserver(module, method_name);
  h.run(module, method_name);
  h.cleanup(module);
  h.propagateQuantizationOps(module);
  return module;
}

/*
 *
 * Assumption: method_name method has observer placed
 * Objective: modify that method to insert calls to:
 * 1. calculate_qparams
 * 2. GetAttr for axis and dtype values
 * 3. Use Values from above two to insert calls to quant + dequant
 * Thus after this step you have a graph of, e.g., observe_forward,
 * that has observer nodes, calculate_qparams run on those observer nodes,
 * output of which is used by quant-dequant nodes. output of dequant is used
 * by the actual op.
 * Later on we will replace dequant + op (e.g. linear) with
 * 1. prepacked_op context
 * 2. unpack
 * 3. dequantize
 * 4. linear
 *
 * Of the above pattern 2, 3, and 4 can be replaced by linear_run op
 */
// Module InsertQuantDeQuantForOnDevicePTQ(
Module InsertQuantDeQuantOnDevicePTQ(
    Module& input_module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type) {
  Module module = input_module.clone(inplace);
  const std::string kObserveString = "observe_";
  const auto matched_pos = method_name.find(kObserveString);
  const auto end_pos = matched_pos + kObserveString.length();
  const std::string orig_method_name = method_name.substr(end_pos);
  TORCH_CHECK(
      matched_pos == 0,
      "Quant dequant nodes can only be added to observe_",
      orig_method_name,
      ". Please make sure to run prepare step for on-device PTQ.");

  std::string quantize_method_name = "quantize_" + orig_method_name;
  cloneMethod(module, method_name, quantize_method_name);
  InsertQuantDeQuantHelper h(quant_type, debug);
  h.runForOnDevicePTQ(module, quantize_method_name);
  h.removeObserverNodes(module);
  // Dont need:
  // ReplicateChooseQParamsQuantDequant: This is propagating dynamic quant's
  // quant dequant RemoveRedundantQuantizationOps: THis is removing activation
  // observers for dynamic quant when the op related to it is not dynamically
  // quantizable. Doesnt really make sense. In our case we wont have those
  // anyway since for dynamic quant activations wont be observed We can still
  // use this function because the above two methods should really be a noop
  h.propagateQuantizationOps(module);
  return module;
}
} // namespace jit
} // namespace torch
