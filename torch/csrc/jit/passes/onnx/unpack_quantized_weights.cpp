#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/subgraph_matcher.h>
#include <stack>

using ::c10::Dispatcher;
using ::c10::TensorTypeId;
namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;

}
template <class Result, class... Args>
inline Result callOpUnboxed(const c10::OperatorHandle& op, Args... args) {
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  return c10::Dispatcher::singleton().template callUnboxed<Result, Args...>(
      op, std::forward<Args>(args)...);
}
using ValueToParamPairMap =
    std::map<Value*, std::pair<std::string, at::Tensor>>;

using ParamMap = std::map<std::string, at::Tensor>;
ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->debugName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

double getScaleFromInput(Node* input_node) {
  c10::optional<IValue> scale;
  std::string input_name = input_node->kind().toQualString();
  if (input_name.find("quantize_per_tensor") != std::string::npos) {
    scale = toIValue(input_node->inputs()[1]);
    if (scale.value().isDouble()) {
      return scale.value().toDouble();
    }
  } else if (input_name.find("quantized::linear") != std::string::npos) {
    // %r = quantized::linear(%input, %unpacked_weight, %bias, %w_scale,
    // %w_zero_point)
    scale = toIValue(input_node->inputs()[3]);
    if (scale.value().isDouble()) {
      return scale.value().toDouble();
    }
  } else if (input_name.find("quantized::conv2d") != std::string::npos) {
    // %r = quantized::conv2d(%input, %unpacked_weight, %bias, %stride,
    // %padding, %dilation, %groups, %w_scale, %w_zero_point)
    scale = toIValue(input_node->inputs()[7]);
    if (scale.value().isDouble()) {
      return scale.value().toDouble();
    }
  } else if (input_name.find("quantized::conv2d_relu") != std::string::npos) {
    // %r = quantized::conv2d_relu(%input, %unpacked_weight, %bias, %stride,
    // %padding, %dilation, %groups, %w_scale, %w_zero_point)
    scale = toIValue(input_node->inputs()[7]);
    if (scale.value().isDouble()) {
      return scale.value().toDouble();
    }
  } else if (input_name.find("quantized::add") != std::string::npos) {
    // %r = quantized::add(%input_a, %input_b, %w_scale, %w_zero_point)
    scale = toIValue(input_node->inputs()[2]);
    if (scale.value().isDouble()) {
      return scale.value().toDouble();
    }
  }
  // For the ops below the scale is not part of the op signature, so we traverse
  // up the graph to get the scale from its input when defined in the graph.
  else if (input_name.find("quantized::max_pool2d") != std::string::npos) {
    auto tmp = input_node->inputs()[0]->node();
    return getScaleFromInput(tmp);
  } else if (input_name.find("aten::relu") != std::string::npos) {
    auto tmp = input_node->inputs()[0]->node();
    return getScaleFromInput(tmp);
  }
  return 1.0;
}

Node* CreateQuantizedWeights(
    std::string data,
    std::shared_ptr<Graph>& graph,
    std::vector<int64_t> shapes,
    double scale,
    int64_t zero_point) {
  Node* cast_node = graph->create(Symbol::caffe2("Int8GivenTensorFill"));
  cast_node->is_(Symbol::attr("shape"), shapes);
  cast_node->i_(Symbol::attr("Y_zero_point"), zero_point);
  cast_node->f_(Symbol::attr("Y_scale"), scale);
  cast_node->s_(Symbol::attr("values"), data);
  return cast_node;
}

Node* CreateQuantizedBias(
    std::vector<int64_t> data,
    std::shared_ptr<Graph>& graph,
    std::vector<int64_t> shapes,
    double scale,
    int64_t zero_point) {
  Node* cast_node = graph->create(Symbol::caffe2("Int8GivenIntTensorFill"));
  cast_node->is_(Symbol::attr("shape"), shapes);
  cast_node->i_(Symbol::attr("Y_zero_point"), zero_point);
  cast_node->f_(Symbol::attr("Y_scale"), scale);
  cast_node->is_(Symbol::attr("values"), data);
  return cast_node;
}

// This is called before the onnx pass. Using pattern matching we
// find the relevant nodes and extract the packed_params. The packed_params are
// passed to the appropriate unpack function using c10::Dispatcher. We insert
// the unpacked weights and bias into the graph using
// caffe2::Int8GivenTensorFill nodes.
void unpackQuantizedWeightsHelper(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, at::Tensor>& paramsDict,
    const std::string& pattern,
    const std::string& unpack_fn) {
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  script::parseIR(pattern, &pattern_graph, vmap);
  const auto& matches = findPatternMatches(pattern_graph, *graph);

  for (const auto& match : matches) {
    auto match_vmap = match.values_map;
    auto qlinear_node = match_vmap.at(vmap.at("r"))->node();
    std::string quantized_weight =
        match_vmap.at(vmap.at("r"))->node()->inputs()[1]->debugName();

    auto itr = paramsDict.find(quantized_weight);
    if (itr == paramsDict.end()) {
      throw std::runtime_error(
          "getValues: Quantized weight value not found amongst constant parameters.");
    }
    at::Tensor packed_weight = itr->second;
    auto op = Dispatcher::singleton().findSchema({unpack_fn, ""});
    assert(op.has_value());
    std::tuple<at::Tensor, c10::optional<at::Tensor>> result = callOpUnboxed<
        std::tuple<at::Tensor, c10::optional<at::Tensor>>,
        at::Tensor>(*op, packed_weight);
    at::Tensor unpacked_weight = std::get<0>(result);

    // Permute weights?
    /*
    if (unpacked_weight.ndimension() == 2) {
      std::cout << "2 dim weight ... permuting \n";
      unpacked_weight.permute({1, 0});
    } else if (unpacked_weight.ndimension() == 4) {
      std::cout << "4 dim weight ... permuting \n";
      unpacked_weight.permute({0, 2, 3, 1});
    }
    */

    // Remove packed_params
    qlinear_node->removeInput(1);

    // Convert from int8 to uint8
    int8_t* inp_data = (int8_t*)unpacked_weight.data_ptr<c10::qint8>();
    auto weight_zp = unpacked_weight.q_zero_point() + 128;
    auto wt_numel = unpacked_weight.numel();

    // Create caffe2::Int8GivenTensorFill node
    std::string w_data;
    for (int64_t i = 0; i < wt_numel; ++i) {
      w_data += static_cast<char>(inp_data[i] + 128);
    }

    Node* c2_weight = CreateQuantizedWeights(
        w_data,
        graph,
        unpacked_weight.sizes().vec(),
        unpacked_weight.q_scale(),
        weight_zp);
    graph->setInsertPoint(qlinear_node);
    c2_weight->insertBefore(qlinear_node);
    qlinear_node->insertInput(1, c2_weight->output());

    // Add bias
    at::Tensor original_bias;
    if (std::get<1>(result).has_value()) {
      original_bias = std::get<1>(result).value();
      original_bias.set_requires_grad(false);
    } else {
      // Caffe2 ops always expect bias tensor so if not present create empty
      // tensor.
      int64_t bias_size = unpacked_weight.size(0);
      original_bias =
          at::zeros(bias_size, unpacked_weight.options().dtype(at::kFloat));
    }

    auto weight_scale = unpacked_weight.q_scale();

    auto input_val = match_vmap.at(vmap.at("r"))->node()->inputs()[0];
    TORCH_INTERNAL_ASSERT(input_val->type()->isSubtypeOf(TensorType::get()));

    auto input_node = match_vmap.at(vmap.at("r"))->node()->inputs()[0]->node();
    auto input_scale = getScaleFromInput(input_node);
    auto q_bias = at::quantize_per_tensor(
        original_bias, weight_scale * input_scale, 0, at::kQInt32);

    std::vector<int64_t> bias_values;
    bias_values.reserve(q_bias.numel());
    auto bias_data = (int32_t*)q_bias.data_ptr<c10::qint32>();
    for (int64_t i = 0; i < q_bias.numel(); ++i) {
      bias_values.push_back(bias_data[i]);
    }
    Node* c2_bias = CreateQuantizedBias(
        bias_values,
        graph,
        q_bias.sizes().vec(),
        q_bias.q_scale(),
        q_bias.q_zero_point());
    c2_bias->insertBefore(qlinear_node);
    qlinear_node->insertInput(2, c2_bias->output());

    auto b = graph->block();
    auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
    eraseUnusedValuesFromMap(valsToParamsMap);
  }
}
void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, at::Tensor>& paramsDict) {
  std::string qlinear = R"(
  graph(%input, %packed_weight, %w_scale, %w_zero_point):
        %r = quantized::linear(%input, %packed_weight, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv = R"(
  graph(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv_relu = R"(
  graph(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d_relu(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  unpackQuantizedWeightsHelper(
      graph, paramsDict, qlinear, "quantized::linear_unpack");
  unpackQuantizedWeightsHelper(
      graph, paramsDict, qconv, "quantized::conv_unpack");
  unpackQuantizedWeightsHelper(
      graph, paramsDict, qconv_relu, "quantized::conv_unpack");
}

} // namespace jit
} // namespace torch
