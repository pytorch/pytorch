#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <stack>

using ::c10::Dispatcher;
using ::c10::DispatchKey;
namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;

}

// Get the scale of the input to quantized op. There are two cases here
// 1. For ops with output_scale specified in op signature, we get the output
// scale
// 2. For ops with no output scale in op signature (like quantized::relu)
// we traverse up the graph to get the scale from its input until we hit a node
// where scale is explicitly specified.
double getScaleFromInput(Node* input_node) {
  c10::optional<IValue> scale;
  std::string input_name = input_node->kind().toQualString();
  std::unordered_set<std::string> noscale_ops = {"quantized::max_pool2d",
                                                 "aten::max_pool2d",
                                                 "aten::relu",
                                                 "prim::ListUnpack",
                                                 "aten::split_with_sizes",
                                                 "quantized::nchw2nhwc",
                                                 "quantized::nhwc2nchw",
                                                 "aten::slice",
                                                 "aten::avg_pool2d",
                                                 "quantized::cat",
                                                 "prim::ListConstruct",
                                                 "aten::upsample_nearest2d",
                                                 "aten::sigmoid",
                                                 "aten::reshape"};
  if (input_name == "aten::quantize_per_tensor") {
    TORCH_CHECK(
        input_node->inputs().size() > 1,
        "aten::quantize_per_tensor expected scale to be 2nd input");
    scale = toIValue(input_node->inputs()[1]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::linear") {
    // %r = quantized::linear(%input, %packed_weight, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::linear expected scale to be 3rd input");
    scale = toIValue(input_node->inputs()[2]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::conv2d") {
    // %r = quantized::conv2d(%input, %packed_weight, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::conv2d expected scale to be 3rd input");
    auto num_inputs = input_node->inputs().size();
    scale = toIValue(input_node->inputs()[num_inputs - 2]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::conv2d_relu") {
    // %r = quantized::conv2d_relu(%input, %packed_weight, %w_scale,
    // %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::conv2d_relu expected scale to be 3rd input");
    auto num_inputs = input_node->inputs().size();
    scale = toIValue(input_node->inputs()[num_inputs - 2]);
    return scale.value().toDouble();
  } else if (input_name == "quantized::add") {
    // %r = quantized::add(%input_a, %input_b, %w_scale, %w_zero_point)
    TORCH_CHECK(
        input_node->inputs().size() > 2,
        "quantized::add expected scale to be 3rd input");
    scale = toIValue(input_node->inputs()[2]);
    return scale.value().toDouble();
  } else if (input_name == "aten::sigmoid") {
    // For the _caffe2::Int8Sigmoid op output scale is 1.0/256
    // And output zero_point is set to 0 (quint8 type).
    return 1.0L / 256;
  }
  // For the ops below the scale is not part of the op signature, so we traverse
  // up the graph to get the scale from its input when defined in the graph.
  else if (noscale_ops.find(input_name) != noscale_ops.end()) {
    return getScaleFromInput(input_node->inputs()[0]->node());
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "Unrecognized quantized operator while trying to compute q_scale for operator ",
      input_name);
}

Node* CreateQuantizedWeights(
    std::string data,
    std::shared_ptr<Graph>& graph,
    std::vector<int64_t> shapes,
    double scale,
    int64_t zero_point) {
  Node* const_node = graph->create(Symbol::caffe2("Int8GivenTensorFill"));
  const_node->is_(Symbol::attr("shape"), shapes);
  const_node->i_(Symbol::attr("Y_zero_point"), zero_point);
  const_node->f_(Symbol::attr("Y_scale"), scale);
  const_node->s_(Symbol::attr("values"), data);
  return const_node;
}

Node* CreateQuantizedBias(
    std::vector<int64_t> data,
    std::shared_ptr<Graph>& graph,
    std::vector<int64_t> shapes,
    double scale,
    int64_t zero_point) {
  Node* const_node = graph->create(Symbol::caffe2("Int8GivenIntTensorFill"));
  const_node->is_(Symbol::attr("shape"), shapes);
  const_node->i_(Symbol::attr("Y_zero_point"), zero_point);
  const_node->f_(Symbol::attr("Y_scale"), scale);
  const_node->is_(Symbol::attr("values"), data);
  return const_node;
}

Node* createIntTuple(
    const std::vector<int64_t>& is,
    std::shared_ptr<Graph>& graph) {
  Node* const_node = graph->create(Symbol::onnx("Constant"));
  const_node->is_(Symbol::attr("value"), is);
  return const_node;
}

Node* createInt(int64_t i, std::shared_ptr<Graph>& graph) {
  Node* const_node = graph->create(Symbol::onnx("Constant"));
  const_node->i_(Symbol::attr("value"), i);
  return const_node;
}

enum class QuantizedParamsType { CONV, LINEAR };

// This is called before the onnx pass. Using pattern matching we
// find the relevant nodes and extract the packed_params. The packed_params are
// passed to the appropriate unpack function using c10::Dispatcher. We insert
// the unpacked weights and bias into the graph using
// caffe2::Int8GivenTensorFill nodes.
void unpackQuantizedWeightsHelper(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict,
    const std::string& pattern,
    const std::string& unpack_fn,
    QuantizedParamsType params_type) {
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(pattern, &pattern_graph, vmap);
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
    at::Tensor unpacked_weight;
    c10::optional<at::Tensor> bias;
    constexpr int64_t stride_idx = 2;
    constexpr int64_t padding_idx = 3;
    constexpr int64_t dilation_idx = 4;
    constexpr int64_t groups_idx = 5;
    c10::optional<torch::List<int64_t>> stride, padding, dilation,
        output_padding;
    c10::optional<int64_t> groups;
    c10::optional<int64_t> transpose;

    torch::List<int64_t> stride_int, padding_int, dilation_int,
        output_padding_int;
    int64_t groups_int;
    int64_t transpose_int;

    if (itr->second.isTuple()) {
      // Pre-unpacked weights. Comes from Conv/Linear weights which are
      // stored as bound C++ classes.
      auto ser_tup = itr->second.toTuple();

      if (params_type == QuantizedParamsType::CONV &&
          ser_tup->elements()[0].isString()) {
        auto elements = ser_tup->elements();
        auto version = elements[0].toStringRef();
        TORCH_INTERNAL_ASSERT(version == "2", "Unknown serialization version");
        std::vector<at::Tensor> non_optional = elements[1].toTensorVector();

        at::Tensor conv_params_packed = non_optional[0];
        unpacked_weight = non_optional[1];

        const int64_t kSpatialDim = conv_params_packed[0].item<int64_t>();
        // skip kSpatialDim
        int64_t idx = 1;
        for (int i = 0; i < kSpatialDim; ++i) {
          stride_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          idx++;
        }
        for (int i = 0; i < kSpatialDim; ++i) {
          padding_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          idx++;
        }
        for (int i = 0; i < kSpatialDim; ++i) {
          dilation_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          idx++;
        }
        for (int i = 0; i < kSpatialDim; ++i) {
          output_padding_int.emplace_back(
              conv_params_packed[idx].item<int64_t>());
          idx++;
        }
        groups_int = conv_params_packed[idx].item<int64_t>();
        idx++;
        transpose_int = conv_params_packed[idx].item<int64_t>();
        idx++;
        TORCH_INTERNAL_ASSERT(
            idx == conv_params_packed.numel(),
            "Unexpected length of conv_params_packed, expected ",
            idx,
            " got ",
            conv_params_packed.numel());

        torch::List<c10::IValue> optional = elements[2].toList();
        bias = optional.get(0).toOptional<at::Tensor>();

        stride = stride_int;
        padding = padding_int;
        dilation = dilation_int;
        groups = groups_int;
        transpose = transpose_int;
      } else { // Legacy
        unpacked_weight = ser_tup->elements()[0].toTensor();
        bias = ser_tup->elements()[1].toOptional<at::Tensor>();
        // conv only parameters
        if (ser_tup->elements().size() > 2) {
          auto stride_ivalue = ser_tup->elements()[stride_idx].toListRef();
          auto padding_ivalue = ser_tup->elements()[padding_idx].toListRef();
          auto dilation_ivalue = ser_tup->elements()[dilation_idx].toListRef();
          auto groups_ivalue = ser_tup->elements()[groups_idx];

          for (const auto& s : stride_ivalue) {
            stride_int.emplace_back(s.toTensor()[0].item<int64_t>());
          }
          for (const auto& p : padding_ivalue) {
            padding_int.emplace_back(p.toTensor()[0].item<int64_t>());
          }
          for (const auto& d : dilation_ivalue) {
            dilation_int.emplace_back(d.toTensor()[0].item<int64_t>());
          }
          groups_int = groups_ivalue.toTensor()[0].item<int64_t>();
          stride = stride_int;
          padding = padding_int;
          dilation = dilation_int;
          groups = groups_int;
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(itr->second.isTensor());
      at::Tensor packed_weight = itr->second.toTensor();
      auto op = Dispatcher::singleton()
                    .findSchemaOrThrow(unpack_fn.c_str(), "")
                    .typed<std::tuple<at::Tensor, c10::optional<at::Tensor>>(
                        at::Tensor)>();
      std::tie(unpacked_weight, bias) = op.call(packed_weight);
    }

    // Permute weights
    std::vector<int64_t> wt_sizes = unpacked_weight.sizes().vec();
    if (unpacked_weight.ndimension() == 4) {
      unpacked_weight.permute({0, 2, 3, 1});
      wt_sizes = {unpacked_weight.size(0),
                  unpacked_weight.size(2),
                  unpacked_weight.size(3),
                  unpacked_weight.size(1)};
    }

    // Remove packed_params
    qlinear_node->removeInput(1);

    // Convert from int8 to uint8
    int8_t* inp_data =
        reinterpret_cast<int8_t*>(unpacked_weight.data_ptr<c10::qint8>());
    const int64_t weight_zp = unpacked_weight.q_zero_point() + 128;
    const int64_t wt_numel = unpacked_weight.numel();

    // Create caffe2::Int8GivenTensorFill node
    std::ostringstream os;
    for (int64_t i = 0; i < wt_numel; ++i) {
      os << static_cast<char>(inp_data[i] + 128);
    }

    Node* c2_weight = CreateQuantizedWeights(
        os.str(), graph, wt_sizes, unpacked_weight.q_scale(), weight_zp);
    graph->setInsertPoint(qlinear_node);
    c2_weight->insertBefore(qlinear_node);
    qlinear_node->insertInput(1, c2_weight->output());

    // Add bias
    at::Tensor original_bias;
    if (bias.has_value()) {
      original_bias = bias.value();
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
    TORCH_INTERNAL_ASSERT(
        input_val->type()->isSubtypeOf(TensorType::get()),
        "Unsupported input type. Expected TensorType, got ",
        input_val->type()->str());

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

    // add conv arguments: stride, padding, dilation, groups
    if (stride.has_value() && padding.has_value() && dilation.has_value() &&
        groups.has_value()) {
      std::vector<c10::optional<torch::List<int64_t>>> conv_ints_args;
      conv_ints_args.push_back(stride);
      conv_ints_args.push_back(padding);
      conv_ints_args.push_back(dilation);
      const size_t arg_offset = 3;
      for (size_t i = 0; i < conv_ints_args.size(); ++i) {
        Node* ints_node =
            createIntTuple(conv_ints_args[i].value().vec(), graph);
        ints_node->insertBefore(qlinear_node);
        qlinear_node->insertInput(arg_offset + i, ints_node->output());
      }
      Node* groups_node = createInt(groups.value(), graph);
      groups_node->insertBefore(qlinear_node);
      qlinear_node->insertInput(groups_idx + 1, groups_node->output());
    }
    auto b = graph->block();
    auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
    eraseUnusedValuesFromMap(valsToParamsMap);
  }
}
void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict) {
  std::string qlinear = R"(
  graph(%input, %packed_weight, %w_scale, %w_zero_point):
        %r = quantized::linear(%input, %packed_weight, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv2d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv2d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv2d_relu = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv2d_relu(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv3d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv3d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv3d_relu = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv3d_relu(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qlinear,
      "quantized::linear_unpack",
      QuantizedParamsType::LINEAR);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv2d,
      "quantized::conv2d_unpack",
      QuantizedParamsType::CONV);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv2d_relu,
      "quantized::conv2d_unpack",
      QuantizedParamsType::CONV);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv3d,
      "quantized::conv3d_unpack",
      QuantizedParamsType::CONV);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv3d_relu,
      "quantized::conv3d_unpack",
      QuantizedParamsType::CONV);
}

// Caffe2 expects quantized ops to be in NHWC format while pytorch inputs are in
// NCHW. This pass inserts permutes to convert from NCHW to NHWC before each
// conv op and add another permute from NHWC to NCHW after the conv op.
void insertPermutesHelper(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict,
    const std::string& pattern) {
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(pattern, &pattern_graph, vmap);

  const auto& matches = findPatternMatches(pattern_graph, *graph);

  for (const auto& match : matches) {
    auto match_vmap = match.values_map;
    auto op_node = match_vmap.at(vmap.at("r"))->node();
    auto input_node = match_vmap.at(vmap.at("r"))->node()->inputs()[0]->node();

    Node* permute_node_before = graph->create(
        Symbol::fromQualString("quantized::nchw2nhwc"), {input_node->output()});
    permute_node_before->insertBefore(op_node);
    op_node->removeInput(0);
    op_node->insertInput(0, permute_node_before->output());

    Node* permute_node_after = graph->create(
        Symbol::fromQualString("quantized::nhwc2nchw"),
        {op_node->outputs()[0]});
    permute_node_after->insertAfter(op_node);
    auto v = op_node->outputs().at(0);
    v->replaceAllUsesWith(permute_node_after->outputs().at(0));
    permute_node_after->removeInput(0);
    permute_node_after->addInput(v);
  }
}

void insertPermutes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict) {
  std::string qconv = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv_relu = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv2d_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv_transpose = R"(
  graph(%input, %weight, %bias, %stride, %padding, %dilation, %output_padding, %groups, %w_scale, %w_zero_point):
        %r = quantized::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";

  insertPermutesHelper(graph, paramsDict, qconv);
  insertPermutesHelper(graph, paramsDict, qconv_relu);
  insertPermutesHelper(graph, paramsDict, qconv_transpose);
}

} // namespace jit
} // namespace torch
