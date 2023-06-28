#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>

#include <ATen/native/quantized/PackedParams.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// TODO: Switch to per operator headers after
// https://github.com/pytorch/pytorch/pull/68693 is merged
#include <ATen/Functions.h>

#include <stack>

using ::c10::Dispatcher;
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
  std::unordered_set<std::string> noscale_ops = {
      "quantized::max_pool2d",
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

Node* CreateQuantizedWeightsCaffe2(
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

Node* CreateQuantizedBiasCaffe2(
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

std::vector<Node*> CreateQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    const at::Tensor& weight,
    int8_t* data,
    const std::vector<int64_t>& shapes,
    const std::vector<int64_t>& strides) {
  auto qscheme = weight.qscheme();
  std::vector<Node*> unpacked_wt;

  // Retrieve scales and zero_points. Their formats are different depending on
  // different weight qscheme.
  std::vector<float> scale_data;
  std::vector<int64_t> scale_shapes;
  std::vector<int64_t> zero_point_data;
  std::vector<int64_t> zero_point_shapes;
  std::vector<int64_t> axis_data;
  switch (qscheme) {
    case c10::kPerTensorAffine: {
      // Cast to float since ONNX (De)QuantizeLinear only supports float scale.
      scale_data = {static_cast<float>(weight.q_scale())};
      scale_shapes = {1};
      zero_point_data = {weight.q_zero_point()};
      zero_point_shapes = {1};
      break;
    }
    case c10::kPerChannelAffine:
    case c10::kPerChannelAffineFloatQParams: {
      auto q_scales = weight.q_per_channel_scales();
      auto* scale_data_raw = q_scales.const_data_ptr<double>();
      scale_shapes = q_scales.sizes().vec();
      TORCH_INTERNAL_ASSERT(
          scale_shapes.size() == 1,
          "quantized per channel scales are expected as 1-d array.");
      scale_data.resize(scale_shapes[0]);
      // Cast to float since ONNX (De)QuantizeLinear only supports float scale.
      std::transform(
          scale_data_raw,
          scale_data_raw + scale_shapes[0],
          scale_data.begin(),
          [](double x) { return static_cast<float>(x); });

      auto q_zero_points = weight.q_per_channel_zero_points();
      auto* zero_point_data_raw = q_zero_points.const_data_ptr<int64_t>();
      zero_point_shapes = q_zero_points.sizes().vec();
      TORCH_INTERNAL_ASSERT(
          zero_point_shapes.size() == 1,
          "quantized per channel zero points are expected as 1-d array.");
      zero_point_data = std::vector<int64_t>(
          zero_point_data_raw, zero_point_data_raw + zero_point_shapes[0]);
      axis_data = {weight.q_per_channel_axis()};
      break;
    }
    default:
      TORCH_CHECK(
          false, "Unsupported qscheme for weight, got ", toString(qscheme));
  }

  Node* data_node = graph->create(prim::Constant);
  auto data_value =
      at::from_blob(
          data, c10::IntArrayRef(shapes), c10::IntArrayRef(strides), at::kChar)
          .to(at::kCPU);
  // Need clone because at::from_blob does not take ownership of data.
  data_node->t_(Symbol::attr("value"), data_value.clone());

  Node* scale_node = graph->create(prim::Constant);
  auto scale_value =
      at::from_blob(
          scale_data.data(), c10::IntArrayRef(scale_shapes), at::kFloat)
          .to(at::kCPU);
  scale_node->t_(Symbol::attr("value"), scale_value.clone());

  Node* zero_point_node = graph->create(prim::Constant);
  auto zero_point_value =
      at::from_blob(
          zero_point_data.data(), c10::IntArrayRef(zero_point_shapes), at::kInt)
          .to(at::kCPU);
  zero_point_node->t_(Symbol::attr("value"), zero_point_value.clone());

  Node* axis_node = graph->create(prim::Constant);
  if (!axis_data.empty()) {
    auto axis_value =
        at::from_blob(
            axis_data.data(), c10::IntArrayRef(axis_data.size()), at::kLong)
            .to(at::kCPU);
    axis_node->t_(attr::value, axis_value.clone());
  } else {
    axis_node->output()->setType(NoneType::get());
  }

  return {data_node, scale_node, zero_point_node, axis_node};
}

Node* CreateQuantizedBias(
    std::vector<float> data,
    std::shared_ptr<Graph>& graph,
    std::vector<int64_t> shapes) {
  Node* const_node_1 = graph->create(prim::Constant);
  auto const_bias =
      at::from_blob(data.data(), c10::IntArrayRef(shapes), at::kFloat)
          .to(at::kCPU);
  auto options = c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor const_bias_copy = at::empty(c10::IntArrayRef(shapes), options);
  const_bias_copy.copy_(const_bias);
  const_node_1->t_(Symbol::attr("value"), const_bias_copy);
  return const_node_1;
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

void ConvertQuantizedWeight(
    std::shared_ptr<Graph>& graph,
    Node* node,
    at::Tensor& weight,
    bool is_caffe2) {
  std::vector<int64_t> wt_sizes = weight.sizes().vec();
  std::vector<int64_t> wt_strides = weight.strides().vec();
  if (weight.ndimension() == 4 && is_caffe2) {
    // Permute weights
    weight.permute({0, 2, 3, 1});
    wt_sizes = {weight.size(0), weight.size(2), weight.size(3), weight.size(1)};
  }

  // Remove packed_params
  node->removeInput(1);

  auto* wt_data =
      reinterpret_cast<int8_t*>(weight.mutable_data_ptr<c10::qint8>());

  if (is_caffe2) {
    // Convert from int8 to uint8
    const int64_t weight_zp = weight.q_zero_point() + 128;
    const int64_t wt_numel = weight.numel();
    // Create caffe2::Int8GivenTensorFill node
    std::ostringstream os;
    for (const auto i : c10::irange(wt_numel)) {
      os << static_cast<char>(wt_data[i] + 128);
    }
    Node* c2_weight = CreateQuantizedWeightsCaffe2(
        os.str(), graph, wt_sizes, weight.q_scale(), weight_zp);
    graph->setInsertPoint(node);
    c2_weight->insertBefore(node);
    node->insertInput(1, c2_weight->output());
  } else {
    std::vector<Node*> unpacked_wt =
        CreateQuantizedWeights(graph, weight, wt_data, wt_sizes, wt_strides);
    graph->setInsertPoint(node);
    Node* quant_node = graph->create(prim::TupleConstruct);
    for (auto* n : unpacked_wt) {
      n->insertBefore(node);
      quant_node->addInput(n->output());
    }
    quant_node->insertBefore(node);
    node->insertInput(1, quant_node->output());
  }
}

// CONV1D needs a different unpacking from CONV, since it's
// packed as CONV2D intentionally at the first place.
// See: https://github.com/pytorch/pytorch/pull/38248
enum class QuantizedParamsType { CONV1D, CONV, LINEAR };

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
    QuantizedParamsType params_type,
    bool caffe2 = true) {
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
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t groups_int;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t transpose_int;

    if (itr->second.isTuple()) {
      // Pre-unpacked weights. Comes from Conv/Linear weights which are
      // stored as bound C++ classes.
      auto ser_tup = itr->second.toTuple();

      if (params_type == QuantizedParamsType::CONV &&
          ser_tup->elements()[0].isInt()) {
        const auto& elements = ser_tup->elements();
        auto version = elements[0].toInt();
        TORCH_INTERNAL_ASSERT(version == 3, "Unknown serialization version");
        TORCH_INTERNAL_ASSERT(elements.size() == 3, "Wrong tuple size.");

        auto config_vals = elements[1].to<std::vector<int64_t>>();
        auto tensors = elements[2].to<std::vector<c10::optional<at::Tensor>>>();

        c10::optional<at::Tensor> weight = tensors[1];
        TORCH_INTERNAL_ASSERT(
            weight, "Weight should always be present in serialized qconv.");
        unpacked_weight = *weight;
        bias = tensors[2];

        const int64_t kSpatialDim = config_vals.at(0);
        // skip kSpatialDim
        unsigned idx = 1;
        for (const auto i : c10::irange(kSpatialDim)) {
          (void)i; // Suppress unused variable warning
          stride_int.emplace_back(config_vals.at(idx));
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          (void)i; // Suppress unused variable warning
          padding_int.emplace_back(config_vals.at(idx));
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          (void)i; // Suppress unused variable warning
          dilation_int.emplace_back(config_vals.at(idx));
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          (void)i; // Suppress unused variable warning
          output_padding_int.emplace_back(config_vals.at(idx));
          idx++;
        }
        int64_t groups_int = config_vals.at(idx);
        idx++;
        int64_t flags = config_vals.at(idx);
        idx++;
        TORCH_INTERNAL_ASSERT(
            idx == config_vals.size(),
            "Unexpected length of config_vals, expected ",
            idx,
            " got ",
            config_vals.size());

        bool transpose_int = flags & (1 << 0);

        int64_t other_flags = flags & ~(1 << 0);
        TORCH_CHECK(other_flags == 0, "Unexpected flags set in ", flags, ".");

        stride = stride_int;
        padding = padding_int;
        dilation = dilation_int;
        groups = groups_int;
        transpose = transpose_int;
      } else if (
          (params_type == QuantizedParamsType::CONV ||
           params_type == QuantizedParamsType::CONV1D) &&
          ser_tup->elements()[0].isString()) {
        const auto& elements = ser_tup->elements();
        auto version = elements[0].toStringRef();
        TORCH_INTERNAL_ASSERT(version == "2", "Unknown serialization version");
        std::vector<at::Tensor> non_optional = elements[1].toTensorVector();

        at::Tensor conv_params_packed = non_optional[0];
        unpacked_weight = non_optional[1];

        const int64_t kSpatialDim = conv_params_packed[0].item<int64_t>();
        // skip kSpatialDim
        int64_t idx = 1;
        // kSpatialDim = 2 even it's for Conv1D from torch.op to adopt Conv2D,
        // so we need a special unpack for Conv1D which has Conv2D dim.
        // See: https://github.com/pytorch/pytorch/pull/38248
        for (const auto i : c10::irange(kSpatialDim)) {
          if (params_type != QuantizedParamsType::CONV1D || i != 0) {
            stride_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          }
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          if (params_type != QuantizedParamsType::CONV1D || i != 0) {
            padding_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          }
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          if (params_type != QuantizedParamsType::CONV1D || i != 0) {
            dilation_int.emplace_back(conv_params_packed[idx].item<int64_t>());
          }
          idx++;
        }
        for (const auto i : c10::irange(kSpatialDim)) {
          if (params_type != QuantizedParamsType::CONV1D || i != 0) {
            output_padding_int.emplace_back(
                conv_params_packed[idx].item<int64_t>());
          }
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

        if (params_type == QuantizedParamsType::CONV1D) {
          unpacked_weight = unpacked_weight.squeeze_(2);
        }
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

    ConvertQuantizedWeight(graph, qlinear_node, unpacked_weight, caffe2);

    // Add bias
    at::Tensor original_bias;
    if (bias.has_value()) {
      original_bias = bias.value();
      original_bias.set_requires_grad(false);
    } else {
      int64_t bias_size = unpacked_weight.size(0);
      original_bias =
          at::zeros(bias_size, unpacked_weight.options().dtype(at::kFloat));
    }

    auto input_val = match_vmap.at(vmap.at("r"))->node()->inputs()[0];
    TORCH_INTERNAL_ASSERT(
        input_val->type()->isSubtypeOf(*TensorType::get()),
        "Unsupported input type. Expected TensorType, got ",
        input_val->type()->str());

    auto input_node = match_vmap.at(vmap.at("r"))->node()->inputs()[0]->node();
    at::Tensor q_bias;

    if (caffe2) {
      auto weight_scale = unpacked_weight.q_scale();
      auto input_scale = getScaleFromInput(input_node);
      q_bias = at::quantize_per_tensor(
          original_bias, weight_scale * input_scale, 0, at::kQInt32);
      std::vector<int64_t> bias_values;
      bias_values.reserve(q_bias.numel());
      auto bias_data = (const int32_t*)q_bias.const_data_ptr<c10::qint32>();
      for (const auto i : c10::irange(q_bias.numel())) {
        bias_values.push_back(bias_data[i]);
      }
      Node* c2_bias = CreateQuantizedBiasCaffe2(
          bias_values,
          graph,
          q_bias.sizes().vec(),
          q_bias.q_scale(),
          q_bias.q_zero_point());
      c2_bias->insertBefore(qlinear_node);
      qlinear_node->insertInput(2, c2_bias->output());
    } else {
      std::vector<float> bias_values(original_bias.numel());
      auto bias_data = original_bias.const_data_ptr<float>();
      for (const auto i : c10::irange(original_bias.numel())) {
        bias_values[i] = bias_data[i];
      }
      Node* bias =
          CreateQuantizedBias(bias_values, graph, original_bias.sizes().vec());
      bias->insertBefore(qlinear_node);
      // For quantized_linear inputs, the order is input, weight, bias, ....
      // Therefore bias is at location 2.
      qlinear_node->insertInput(2, bias->output());
    }

    // add conv arguments: stride, padding, dilation, groups
    if (stride.has_value() && padding.has_value() && dilation.has_value() &&
        groups.has_value()) {
      std::vector<c10::optional<torch::List<int64_t>>> conv_ints_args;
      conv_ints_args.push_back(stride);
      conv_ints_args.push_back(padding);
      conv_ints_args.push_back(dilation);
      // skip (input, weight, bias)
      const size_t arg_offset = 3;
      for (const auto i : c10::irange(conv_ints_args.size())) {
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

static std::
    unordered_map<c10::ScalarType, c10::ScalarType, ScalarTypeHashFunction>
        qTypeToValType = {
            {c10::ScalarType::QInt8, c10::ScalarType::Char},
            {c10::ScalarType::QUInt8, c10::ScalarType::Byte},
            {c10::ScalarType::QInt32, c10::ScalarType::Int},
            {c10::ScalarType::QUInt4x2, c10::ScalarType::Byte},
};

// Unpack quantized tensor inputs into {value, scale, zero_point},
// Then create a prim::TupleConstruct node based on these three values.
void UnpackQuantizedTensorInputs(std::shared_ptr<Graph>& graph) {
  for (size_t index = 0; index < graph->inputs().size();) {
    auto g_input = graph->inputs()[index];
    TensorTypePtr shape_type = g_input->type()->cast<TensorType>();
    if (!shape_type || !shape_type->scalarType().has_value()) {
      index++;
      continue;
    }
    auto scalar_type = shape_type->scalarType().value();
    if (qTypeToValType.find(scalar_type) == qTypeToValType.end()) {
      index++;
      continue;
    }
    std::string input_name = g_input->debugName();
    auto input_value =
        graph->insertInput(index, input_name + "_value")
            ->setType(shape_type->withScalarType(qTypeToValType[scalar_type]));
    // scale and zero_point type can be found at torch/include/ATen/Operators.h
    auto input_scale =
        graph->insertInput(index + 1, input_name + "_scale")
            ->setType(TensorType::create(
                at::kDouble, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
    auto input_zero_point =
        graph->insertInput(index + 2, input_name + "_zero_point")
            ->setType(TensorType::create(
                at::kLong, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
    std::vector<Value*> converted{input_value, input_scale, input_zero_point};
    auto input_tuple =
        graph->prependNode(graph->createTuple(converted))->output();
    g_input->replaceAllUsesWith(input_tuple);
    // Erase the original quantized tensor input.
    graph->eraseInput(index + converted.size());
    index += 3;
  }
}

// https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict,
    bool caffe2) {
  std::string qlinear = R"(
  graph(%input, %packed_weight, %w_scale, %w_zero_point):
        %r = quantized::linear(%input, %packed_weight, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv1d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv1d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv1d_relu = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv1d_relu(%input, %packed_params, %scale, %zero_point)
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
  std::string qconv_transpose1d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv_transpose1d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv_transpose2d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv_transpose2d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  std::string qconv_transpose3d = R"(
  graph(%input, %packed_params, %scale, %zero_point):
        %r = quantized::conv_transpose3d(%input, %packed_params, %scale, %zero_point)
        return (%r) )";
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qlinear,
      "quantized::linear_unpack",
      QuantizedParamsType::LINEAR,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv1d,
      "quantized::conv1d_unpack",
      QuantizedParamsType::CONV1D,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv2d,
      "quantized::conv2d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv1d_relu,
      "quantized::conv1d_unpack",
      QuantizedParamsType::CONV1D,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv2d_relu,
      "quantized::conv2d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv3d,
      "quantized::conv3d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv3d_relu,
      "quantized::conv3d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv_transpose1d,
      "quantized::conv_transpose1d_unpack",
      QuantizedParamsType::CONV1D,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv_transpose2d,
      "quantized::conv_transpose2d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  unpackQuantizedWeightsHelper(
      graph,
      paramsDict,
      qconv_transpose3d,
      "quantized::conv_transpose3d_unpack",
      QuantizedParamsType::CONV,
      caffe2);
  if (!caffe2) {
    UnpackQuantizedTensorInputs(graph);
  }
  GRAPH_DUMP("After UnpackQuantizedWeights: ", graph);
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
  GRAPH_DUMP("After insertPermutes: ", graph);
}

} // namespace jit
} // namespace torch
