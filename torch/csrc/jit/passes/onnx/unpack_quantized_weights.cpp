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
inline Result callOpUnboxed(
    const c10::OperatorHandle& op,
    c10::TensorTypeId dispatchKey,
    Args... args) {
  return c10::Dispatcher::singleton().template callUnboxed<Result, Args...>(
      op, dispatchKey, std::forward<Args>(args)...);
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
// This is called after onnx optimize_graph so the graph already contains
// "_caffe2" nodes at this point for quantized ops. Using pattern matching we
// find the relevant nodes and extract the packed_params The packed_params are
// passed to the appropriate unpack function using c10::Dispatcher. We insert
// the unpacked weights and bias into the graph using prim::Constant nodes.
void unpackQuantizedWeightsHelper(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, at::Tensor>& paramsDict,
    std::string pattern,
    std::string unpack_fn) {
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  script::parseIR(pattern, &pattern_graph, vmap);
  const auto& matches = findPatternMatches(pattern_graph, *graph);
  for (const auto& match : matches) {
    auto match_vmap = match.values_map;
    auto qlinear_node = match_vmap.at(vmap.at("r"))->node();
    std::string quantized_weight =
        match_vmap.at(vmap.at("r"))->node()->inputs()[1]->debugName();
    auto packed_node = match_vmap.at(vmap.at("r"))->node()->inputs()[1]->node();
    auto itr = paramsDict.find(quantized_weight);
    if (itr == paramsDict.end()) {
      throw std::runtime_error(
          "getValues: Quantized weight value not found amongst constant parameters.");
    }
    at::Tensor packed_weight = itr->second;
    auto op = Dispatcher::singleton().findSchema({unpack_fn, ""});
    assert(op.has_value());
    auto key = c10::TensorTypeId::CPUTensorId;
    std::tuple<at::Tensor, c10::optional<at::Tensor>> result = callOpUnboxed<
        std::tuple<at::Tensor, c10::optional<at::Tensor>>,
        at::Tensor>(*op, key, packed_weight);
    at::Tensor unpacked_weight = std::get<0>(result);
    // Remove packed_params
    qlinear_node->removeInput(1);

    // Update the input
    auto val = graph->insertConstant(unpacked_weight);
    qlinear_node->insertInput(1, val);
    std::cout << "Inserted weight " << val->debugName();
    // TODO add to paramsDict?
    // auto tmp = std::get<0>(result);
    // paramsDict[val->debugName()] = std::get<0>(result);

    // Add bias
    if (std::get<1>(result).has_value()) {
      at::Tensor original_bias = std::get<1>(result).value();
      original_bias.set_requires_grad(false);
      auto val = graph->insertConstant(original_bias);
      qlinear_node->insertInput(2, val);
      std::cout << "Inserted bias " << val->debugName();
    }
    auto b = graph->block();
    auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
    eraseUnusedValuesFromMap(valsToParamsMap);

    // Delete original node??
    // packed_node->destroy();
  }
}
void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, at::Tensor>& paramsDict) {
  std::string qlinear = R"(
  graph(%input, %packed_weight, %w_scale, %w_zero_point):
        %r = _caffe2::Int8FC(%input, %packed_weight, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv = R"(
  graph(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = _caffe2::Int8Conv(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  std::string qconv_relu = R"(
  graph(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point):
        %r = _caffe2::Int8ConvRelu(%input, %packed_weight, %stride, %padding, %dilation, %groups, %w_scale, %w_zero_point)
        return (%r) )";
  unpackQuantizedWeightsHelper(
      graph, paramsDict, qlinear, "quantized::linear_unpack");
  unpackQuantizedWeightsHelper(
      graph, paramsDict, qconv, "quantized::conv_unpack");
  // unpackQuantizedWeightsHelper(graph, paramsDict, qconv_relu,
  // "quantized::conv_unpack");
}

} // namespace jit
} // namespace torch
