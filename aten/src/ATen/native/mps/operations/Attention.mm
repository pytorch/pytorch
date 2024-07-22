#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <iostream>
#include <optional>

#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/empty_native.h>
#endif

namespace at::native {

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  const std::optional<Tensor>& attn_mask,
                                                                  double dropout_p,
                                                                  bool is_causal,
                                                                  const std::optional<Tensor>& dropout_mask,
                                                                  std::optional<double> scale) {
  using namespace mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* qTensor = nil;
    MPSGraphTensor* kTensor = nil;
    MPSGraphTensor* vTensor = nil;
    MPSGraphTensor* attnTensor = nil;
    MPSGraphTensor* outputTensor = nil;
  };
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(3);
  auto out = at::empty({batchSize, num_head, qSize, headSize}, query.options());
  auto scale_factor = sdp::calculate_scale(query, scale).sqrt().as_float_unchecked();
  @autoreleasepool {
    auto mkey = __func__ + getTensorsStringKey({query, key, value}) + ":" + std::to_string(is_causal) + ":" + std::to_string(attn_mask.has_value());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&](auto mpsGraph, auto graph) {
      auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, query);
      auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, key);
      auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, value);
      auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
      auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:kT.dataType];
      auto sqTensor = [mpsGraph multiplicationWithPrimaryTensor:qTensor secondaryTensor:scaleTensor name:nil];
      auto skT = [mpsGraph multiplicationWithPrimaryTensor:kT secondaryTensor:scaleTensor name:nil];
      auto maskedMM = [mpsGraph matrixMultiplicationWithPrimaryTensor:sqTensor secondaryTensor:skT name:nil];
      if (is_causal) {
        auto causalMask = [mpsGraph constantWithScalar:1.0f shape:getMPSShape({qSize, qSize}) dataType:MPSDataTypeBool];
        causalMask = [mpsGraph bandPartWithTensor:causalMask numLower:-1 numUpper:0 name:nil];
        // auto minusInf = [mpsGraph constantWithScalar:std::numeric_limits<double>::infinity() shape:maskedMM.shape
        // dataType:maskedMM.dataType];
        auto minusInf = [mpsGraph constantWithScalar:-1e20 shape:maskedMM.shape dataType:maskedMM.dataType];
        maskedMM = [mpsGraph selectWithPredicateTensor:causalMask
                                   truePredicateTensor:maskedMM
                                  falsePredicateTensor:minusInf
                                                  name:nil];
      } else if (attn_mask) {
        graph->attnTensor = mpsGraphRankedPlaceHolder(mpsGraph, *attn_mask);
        maskedMM = [mpsGraph additionWithPrimaryTensor:maskedMM secondaryTensor:graph->attnTensor name:nil];
      }
      auto sm = [mpsGraph softMaxWithTensor:maskedMM axis:3 name:nil];
      auto output = [mpsGraph matrixMultiplicationWithPrimaryTensor:sm secondaryTensor:vTensor name:nil];
      graph->qTensor = qTensor;
      graph->kTensor = kTensor;
      graph->vTensor = vTensor;
      graph->outputTensor = output;
    });
    auto qPlaceholder = Placeholder(cachedGraph->qTensor, query);
    auto kPlaceholder = Placeholder(cachedGraph->kTensor, key);
    auto vPlaceholder = Placeholder(cachedGraph->vTensor, value);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out);
    NSDictionary *feeds = nil;
    if (!attn_mask) {
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
    } else {
      auto mPlaceholder = Placeholder(cachedGraph->attnTensor, *attn_mask);
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder, mPlaceholder);
    }
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return {out, Tensor()};
}

} // namespace at::native
