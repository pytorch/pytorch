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
    MPSGraphTensor* maskTensor = nil;
    MPSGraphTensor* outputTensor = nil;
    MPSGraphTensor* attnTensor = nil;
  };
  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t qSize = query.size(2);
  int64_t headSize = query.size(3);
  int64_t maxSeqLength = key.size(2);
  auto out = at::empty({batchSize, num_head, qSize, headSize}, query.options());
  auto attn = at::empty({batchSize, num_head, qSize, maxSeqLength}, query.options());
  auto scale_factor = sdp::calculate_scale(query, scale).sqrt().as_float_unchecked();
  @autoreleasepool {
    auto mkey = __func__ + getTensorsStringKey({query, key, value}) + ":" + std::to_string(is_causal) + ":" + std::to_string(attn_mask.has_value());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&](auto mpsGraph, auto graph) {
      auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, query);
      auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, key);
      auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, value);
      auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
      auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];

      // sqTensor = qTensor * scale_factor
      auto castedQ = qTensor;
      if ([qTensor dataType] != MPSDataTypeFloat32) {
        castedQ = [mpsGraph castTensor:qTensor toType:MPSDataTypeFloat32 name:nil];
      }
      auto sqTensor = [mpsGraph multiplicationWithPrimaryTensor:castedQ secondaryTensor:scaleTensor name:nil];
      if ([sqTensor dataType] != qTensor.dataType) {
        sqTensor = [mpsGraph castTensor:sqTensor toType:qTensor.dataType name:nil];
      }

      // skT = kT * scale_factor
      auto castedKT = kT;
      if ([kT dataType] != MPSDataTypeFloat32) {
        castedKT = [mpsGraph castTensor:kT toType:MPSDataTypeFloat32 name:nil];
      }
      auto skT = [mpsGraph multiplicationWithPrimaryTensor:castedKT secondaryTensor:scaleTensor name:nil];
      if ([skT dataType] != kT.dataType) {
        skT = [mpsGraph castTensor:skT toType:kT.dataType name:nil];
      }

      auto maskedMM = [mpsGraph matrixMultiplicationWithPrimaryTensor:sqTensor secondaryTensor:skT name:nil];
      if (is_causal) {
        auto minusInf = [mpsGraph constantWithScalar:std::numeric_limits<double>::infinity() shape:maskedMM.shape dataType:maskedMM.dataType];
        auto causalMask = [mpsGraph bandPartWithTensor:minusInf numLower:-1 numUpper:0 name:nil];
        maskedMM = [mpsGraph additionWithPrimaryTensor:maskedMM secondaryTensor:causalMask name:nil];
      } else if (attn_mask) {
        graph->maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, *attn_mask);
        maskedMM = [mpsGraph additionWithPrimaryTensor:maskedMM secondaryTensor:graph->maskTensor name:nil];
      }
      auto sm = [mpsGraph softMaxWithTensor:maskedMM axis:3 name:nil];
      auto output = [mpsGraph matrixMultiplicationWithPrimaryTensor:sm secondaryTensor:vTensor name:nil];
      graph->qTensor = qTensor;
      graph->kTensor = kTensor;
      graph->vTensor = vTensor;
      graph->outputTensor = output;
      graph->attnTensor = sm;
    });
    auto qPlaceholder = Placeholder(cachedGraph->qTensor, query);
    auto kPlaceholder = Placeholder(cachedGraph->kTensor, key);
    auto vPlaceholder = Placeholder(cachedGraph->vTensor, value);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out);
    auto attnPlaceholder = Placeholder(cachedGraph->attnTensor, attn);
    NSDictionary *feeds = nil;
    if (!attn_mask) {
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
    } else {
      auto mPlaceholder = Placeholder(cachedGraph->maskTensor, *attn_mask);
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder, mPlaceholder);
    }
    NSDictionary *outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
  }
  return {out, attn};
}

} // namespace at::native
