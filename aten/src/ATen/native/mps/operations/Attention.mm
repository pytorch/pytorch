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

namespace at {
namespace native {

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  const std::optional<Tensor>& attn_mask,
                                                                  double dropout_p,
                                                                  bool is_causal,
                                                                  const std::optional<Tensor>& dropout_mask,
                                                                  std::optional<double> scale) {
  const auto macOS15_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  if (is_causal) {
    TORCH_CHECK(!attn_mask.has_value(),
                "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
  }

  TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_for_mps: dropout_p != 0.0 is not supported");
  TORCH_CHECK(macOS15_0_plus || (query.is_contiguous() && key.is_contiguous() && value.is_contiguous()),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must be contiguous");
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must not be nested");

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
  auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
  @autoreleasepool {
    auto mkey = __func__ + getTensorsStringKey({query, key, value}) + ":" + std::to_string(is_causal) + ":" +
        std::to_string(attn_mask.has_value());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&](auto mpsGraph, auto graph) {
      auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, query);
      auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, key);
      auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, value);
      auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
      auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];

      auto maskedMM = [mpsGraph matrixMultiplicationWithPrimaryTensor:qTensor secondaryTensor:kT name:nil];

      if (macOS15_0_plus && [maskedMM dataType] == MPSDataTypeFloat32) {
        // TODO: In MacOS15 beta, there is a MPSGraph issue when the SDPA sequence gets remapped to use
        // an improved kernel for the computation, causing NaNs in the result. This identity prevents the remapping.
        // Limit the availability check once a fix lands.
        maskedMM = [mpsGraph identityWithTensor:maskedMM name:nil];
      }

      // upcasting to float32 if needed to improve precision when multiplying by the scale factor
      if ([maskedMM dataType] != MPSDataTypeFloat32) {
        maskedMM = [mpsGraph castTensor:maskedMM toType:MPSDataTypeFloat32 name:nil];
      }
      maskedMM = [mpsGraph multiplicationWithPrimaryTensor:maskedMM secondaryTensor:scaleTensor name:nil];
      if ([maskedMM dataType] != qTensor.dataType) {
        maskedMM = [mpsGraph castTensor:maskedMM toType:qTensor.dataType name:nil];
      }

      if (is_causal) {
        auto causalMask = [mpsGraph constantWithScalar:1.0f
                                                 shape:getMPSShape({qSize, maxSeqLength})
                                              dataType:MPSDataTypeBool];
        causalMask = [mpsGraph bandPartWithTensor:causalMask numLower:-1 numUpper:0 name:nil];
        auto minusInf = [mpsGraph constantWithScalar:-1e20 shape:maskedMM.shape dataType:maskedMM.dataType];
        maskedMM = [mpsGraph selectWithPredicateTensor:causalMask
                                   truePredicateTensor:maskedMM
                                  falsePredicateTensor:minusInf
                                                  name:nil];
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
    NSDictionary* feeds = nil;
    if (!attn_mask) {
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
    } else {
      auto mPlaceholder = Placeholder(cachedGraph->maskTensor, *attn_mask);
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder, mPlaceholder);
    }
    NSDictionary* outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
  }
  return {out, attn};
}

} // namespace native
} // namespace at
