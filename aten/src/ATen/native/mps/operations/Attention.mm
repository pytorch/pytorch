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

// expand potential 3d to 4d tensor
static inline std::tuple<Tensor, bool> ensure_4d(const Tensor& x) {
  if (x.dim() == 3) {
    return {x.unsqueeze(0), true};
  } else if (x.dim() > 4) {
    auto batchSize = c10::multiply_integers(x.sizes().begin(), x.sizes().end() - 3);
    return {x.view({batchSize, x.size(-3), x.size(-2), x.size(-1)}), true};
  } else {
    return {x, false};
  }
}

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
  TORCH_CHECK(query.size(-3) == key.size(-3) && key.size(-3) == value.size(-3),
              "number of heads in query/key/value should match");
  TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_for_mps: dropout_p != 0.0 is not supported");
  TORCH_CHECK(macOS15_0_plus || (query.is_contiguous() && key.is_contiguous() && value.is_contiguous()),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must be contiguous");
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must not be nested");

  // Ensure 4D tensors
  auto [q_, sq] = ensure_4d(query);
  auto [k_, sk] = ensure_4d(key);
  auto [v_, sv] = ensure_4d(value);

  std::optional<Tensor> mask_;
  if (attn_mask) {
    auto maskExpandedDims = query.sizes().vec();
    maskExpandedDims[maskExpandedDims.size() - 1] = k_.size(2);
    mask_ = attn_mask->expand(maskExpandedDims);
    std::tie(*mask_, std::ignore) = ensure_4d(*mask_);
  }

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
  int64_t batchSize = q_.size(0);
  int64_t num_head = q_.size(1);
  int64_t qSize = q_.size(2);
  int64_t headSize = q_.size(3);
  int64_t maxSeqLength = k_.size(2);
  auto out = at::empty({batchSize, num_head, qSize, headSize}, query.options());
  auto attn = at::empty({batchSize, num_head, qSize, maxSeqLength}, query.options());
  auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
  @autoreleasepool {
    auto mkey = __func__ + getTensorsStringKey({q_, k_, v_}) + ":" + std::to_string(is_causal) + ":" +
        std::to_string(attn_mask.has_value());
    auto cachedGraph =
        LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&, q_ = q_, k_ = k_, v_ = v_](auto mpsGraph, auto graph) {
          auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, q_);
          auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, k_);
          auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, v_);
          auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
          auto scaleTensor = [mpsGraph constantWithScalar:scale_factor
                                                    shape:getMPSShape({1})
                                                 dataType:MPSDataTypeFloat32];

          auto maskedMM = [mpsGraph matrixMultiplicationWithPrimaryTensor:qTensor secondaryTensor:kT name:nil];

          if (macOS15_0_plus && [maskedMM dataType] == MPSDataTypeFloat32) {
            // bug in MacOS15, without this trick SDPA leaks memory, adding 0.0f gets ignored(still takes SDPA sequence
            // path which leaks)
            auto oneTensor = [mpsGraph constantWithScalar:1e-20f shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];
            maskedMM = [mpsGraph additionWithPrimaryTensor:maskedMM secondaryTensor:oneTensor name:nil];
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
          } else if (mask_) {
            graph->maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, *mask_);
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
    auto qPlaceholder = Placeholder(cachedGraph->qTensor, q_);
    auto kPlaceholder = Placeholder(cachedGraph->kTensor, k_);
    auto vPlaceholder = Placeholder(cachedGraph->vTensor, v_);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out);
    auto attnPlaceholder = Placeholder(cachedGraph->attnTensor, attn);
    NSDictionary* feeds = nil;
    if (!mask_) {
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
    } else {
      auto mPlaceholder = Placeholder(cachedGraph->maskTensor, *mask_);
      feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder, mPlaceholder);
    }
    NSDictionary* outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
  }

  // reshape back to original dimension
  auto final_out = sq ? out.view_as(query) : out;
  auto final_attn = sq ? (query.dim() == 3 ? attn.squeeze(0) : [&]{
    std::vector<int64_t> shape(query.sizes().begin(), query.sizes().end() - 3);
    shape.insert(shape.end(), {attn.size(1), attn.size(2), attn.size(3)});
    return attn.view(shape);
  }()) : attn;

  return {std::move(final_out), std::move(final_attn)};
}

} // namespace native
} // namespace at
