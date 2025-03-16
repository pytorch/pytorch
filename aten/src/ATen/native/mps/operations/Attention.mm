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

// Ensure tensor is 4D
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
  TORCH_CHECK(!is_causal || !attn_mask.has_value(),
              "_scaled_dot_product_attention: attn_mask should not be set when is_causal=True");

  TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_for_mps: dropout_p != 0.0 is not supported");
  TORCH_CHECK(macOS15_0_plus || (query.is_contiguous() && key.is_contiguous() && value.is_contiguous()),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must be contiguous");
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
              "_scaled_dot_product_attention_math_for_mps: query, key, and value must not be nested");

  // Ensure 4D tensors
  auto [q__, sq] = ensure_4d(query);
  auto [k__, sk] = ensure_4d(key);
  auto [v__, sv] = ensure_4d(value);

  std::optional<Tensor> mask_;
  if (attn_mask) {
    auto maskExpandedDims = query.sizes().vec();
    maskExpandedDims[maskExpandedDims.size() - 1] = k__.size(2);
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

  int64_t batchSize = q__.size(0);
  int64_t num_head = q__.size(1);
  int64_t qSize = q__.size(2);
  int64_t headSize = q__.size(3);
  int64_t maxSeqLength = k__.size(2);
  auto out = at::empty({batchSize, num_head, qSize, headSize}, query.options());
  auto attn = at::empty({batchSize, num_head, qSize, maxSeqLength}, query.options());
  auto scale_factor = sdp::calculate_scale(query, scale).expect_float();

  // 👉 **Chunking logic (moved outside graph creation)**
  const int64_t chunk_size = 4096; 
  for (int64_t i = 0; i < qSize; i += chunk_size) {
      int64_t end = std::min(i + chunk_size, qSize);
      auto q_chunk = q__.narrow(2, i, end - i);
      auto k_chunk = k__.narrow(2, i, end - i);
      auto v_chunk = v__.narrow(2, i, end - i);

      @autoreleasepool {
        auto mkey = __func__ + getTensorsStringKey({q_chunk, k_chunk, v_chunk}) + ":" + std::to_string(is_causal) + ":" +
            std::to_string(attn_mask.has_value());

        auto cachedGraph =
            LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&, q_chunk, k_chunk, v_chunk](auto mpsGraph, auto graph) {
              auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, q_chunk);
              auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, k_chunk);
              auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, v_chunk);
              auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
              auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];

              auto maskedMM = [mpsGraph matrixMultiplicationWithPrimaryTensor:qTensor secondaryTensor:kT name:nil];
              auto sm = [mpsGraph softMaxWithTensor:maskedMM axis:3 name:nil];
              auto output = [mpsGraph matrixMultiplicationWithPrimaryTensor:sm secondaryTensor:vTensor name:nil];

              graph->qTensor = qTensor;
              graph->kTensor = kTensor;
              graph->vTensor = vTensor;
              graph->outputTensor = output;
              graph->attnTensor = sm;
            });

        auto qPlaceholder = Placeholder(cachedGraph->qTensor, q_chunk);
        auto kPlaceholder = Placeholder(cachedGraph->kTensor, k_chunk);
        auto vPlaceholder = Placeholder(cachedGraph->vTensor, v_chunk);
        auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out.narrow(2, i, end - i));
        auto attnPlaceholder = Placeholder(cachedGraph->attnTensor, attn.narrow(2, i, end - i));

        NSDictionary* feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
        NSDictionary* outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
        runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
      }
  }

  return {std::move(out), std::move(attn)};
}

} // namespace native
} // namespace at
