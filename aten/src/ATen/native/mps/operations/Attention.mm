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

// Helper function to determine chunk size dynamically
static int64_t get_chunk_size(int64_t seq_len) {
    if (seq_len > 26000) {
        std::cout << "[DEBUG] Using chunk size: 4096 for seq_len=" << seq_len << std::endl;
        return 4096;
    }
    if (seq_len > 13000) {
        std::cout << "[DEBUG] Using chunk size: 2048 for seq_len=" << seq_len << std::endl;
        return 2048;
    }
    std::cout << "[DEBUG] Using full seq_len=" << seq_len << std::endl;
    return seq_len;
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

    // Ensure dropout is disabled
    TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_mps: dropout_p != 0.0 is not supported");

    // Ensure tensors are contiguous
    TORCH_CHECK(macOS15_0_plus || (query.is_contiguous() && key.is_contiguous() && value.is_contiguous()),
                "_scaled_dot_product_attention_math_mps: query, key, and value must be contiguous");

    // Extract dimensions
    int64_t batchSize = query.size(0);
    int64_t num_heads = query.size(1);
    int64_t seq_len = query.size(2);
    int64_t head_dim = query.size(3);
    int64_t maxSeqLength = key.size(2);

    // Determine optimal chunk size
    int64_t chunk_size = get_chunk_size(seq_len);
    std::cout << "[DEBUG] Using chunk size: " << chunk_size << " for seq_len=" << seq_len << std::endl;

    // Create output tensors
    auto out = at::empty({batchSize, num_heads, seq_len, head_dim}, query.options());
    auto attn = at::empty({batchSize, num_heads, seq_len, maxSeqLength}, query.options());

    // Loop through chunks
    for (int64_t i = 0; i < seq_len; i += chunk_size) {
        int64_t end = std::min(i + chunk_size, seq_len);

        // Allocate memory for each chunk
        auto out_chunk = out.narrow(2, i, end - i);
        auto attn_chunk = attn.narrow(2, i, end - i);

        // Debug logging
        std::cout << "[DEBUG] Processing chunk: " << i << " to " << end << std::endl;
        std::cout << "[DEBUG] Q shape: " << query.sizes() << " | Q_chunk shape: " << query.narrow(2, i, end - i).sizes() << std::endl;
        std::cout << "[DEBUG] K shape: " << key.sizes() << " | K_chunk shape: " << key.sizes() << std::endl;
        std::cout << "[DEBUG] V shape: " << value.sizes() << " | V_chunk shape: " << value.sizes() << std::endl;

        auto q_chunk = query.narrow(2, i, end - i);
        auto k_chunk = key; // Don't chunk K/V
        auto v_chunk = value;

        using namespace mps;
        struct CachedGraph : public MPSCachedGraph {
            CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
            MPSGraphTensor* qTensor = nil;
            MPSGraphTensor* kTensor = nil;
            MPSGraphTensor* vTensor = nil;
            MPSGraphTensor* outputTensor = nil;
            MPSGraphTensor* attnTensor = nil;
        };

        @autoreleasepool {
            auto mkey = __func__ + std::to_string(chunk_size) + ":" + std::to_string(is_causal);
            auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&](auto mpsGraph, auto graph) {
                auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, q_chunk);
                auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, k_chunk);
                auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, v_chunk);

                auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
                auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
                auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];

                auto qk_chunk = [mpsGraph matrixMultiplicationWithPrimaryTensor:qTensor secondaryTensor:kT name:nil];
                qk_chunk = [mpsGraph multiplicationWithPrimaryTensor:qk_chunk secondaryTensor:scaleTensor name:nil];

                auto sm_chunk = [mpsGraph softMaxWithTensor:qk_chunk axis:3 name:nil];
                auto output_chunk = [mpsGraph matrixMultiplicationWithPrimaryTensor:sm_chunk secondaryTensor:vTensor name:nil];

                graph->qTensor = qTensor;
                graph->kTensor = kTensor;
                graph->vTensor = vTensor;
                graph->outputTensor = output_chunk;
                graph->attnTensor = sm_chunk;
            });

            auto qPlaceholder = Placeholder(cachedGraph->qTensor, q_chunk);
            auto kPlaceholder = Placeholder(cachedGraph->kTensor, k_chunk);
            auto vPlaceholder = Placeholder(cachedGraph->vTensor, v_chunk);
            auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out_chunk);
            auto attnPlaceholder = Placeholder(cachedGraph->attnTensor, attn_chunk);

            NSDictionary* feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
            NSDictionary* outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
            runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
        }
    }

    return {std::move(out), std::move(attn)};
}

} // namespace native
} // namespace at
