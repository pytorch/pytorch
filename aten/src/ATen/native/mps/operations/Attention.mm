#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <fmt/format.h>
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
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Attention_metallib.h>
#endif

static constexpr int SIMD_SIZE = 32;

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

// general version
static std::tuple<Tensor, Tensor> sdpa_general_mps(const Tensor& query,
                                                   const Tensor& key,
                                                   const Tensor& value,
                                                   const std::optional<Tensor>& attn_mask,
                                                   double dropout_p,
                                                   bool is_causal,
                                                   const std::optional<Tensor>& dropout_mask,
                                                   std::optional<double> scale,
                                                   const Tensor& orig_query,
                                                   bool unsqueezed) {
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
  const auto macOS15_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
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
    auto cachedGraph =
        LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&, q_ = query, k_ = key, v_ = value](auto mpsGraph, auto graph) {
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
          } else if (attn_mask) {
            graph->maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, *attn_mask);
            maskedMM = [mpsGraph additionWithPrimaryTensor:maskedMM secondaryTensor:graph->maskTensor name:nil];
          }

          // Account for case where all values were masked causing division by 0 in softmax (issue:#156707)
          // Overwrites expected NANs in sm with zeros.
          auto negInfTensor = [mpsGraph constantWithScalar:-INFINITY shape:maskedMM.shape dataType:maskedMM.dataType];
          auto elem_neg_inf = [mpsGraph equalWithPrimaryTensor:maskedMM secondaryTensor:negInfTensor name:nil];
          auto all_neg_infs_along_axis = [mpsGraph reductionAndWithTensor:elem_neg_inf axis:3 name:nil];
          auto zero_mask = [mpsGraph broadcastTensor:all_neg_infs_along_axis toShape:maskedMM.shape name:nil];
          auto zeroTensor = [mpsGraph constantWithScalar:0.0 shape:maskedMM.shape dataType:maskedMM.dataType];

          auto sm = [mpsGraph softMaxWithTensor:maskedMM axis:3 name:nil];
          MPSGraphTensor* correctedSM = [mpsGraph selectWithPredicateTensor:zero_mask
                                                        truePredicateTensor:zeroTensor
                                                       falsePredicateTensor:sm
                                                                       name:nil];

          auto output = [mpsGraph matrixMultiplicationWithPrimaryTensor:correctedSM secondaryTensor:vTensor name:nil];
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

  auto final_out = unsqueezed ? out.view_as(orig_query) : out;
  auto final_attn = unsqueezed ? (orig_query.dim() == 3 ? attn.squeeze(0) : [&]{
    std::vector<int64_t> shape(orig_query.sizes().begin(), orig_query.sizes().end() - 3);
    shape.insert(shape.end(), {attn.size(1), attn.size(2), attn.size(3)});
    return attn.view(shape);
  }()) : attn;

  return {std::move(final_out), std::move(final_attn)};
}

// Vector mode (One–pass variant)
static std::tuple<Tensor, Tensor> sdpa_vector_fast_mps(const Tensor& q_,
                                                       const Tensor& k_,
                                                       const Tensor& v_,
                                                       const std::optional<Tensor>& mask_,
                                                       double dropout_p,
                                                       bool is_causal,
                                                       const std::optional<Tensor>& dropout_mask,
                                                       std::optional<double> scale,
                                                       const Tensor& orig_query,
                                                       bool unsqueezed) {
  const auto macOS15_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  using namespace mps;
  uint batchSize = q_.size(0);
  uint num_head = q_.size(1);
  uint qSize = q_.size(2);
  uint headSize = q_.size(3);
  uint maxSeqLength = k_.size(2);
  uint N = k_.size(2);
  uint B = q_.size(0) * q_.size(1);
  uint q_head_stride = q_.stride(1);
  uint q_seq_stride = q_.stride(2);
  uint k_head_stride = k_.stride(1);
  uint k_seq_stride = k_.stride(2);
  uint v_head_stride = v_.stride(1);
  uint v_seq_stride = v_.stride(2);

  auto out = at::empty({batchSize, num_head, qSize, headSize}, q_.options());
  auto attn = at::empty({batchSize, num_head, qSize, maxSeqLength}, q_.options());
  auto scale_factor = sdp::calculate_scale(q_, scale).expect_float();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto group_dims = MTLSizeMake(1024, 1, 1);
      auto grid_dims = MTLSizeMake(batchSize * num_head, q_.size(2), 1);
      bool has_mask = mask_.has_value();

      const std::string kname =
          fmt::format("sdpa_vector_{}_{}_{}", scalarToMetalTypeString(q_), q_.size(-1), v_.size(-1));
      auto attentionPSO = lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:attentionPSO];
      mtl_setArgs(computeEncoder,
                  q_,
                  k_,
                  v_,
                  out,
                  1,
                  N,
                  std::array<uint32_t, 3>{q_head_stride, k_head_stride, v_head_stride},
                  std::array<uint32_t, 3>{q_seq_stride, k_seq_stride, v_seq_stride},
                  scale_factor);

      if (has_mask) {
        int nd = mask_.value().dim();
        uint kv_seq_stride = (nd >= 1 && mask_.value().size(nd - 1) > 1) ? mask_.value().stride(nd - 1) : 0;
        uint q_seq_stride = (nd >= 2 && mask_.value().size(nd - 2) > 1) ? mask_.value().stride(nd - 2) : 0;
        uint head_stride = (nd >= 3 && mask_.value().size(nd - 3) > 1) ? mask_.value().stride(nd - 3) : 0;
        mtl_setArgs<9>(
            computeEncoder, mask_.value(), std::array<uint32_t, 3>{kv_seq_stride, q_seq_stride, head_stride});
      }
      mtl_setArgs<11>(computeEncoder, has_mask);
      [computeEncoder dispatchThreadgroups:grid_dims threadsPerThreadgroup:group_dims];
    }
  });
  // reshape back to original dimension
  auto final_out = unsqueezed ? out.view_as(orig_query) : out;
  auto final_attn = unsqueezed ? (orig_query.dim() == 3 ? attn.squeeze(0) : [&]{
    std::vector<int64_t> shape(orig_query.sizes().begin(), orig_query.sizes().end() - 3);
    shape.insert(shape.end(), {attn.size(1), attn.size(2), attn.size(3)});
    return attn.view(shape);
  }()) : attn;

  return {std::move(final_out), std::move(final_attn)};
}

// Vector mode (Two–pass variant)
static std::tuple<Tensor, Tensor> sdpa_vector_2pass_mps(const Tensor& q_,
                                                        const Tensor& k_,
                                                        const Tensor& v_,
                                                        const std::optional<Tensor>& mask_,
                                                        double dropout_p,
                                                        bool is_causal,
                                                        const std::optional<Tensor>& dropout_mask,
                                                        std::optional<double> scale,
                                                        const Tensor& orig_query,
                                                        bool unsqueezed) {
  using namespace mps;
  uint batchSize = q_.size(0);
  uint num_heads = q_.size(1);
  uint seq_len_q = q_.size(2);
  uint headSize = q_.size(3);
  uint N = k_.size(2);
  const uint blocks = 32;
  uint B = batchSize * num_heads;
  uint gqa_factor = q_.size(1) / k_.size(1);

  uint q_head_stride = q_.stride(1);
  uint q_seq_stride = q_.stride(2);
  uint k_head_stride = k_.stride(1);
  uint k_seq_stride = k_.stride(2);
  uint v_head_stride = v_.stride(1);
  uint v_seq_stride = v_.stride(2);

  auto out = at::empty({batchSize, num_heads, seq_len_q, headSize}, q_.options());
  auto intermediate = at::empty({batchSize, num_heads, seq_len_q, blocks, headSize}, q_.options());
  auto sums = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options());
  auto maxs = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options());

  auto scale_factor = sdp::calculate_scale(orig_query, scale).expect_float();
  bool has_mask = mask_.has_value();

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      const std::string kname_pass1 =
          fmt::format("sdpa_vector_2pass_1_{}_{}_{}", scalarToMetalTypeString(q_), q_.size(-1), v_.size(-1));
      const std::string kname_pass2 =
          fmt::format("sdpa_vector_2pass_2_{}_{}", scalarToMetalTypeString(q_), v_.size(-1));
      auto sdpa_vector_pass1PSO = lib.getPipelineStateForFunc(kname_pass1);
      auto sdpa_vector_pass2PSO = lib.getPipelineStateForFunc(kname_pass2);
      MTLSize group_dims = MTLSizeMake(8 * SIMD_SIZE, 1, 1);
      MTLSize grid_dims = MTLSizeMake(B, seq_len_q, blocks);
      auto computeEncoder = mpsStream->commandEncoder();

      [computeEncoder setComputePipelineState:sdpa_vector_pass1PSO];
      mtl_setArgs(computeEncoder,
                  q_,
                  k_,
                  v_,
                  intermediate,
                  sums,
                  maxs,
                  gqa_factor,
                  N,
                  std::array<uint32_t, 3>{q_head_stride, k_head_stride, v_head_stride},
                  std::array<uint32_t, 3>{q_seq_stride, k_seq_stride, v_seq_stride},
                  scale_factor);

      if (has_mask) {
        Tensor mask = mask_.value();
        int nd = mask.dim();
        uint kv_seq_stride = (nd >= 1 && mask.size(nd - 1) > 1) ? mask.stride(nd - 1) : 0;
        uint q_seq_stride = (nd >= 2 && mask.size(nd - 2) > 1) ? mask.stride(nd - 2) : 0;
        uint head_stride = (nd >= 3 && mask.size(nd - 3) > 1) ? mask.stride(nd - 3) : 0;
        mtl_setArgs<11>(computeEncoder, mask, std::array<uint32_t, 3>{kv_seq_stride, q_seq_stride, head_stride});
      }
      mtl_setArgs<13>(computeEncoder, has_mask);
      [computeEncoder dispatchThreadgroups:grid_dims threadsPerThreadgroup:group_dims];
      // 2nd pass
      [computeEncoder setComputePipelineState:sdpa_vector_pass2PSO];
      mtl_setArgs(computeEncoder, intermediate, sums, maxs, out);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(B, seq_len_q, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
  });

  auto final_out = unsqueezed ? out.view_as(orig_query) : out;
  return {std::move(final_out), std::move(intermediate)};
}

// Implementation 3: Full attention mode
static std::tuple<Tensor, Tensor> sdpa_full_attention_mps(const Tensor& q_,
                                                          const Tensor& k_,
                                                          const Tensor& v_,
                                                          const std::optional<Tensor>& mask_,
                                                          double dropout_p,
                                                          bool is_causal,
                                                          const std::optional<Tensor>& dropout_mask,
                                                          std::optional<double> scale,
                                                          const Tensor& orig_query,
                                                          bool unsqueezed) {
  using namespace mps;

  int64_t batchSize = q_.size(0);
  int64_t num_heads = q_.size(1);
  int64_t qL = q_.size(2);
  int64_t kL = k_.size(2);
  int64_t headSize = q_.size(3);

  auto q_batch_stride = q_.stride(0);
  auto q_head_stride = q_.stride(1);
  auto q_seq_stride = q_.stride(2);

  auto k_batch_stride = k_.stride(0);
  auto k_head_stride = k_.stride(1);
  auto k_seq_stride = k_.stride(2);

  auto v_batch_stride = v_.stride(0);
  auto v_head_stride = v_.stride(1);
  auto v_seq_stride = v_.stride(2);

  int mask_batch_stride = 0;
  int mask_head_stride = 0;
  int mask_q_seq_stride = 0;
  int mask_kv_seq_stride = 0;
  Tensor mask_tensor;
  bool has_mask = mask_.has_value();
  if (has_mask) {
    mask_tensor = mask_.value();
    mask_batch_stride = mask_tensor.stride(0);
    mask_head_stride = mask_tensor.stride(1);
    mask_q_seq_stride = mask_tensor.stride(2);
    mask_kv_seq_stride = mask_tensor.stride(3);
  }

  float scale_factor = sdp::calculate_scale(orig_query, scale).expect_float();
  auto out = at::empty_like(q_);

  constexpr uint wm = 4;
  constexpr uint wn = 1;
  constexpr uint bq = 32;
  auto bd = headSize;
  auto bk = (bd < 128 ? 32 : 16);
  auto gqa_factor = static_cast<int>(q_.size(1) / k_.size(1));

  const auto NQ = (qL + bq - 1) / bq;
  const auto NK = (kL + bk - 1) / bk;

  std::string kname =
      fmt::format("attention_{}_bq{}_bk{}_bd{}_wm{}_wn{}", scalarToMetalTypeString(q_), bq, bk, bd, wm, wn);

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^{
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto attentionPSO = lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:attentionPSO];
      mtl_setArgs(computeEncoder,
                  q_,
                  k_,
                  v_,
                  out,
                  static_cast<int>(qL),
                  static_cast<int>(kL),
                  gqa_factor,
                  scale_factor,
                  NK,
                  std::array<uint32_t, 3>{static_cast<uint32_t>(q_batch_stride),
                                          static_cast<uint32_t>(q_head_stride),
                                          static_cast<uint32_t>(q_seq_stride)},
                  std::array<uint32_t, 3>{static_cast<uint32_t>(k_batch_stride),
                                          static_cast<uint32_t>(k_head_stride),
                                          static_cast<uint32_t>(k_seq_stride)},
                  std::array<uint32_t, 3>{static_cast<uint32_t>(v_batch_stride),
                                          static_cast<uint32_t>(v_head_stride),
                                          static_cast<uint32_t>(v_seq_stride)},
                  std::array<uint32_t, 3>{static_cast<uint32_t>(out.stride(0)),
                                          static_cast<uint32_t>(out.stride(1)),
                                          static_cast<uint32_t>(out.stride(2))});

      MTLSize group_dims = MTLSizeMake(NQ, num_heads, batchSize);
      MTLSize threadsPerGroup = MTLSizeMake(SIMD_SIZE, wm, wn);
      [computeEncoder dispatchThreadgroups:group_dims threadsPerThreadgroup:threadsPerGroup];
    }
  });

  auto final_out = unsqueezed ? out.view_as(orig_query) : out;
  return {std::move(final_out), std::move(final_out)};
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  const std::optional<Tensor>& attn_mask,
                                                                  double dropout_p,
                                                                  bool is_causal,
                                                                  const std::optional<Tensor>& dropout_mask,
                                                                  std::optional<double> scale) {
  auto query_tuple = ensure_4d(query);
  Tensor q_ = std::get<0>(query_tuple);
  bool unsqueezed = std::get<1>(query_tuple);

  auto key_tuple = ensure_4d(key);
  Tensor k_ = std::get<0>(key_tuple);

  auto value_tuple = ensure_4d(value);
  Tensor v_ = std::get<0>(value_tuple);

  std::optional<Tensor> mask_;
  if (attn_mask) {
    auto maskExpandedDims = query.sizes().vec();
    maskExpandedDims[maskExpandedDims.size() - 1] = k_.size(2);
    mask_ = attn_mask->expand(maskExpandedDims);
    std::tie(*mask_, std::ignore) = ensure_4d(*mask_);
  }

  int query_head_dim = q_.size(3);
  int value_head_dim = v_.size(3);

  // For a vector fast implementation support {64, 96, 128} and for full support {64, 80, 128} head_dims
  bool sdpa_vector_supported_head_dim =
      (query_head_dim == value_head_dim) && (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128);

  int query_seq_len = q_.size(2);
  // Fast vector attention: when the sequence length is very short,
  // the key sequence length is large,
  // the mask is boolean and head dims are supported
  bool supports_sdpa_vector = (query_seq_len <= 8) && (query_seq_len <= k_.size(2)) &&
      ((!mask_.has_value()) || (mask_.value().dtype() == at::kBool)) && sdpa_vector_supported_head_dim;

  // boolean to decide if we can use kernel paths
  bool supports_fast_sdpa = !is_causal && supports_sdpa_vector;

  // if none of the fast paths apply, fall back to the generic mps graph solution
  if (!supports_fast_sdpa) {
    return sdpa_general_mps(q_, k_, v_, mask_, dropout_p, is_causal, dropout_mask, scale, query, unsqueezed);
  }

  // dispatch to the fast SDPA implementation
  auto is_contiguous_or_head_seq_transposed = [](const Tensor& t) -> bool {
    if (t.is_contiguous())
      return true;
    auto sizes = t.sizes();
    auto strides = t.strides();
    return (strides[3] == 1) && (strides[2] == sizes[3] * sizes[1]) && (strides[1] == sizes[3]) &&
        (strides[0] == strides[2] * sizes[2]);
  };

  Tensor q_contig = is_contiguous_or_head_seq_transposed(q_) ? q_ : q_.contiguous();
  Tensor k_contig = k_.contiguous();
  Tensor v_contig = v_.contiguous();

  // for short sequences, differentiate based on key sequence length
  if ((k_.size(2) >= 1024) || (k_.size(1) < q_.size(1) && k_.size(2) >= 4096)) {
    return sdpa_vector_2pass_mps(
        q_contig, k_contig, v_contig, mask_, dropout_p, is_causal, dropout_mask, scale, query, unsqueezed);
  } else {
    return sdpa_vector_fast_mps(
        q_contig, k_contig, v_contig, mask_, dropout_p, is_causal, dropout_mask, scale, query, unsqueezed);
  }
}
} // namespace native
} // namespace at
