#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <fmt/format.h>
#include <iostream>
#include <optional>

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Attention.h>
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

static inline Tensor ensure_4d(const Tensor& x) {
  if (x.dim() == 3) {
    return x.unsqueeze(0);
  } else if (x.dim() == 2) {
    return x.view({1, 1, x.size(-2), x.size(-1)});
  } else if (x.dim() > 4) {
    auto batchSize = c10::multiply_integers(x.sizes().begin(), x.sizes().end() - 3);
    return x.reshape({batchSize, x.size(-3), x.size(-2), x.size(-1)});
  } else {
    return x;
  }
}

static inline std::tuple<Tensor, Tensor, Tensor, std::vector<int64_t>, std::vector<int64_t>> broadcast_sdpa_batch_dims(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  auto L = query.size(-2);
  auto S = key.size(-2);
  auto Ev = value.size(-1);

  std::vector<int64_t> query_batch(query.sizes().begin(), query.sizes().end() - 2);
  std::vector<int64_t> key_batch(key.sizes().begin(), key.sizes().end() - 2);
  std::vector<int64_t> value_batch(value.sizes().begin(), value.sizes().end() - 2);

  auto batch_sizes = at::infer_size(at::infer_size(query_batch, key_batch), value_batch);

  std::vector<int64_t> query_sizes(batch_sizes);
  std::vector<int64_t> key_sizes(batch_sizes);
  std::vector<int64_t> value_sizes(batch_sizes);
  std::vector<int64_t> attn_sizes(batch_sizes);
  std::vector<int64_t> out_sizes(batch_sizes);

  query_sizes.insert(query_sizes.end(), {query.size(-2), query.size(-1)});
  key_sizes.insert(key_sizes.end(), {key.size(-2), key.size(-1)});
  value_sizes.insert(value_sizes.end(), {value.size(-2), value.size(-1)});
  attn_sizes.insert(attn_sizes.end(), {L, S});
  out_sizes.insert(out_sizes.end(), {L, Ev});

  auto q_ = ensure_4d(query.expand(query_sizes));
  auto k_ = ensure_4d(key.expand(key_sizes));
  auto v_ = ensure_4d(value.expand(value_sizes));

  return {std::move(q_), std::move(k_), std::move(v_), std::move(attn_sizes), std::move(out_sizes)};
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
                                                       IntArrayRef out_sizes,
                                                       IntArrayRef attn_sizes) {
  TORCH_CHECK(q_.size(3) == k_.size(3) && q_.size(3) == v_.size(3),
              "sdpa_vector_fast_mps expects query, key, and value to have the same head dimension");
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
  auto final_out = out.view(out_sizes);
  auto final_attn = attn.view(attn_sizes);
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
                                                        IntArrayRef out_sizes) {
  TORCH_CHECK(q_.size(3) == k_.size(3) && q_.size(3) == v_.size(3),
              "sdpa_vector_2pass_mps expects query, key, and value to have the same head dimension");
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
  auto sums = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options().dtype(kFloat));
  auto maxs = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options().dtype(kFloat));

  auto scale_factor = sdp::calculate_scale(q_, scale).expect_float();
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

  auto final_out = out.view(out_sizes);
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

static std::tuple<Tensor, Tensor> sdpa(const Tensor& q,
                                       const Tensor& k,
                                       const Tensor& v,
                                       const std::optional<Tensor>& attn_mask,
                                       bool is_causal,
                                       std::optional<double> scale,
                                       IntArrayRef out_sizes,
                                       IntArrayRef attn_sizes) {
  using namespace mps;

  TORCH_INTERNAL_ASSERT(!(is_causal && attn_mask.has_value()));

  auto batch_size = q.size(0);
  auto num_heads = q.size(1);
  auto L = q.size(2);
  auto E = q.size(3);
  auto S = v.size(2);
  auto Ev = v.size(3);

  auto out = at::empty({batch_size, num_heads, L, Ev}, q.options());
  auto attn = at::empty({batch_size, num_heads, L, S}, q.options());
  auto scale_factor = sdp::calculate_scale(q, scale).expect_float();

  SDPAParams params;

  params.batch_size = batch_size;
  params.num_heads = num_heads;
  params.L = L;
  params.E = E;
  params.S = S;
  params.Ev = Ev;
  params.scale = scale_factor;

  for (const auto dim : c10::irange(4)) {
    params.q_strides[dim] = q.stride(dim);
    params.k_strides[dim] = k.stride(dim);
    params.v_strides[dim] = v.stride(dim);
    params.mask_strides[dim] = attn_mask.has_value() ? attn_mask->stride(dim) : 0;
    params.out_strides[dim] = out.stride(dim);
    params.attn_strides[dim] = attn.stride(dim);
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(
          fmt::format("sdpa_{}_{}{}",
                      scalarToMetalTypeString(q),
                      // `void` type mask indicates not to apply the mask
                      attn_mask.has_value() ? scalarToMetalTypeString(attn_mask.value()) : "void",
                      is_causal ? "_causal" : ""));

      getMPSProfiler().beginProfileKernel(pso, "scaled_dot_product_attention", {q, k, v});
      [compute_encoder setComputePipelineState:pso];
      mtl_setArgs(compute_encoder, out, attn, q, k, v, attn_mask, params);

      // Execute one threadgroup per (batch, head) pair, and TILE_SIZE×TILE_SIZE
      // threads within each group
      [compute_encoder dispatchThreadgroups:MTLSizeMake(batch_size * num_heads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(TILE_SIZE, TILE_SIZE, 1)];

      getMPSProfiler().endProfileKernel(pso);
    }
  });

  auto final_out = out.view(out_sizes);
  auto final_attn = attn.view(attn_sizes);
  return {std::move(final_out), std::move(final_attn)};
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                  const Tensor& key_,
                                                                  const Tensor& value_,
                                                                  const std::optional<Tensor>& attn_mask,
                                                                  double dropout_p,
                                                                  bool is_causal,
                                                                  const std::optional<Tensor>& dropout_mask,
                                                                  std::optional<double> scale,
                                                                  bool enable_gqa) {
  TORCH_CHECK_NOT_IMPLEMENTED(c10::isFloatingType(query.scalar_type()),
                              "scaled_dot_product_attention for MPS does not support dtype ",
                              query.scalar_type());
  TORCH_CHECK_NOT_IMPLEMENTED(c10::isFloatingType(key_.scalar_type()),
                              "scaled_dot_product_attention for MPS does not support dtype ",
                              key_.scalar_type());
  TORCH_CHECK_NOT_IMPLEMENTED(c10::isFloatingType(value_.scalar_type()),
                              "scaled_dot_product_attention for MPS does not support dtype ",
                              value_.scalar_type());
  const auto any_nested = query.is_nested() || key_.is_nested() || value_.is_nested();
  const auto all_contiguous =
      query.is_contiguous_or_false() && key_.is_contiguous_or_false() && value_.is_contiguous_or_false();
  auto key = key_;
  auto value = value_;
  if (enable_gqa) {
    int64_t q_heads = query.size(-3);
    int64_t k_heads = key_.size(-3);
    int64_t repeat_factor = q_heads / k_heads;

    if (repeat_factor > 1) {
      TORCH_CHECK(q_heads % k_heads == 0,
                  "For GQA, the query tensor's head dimension (" + std::to_string(q_heads) +
                      ") must be divisible by the key tensor's head dimension (" + std::to_string(k_heads) + ").");
      key = key_.repeat_interleave(repeat_factor, /*dim=*/-3);
      value = value_.repeat_interleave(repeat_factor, /*dim=*/-3);
    }
  }

  auto [q_, k_, v_, attn_sizes, out_sizes] = broadcast_sdpa_batch_dims(query, key, value);

  std::optional<Tensor> mask_;
  if (attn_mask) {
    auto maskExpandedDims = query.sizes().vec();
    maskExpandedDims[maskExpandedDims.size() - 1] = k_.size(2);
    mask_ = attn_mask->expand(maskExpandedDims);
    *mask_ = ensure_4d(*mask_);
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
    return sdpa(q_, k_, v_, mask_, is_causal, scale, out_sizes, attn_sizes);
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
        q_contig, k_contig, v_contig, mask_, dropout_p, is_causal, dropout_mask, scale, out_sizes);
  } else {
    return sdpa_vector_fast_mps(
        q_contig, k_contig, v_contig, mask_, dropout_p, is_causal, dropout_mask, scale, out_sizes, attn_sizes);
  }
}
} // namespace native
} // namespace at
