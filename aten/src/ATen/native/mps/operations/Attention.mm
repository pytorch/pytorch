#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <fmt/format.h>
#include <optional>

#include <ATen/ExpandUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Attention.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/safe_conv.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Attention_metallib.h>
#endif

using namespace std::literals;

// Decode tile (padded head dim), or 0 when the head dims do not fit the decode
// kernels: a lane's window of tile / 32 columns must fit inside both dims.
static int64_t decode_bucket(int64_t d, int64_t dv) {
  const auto it = std::lower_bound(kDecodeTiles.begin(), kDecodeTiles.end(), std::max(d, dv));
  return it != kDecodeTiles.end() && std::min(d, dv) >= *it / c10::metal::simdgroup_size ? *it : 0;
}

static std::string_view padded_suffix(const Tensor& q, const Tensor& v, int64_t bd) {
  return q.size(3) == bd && v.size(3) == bd ? ""sv : "_padded"sv;
}

static std::string sdpa_vector_mask_suffix(const std::optional<Tensor>& mask) {
  return mask.has_value() ? mps::scalarToMetalTypeString(mask.value()) : "none";
}

// Vector mode (One–pass variant)
static void sdpa_vector_fast_mps(const Tensor& q_,
                                 const Tensor& k_,
                                 const Tensor& v_,
                                 const Tensor& out,
                                 const std::optional<Tensor>& mask_,
                                 bool is_causal,
                                 float scale_factor,
                                 int64_t bd) {
  using namespace mps;
  uint batchSize = q_.size(0);
  uint num_head = q_.size(1);
  uint headSize = q_.size(3);
  uint N = c10::safe_conv<int32_t>(k_.size(2));
  uint gqa_factor = q_.size(1) / k_.size(1);
  uint q_batch_stride = q_.stride(0);
  uint q_head_stride = q_.stride(1);
  uint q_seq_stride = q_.stride(2);
  uint k_batch_stride = k_.stride(0);
  uint k_head_stride = k_.stride(1);
  uint k_seq_stride = k_.stride(2);
  uint v_batch_stride = v_.stride(0);
  uint v_head_stride = v_.stride(1);
  uint v_seq_stride = v_.stride(2);

  const std::array<uint32_t, 2> head_dims{headSize, c10::safe_conv<uint32_t>(v_.size(3))};
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto group_dims = MTLSizeMake(kDecodeSimdgroups * c10::metal::simdgroup_size, 1, 1);
      auto grid_dims = MTLSizeMake(batchSize * num_head, q_.size(2), 1);
      const bool has_mask = mask_.has_value();

      const std::string kname = fmt::format("sdpa_vector{}_{}_{}_{}_mask{}{}",
                                            padded_suffix(q_, v_, bd),
                                            scalarToMetalTypeString(q_),
                                            bd,
                                            bd,
                                            sdpa_vector_mask_suffix(mask_),
                                            is_causal ? "_causal" : "");
      auto attentionPSO = lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:attentionPSO];
      mtl_setArgs(computeEncoder,
                  q_,
                  k_,
                  v_,
                  out,
                  gqa_factor,
                  N,
                  c10::metal::vec3<uint32_t>{q_head_stride, k_head_stride, v_head_stride},
                  c10::metal::vec3<uint32_t>{q_seq_stride, k_seq_stride, v_seq_stride},
                  scale_factor);

      if (has_mask) {
        auto mask = mask_.value();
        uint kv_seq_stride = (mask.size(3) > 1) ? mask.stride(3) : 0;
        uint q_seq_stride = (mask.size(2) > 1) ? mask.stride(2) : 0;
        uint head_stride = (mask.size(1) > 1) ? mask.stride(1) : 0;
        uint batch_stride = (mask.size(0) > 1) ? mask.stride(0) : 0;
        mtl_setArgs<9>(
            computeEncoder, mask, std::array<uint32_t, 4>{kv_seq_stride, q_seq_stride, head_stride, batch_stride});
      }
      mtl_setArgs<11>(
          computeEncoder, std::array<uint32_t, 4>{q_batch_stride, k_batch_stride, v_batch_stride, num_head}, head_dims);
      [computeEncoder dispatchThreadgroups:grid_dims threadsPerThreadgroup:group_dims];
    }
  });
}

// Vector mode (Two–pass variant)
static void sdpa_vector_2pass_mps(const Tensor& q_,
                                  const Tensor& k_,
                                  const Tensor& v_,
                                  const Tensor& out,
                                  const std::optional<Tensor>& mask_,
                                  bool is_causal,
                                  float scale_factor,
                                  int64_t bd) {
  using namespace mps;
  uint batchSize = q_.size(0);
  uint num_heads = q_.size(1);
  uint seq_len_q = q_.size(2);
  uint headSize = q_.size(3);
  uint N = c10::safe_conv<int32_t>(k_.size(2));
  const uint blocks = kDecodeBlocks;
  uint B = batchSize * num_heads;
  uint gqa_factor = q_.size(1) / k_.size(1);

  uint q_batch_stride = q_.stride(0);
  uint q_head_stride = q_.stride(1);
  uint q_seq_stride = q_.stride(2);
  uint k_batch_stride = k_.stride(0);
  uint k_head_stride = k_.stride(1);
  uint k_seq_stride = k_.stride(2);
  uint v_batch_stride = v_.stride(0);
  uint v_head_stride = v_.stride(1);
  uint v_seq_stride = v_.stride(2);

  auto intermediate = at::empty({batchSize, num_heads, seq_len_q, blocks, bd}, q_.options());
  auto sums = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options().dtype(kFloat));
  auto maxs = at::empty({batchSize, num_heads, seq_len_q, blocks}, q_.options().dtype(kFloat));

  const std::array<uint32_t, 2> head_dims{headSize, c10::safe_conv<uint32_t>(v_.size(3))};
  const bool has_mask = mask_.has_value();

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      const std::string kname_pass1 = fmt::format("sdpa_vector_2pass_1{}_{}_{}_{}_mask{}{}",
                                                  padded_suffix(q_, v_, bd),
                                                  scalarToMetalTypeString(q_),
                                                  bd,
                                                  bd,
                                                  sdpa_vector_mask_suffix(mask_),
                                                  is_causal ? "_causal" : "");
      const std::string kname_pass2 = fmt::format("sdpa_vector_2pass_2_{}_{}", scalarToMetalTypeString(q_), bd);
      auto sdpa_vector_pass1PSO = lib.getPipelineStateForFunc(kname_pass1);
      auto sdpa_vector_pass2PSO = lib.getPipelineStateForFunc(kname_pass2);
      MTLSize group_dims = MTLSizeMake(kDecode2PassSimdgroups * c10::metal::simdgroup_size, 1, 1);
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
                  c10::metal::vec3<uint32_t>{q_head_stride, k_head_stride, v_head_stride},
                  c10::metal::vec3<uint32_t>{q_seq_stride, k_seq_stride, v_seq_stride},
                  scale_factor);

      if (has_mask) {
        auto mask = mask_.value();
        uint kv_seq_stride = (mask.size(3) > 1) ? mask.stride(3) : 0;
        uint q_seq_stride = (mask.size(2) > 1) ? mask.stride(2) : 0;
        uint head_stride = (mask.size(1) > 1) ? mask.stride(1) : 0;
        uint batch_stride = (mask.size(0) > 1) ? mask.stride(0) : 0;
        mtl_setArgs<11>(
            computeEncoder, mask, std::array<uint32_t, 4>{kv_seq_stride, q_seq_stride, head_stride, batch_stride});
      }
      mtl_setArgs<13>(computeEncoder,
                      std::array<uint32_t, 4>{q_batch_stride, k_batch_stride, v_batch_stride, num_heads},
                      head_dims);
      [computeEncoder dispatchThreadgroups:grid_dims threadsPerThreadgroup:group_dims];
      // 2nd pass
      [computeEncoder setComputePipelineState:sdpa_vector_pass2PSO];
      mtl_setArgs(computeEncoder, intermediate, sums, maxs, out, head_dims[1]);
      const auto pass2_group_dims = MTLSizeMake(blocks * c10::metal::simdgroup_size, 1, 1);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(B, seq_len_q, 1) threadsPerThreadgroup:pass2_group_dims];
    }
  });
}

// The MPP kernel takes all full-width V/O slabs in one launch while the grid
// stays below kGridThreadgroupsPerCore threadgroups per GPU core; past that,
// one launch per slab keeps the slab's V hot across the wave (M4, M5 Pro).
static constexpr int64_t kGridThreadgroupsPerCore = 8;
// A QK dim in (128, kMppMaxSimdTile] with V/O past 128 keeps its exact
// simdgroup tile: MPP would chunk QK and recompute the scores per V/O slab
// (M5 Pro: 0.66x). Above it both kernels chunk and MPP wins.
static constexpr int64_t kMppMaxSimdTile = 192;

// Prefill: flash attention on simdgroup matrices (PrefillAttention.h) or, on
// macOS 26.2+, Metal 4 cooperative tensors (MppAttention.h) for half and
// bfloat. One threadgroup per (Q block, head, batch, V/O slab).
static void sdpa_prefill_mps(const Tensor& q,
                             const Tensor& k,
                             const Tensor& v,
                             const Tensor& out,
                             const std::optional<Tensor>& mask_,
                             bool is_causal,
                             float scale) {
  using namespace mps;
  // matmul2d cooperative_tensor's lane->element layout differs across Apple GPU
  // generations (see MppAttention.h). Before Apple10's matrix units, MPP only
  // beats the simdgroup kernel on an exact tile (M3 Ultra, M4).
  static const bool is_apple10_or_newer = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
  static const int64_t core_grid = kGridThreadgroupsPerCore * std::max(MPSDevice::getInstance()->getCoreCount(), 1u);
  const int64_t d = q.size(3), dv = v.size(3), widest = std::max(d, dv);
  const bool exact_mpp_tile = d == dv &&
      (std::binary_search(kMppTiles.begin(), kMppTiles.end(), d) || (!is_apple10_or_newer && d == kMppWideTile));
  const bool exact_simd_tile = widest > kMppTiles.back() && widest <= kMppMaxSimdTile && dv > kMppTiles.back();
  const bool use_mpp = at::mps::has_mpp() && q.scalar_type() != kFloat && k.size(2) > 0 &&
      (exact_mpp_tile || (is_apple10_or_newer && !exact_simd_tile));
  const c10::ArrayRef<int> tiles = use_mpp ? c10::ArrayRef<int>(kMppTiles) : c10::ArrayRef<int>(kPrefillTiles);
  const int64_t bd = use_mpp && exact_mpp_tile ? d
      : widest > tiles.back()                  ? kChunkTile
                                               : *std::lower_bound(tiles.begin(), tiles.end(), widest);
  const int64_t simdgroups = use_mpp ? kPrefillSimdgroups : prefill_simdgroups(bd);
  const int64_t bq = use_mpp ? kMppBQ : prefill_bq(bd);

  std::optional<Tensor> mask = mask_;
  PrefillAttnMaskParams mask_params{};
  if (mask) {
    // Kernels assume a unit kL stride; materialize if not.
    if (mask->stride(-1) != 1) {
      mask = mask->contiguous();
    }
    mask_params = {.M_strides = {mask->stride(0), mask->stride(1), mask->stride(2), mask->stride(3)}};
  }

  const auto dtype = scalarToMetalTypeString(q);
  const auto mask_dtype = mask ? scalarToMetalTypeString(*mask) : dtype;
  const auto kname = use_mpp ? fmt::format("prefill_attention_mpp_{}_{}_bd{}_hm{:d}_dc{:d}_ck{:d}_mask{}",
                                           is_apple10_or_newer ? "a10"sv : "pre_a10"sv,
                                           dtype,
                                           bd,
                                           mask.has_value(),
                                           is_causal,
                                           d > bd,
                                           mask_dtype)
                             : fmt::format("prefill_attention_{}_bd{}_hm{:d}_dc{:d}_ck{:d}_pd{:d}_mask{}",
                                           dtype,
                                           bd,
                                           mask.has_value(),
                                           is_causal,
                                           d > bd,
                                           d != bd || dv != bd,
                                           mask_dtype);
  const auto pso = lib.getPipelineStateForFunc(kname);
  const int64_t nq = at::ceil_div(q.size(2), bq);
  const int64_t step = use_mpp && nq * q.size(0) * q.size(1) < core_grid ? std::max(dv / bd * bd, bd) : bd;
  const MTLSize threadgroup = MTLSizeMake(c10::metal::simdgroup_size, simdgroups, 1);
  MPSStream* mpsStream = getCurrentMPSStream();
  for (int64_t v0 = 0; v0 < dv; v0 += step) {
    const auto vs = v.narrow(3, v0, std::min(step, dv - v0));
    const auto os = out.narrow(3, v0, vs.size(3));
    const int64_t slabs = at::ceil_div(vs.size(3), bd);
    TORCH_INTERNAL_ASSERT(vs.size(3) % slabs == 0);
    const MTLSize grid = MTLSizeMake(nq, q.size(1), q.size(0) * slabs);
    const PrefillAttnParams params{
        .B = c10::safe_conv<int32_t>(q.size(0)),
        .H = c10::safe_conv<int32_t>(q.size(1)),
        .D = c10::safe_conv<int32_t>(d),
        .qL = c10::safe_conv<int32_t>(q.size(2)),
        .kL = c10::safe_conv<int32_t>(k.size(2)),
        .gqa_factor = c10::safe_conv<int32_t>(q.size(1) / k.size(1)),
        .scale = scale,
        .Dv = c10::safe_conv<int32_t>(vs.size(3) / slabs),
        .Q_strides = {q.stride(0), q.stride(1), q.stride(2)},
        .K_strides = {k.stride(0), k.stride(1), k.stride(2)},
        .V_strides = {vs.stride(0), vs.stride(1), vs.stride(2)},
        .O_strides = {os.stride(0), os.stride(1), os.stride(2)},
    };
    dispatch_sync_with_rethrow(mpsStream->queue(), ^{
      @autoreleasepool {
        auto computeEncoder = mpsStream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, q, k, vs, os, params, mask_params);
        // The kernels reference the mask in every instantiation; Metal
        // validation wants it bound even when unused.
        mtl_setArgs<6>(computeEncoder, uint32_t(0));
        if (mask) {
          mtl_setArgs<6>(computeEncoder, *mask);
        }
        [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
      }
    });
  }
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
  const auto dtype = query.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(
      c10::isFloatingType(dtype), "scaled_dot_product_attention for MPS does not support dtype ", dtype);
  TORCH_CHECK_TYPE(key_.scalar_type() == dtype && value_.scalar_type() == dtype,
                   "scaled_dot_product_attention for MPS expects query, key and value to share a dtype");
  TORCH_CHECK_VALUE(
      query.size(-1) == key_.size(-1) && key_.size(-2) == value_.size(-2),
      "scaled_dot_product_attention for MPS expects key to share the head dim with query and the sequence length with value");
  TORCH_CHECK_NOT_IMPLEMENTED(dropout_p == 0.0f, "scaled_dot_product_attention for MPS does not support dropout.");
  TORCH_CHECK(!(is_causal && attn_mask.has_value()),
              "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
  // The kernels broadcast K/V across GQA groups via their gqa_factor parameter,
  // whether or not enable_gqa is set; a non-integral ratio would index past K/V.
  const auto q_heads = query.size(-3), k_heads = key_.size(-3);
  TORCH_CHECK_VALUE(k_heads == value_.size(-3),
                    "scaled_dot_product_attention for MPS expects key and value to share the number of heads");
  TORCH_CHECK_VALUE(q_heads % k_heads == 0,
                    "For GQA, the query tensor's head dimension (",
                    q_heads,
                    ") must be divisible by the key tensor's head dimension (",
                    k_heads,
                    ").");

  auto batch_dims = [](const Tensor& t) { return t.sym_sizes().slice(0, t.dim() - 3); };
  auto batch_shape = at::infer_size_symdimvector(batch_dims(query), batch_dims(key_));
  batch_shape = at::infer_size_symdimvector(batch_shape, batch_dims(value_));
  const auto batch_numel = c10::multiply_integers(batch_shape);
  // A zero QK head dim makes every score zero, so attention is uniform over the
  // unmasked keys as the CPU math path computes it; the kernels get a single
  // zero column instead (which also keeps the default scale finite).
  auto zero_column = [](const Tensor& t) {
    auto shape = t.sym_sizes().vec();
    shape.back() = 1;
    return t.new_zeros_symint(shape);
  };
  const bool zero_head_dim = query.size(-1) == 0;
  // Broadcasts the batch dims and flattens them into one. Kernels load head-dim
  // elements linearly, so stride(-1) == 1 is the only layout requirement.
  auto expand_batch = [&](const Tensor& t) {
    auto shape = batch_shape;
    shape.append(t.sym_sizes().end() - 3, t.sym_sizes().end());
    const auto t4d =
        t.expand_symint(shape).reshape_symint({batch_numel, t.sym_size(-3), t.sym_size(-2), t.sym_size(-1)});
    return t4d.stride(-1) == 1 ? t4d : t4d.contiguous();
  };
  const auto q_ = expand_batch(zero_head_dim ? zero_column(query) : query);
  const auto k_ = expand_batch(zero_head_dim ? zero_column(key_) : key_);
  const auto v_ = expand_batch(value_);

  auto out_shape = batch_shape;
  out_shape.append({query.sym_size(-3), query.sym_size(-2), value_.sym_size(-1)});
  std::optional<Tensor> mask;
  if (attn_mask) {
    // An additive mask is added in fp32 like the CPU math path, so an fp32 mask
    // stays fp32 for any query dtype (to() is then a no-op). Cast before
    // expanding so a broadcast mask stays a view.
    const auto m =
        attn_mask->scalar_type() == kBool || attn_mask->scalar_type() == dtype ? *attn_mask : attn_mask->to(kFloat);
    auto mask_shape = out_shape;
    mask_shape.back() = key_.sym_size(-2);
    mask = m.expand_symint(mask_shape)
               .reshape_symint({batch_numel, query.sym_size(-3), query.sym_size(-2), key_.sym_size(-2)});
  }

  const int64_t q_len = q_.size(2), d = q_.size(3), dv = v_.size(3);
  const float scale_factor = sdp::calculate_scale(q_, scale).expect_float();
  auto out = at::empty({q_.size(0), q_.size(1), q_len, dv}, q_.options());
  // Decode forms uint32 offsets up to the first 1024 (kTwoPassMinKeys) keys of a head and advances pointers past them,
  // and pass 1 int32 offsets into its {B, H, qL, kDecodeBlocks, bd} partials; prefill takes 64-bit offsets.
  const auto fits_u32 = [](const Tensor& t, int64_t key_dim = -1) {
    int64_t offset = 0;
    for (const auto i : c10::irange(t.dim())) {
      offset += (std::min(t.size(i), i == key_dim ? kTwoPassMinKeys : t.size(i)) - 1) * t.stride(i);
    }
    return offset <= std::numeric_limits<uint32_t>::max();
  };
  const bool u32 = fits_u32(q_) && fits_u32(k_, 2) && fits_u32(v_, 2) && fits_u32(out) && (!mask || fits_u32(*mask, 3));
  const int64_t bd = q_len <= kDecodeMaxQueries && u32 ? decode_bucket(d, dv) : 0;
  if (out.numel() > 0 && bd > 0 && k_.size(2) >= kTwoPassMinKeys &&
      out.numel() / dv * kDecodeBlocks * bd <= std::numeric_limits<int32_t>::max()) {
    sdpa_vector_2pass_mps(q_, k_, v_, out, mask, is_causal, scale_factor, bd);
  } else if (out.numel() > 0 && bd > 0) {
    sdpa_vector_fast_mps(q_, k_, v_, out, mask, is_causal, scale_factor, bd);
  } else if (out.numel() > 0) {
    sdpa_prefill_mps(q_, k_, v_, out, mask, is_causal, scale_factor);
  }
  // The attention weights are never materialized (nothing consumes them); an
  // empty tensor stands in, as the flash attention ops do for unrequested outputs.
  // TODO: implement the flash, efficient and math attention ops for MPS properly
  // instead of routing them all through this op.
  return {out.view_symint(out_shape), at::empty({0}, q_.options())};
}
} // namespace native
} // namespace at
