#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <fmt/format.h>
#include <optional>

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Attention.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/env.h>
#include <c10/util/safe_conv.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at {
namespace native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Attention_metallib.h>
#endif

using namespace std::literals;

// out[rows, Dv] = merge of partials[rows, blocks, width] with their row max / sum.
static void merge_partials(id<MTLComputeCommandEncoder> encoder,
                           const Tensor& partials,
                           const Tensor& sums,
                           const Tensor& maxs,
                           const Tensor& out) {
  using namespace mps;
  const auto dv = out.size(-1);
  const auto pso = lib.getPipelineStateForFunc("sdpa_merge_partials_" + scalarToMetalTypeString(out));
  [encoder setComputePipelineState:pso];
  const std::array<uint32_t, 4> dims{c10::safe_conv<uint32_t>(dv),
                                     c10::safe_conv<uint32_t>(partials.size(-2)),
                                     c10::safe_conv<uint32_t>(out.stride(-2)),
                                     c10::safe_conv<uint32_t>(partials.size(-1))};
  mtl_setArgs(encoder, partials, sums, maxs, out, dims);
  // Threadgroups of kMergeLanes x columns: a decode merge has few rows and
  // should still spread across the GPU cores.
  const auto cols = std::min<NSUInteger>(dv, [pso maxTotalThreadsPerThreadgroup] / kMergeLanes);
  [encoder dispatchThreads:MTLSizeMake(kMergeLanes, dv, out.numel() / dv)
      threadsPerThreadgroup:MTLSizeMake(kMergeLanes, cols, 1)];
}

// Decode tile (padded head dim) for qL == 1, or 0 when the head dims do not
// fit the decode kernel: a lane's window of tile / 32 columns must fit inside
// both dims. Keep the tiles in sync with the instantiations in DecodeAttention.h.
static int64_t decode_bucket(int64_t d, int64_t dv) {
  static constexpr std::array<int64_t, 6> tiles{32, 64, 96, 128, 256, 512};
  const auto it = std::lower_bound(tiles.begin(), tiles.end(), std::max(d, dv));
  return it != tiles.end() && std::min(d, dv) >= *it / c10::metal::simdgroup_size ? *it : 0;
}

// Two-pass decode trades a merge for a kDecodeBlocks times larger grid, so it
// only wins while one threadgroup per (batch, head) row leaves the GPU idle:
// below these row counts, by key length. Crossovers measured on an M5 Pro.
static constexpr uint32_t kTwoPassMinKeys = 1024, kTwoPassMaxRows = 16;
static constexpr uint32_t kTwoPassLongKeys = 4096, kTwoPassLongMaxRows = 96;

// Decode (qL == 1): DecodeAttention.h, one simdgroup per key.
static void sdpa_vector_mps(const Tensor& q,
                            const Tensor& k,
                            const Tensor& v,
                            const Tensor& out,
                            const std::optional<Tensor>& mask,
                            bool is_causal,
                            float scale,
                            int64_t bd) {
  using namespace mps;
  const uint32_t rows = c10::safe_conv<uint32_t>(q.size(0) * q.size(1)), n = c10::safe_conv<uint32_t>(k.size(2));
  const bool two_pass = n >= kTwoPassMinKeys && rows < (n >= kTwoPassLongKeys ? kTwoPassLongMaxRows : kTwoPassMaxRows);
  auto stride = [](const Tensor& t, int d) { return c10::safe_conv<uint32_t>(t.stride(d)); };
  auto mask_stride = [&](int d) { return mask && mask->size(d) > 1 ? stride(*mask, d) : 0u; };
  const DecodeAttnParams params{
      .N = n,
      .D = c10::safe_conv<uint32_t>(q.size(3)),
      .Dv = c10::safe_conv<uint32_t>(v.size(3)),
      .gqa_factor = c10::safe_conv<uint32_t>(q.size(1) / k.size(1)),
      .num_heads = c10::safe_conv<uint32_t>(q.size(1)),
      .scale = scale,
      .Q_strides = {stride(q, 0), stride(q, 1)},
      .K_strides = {stride(k, 0), stride(k, 1), stride(k, 2)},
      .V_strides = {stride(v, 0), stride(v, 1), stride(v, 2)},
      .M_strides = {mask_stride(0), mask_stride(1), mask_stride(3)},
  };
  Tensor partials, sums, maxs;
  if (two_pass) {
    partials = at::empty({rows, kDecodeBlocks, bd}, q.options().dtype(kFloat));
    sums = at::empty({rows, kDecodeBlocks}, q.options().dtype(kFloat));
    maxs = at::empty_like(sums);
  }
  const auto kname = fmt::format("sdpa_vector{}_{}_{}_mask{}{}",
                                 two_pass ? "_2pass"sv : ""sv,
                                 scalarToMetalTypeString(q),
                                 bd,
                                 mask ? scalarToMetalTypeString(*mask) : "none"s,
                                 is_causal ? "_causal"sv : ""sv);
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      [computeEncoder setComputePipelineState:lib.getPipelineStateForFunc(kname)];
      // Two-pass blocks write their partial rows in place of the output.
      mtl_setArgs(computeEncoder, q, k, v, two_pass ? partials : out, params);
      if (mask) {
        mtl_setArgs<5>(computeEncoder, *mask);
      }
      if (two_pass) {
        mtl_setArgs<6>(computeEncoder, sums, maxs);
      }
      const auto simdgroups = two_pass ? kDecodeBlockSimdgroups : kDecodeSimdgroups;
      [computeEncoder dispatchThreadgroups:MTLSizeMake(rows, two_pass ? kDecodeBlocks : 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(simdgroups * c10::metal::simdgroup_size, 1, 1)];
      if (two_pass) {
        merge_partials(computeEncoder, partials, sums, maxs, out);
      }
    }
  });
}

// Prefill tiles (padded head dims); keep in sync with the instantiations in
// PrefillAttention.h and MppAttention.h. Head dims above the widest tile run
// on kChunkTile: the QK dim in tile-wide chunks, V/O in tile-wide column slabs.
static constexpr std::array<int64_t, 8> kPrefillTiles{32, 64, 72, 80, 96, 128, 160, 192};
static constexpr std::array<int64_t, 3> kMppTiles{64, 96, 128};
static constexpr int64_t kChunkTile = 128;
// Split-K spreads the keys of a Q block over threadgroups until the grid
// reaches kSplitTargetThreadgroups (fills an M5 Pro), keeping at least
// kSplitMinKBlocks K blocks per split so the partial rows and merge pay off,
// and at most kMaxSplits of them.
static constexpr int64_t kSplitTargetThreadgroups = 128, kSplitMinKBlocks = 8, kMaxSplits = 32;
// The staged V/O slab passes materialize the fp32 scores S[B * H * qL, kL];
// above this many bytes each slab recomputes the scores instead.
static constexpr int64_t kStagedScoresMaxBytes = int64_t(1) << 30;
// PYTORCH_MPS_SDPA_STAGED_SCORES_MAX_BYTES overrides the cap (0 disables
// staging); read per call, on the wide V/O path only, so tests can set it.
static int64_t staged_scores_max_bytes() {
  const auto env = c10::utils::get_env("PYTORCH_MPS_SDPA_STAGED_SCORES_MAX_BYTES");
  return env ? strtoll(env->c_str(), nullptr, 10) : kStagedScoresMaxBytes;
}
// The full-width V/O slabs of the MPP and P @ V kernels run in one launch,
// and the staged scores pass splits the keys, while the grid stays below
// kGridThreadgroupsPerCore per GPU core (an M5 Pro fills at 128); past that,
// one launch per slab keeps the slab's V hot across the wave (M4, M5 Pro).
static constexpr int64_t kGridThreadgroupsPerCore = 8;

enum class Stage { fused, scores, pv };
static constexpr std::array<std::string_view, 3> kStageSuffix{""sv, "_scores"sv, "_pv"sv};

// Prefill: flash attention on simdgroup matrices (PrefillAttention.h) or, on
// macOS 26.2+, Metal 4 cooperative tensors (MppAttention.h). One threadgroup
// per (Q block, head, batch), with the keys split across threadgroups when that
// grid is too small to fill the GPU.
static void sdpa_prefill_mps(const Tensor& q,
                             const Tensor& k,
                             const Tensor& v,
                             const Tensor& out,
                             const std::optional<Tensor>& mask_,
                             bool is_causal,
                             float scale) {
  using namespace mps;
  // matmul2d cooperative_tensor's lane->element layout differs across Apple GPU
  // generations (see MppAttention.h).
  static const bool is_apple10_or_newer = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
  // MPP fp32 is only bfloat-accurate in relaxed precision, and strict precision
  // (its own fragment layout) measured 1.3x the simdgroup kernel: not worth it.
  // Before Apple10's matrix units, MPP only beats the simdgroup kernel on an
  // unpadded, unchunked tile (M3 Ultra, M4).
  const int64_t d = q.size(3), dv = v.size(3);
  const bool exact_mpp_tile = d == dv && std::find(kMppTiles.begin(), kMppTiles.end(), d) != kMppTiles.end();
  const bool use_mpp = at::mps::has_mpp() && q.scalar_type() != kFloat && (is_apple10_or_newer || exact_mpp_tile);
  const c10::IntArrayRef tiles = use_mpp ? c10::IntArrayRef(kMppTiles) : c10::IntArrayRef(kPrefillTiles);
  const int64_t bd =
      std::max(d, dv) > tiles.back() ? kChunkTile : *std::lower_bound(tiles.begin(), tiles.end(), std::max(d, dv));
  const int64_t bq = use_mpp ? kMppBQ : kPrefillBQ, bk = use_mpp ? kMppBK : prefill_bk(bd);

  std::optional<Tensor> mask = mask_;
  PrefillAttnMaskParams mask_params{};
  if (mask) {
    // Kernels assume a unit kL stride; materialize if not.
    if (mask->stride(-1) != 1) {
      mask = mask->contiguous();
    }
    mask_params = {.M_strides = {mask->stride(0), mask->stride(1), mask->stride(2), mask->stride(3)}};
  }

  const int64_t rows = q.size(0) * q.size(1) * q.size(2);
  const int64_t nq = c10::metal::ceil_div(q.size(2), bq), nk = c10::metal::ceil_div(k.size(2), bk);
  // Wide V/O on the simdgroup kernel: prefill_attention_scores writes the
  // scaled, masked scores and row stats once, then prefill_attention_pv runs
  // P @ V over the slabs.
  const bool staged = !use_mpp && dv > bd && rows * k.size(2) * int64_t(sizeof(float)) <= staged_scores_max_bytes();
  static const int64_t core_grid = kGridThreadgroupsPerCore * std::max(MPSDevice::getInstance()->getCoreCount(), 1u);
  const int64_t grid_xy = nq * q.size(0) * q.size(1);
  const int64_t max_splits = std::clamp(nk / kSplitMinKBlocks, int64_t(1), kMaxSplits);
  const int64_t n_splits = std::clamp(kSplitTargetThreadgroups / grid_xy, int64_t(1), max_splits);
  const int64_t score_grid = std::max(kSplitTargetThreadgroups, core_grid);
  const int64_t n_score_splits = std::clamp(score_grid / grid_xy, int64_t(1), max_splits);
  const auto float_opts = q.options().dtype(kFloat);
  Tensor sums, maxs, scores, score_sums, score_maxs;
  if (n_splits > 1) {
    sums = at::empty({rows, n_splits}, float_opts);
    maxs = at::empty_like(sums);
  }
  if (staged) {
    scores = at::empty({rows, k.size(2)}, float_opts);
    score_sums = at::empty({rows, n_score_splits}, float_opts);
    score_maxs = at::empty_like(score_sums);
  }
  const auto dtype = scalarToMetalTypeString(q);
  const auto family = use_mpp ? (is_apple10_or_newer ? "_mpp_a10"sv : "_mpp_a9"sv) : ""sv;
  const MTLSize threadgroup = MTLSizeMake(c10::metal::simdgroup_size, kPrefillSimdgroups, 1);
  MPSStream* mpsStream = getCurrentMPSStream();
  auto launch = [&](Stage stage, const Tensor& vs, const Tensor& os) {
    const bool has_mask = mask && stage != Stage::pv;
    const bool chunked = stage != Stage::pv && d > bd;
    const bool split = stage != Stage::scores && n_splits > 1;
    const int64_t slabs = stage == Stage::scores ? 1 : c10::metal::ceil_div(vs.size(3), bd);
    const int64_t splits = stage == Stage::scores ? n_score_splits : n_splits;
    const MTLSize grid = MTLSizeMake(nq, q.size(1), q.size(0) * splits * slabs);
    Tensor partials;
    if (split) {
      // The simdgroup fused kernel writes tile-wide partial rows.
      partials = at::empty({rows, n_splits, use_mpp || stage == Stage::pv ? vs.size(3) : bd}, float_opts);
    }
    const auto mask_dtype = has_mask ? scalarToMetalTypeString(*mask) : dtype;
    auto kname = fmt::format("prefill_attention{}{}_{}_bd{}", family, kStageSuffix[size_t(stage)], dtype, bd);
    kname += fmt::format("_hm{:d}_dc{:d}_ck{:d}_split{:d}_mask{}", has_mask, is_causal, chunked, split, mask_dtype);
    const auto pso = lib.getPipelineStateForFunc(kname);
    const PrefillAttnParams params{
        .B = c10::safe_conv<int32_t>(q.size(0)),
        .H = c10::safe_conv<int32_t>(q.size(1)),
        .D = c10::safe_conv<int32_t>(d),
        .qL = c10::safe_conv<int32_t>(q.size(2)),
        .kL = c10::safe_conv<int32_t>(k.size(2)),
        .gqa_factor = c10::safe_conv<int32_t>(q.size(1) / k.size(1)),
        .scale = scale,
        .Dv = c10::safe_conv<int32_t>(vs.size(3) / slabs),
        .n_splits = c10::safe_conv<int32_t>(splits),
        .n_slabs = c10::safe_conv<int32_t>(slabs),
        .n_score_splits = c10::safe_conv<int32_t>(n_score_splits),
        .Q_strides = {q.stride(0), q.stride(1), q.stride(2)},
        .K_strides = {k.stride(0), k.stride(1), k.stride(2)},
        .V_strides = {vs.stride(0), vs.stride(1), vs.stride(2)},
        .O_strides = {os.stride(0), os.stride(1), os.stride(2)},
    };
    dispatch_sync_with_rethrow(mpsStream->queue(), ^{
      @autoreleasepool {
        auto computeEncoder = mpsStream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        // Split-K launches write their partial rows in place of the output.
        mtl_setArgs(computeEncoder, q, k, vs, split ? partials : os, params, mask_params);
        // The kernels reference the mask and row-stats buffers in every
        // instantiation; Metal validation wants them bound even when unused, and
        // the writable row stats bound to a real buffer (the output stands in).
        mtl_setArgs<6>(computeEncoder, uint32_t(0), os, os);
        if (has_mask) {
          mtl_setArgs<6>(computeEncoder, *mask);
        }
        if (stage == Stage::scores) {
          mtl_setArgs<7>(computeEncoder, score_sums, score_maxs);
        } else if (split) {
          mtl_setArgs<7>(computeEncoder, sums, maxs);
        }
        if (staged) {
          mtl_setArgs<9>(computeEncoder, scores, score_maxs);
        }
        [computeEncoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        if (split) {
          merge_partials(computeEncoder, partials, sums, maxs, os);
        }
      }
    });
  };
  if (staged) {
    launch(Stage::scores, v, out);
  }
  // Only the MPP fused kernel and P @ V take slabs in one launch.
  const int64_t full = dv / bd * bd;
  const int64_t step = (use_mpp || staged) && grid_xy * n_splits < core_grid ? std::max(full, bd) : bd;
  for (int64_t v0 = 0; v0 < dv; v0 += step) {
    const auto vs = v.narrow(3, v0, std::min(step, dv - v0));
    launch(staged ? Stage::pv : Stage::fused, vs, out.narrow(3, v0, vs.size(3)));
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
  TORCH_CHECK_NOT_IMPLEMENTED(dtype == kFloat || dtype == kHalf || dtype == kBFloat16,
                              "scaled_dot_product_attention for MPS does not support dtype ",
                              dtype);
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
  // The kernels index with 32-bit sizes (strides stay 64-bit).
  TORCH_CHECK_NOT_IMPLEMENTED(
      std::max({q_.numel(), k_.numel(), v_.numel()}) <= std::numeric_limits<int32_t>::max(),
      "scaled_dot_product_attention for MPS does not support tensors with more than 2^31 elements");

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
  const int64_t bd = q_len == 1 ? decode_bucket(d, dv) : 0;
  if (out.numel() > 0 && bd > 0) {
    sdpa_vector_mps(q_, k_, v_, out, mask, is_causal, scale_factor, bd);
  } else if (out.numel() > 0) {
    sdpa_prefill_mps(q_, k_, v_, out, mask, is_causal, scale_factor);
  }
  // The attention weights are never materialized (nothing consumes them); an
  // empty tensor stands in, as the flash attention ops do for unrequested outputs.
  return {out.view_symint(out_shape), at::empty({0}, q_.options())};
}
} // namespace native
} // namespace at
