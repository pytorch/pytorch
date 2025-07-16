/******************************************************************************
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <cstdint>
#include <tuple>

#include <ATen/ops/zeros.h>

#ifdef USE_FLASH_ATTENTION
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#endif

#include <ATen/native/transformers/hip/aotriton_adapter.h>
#include <ATen/native/transformers/hip/flash_attn/flash_api.h>

#include <c10/util/Exception.h>

// AOTriton headers
#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>

#if AOTRITON_VERSION_MINOR < 9
#error "This adaptor code is only tested with AOTriton >= 0.9"
#endif

#if (AOTRITON_VERSION_MAJOR * 100 + AOTRITON_VERSION_MINOR) >= 10
#define V3_API 1
#else
#define V3_API 0
#endif

namespace pytorch_flash {

namespace {

void check_gpu_arch(hipStream_t stream) {
  auto ret = aotriton::v2::flash::check_gpu(stream);
  if (hipSuccess != ret) {
      TORCH_CHECK(false,
                  "[AOTriton] Accelerated SDPA only supports MI200/MI300X/Navi31 GPUs"
                  " (gfx90a:sramecc+:xnack-/gfx942:sramecc+:xnack-/gfx1100)")
  }
}

std::tuple<bool, int, int>
calculate_swa(std::optional<int64_t> window_size_left,
              std::optional<int64_t> window_size_right,
              int max_seqlen_q,
              int max_seqlen_k,
              bool is_causal) {
#if V3_API  // SWA is exposed through V3 API
  bool needs_swa = false;
  using aotriton::v3::flash::WindowValue;
  // Default values when std::optional window_size_left/right have no value
  int window_left = max_seqlen_q;
  int window_right = max_seqlen_k;
  if (is_causal) {
    window_left = WindowValue::TopLeftAligned;
    window_right = WindowValue::TopLeftAligned;
  }
  if (window_size_left.has_value() || window_size_right.has_value()) {
    needs_swa = true;
    window_left = window_size_left.value_or(window_left);
    window_right = window_size_right.value_or(window_right);
  }
  return std::make_tuple(needs_swa, window_left, window_right);
#else
  if (window_size_left.has_value() || window_size_right.has_value()) {
    TORCH_WARN_ONCE("Current AOTriton does not support sliding window attention (SWA)."
                    " Both window_size_left and window_size_right will be ignored."
                    " Re-compile PyTorch with AOTriton >= 0.10b to enable SWA support.");
  }
  return std::make_tuple(false, 0, 0);
#endif
}

// We want to checkpoint and save the RNG state for backward if dropout
// We get the default generator and return the seed and offset which will
// be used in the backward function
std::tuple<at::Tensor, at::Tensor, at::PhiloxCudaState, bool>
prepare_philox_arguments(float p_dropout, int64_t counter_offset) {
  at::Tensor seed_t, offset_t;
  at::PhiloxCudaState philox_state;
  bool use_philox_state = false;
  if (p_dropout <= 0.0)  {
    seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    return { seed_t, offset_t, philox_state, use_philox_state };
  }
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  philox_state = gen->philox_cuda_state(counter_offset);
  if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
    auto [seed, offset] = at::cuda::philox::unpack(philox_state);
    seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong).device(at::kCUDA));
    offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong).device(at::kCUDA));
  } else {
    // See Note [CUDA Graph-safe RNG states] about the design
    use_philox_state = true;
    seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
  }

  return { seed_t, offset_t, philox_state, use_philox_state };
}


}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_aot(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
            const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
            std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
            std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
            const float p_dropout,
            const float softmax_scale,
            bool is_causal,
            std::optional<int64_t> window_size_left,
            std::optional<int64_t> window_size_right,
            const bool return_softmax,
            const std::optional<at::Generator>& gen_) {
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  // FIXME: ROCM probably does not need this
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(head_size_og <= 512, "FlashAttention on ROCm forward only supports head dimension at most 512");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (seqlen_q == 1) { is_causal = false; }  // causal=true is the same as causal=false in this case

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

  at::Tensor q_padded, k_padded, v_padded;
  q_padded = q;
  k_padded = k;
  v_padded = v;

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
    if (head_size_og % 8 != 0) { out = at::empty_like(q_padded); }
  } else {
    out = at::empty_like(q_padded);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);

  auto [seed_t, offset_t, philox_state, use_philox_state] =
    prepare_philox_arguments(p_dropout, batch_size * num_heads * 32);

  // Transpose tensors to meet AOTriton's Flash API
  at::Tensor q_t = q_padded.permute({0,2,1,3});
  at::Tensor k_t = k_padded.permute({0,2,1,3});
  at::Tensor v_t = v_padded.permute({0,2,1,3});
  at::Tensor output_t = out.permute({0,2,1,3});

  auto opts = q.options();
  at::Tensor M = at::empty({batch_size * num_heads, seqlen_q}, opts.dtype(at::kFloat)); // aka softmax_lse

  at::Tensor softmax_fa_t;
  if (return_softmax) {
    softmax_fa_t = at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts);
  } else {
    softmax_fa_t = at::empty({ 0, 0, 0, 0 }, opts);
  }

  at::Tensor atomic_counter;
  if (is_causal) {
    atomic_counter = at::zeros({1}, opts.dtype(at::kInt));
  }

  auto [needs_swa, window_left, window_right] = calculate_swa(window_size_left,
                                                              window_size_right,
                                                              seqlen_q,
                                                              seqlen_k,
                                                              is_causal);
#if V3_API
  const bool uses_swa = needs_swa;
#else
  // When V3_API = 0, uses_swa is constexpr and the if (uses_swa) branch can be
  // optimized out (hopefully).
  constexpr bool uses_swa = false;
#endif

  hipError_t err; // TODO: Error handling
  using aotriton::v2::flash::attn_fwd;
  using sdp::aotriton_adapter::mk_aotensor;
  using sdp::aotriton_adapter::mk_aoscalartensor;
  using sdp::aotriton_adapter::mk_philoxtensor;
  using sdp::aotriton_adapter::mk_atomictensor;
  using sdp::aotriton_adapter::cast_dtype;
  aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
  auto seed = use_philox_state ? mk_philoxtensor(philox_state.seed_.ptr) : mk_aoscalartensor(seed_t);
  auto offset1 = use_philox_state ? mk_philoxtensor(philox_state.offset_.ptr) : mk_aoscalartensor(offset_t);
  auto offset2 = use_philox_state ? philox_state.offset_intragraph_ : 0;
  auto seed_output = mk_philoxtensor(use_philox_state ? seed_t.data_ptr<int64_t>() : nullptr);
  auto offset_output = mk_philoxtensor(use_philox_state ? offset_t.data_ptr<int64_t>() : nullptr);
  auto persistent_counter = mk_atomictensor(is_causal ? atomic_counter.data_ptr<int32_t>() : nullptr);
  if (uses_swa) {
#if V3_API
    using aotriton::v3::flash::CausalType;
    using aotriton::v3::flash::VarlenType;
    aotriton::v3::flash::attn_fwd_params params;
    params.Q = mk_aotensor(q_t, "q");
    params.K = mk_aotensor(k_t, "k");
    params.V = mk_aotensor(v_t, "v");
    params.Sm_scale = softmax_scale;
    params.L = mk_aotensor<2>(M, "M");
    params.Out = mk_aotensor(output_t, "Out");
    params.Max_seqlen_q = seqlen_q;    // Unused if cu_seqlens_q is empty
    params.Max_seqlen_k = seqlen_k;    // Unused if cu_seqlens_k is empty
    params.dropout_p = p_dropout;
    params.philox_seed_ptr = seed;
    params.philox_offset1 = offset1;
    params.philox_offset2 = offset2;
    params.philox_seed_output = seed_output;
    params.philox_offset_output = offset_output;
    params.encoded_softmax = mk_aotensor(softmax_fa_t, "encoded_softmax");
    params.persistent_atomic_counter = persistent_counter;
    params.causal_type = CausalType::WindowedAttention;
    params.varlen_type = VarlenType::None;
    params.window_left = window_left;
    params.window_right = window_right;
    err = aotriton::v3::flash::attn_fwd(params,
                                        aotriton::v3::flash::attn_fwd_params::kVersion,
                                        stream);
#endif
  } else {
    err = attn_fwd(mk_aotensor(q_t, "q"),
                   mk_aotensor(k_t, "k"),
                   mk_aotensor(v_t, "v"),
                   empty_bias,
                   softmax_scale,
                   mk_aotensor<2>(M, "M"),
                   mk_aotensor(output_t, "Out"),
                   p_dropout,
                   seed,
                   offset1,
                   offset2,
                   seed_output,
                   offset_output,
                   mk_aotensor(softmax_fa_t, "encoded_softmax"),
                   is_causal,
                   persistent_counter,
                   stream);
  }

  return {out, q_padded, k_padded, v_padded, M.view({batch_size, num_heads, seqlen_q}), seed_t, offset_t, softmax_fa_t};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_aot(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               std::optional<int64_t> window_size_left,
               std::optional<int64_t> window_size_right,
               const bool return_softmax,
               const std::optional<at::Generator>& gen_) {
  TORCH_CHECK(!seqused_k.has_value(), "[ROCm] mha_varlen_fwd: seqused_k must be nullopt");
  const bool paged_KV = block_table_.has_value();
  TORCH_CHECK(!paged_KV, "[ROCm] mha_varlen_fwd: block_table_ must be nullopt");
  TORCH_CHECK(!alibi_slopes_.has_value(), "[ROCm] mha_varlen_fwd: alibi_slopes_ must be nullopt");

  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();

  const int batch_size = cu_seqlens_q.numel() - 1;
  int num_heads = sizes[1];
  const int head_size_og = sizes[2];
  const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

  if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) {
    is_causal = false;
  }  // causal=true is the same as causal=false in this case

  at::Tensor temp_q = q;
  const int total_q = temp_q.sizes()[0];

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og <= 512, "FlashAttention on ROCm forward only supports head dimension at most 512");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  CHECK_SHAPE(temp_q, total_q, num_heads, head_size_og);
  const int total_k = k.size(0);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  // AOTriton's varlen API needs input shapes be
  // (1, num_heads, total sequence lenght, head dimension)
  at::Tensor q_padded, k_padded, v_padded;
  at::Tensor out, out_padded;
  q_padded = q.unsqueeze(0).transpose(1, 2);
  k_padded = k.unsqueeze(0).transpose(1, 2);
  v_padded = v.unsqueeze(0).transpose(1, 2);
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, total_q, num_heads, head_size_og);
  } else {
    out = at::empty_like(q);
  }
  out_padded = out.unsqueeze(0).transpose(1, 2);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = head_size_og;

  auto opts = q.options();

  auto softmax_lse = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor M = softmax_lse.view({batch_size * num_heads, max_seqlen_q});
  at::Tensor softmax_fa_t;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    softmax_fa_t = at::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts);
  } else {
    softmax_fa_t = at::empty({ 0, 0, 0, 0 }, opts);
  }

  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    if (return_softmax) {
      softmax_fa_t.zero_();
    }
  }

  auto [needs_swa, window_left, window_right] = calculate_swa(window_size_left,
                                                              window_size_right,
                                                              max_seqlen_q,
                                                              max_seqlen_k,
                                                              is_causal);
#if V3_API
  const bool uses_swa = needs_swa;
#else
  // When V3_API = 0, uses_swa is constexpr and the if (uses_swa) branch can be
  // optimized out (hopefully).
  constexpr bool uses_swa = false;
#endif

  auto [seed_t, offset_t, philox_state, use_philox_state] =
    prepare_philox_arguments(p_dropout, batch_size * num_heads * 32);

  if (max_seqlen_k > 0) {
    hipError_t err; // TODO: Error handling
    using aotriton::v2::flash::attn_fwd_compact_varlen;
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    using sdp::aotriton_adapter::mk_philoxtensor;
    using sdp::aotriton_adapter::cast_dtype;
    at::Tensor atomic_counter;
    if (is_causal) {
      atomic_counter = at::zeros({1}, q.options());
    }
    aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
    auto seed = use_philox_state ? mk_philoxtensor(philox_state.seed_.ptr) : mk_aoscalartensor(seed_t);
    auto offset1 = use_philox_state ? mk_philoxtensor(philox_state.offset_.ptr) : mk_aoscalartensor(offset_t);
    auto offset2 = use_philox_state ? philox_state.offset_intragraph_ : 0;
    auto nullscalar = mk_philoxtensor(nullptr);
    auto seed_output = use_philox_state ? mk_philoxtensor(seed_t.data_ptr<int64_t>()) : nullscalar;
    auto offset_output = use_philox_state ? mk_philoxtensor(offset_t.data_ptr<int64_t>()) : nullscalar;
    auto persistent_counter = is_causal ? mk_philoxtensor(atomic_counter.data_ptr<int64_t>()) : nullscalar;
    if (uses_swa) {
#if V3_API
      using aotriton::v3::flash::CausalType;
      using aotriton::v3::flash::VarlenType;
      aotriton::v3::flash::attn_fwd_params params;
      params.Q = mk_aotensor(q_padded, "q");
      params.K = mk_aotensor(k_padded, "k");
      params.V = mk_aotensor(v_padded, "v");
      params.Sm_scale = softmax_scale;
      params.L = mk_aotensor<2>(M, "M");
      params.Out = mk_aotensor(out_padded, "Out");
      params.cu_seqlens_q = mk_aotensor<1>(cu_seqlens_q, "cu_seqlens_q");
      params.cu_seqlens_k = mk_aotensor<1>(cu_seqlens_k, "cu_seqlens_k");
      params.Max_seqlen_q = max_seqlen_q;    // Unused if cu_seqlens_q is empty
      params.Max_seqlen_k = max_seqlen_k;    // Unused if cu_seqlens_k is empty
      params.dropout_p = p_dropout;
      params.philox_seed_ptr = seed;
      params.philox_offset1 = offset1;
      params.philox_offset2 = offset2;
      params.philox_seed_output = seed_output;
      params.philox_offset_output = offset_output;
      params.encoded_softmax = mk_aotensor(softmax_fa_t, "encoded_softmax");
      params.persistent_atomic_counter = persistent_counter;
      params.causal_type = CausalType::WindowedAttention;
      params.varlen_type = VarlenType::CompactVarlen;
      params.window_left = window_left;
      params.window_right = window_right;
      err = aotriton::v3::flash::attn_fwd(params,
                                          aotriton::v3::flash::attn_fwd_params::kVersion,
                                          stream);
#endif
    } else {
      err = attn_fwd_compact_varlen(mk_aotensor(q_padded, "q"),
                                    mk_aotensor(k_padded, "k"),
                                    mk_aotensor(v_padded, "v"),
                                    empty_bias,
                                    mk_aotensor<1>(cu_seqlens_q, "cu_seqlens_q"),
                                    mk_aotensor<1>(cu_seqlens_k, "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    softmax_scale,
                                    mk_aotensor<2>(M, "M"),
                                    mk_aotensor(out_padded, "Out"),
                                    p_dropout,
                                    seed,
                                    offset1,
                                    offset2,
                                    seed_output,
                                    offset_output,
                                    mk_aotensor(softmax_fa_t, "encoded_softmax"),
                                    is_causal,
                                    persistent_counter,
                                    stream);
    }
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  return {out, q, k, v, softmax_lse, seed_t, offset_t, softmax_fa_t};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd_aot(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        std::optional<int64_t> window_size_left,
        std::optional<int64_t> window_size_right,
        const bool deterministic,
        const at::Tensor& philox_seed,
        const at::Tensor& philox_offset) {
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

  bool is_dropout = p_dropout > 0.0;

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);

  if (is_causal){
    TORCH_CHECK((seqlen_q == seqlen_k), "For backwards kernel seqlen_q must equal seqlen_k for causal kernels");
  }

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_og % 8 == 0, "head_size_og should be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(head_size <= 512, "FlashAttention on ROCm backward only supports head dimension at most 512");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);

  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  } else {
    dq = at::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dk = at::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dv = at::empty_like(k);
  }

  auto [needs_swa, window_left, window_right] = calculate_swa(window_size_left,
                                                              window_size_right,
                                                              seqlen_q,
                                                              seqlen_k,
                                                              is_causal);
#if V3_API
  const bool uses_swa = needs_swa;
#else
  // When V3_API = 0, uses_swa is constexpr and the if (uses_swa) branch can be
  // optimized out (hopefully).
  constexpr bool uses_swa = false;
#endif

  auto opts = q.options();
  auto softmax_d = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  at::Tensor q_t = q.permute({0,2,1,3});
  at::Tensor k_t = k.permute({0,2,1,3});
  at::Tensor v_t = v.permute({0,2,1,3});
  at::Tensor out_t = out.permute({0,2,1,3});
  at::Tensor dq_t = dq.permute({0,2,1,3});
  at::Tensor dk_t = dk.permute({0,2,1,3});
  at::Tensor dv_t = dv.permute({0,2,1,3});
  at::Tensor dout_t = dout.permute({0,2,1,3});

  at::Tensor softmax_lse_cont = softmax_lse.view({batch_size * num_heads, seqlen_q}).contiguous();

  int d_head = head_size_og;
  bool use_fused_bwd = d_head <= 192 && d_head * seqlen_q < 64 * 512;
  hipError_t err; // TODO: Error handling
  using sdp::aotriton_adapter::mk_aotensor;
  using sdp::aotriton_adapter::mk_aoscalartensor;
  if (uses_swa) {
#if V3_API
    // Fused BWD does not support SWA
    at::Tensor delta = at::empty_like(softmax_lse_cont).contiguous();
    using aotriton::v3::flash::CausalType;
    using aotriton::v3::flash::VarlenType;
    aotriton::v3::flash::attn_bwd_params params;
    params.Q = mk_aotensor(q_t, "q");
    params.K = mk_aotensor(k_t, "k");
    params.V = mk_aotensor(v_t, "v");
    params.Sm_scale = softmax_scale;
    params.Out = mk_aotensor(out_t, "out");
    params.DO = mk_aotensor(dout_t, "dout");
    params.DK = mk_aotensor(dq_t, "dq");
    params.DV = mk_aotensor(dk_t, "dk");
    params.DQ = mk_aotensor(dv_t, "dv");
    params.L = mk_aotensor<2>(softmax_lse_cont, "L");
    params.D = mk_aotensor<2>(delta, "delta");
    params.Max_seqlen_q = seqlen_q;        // Unused if cu_seqlens_q is empty
    params.Max_seqlen_k = seqlen_k;        // Unused if cu_seqlens_k is empty
    params.dropout_p = p_dropout;
    params.philox_seed_ptr =  mk_aoscalartensor(philox_seed);
    params.philox_offset1 = mk_aoscalartensor(philox_offset);
    params.philox_offset2 = 0;
    params.causal_type = CausalType::WindowedAttention;
    params.varlen_type = VarlenType::None;
    params.window_left = window_left;
    params.window_right = window_right;
    err = aotriton::v3::flash::attn_bwd(params,
                                        aotriton::v3::flash::attn_bwd_params::kVersion,
                                        stream);
#endif
  } else if (use_fused_bwd) {
    using aotriton::v2::flash::attn_bwd_fused;
    using sdp::aotriton_adapter::cast_dtype;
    aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
    err = attn_bwd_fused(mk_aotensor(q_t, "q"),
                         mk_aotensor(k_t, "k"),
                         mk_aotensor(v_t, "v"),
                         empty_bias,
                         softmax_scale,
                         mk_aotensor(out_t, "out"),
                         mk_aotensor(dout_t, "dout"),
                         mk_aotensor(dq_t, "dq"),
                         mk_aotensor(dk_t, "dk"),
                         mk_aotensor(dv_t, "dv"),
                         empty_bias,  // dbb
                         mk_aotensor<2>(softmax_lse_cont, "L"),
                         p_dropout,
                         mk_aoscalartensor(philox_seed),
                         mk_aoscalartensor(philox_offset),
                         0,
                         is_causal,
                         stream);
  } else {
    at::Tensor delta = at::empty_like(softmax_lse_cont).contiguous();
    using aotriton::v2::flash::attn_bwd;
    using sdp::aotriton_adapter::cast_dtype;
    aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
    err = attn_bwd(mk_aotensor(q_t, "q"),
                   mk_aotensor(k_t, "k"),
                   mk_aotensor(v_t, "v"),
                   empty_bias,
                   softmax_scale,
                   mk_aotensor(out_t, "out"),
                   mk_aotensor(dout_t, "dout"),
                   mk_aotensor(dq_t, "dq"),
                   mk_aotensor(dk_t, "dk"),
                   mk_aotensor(dv_t, "dv"),
                   empty_bias,  // db
                   mk_aotensor<2>(softmax_lse_cont, "L"),
                   mk_aotensor<2>(delta, "delta"),
                   p_dropout,
                   mk_aoscalartensor(philox_seed),
                   mk_aoscalartensor(philox_offset),
                   0,
                   is_causal,
                   stream);
  }

  return { dq, dk, dv, softmax_d };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd_aot(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               std::optional<int64_t> window_size_left,
               std::optional<int64_t> window_size_right,
               const bool deterministic,
               const at::Tensor& philox_seed,
               const at::Tensor& philox_offset)
{
  TORCH_CHECK(!alibi_slopes_.has_value(), "[ROCm] mha_varlen_fwd: alibi_slopes_ must be nullopt");

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

  bool is_dropout = p_dropout > 0.0;

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
  CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = sizes[1];
  const int head_size_og = dout.size(2);
  const int head_size = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size <= 512, "FlashAttention on ROCm backward only supports head dimension at most 512");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  CHECK_SHAPE(out, total_q, num_heads, head_size);
  CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  at::Tensor softmax_lse_cont = softmax_lse.view({batch_size * num_heads, max_seqlen_q}).contiguous();
  at::Tensor delta = at::empty_like(softmax_lse_cont).contiguous();

  at::Tensor q_padded, k_padded, v_padded;
  q_padded = q.unsqueeze(0).transpose(1, 2);
  k_padded = k.unsqueeze(0).transpose(1, 2);
  v_padded = v.unsqueeze(0).transpose(1, 2);
  at::Tensor out_t, dout_t;
  out_t = out.unsqueeze(0).transpose(1, 2);
  dout_t = dout.unsqueeze(0).transpose(1, 2);

  at::Tensor dq, dk, dv;
  at::Tensor dq_padded, dk_padded, dv_padded;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
  } else {
    dq = at::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
  } else {
    dk = at::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
  } else {
    dv = at::empty_like(v);
  }
  dq_padded = dq.unsqueeze(0).transpose(1, 2);
  dk_padded = dk.unsqueeze(0).transpose(1, 2);
  dv_padded = dv.unsqueeze(0).transpose(1, 2);

  auto opts = q.options();
  auto softmax_d = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

  if( zero_tensors ) {
    dq.zero_();
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  }

  auto [needs_swa, window_left, window_right] = calculate_swa(window_size_left,
                                                              window_size_right,
                                                              max_seqlen_q,
                                                              max_seqlen_k,
                                                              is_causal);
#if V3_API
  const bool uses_swa = needs_swa;
#else
  // When V3_API = 0, uses_swa is constexpr and the if (uses_swa) branch can be
  // optimized out (hopefully).
  constexpr bool uses_swa = false;
#endif

  at::PhiloxCudaState philox_args;
  if (is_dropout) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None)
    {
      philox_args = at::PhiloxCudaState(*philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      philox_args = at::PhiloxCudaState(
          philox_seed.data_ptr<int64_t>(), philox_offset.data_ptr<int64_t>(), 0);
    }
  }
  if (max_seqlen_q > 0) {
    hipError_t err; // TODO: Error handling
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    if (uses_swa) {
#if V3_API
      using aotriton::v3::flash::CausalType;
      using aotriton::v3::flash::VarlenType;
      aotriton::v3::flash::attn_bwd_params params;
      params.Q = mk_aotensor(q_padded, "q");
      params.K = mk_aotensor(k_padded, "k");
      params.V = mk_aotensor(v_padded, "v");
      params.Sm_scale = softmax_scale;
      params.Out = mk_aotensor(out_t, "out");
      params.DO = mk_aotensor(dout_t, "dout");
      params.DK = mk_aotensor(dq_padded, "dq");
      params.DV = mk_aotensor(dk_padded, "dk");
      params.DQ = mk_aotensor(dv_padded, "dv");
      params.L = mk_aotensor<2>(softmax_lse_cont, "L");
      params.D = mk_aotensor<2>(delta, "delta");
      params.cu_seqlens_q = mk_aotensor<1>(cu_seqlens_q, "cu_seqlens_q");
      params.cu_seqlens_k = mk_aotensor<1>(cu_seqlens_k, "cu_seqlens_k");
      params.Max_seqlen_q = max_seqlen_q;        // Unused if cu_seqlens_q is empty
      params.Max_seqlen_k = max_seqlen_k;        // Unused if cu_seqlens_k is empty
      params.dropout_p = p_dropout;
      params.philox_seed_ptr =  mk_aoscalartensor(philox_seed);
      params.philox_offset1 = mk_aoscalartensor(philox_offset);
      params.philox_offset2 = 0;
      params.causal_type = CausalType::WindowedAttention;
      params.varlen_type = VarlenType::CompactVarlen;
      params.window_left = window_left;
      params.window_right = window_right;
      err = aotriton::v3::flash::attn_bwd(params,
                                          aotriton::v3::flash::attn_bwd_params::kVersion,
                                          stream);
#endif
    } else {
      using aotriton::v2::flash::attn_bwd_compact_varlen;
      using sdp::aotriton_adapter::cast_dtype;
      aotriton::TensorView<4> empty_bias(0, {0,0,0,0}, {0,0,0,0}, cast_dtype(q.dtype()));
      err = attn_bwd_compact_varlen(mk_aotensor(q_padded, "q"),
                                    mk_aotensor(k_padded, "k"),
                                    mk_aotensor(v_padded, "v"),
                                    mk_aotensor<1>(cu_seqlens_q, "cu_seqlens_q"),
                                    mk_aotensor<1>(cu_seqlens_k, "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    empty_bias,
                                    softmax_scale,
                                    mk_aotensor(out_t, "out"),
                                    mk_aotensor(dout_t, "dout"),
                                    mk_aotensor(dq_padded, "dq"),
                                    mk_aotensor(dk_padded, "dk"),
                                    mk_aotensor(dv_padded, "dv"),
                                    empty_bias,
                                    mk_aotensor<2>(softmax_lse_cont, "L"),
                                    mk_aotensor<2>(delta, "delta"),
                                    p_dropout,
                                    mk_aoscalartensor(philox_seed),
                                    mk_aoscalartensor(philox_offset),
                                    0,
                                    is_causal,
                                    stream);
    }
  } else {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
    dq.zero_();
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  }

  return { dq, dk, dv, softmax_d };
}
} // namespace pytorch_flash

#endif  // USE_FLASH_ATTENTION
