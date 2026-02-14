#include <string_view>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <cstdint>
#include <type_traits>

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/Exception.h>
#include <c10/util/bit_cast.h>

#include <c10/core/TensorImpl.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/_cudnn_attention_backward.h>
#include <ATen/ops/_cudnn_attention_backward_native.h>
#include <ATen/ops/_flash_attention_backward.h>
#include <ATen/ops/_flash_attention_backward_native.h>
#include <ATen/ops/_efficient_attention_backward.h>
#include <ATen/ops/_efficient_attention_backward_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward_native.h>
#endif

#ifdef USE_FLASH_ATTENTION
// FlashAttention Specific Imports
#include <ATen/native/transformers/cuda/flash_attn/flash_api.h>
#endif
#ifdef USE_MEM_EFF_ATTENTION
#ifndef USE_ROCM
// MemoryEfficient Attention Specific Imports for CUDA
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h>
#else
#include <ATen/native/transformers/hip/gemm_kernel_utils.h>
// MemoryEfficient Attention Specific Imports for ROCM
#ifndef DISABLE_AOTRITON
#include <ATen/native/transformers/hip/aotriton_adapter.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#endif
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>
#endif
#endif

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/MHA.h>
#else
#include <ATen/native/cudnn/MHA.h>
#endif

namespace at::native {

std::tuple<Tensor, Tensor, Tensor> _flash_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

#ifndef USE_ROCM  // ROCM backend accepts std::optional for window_size_left/right directly.
  const int non_null_window_left = window_size_left.has_value() ? window_size_left.value() : -1;
  const int non_null_window_right = window_size_right.has_value() ? window_size_right.value() : -1;
#endif

  std::optional<at::Tensor> dq{std::nullopt};
  std::optional<at::Tensor> dk{std::nullopt};
  std::optional<at::Tensor> dv{std::nullopt};

  //  The kernel computes regardless we will drop for this functions return
  Tensor grad_softmax;

  // Currently unused args:
  std::optional<at::Tensor> alibi_slopes{std::nullopt};
  const float softcap = 0.0;

  bool deterministic{false};
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      deterministic = true;
    }
  }

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense forward
  if (cumulative_sequence_length_q.defined()) {
    // Varlen forward
    auto [dQuery, dKey, dValue, dSoftmax] = FLASH_NAMESPACE::mha_varlen_bwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        alibi_slopes,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        false /*zero_tensors*/,
        is_causal,
#ifdef USE_ROCM
        window_size_left,
        window_size_right,
#else
        non_null_window_left,
        non_null_window_right,
#endif
        softcap,
        deterministic,
        philox_seed,
        philox_offset);
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
  } else {
    // Dense forward
    auto [dQuery, dKey, dValue, dSoftmax] = FLASH_NAMESPACE::mha_bwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        is_causal,
#ifdef USE_ROCM
        window_size_left,
        window_size_right,
#else
        non_null_window_left,
        non_null_window_right,
#endif
        softcap,
        deterministic,
        philox_seed,
        philox_offset);
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& attn_bias,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

    auto& ctx = at::globalContext();
    if (ctx.deterministicAlgorithms()) {
      if (ctx.deterministicAlgorithmsWarnOnly()) {
        TORCH_WARN_ONCE(
            "cuDNN Attention defaults to a non-deterministic algorithm. ",
            "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
      }
    }

    const bool is_nested = cum_seq_q.defined();
    const int64_t max_seqlen_batch_q = query.size(2);
    const int64_t max_seqlen_batch_k = key.size(2);

    if (!is_nested) {
      const int64_t batch_size = query.size(0);
      const int64_t num_heads = query.size(1);
      const int64_t head_dim_qk = query.size(3);
      const int64_t head_dim_v = value.size(3);

      // This is needed because SaveVariable automatically converts
      // std::optional to undefined tensor
      std::optional<Tensor> attn_bias_;
      if (attn_bias.defined()) {
        attn_bias_ = attn_bias;
      }
      if (attn_bias_.has_value()) {
        const auto bias_dim = attn_bias_.value().dim();
        if (bias_dim == 2) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else if (bias_dim == 3) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else {
          TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
          attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_k});
        }
      }

      const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
      auto dq = at::empty_like(query);
      auto dk = at::empty_like(key);
      auto dv = at::empty_like(value);
      run_cudnn_SDP_bprop(batch_size /*int64_t b*/,
                          num_heads /*int64_t h*/,
                          max_q/*int64_t s_q*/,
                          max_k/*int64_t s_kv*/,
                          head_dim_qk /*int64_t d_qk*/,
                          head_dim_v /*int64_t d_v*/,
                          softmax_scale /*float scaling_factor*/,
                          is_causal /*bool is_causal*/,
                          dropout_p /*float dropout_probability*/,
                          query /*const Tensor& q*/,
                          key /*const Tensor& k*/,
                          value /*const Tensor& v*/,
                          attn_bias_ /*const std::optional<Tensor>& attn_bias*/,
                          out /*const Tensor& o*/,
                          grad_out/*const Tensor& dO*/,
                          logsumexp/*const Tensor& softmaxstats*/,
                          dq/*Tensor& dQ*/,
                          dk/*Tensor& dK*/,
                          dv/*Tensor& dV*/,
                          philox_seed/*Tensor& dropoutseed*/,
                          philox_offset/*Tensor& dropoutoffset*/);
      return std::make_tuple(std::move(dq), std::move(dk), std::move(dv));
    } else {
      // BHSD ...
      const int64_t batch_size = cum_seq_q.size(0) - 1;
      const int64_t num_heads_q = query.size(-2);
      const int64_t num_heads_k = key.size(-2);
      const int64_t num_heads_v = value.size(-2);
      const int64_t head_dim_qk = query.size(-1);
      const int64_t head_dim_v = value.size(-1);
      std::optional<Tensor> attn_bias_;
      if (attn_bias.defined()) {
        attn_bias_ = attn_bias;
      }
      if (attn_bias_.has_value()) {
        const auto bias_dim = attn_bias_.value().dim();
        if (bias_dim == 2) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else if (bias_dim == 3) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else {
          attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_k});
          TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
        }
      }

      auto dq = at::empty_like(query);
      auto dk = at::empty_like(key);
      auto dv = at::empty_like(value);

      const auto softmax_scale = sdp::calculate_scale(query, scale).as_float_unchecked();
      run_cudnn_SDP_bprop_nestedtensor(
        batch_size,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        head_dim_qk,
        head_dim_v,
        softmax_scale,
        is_causal,
        dropout_p,
        cum_seq_q,
        cum_seq_k,
        query,
        key,
        value,
        attn_bias_,
        out,
        grad_out,
        logsumexp,
        dq,
        dk,
        dv,
        philox_seed,
        philox_offset);
      return std::make_tuple(std::move(dq), std::move(dk), std::move(dv));
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& kernel_bias, // additive attention bias
    const at::Tensor& out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_k_dummy,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_q,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p, // dropout probability
    const at::Tensor& philox_seed, // seed using for generating random numbers for dropout
    const at::Tensor& philox_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const bool bias_requires_grad,
    const std::optional<double> scale,
    std::optional <int64_t> num_splits_key,
    const std::optional<int64_t> window_size,
    const bool shared_storage_dqdkdv) {
  #if defined(USE_MEM_EFF_ATTENTION)
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  // This path is used when we directly call _efficient_attention_forward
  // from python.
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<Tensor> bias, cu_seqlens_q, cu_seqlens_k;
  bias = kernel_bias.has_value() && !kernel_bias->defined() ? std::nullopt : kernel_bias;
  cu_seqlens_q = cu_seqlens_q_dummy.has_value() && !cu_seqlens_q_dummy->defined() ? std::nullopt : cu_seqlens_q_dummy;
  cu_seqlens_k = cu_seqlens_k_dummy.has_value() && !cu_seqlens_k_dummy->defined() ? std::nullopt : cu_seqlens_k_dummy;

    // ndim
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out_.size(3));

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(grad_out);

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  TORCH_CHECK(
      !(cu_seqlens_q.has_value() && bias.has_value()),
      "cu seqlen + bias not supported");
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q required with `cu_seqlens_q`");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k required with `cu_seqlens_k`");
    TORCH_CHECK(
        max_seqlen_k <= key.size(1), "Invalid max_seqlen_k:", max_seqlen_k);
    TORCH_CHECK(
        max_seqlen_q <= query.size(1), "Invalid max_seqlen_q:", max_seqlen_q);
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;
  if (shared_storage_dqdkdv) {
    // Create one big contiguous chunk
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    TORCH_CHECK(
      query.size(1) == key.size(1),
      "`shared_storage_dqdkdv` is only supported when Q/K/V "
      "have the same sequence length: got ", query.size(1),
      " query tokens and ", key.size(1), " key/value tokens"
    );
    TORCH_CHECK(
      query.size(3) == key.size(3),
      "`shared_storage_dqdkdv` is only supported when Q/K/V "
      "have the same embed dim: got ", query.size(3),
      " for Q, and ", key.size(3), " for K"
    );
    at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else {
    grad_q = at::empty(query.sizes(), query.options());
    grad_k = at::empty(key.sizes(), key.options());
    grad_v = at::empty(value.sizes(), value.options());
  }

  if (bias_requires_grad) {
    TORCH_CHECK(
        bias.has_value(),
        "bias_requires_grad is true but no bias was provided");
    // force alignment for the last dim
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    grad_bias = at::empty(sz, bias->options())
                    .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
  }

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // See Note [Seed and Offset Device]
  at::PhiloxCudaState rng_engine_inputs;
  if (use_dropout) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None) {
      rng_engine_inputs = at::PhiloxCudaState(
          *philox_seed.data_ptr<int64_t>(),
          *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      rng_engine_inputs = at::PhiloxCudaState(
          philox_seed.data_ptr<int64_t>(),
          philox_offset.data_ptr<int64_t>(),
          0);
    }
  }

#ifdef USE_ROCM
  // ROCM Implementation
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck)
  {
#if defined(USE_ROCM_CK_SDPA)
    const auto my_softmax_scale = sdp::calculate_scale(query, scale).expect_float();
    // Store grad_bias in optional
    std::optional<at::Tensor> opt_grad_bias = grad_bias;
    auto
        [dQ,
         dK,
         dV,
         dBias] =
             pytorch_flash::mem_eff_backward_ck(
                     grad_out,
                     query,
                     key,
                     value,
                     out,
                     logsumexp,
                     grad_q,
                     grad_k,
                     grad_v,
                     bias,
                     bias_requires_grad,
                     opt_grad_bias,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     max_seqlen_q,
                     max_seqlen_k,
                     float(dropout_p),
                     my_softmax_scale,
                     custom_mask_type == 0 ? false : true, // is_causal
                     false, // deterministic
                     false, // zero_tensors
                     philox_seed,
                     philox_offset);
    grad_bias = dBias;
#else
    TORCH_CHECK(false, "Attempting to use CK mem_eff_backward backend in a build that has not built CK");
#endif
  } else {
#ifndef DISABLE_AOTRITON
    TORCH_CHECK(!num_splits_key.has_value(),
              "ROCM does not support num_split_keys in _efficient_attention_forward");
    TORCH_CHECK(!window_size.has_value(),
              "ROCM does not support window_size in _efficient_attention_forward");
    auto ret = aotriton::v2::flash::check_gpu(stream);
    if (hipSuccess != ret) {
      TORCH_CHECK(false,
                "[AOTriton] Accelerated SDPA only supports MI200/MI300X/7900XTX/9070XT GPUs"
                " (gfx90a/gfx942/gfx1100/gfx1201)")
    }
    const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
    bool is_causal;
    if (static_cast<int64_t>(sdp::CustomMaskType::NoCustomMask) == custom_mask_type) {
      is_causal = false;
    } else {
      is_causal = true;
#if AOTRITON_V3_API == 0
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) != custom_mask_type) {
        TORCH_CHECK(false, "[_efficient_attention_forward] Unsupported mask type on ROCM, for now");
      }
#endif
    }
    at::Tensor q_t = query.permute({0,2,1,3});
    at::Tensor k_t = key.permute({0,2,1,3});
    at::Tensor v_t = value.permute({0,2,1,3});
    at::Tensor out_t = out.permute({0,2,1,3});
    at::Tensor dq_t = grad_q.permute({0,2,1,3});
    at::Tensor dk_t = grad_k.permute({0,2,1,3});
    at::Tensor dv_t = grad_v.permute({0,2,1,3});
    at::Tensor dout_t = grad_out.permute({0,2,1,3});
    at::Tensor softmax_lse = logsumexp.view({B * nH, max_seqlen_q});
    hipError_t err;
    using aotriton::v2::flash::attn_bwd;
    using aotriton::v2::flash::attn_bwd_fused;
    using aotriton::v2::flash::attn_bwd_compact_varlen;
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    using sdp::aotriton_adapter::cast_dtype;
    aotriton::TensorView<4> empty_t4(0, {0, 0, 0, 0}, {0, 0, 0, 0}, cast_dtype(query.dtype()));
    if constexpr (AOTRITON_ALWAYS_V3_API) {  // Better readability than nesting ifdef
#if AOTRITON_V3_API  // if constexpr does not stop errors from undefined functions
      using aotriton::v3::flash::CausalType;
      using aotriton::v3::flash::VarlenType;
      using aotriton::v3::flash::WindowValue;
      aotriton::v3::flash::attn_bwd_params params;
      params.Q = mk_aotensor(q_t, "q");
      params.K = mk_aotensor(k_t, "k");
      params.V = mk_aotensor(v_t, "v");
      params.B = bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4;
      params.Sm_scale = softmax_scale;
      params.Out = mk_aotensor(out_t, "out");
      params.DO = mk_aotensor(dout_t, "dout");
      params.DK = mk_aotensor(dk_t, "dk");
      params.DV = mk_aotensor(dv_t, "dv");
      params.DQ = mk_aotensor(dq_t, "dq");
      params.DB = bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4;
      params.L = mk_aotensor<2>(softmax_lse, "L");
      params.Max_seqlen_q = max_seqlen_q;        // Unused if cu_seqlens_q is empty
      params.Max_seqlen_k = max_seqlen_k;        // Unused if cu_seqlens_k is empty
      params.dropout_p = float(dropout_p);
      params.philox_seed_ptr =  mk_aoscalartensor(philox_seed);
      params.philox_offset1 = mk_aoscalartensor(philox_offset);
      params.philox_offset2 = 0;
      params.causal_type = is_causal ? CausalType::WindowedAttention : CausalType::None;
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) == custom_mask_type) {
        params.window_left = WindowValue::TopLeftAligned;
        params.window_right = WindowValue::TopLeftAligned;
      } else if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromBottomRight) == custom_mask_type) {
        params.window_left = WindowValue::BottomRightAligned;
        params.window_right = WindowValue::BottomRightAligned;
      }
#if AOTRITON_ALWAYS_V3_API
      using sdp::aotriton_adapter::mklazy_empty_like;
      using sdp::aotriton_adapter::mklazy_fp32zeros;
      using sdp::aotriton_adapter::LazyTensorContext;
      LazyTensorContext lazy_delta { .like_tensor = softmax_lse, .tensor_name = "delta" };
      LazyTensorContext lazy_dq_acc { .like_tensor = dq_t, .tensor_name = "dq_acc" };
      params.D = mklazy_empty_like<2>(&lazy_delta);
      params.DQ_ACC = mklazy_fp32zeros<4>(&lazy_dq_acc);
#else
      at::Tensor delta = at::empty_like(softmax_lse).contiguous();
      params.D = mk_aotensor<2>(delta, "delta");
#endif
      if (cu_seqlens_q.has_value()) {
        params.varlen_type = VarlenType::CompactVarlen;
        params.cu_seqlens_q = mk_aotensor<1>(cu_seqlens_q.value(), "cu_seqlens_q");
        params.cu_seqlens_k = mk_aotensor<1>(cu_seqlens_k.value(), "cu_seqlens_k");
      } else {
        params.varlen_type = VarlenType::None;
      }
      err = aotriton::v3::flash::attn_bwd(params,
                                          aotriton::v3::flash::attn_bwd_params::kVersion,
                                          stream);
#endif  // AOTRITON_V3_API
    } else if (cu_seqlens_q.has_value()) {
      at::Tensor delta = at::empty_like(softmax_lse).contiguous();
      // varlen aka Nested tensor
      err = attn_bwd_compact_varlen(mk_aotensor(q_t, "q"),
                                    mk_aotensor(k_t, "k"),
                                    mk_aotensor(v_t, "v"),
                                    mk_aotensor<1>(cu_seqlens_q.value(), "cu_seqlens_q"),
                                    mk_aotensor<1>(cu_seqlens_k.value(), "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                                    softmax_scale,
                                    mk_aotensor(out_t, "out"),
                                    mk_aotensor(dout_t, "dout"),
                                    mk_aotensor(dq_t, "dq"),
                                    mk_aotensor(dk_t, "dk"),
                                    mk_aotensor(dv_t, "dv"),
                                    bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                                    mk_aotensor<2>(softmax_lse, "L"),
                                    mk_aotensor<2>(delta, "delta"),
                                    float(dropout_p),
                                    mk_aoscalartensor(philox_seed),
                                    mk_aoscalartensor(philox_offset),
                                    0,
                                    is_causal,
                                    stream);
    } else { // cu_seqlens.has_value
      auto d_head = Kv;
      bool use_fused_bwd = d_head <= 192 && d_head * max_seqlen_q < 64 * 512;
      if (use_fused_bwd) {
        err = attn_bwd_fused(mk_aotensor(q_t, "q"),
                             mk_aotensor(k_t, "k"),
                             mk_aotensor(v_t, "v"),
                             bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                             softmax_scale,
                             mk_aotensor(out_t, "out"),
                             mk_aotensor(dout_t, "dout"),
                             mk_aotensor(dq_t, "dq"),
                             mk_aotensor(dk_t, "dk"),
                             mk_aotensor(dv_t, "dv"),
                             bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                             mk_aotensor<2>(softmax_lse, "L"),
                             float(dropout_p),
                             mk_aoscalartensor(philox_seed),
                             mk_aoscalartensor(philox_offset),
                             0,
                             is_causal,
                             stream);
      } else {
        at::Tensor delta = at::empty_like(softmax_lse).contiguous();
        err = attn_bwd(mk_aotensor(q_t, "q"),
                     mk_aotensor(k_t, "k"),
                     mk_aotensor(v_t, "v"),
                     bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                     softmax_scale,
                     mk_aotensor(out_t, "out"),
                     mk_aotensor(dout_t, "dout"),
                     mk_aotensor(dq_t, "dq"),
                     mk_aotensor(dk_t, "dk"),
                     mk_aotensor(dv_t, "dv"),
                     bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                     mk_aotensor<2>(softmax_lse, "L"),
                     mk_aotensor<2>(delta, "delta"),
                     float(dropout_p),
                     mk_aoscalartensor(philox_seed),
                     mk_aoscalartensor(philox_offset),
                     0,
                     is_causal,
                     stream);
      } //used_fused_bwd
    } // cuseqlen.has_value
#else  // DISABLE_AOTRITON
    TORCH_CHECK(false, "Attempting to use aotriton mem_eff_backward backend in a build that has not built AOTriton");
#endif
  } // Use CK
#else // USE_CUDA
  at::Tensor workspace;
  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  int computeCapability = p->major * 10 + p->minor;
  if (computeCapability == 121) {
    computeCapability = 120;
  }

  bool kernel_launched = false;
  const auto maxK = std::max(query.size(3), value.size(3));
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (Kernel::kMaxK < maxK) {
      return;
    }
    // Dropout must be supported if we need it
    if (use_dropout && !Kernel::kApplyDropout) {
      return;
    }
    if (Kernel::kKeysQueriesAlignedToBlockSize &&
        (cu_seqlens_q.has_value() || M % Kernel::kBlockSizeI ||
         N % Kernel::kBlockSizeJ)) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kMinimumAlignment) ||
        (key.stride(2) % Kernel::kMinimumAlignment) ||
        (value.stride(2) % Kernel::kMinimumAlignment)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }

    kernel_launched = true;

    // TODO: Fuse this into a kernel?
    // This is a bottleneck for smaller sequences (M <= 128)
    auto delta = Kernel::kKernelComputesDelta
        ? at::empty({B, nH, M}, query.options().dtype(at::ScalarType::Float))
        : (grad_out.to(at::kFloat) * out.to(at::kFloat))
              .sum(-1)
              .transpose(-2, -1)
              .contiguous();
    TORCH_INTERNAL_ASSERT(delta.size(0) == B);
    TORCH_INTERNAL_ASSERT(delta.size(1) == nH);
    TORCH_INTERNAL_ASSERT(delta.size(2) == M);

    typename Kernel::Params p;
    p.query_ptr = (const scalar_t*)query.const_data_ptr();
    p.key_ptr = (const scalar_t*)key.const_data_ptr();
    p.value_ptr = (const scalar_t*)value.const_data_ptr();
    p.logsumexp_ptr = (typename Kernel::lse_scalar_t const *)logsumexp.const_data_ptr();
    p.output_ptr = (const scalar_t*)out.const_data_ptr();
    p.grad_output_ptr = (const scalar_t*)grad_out.const_data_ptr();
    p.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
    p.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
    p.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
    p.delta_ptr = (float*)delta.data_ptr();
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = cu_seqlens_q.has_value() ? cu_seqlens_q->size(0) - 1 : B;
    p.num_heads = nH;
    p.custom_mask_type = custom_mask_type;
    p.scale = sdp::calculate_scale(query, scale).expect_float();
    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (const int32_t*)cu_seqlens_q->const_data_ptr();
      p.cu_seqlens_k_ptr = (const int32_t*)cu_seqlens_k->const_data_ptr();
    }
    if (window_size.has_value()) {
      p.window_size = *window_size;
    }

    ASSIGN_CHECK_OVERFLOW(p.lse_strideB, logsumexp.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.lse_strideH, logsumexp.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideB, grad_out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideM, grad_out.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideH, grad_out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.o_strideB, out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.o_strideH, out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.gQ_strideB, grad_q.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideB, grad_k.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideB, grad_v.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gQ_strideH, grad_q.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideH, grad_k.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideH, grad_v.stride(2));
    p.gQKV_strideM_multiplier = shared_storage_dqdkdv ? 3 : 1;
    TORCH_INTERNAL_ASSERT(p.gQ_strideM() == grad_q.stride(1));
    TORCH_INTERNAL_ASSERT(p.gK_strideM() == grad_k.stride(1));
    TORCH_INTERNAL_ASSERT(p.gV_strideM() == grad_v.stride(1));

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideB, delta.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideH, delta.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");

      p.bias_ptr = (scalar_t*)bias->data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));

      if (bias_requires_grad) {
        p.grad_bias_ptr = (scalar_t*)grad_bias.data_ptr();

        ASSIGN_CHECK_OVERFLOW(p.gB_strideB, grad_bias.stride(0));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideH, grad_bias.stride(1));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideM, grad_bias.stride(2));
      }
    }

    if (use_dropout) {
      p.rng_engine_inputs = rng_engine_inputs;
      p.dropout_prob = dropout_p;
    }

    // Heuristic for finding optimal number of splits
    auto parallelism_without_split_key =
        p.getBlocksGrid().x * p.getBlocksGrid().y * p.getBlocksGrid().z;
    p.num_splits_key = cutlass::ceil_div(p.num_keys, Kernel::kBlockSizeJ);
    if (num_splits_key.has_value()) {
      p.num_splits_key =
          std::min<int64_t>(p.num_splits_key, num_splits_key.value());
    } else {
      // Keys splitting heuristic

      // If we already have enough parallelism, split-keys can help
      // better use L2 cache.
      // This is negligible when the seqlen is too small tho
      if (parallelism_without_split_key >= 256 &&
          p.num_keys <= 2 * Kernel::kBlockSizeJ) {
        p.num_splits_key = 1;
      }
      // Increasing `split_keys` leads to using more gmem for temporary storage
      // when we need a staging area for gK/gV. let's avoid that
      if (Kernel::kNeedsAccumGradK || Kernel::kNeedsAccumGradV) {
        p.num_splits_key = std::min(
            int32_t(p.num_splits_key), 200 / ((int32_t)(p.num_batches * p.num_heads)));
      }
    }
    if (!Kernel::kEnableSplitKeys || p.num_splits_key < 1) {
      p.num_splits_key = 1;
    }

    auto& ctx = at::globalContext();
    if (ctx.deterministicAlgorithms()) {
      if (ctx.deterministicAlgorithmsWarnOnly()) {
        TORCH_WARN_ONCE(
            "Memory Efficient attention defaults to a non-deterministic algorithm. ",
            "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
      } else {
        TORCH_CHECK(
            num_splits_key.value_or(1) <= 1,
            "Using `num_splits_key > 1` makes the algorithm non-deterministic, and pytorch's deterministic mode is enabled");
        p.num_splits_key = 1;
      }
    }
    int64_t size_bytes = p.workspace_size();
    if (size_bytes) {
      workspace =
          at::empty({size_bytes}, query.options().dtype(at::ScalarType::Byte));
      p.workspace = (float*)workspace.data_ptr();
      if (p.should_zero_workspace()) {
        workspace.zero_();
      }
    }

    // Handle the edge-cases where some tensors are empty
    if (p.num_queries == 0 || p.num_keys == 0 || p.num_batches == 0 ||
        p.num_heads == 0) {
      grad_k.zero_();
      grad_v.zero_();
      grad_q.zero_();
      return;
    }
    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }

    // second syntax resulted in the error below on windows
    // error C3495: 'kernel_fn': a simple capture must be a variable
    // with automatic storage duration declared
    // in the reaching scope of the lambda
#ifdef _WIN32
    cudaFuncAttributes attr;
    AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    TORCH_INTERNAL_ASSERT(
        attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability,
        "Something went wrong in the build process");
#else
    auto checkBinaryArchMatches = [&]() {
      cudaFuncAttributes attr;
      AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    };
    TORCH_INTERNAL_ASSERT(
        checkBinaryArchMatches(), "Something went wrong in the build process");
#endif

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassB<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassB: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());
#endif // USE_ROCM
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), std::move(grad_bias));
  #endif // defined(USE_MEM_EFF_ATTENTION)
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_cuda(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale){
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  Tensor grad_out_t = grad_out_.transpose(1,2);
  Tensor out_t = out.transpose(1,2);

  auto [grad_q, grad_k, grad_v] = at::_flash_attention_backward(
    grad_out_t,
    q_t,
    k_t,
    v_t,
    out_t,
    logsumexp,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_seqlen_batch_q,
    max_seqlen_batch_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale);

  grad_q = grad_q.transpose(1,2);
  grad_k = grad_k.transpose(1,2);
  grad_v = grad_v.transpose(1,2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_efficient_attention_backward_cuda(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool causal,
    std::optional<double> scale) {

  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  constexpr int64_t MAX_BATCH_SIZE = (1LL << 16) - 1;
  int64_t batch_size = query.size(0);

  if (batch_size > MAX_BATCH_SIZE) {
    TORCH_CHECK(dropout_p == 0.0,
                "Efficient attention backward cannot handle dropout when "
                "the batch size exceeds (", MAX_BATCH_SIZE, ").");
  }
  auto grad_out_t = grad_out_.transpose(1, 2);
  auto query_t = query.transpose(1, 2);
  auto key_t = key.transpose(1, 2);
  auto value_t = value.transpose(1, 2);
  auto out_t = out.transpose(1, 2);

  auto process_chunk = [&](const Tensor& grad_out_chunk,
                          const Tensor& query_chunk,
                          const Tensor& key_chunk,
                          const Tensor& value_chunk,
                          const std::optional<Tensor>& attn_bias_chunk,
                          const Tensor& out_chunk,
                          const Tensor& logsumexp_chunk)
      -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<Tensor> kernel_bias;
  if (attn_bias_chunk.has_value() && attn_bias_chunk.value().defined()) {
    kernel_bias = attn_bias_chunk.value();
  }
  // Will add with signauter changes for dropout and bias
  // We are only handling Dense inputs, but this should be passed
  // from forward to backward
  int64_t max_seqlen_q = query_chunk.size(2);
  int64_t max_seqlen_k = key_chunk.size(2);

  sdp::CustomMaskType custom_mask_type = causal
    ? sdp::CustomMaskType::CausalFromTopLeft
    : sdp::CustomMaskType::NoCustomMask;
  auto [grad_q, grad_k, grad_v, grad_bias] =
      at::_efficient_attention_backward(
          grad_out_chunk,
          query_chunk,
          key_chunk,
          value_chunk,
          kernel_bias,
          out_chunk,
          std::nullopt,
          std::nullopt,
          max_seqlen_q,
          max_seqlen_k,
          logsumexp_chunk,
          dropout_p,
          philox_seed,
          philox_offset,
          static_cast<int64_t>(custom_mask_type),
          grad_input_mask[3],
          scale,
          std::nullopt);  // num_split_keys
  return std::make_tuple(
      grad_q.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2), std::move(grad_bias));
  };

  // process in chunks if batch size exceeds maximum
  if (batch_size > MAX_BATCH_SIZE) {
    Tensor final_grad_q, final_grad_k, final_grad_v, final_grad_bias;

    auto create_strided_output = [batch_size](const Tensor& tensor) -> Tensor {
      if (!tensor.defined()) {
        return Tensor{};
      }
      int dim = tensor.dim();
      std::vector<int64_t> sizes;
      sizes.reserve(dim);
      sizes.push_back(batch_size);
      for (int i = 1; i < dim; i++) {
        sizes.push_back(tensor.size(i));
      }
      return at::empty_strided(std::move(sizes), tensor.strides(), tensor.options());
    };

    if (grad_input_mask[0]) {
      final_grad_q = create_strided_output(query);
    }

    if (grad_input_mask[1]) {
      final_grad_k = create_strided_output(key);
    }

    if (grad_input_mask[2]) {
      final_grad_v = create_strided_output(value);
    }
    if (grad_input_mask[3] && attn_bias.defined()) {
      final_grad_bias = at::zeros_like(attn_bias);
    }

    for (int64_t start = 0; start < batch_size; start += MAX_BATCH_SIZE) {
      int64_t end = std::min(start + MAX_BATCH_SIZE, batch_size);

      Tensor grad_out_chunk = grad_out_t.slice(0, start, end);
      Tensor query_chunk = query_t.slice(0, start, end);
      Tensor key_chunk = key_t.slice(0, start, end);
      Tensor value_chunk = value_t.slice(0, start, end);
      Tensor attn_bias_chunk;
      if (attn_bias.defined()) {
        attn_bias_chunk = attn_bias.slice(0, start, end);
      } else {
        attn_bias_chunk.reset();
      }
      Tensor out_chunk = out_t.slice(0, start, end);
      Tensor logsumexp_chunk = logsumexp.numel() > 0 ? logsumexp.slice(0, start, end) : logsumexp;

      auto [chunk_grad_q, chunk_grad_k, chunk_grad_v, chunk_grad_bias] =
          process_chunk(grad_out_chunk, query_chunk, key_chunk, value_chunk,
                      attn_bias_chunk, out_chunk, logsumexp_chunk);

      if (grad_input_mask[0] && chunk_grad_q.defined()) {
        final_grad_q.slice(0, start, end).copy_(chunk_grad_q);
      }
      if (grad_input_mask[1] && chunk_grad_k.defined()) {
        final_grad_k.slice(0, start, end).copy_(chunk_grad_k);
      }
      if (grad_input_mask[2] && chunk_grad_v.defined()) {
        final_grad_v.slice(0, start, end).copy_(chunk_grad_v);
      }
      if (grad_input_mask[3] && chunk_grad_bias.defined()) {
        final_grad_bias.add_(chunk_grad_bias);
      }
    }

    return std::make_tuple(
        std::move(final_grad_q),
        std::move(final_grad_k),
        std::move(final_grad_v),
        std::move(final_grad_bias));
  }
  // when batch size is within allowed size, no chunking needed
  else {
    std::optional<Tensor> attn_bias_opt;
    if (attn_bias.defined()) {
      attn_bias_opt = attn_bias;
    }
    return process_chunk(grad_out_t, query_t, key_t, value_t, attn_bias_opt, out_t, logsumexp);
  }
}

std::tuple<Tensor, Tensor, Tensor> _scaled_dot_product_cudnn_attention_backward_cuda(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& attn_bias,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
        return at::_cudnn_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
            attn_bias,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            scale);
}

} // namespace at::native
