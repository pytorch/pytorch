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
// MemoryEfficient Attention Specific Imports
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h>
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
    c10::optional<double> scale) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale = sdp::calculate_scale(query, scale).as_float_unchecked();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

  c10::optional<at::Tensor> dq{c10::nullopt};
  c10::optional<at::Tensor> dk{c10::nullopt};
  c10::optional<at::Tensor> dv{c10::nullopt};

  //  The kernel computes irregardless we will drop for this functions return
  Tensor grad_softmax;

  // Currently unused args:
  c10::optional<at::Tensor> alibi_slopes{c10::nullopt};

  bool determinisitic{false};
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      determinisitic = true;
    }
  }

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense forward
  if (cumulative_sequence_length_q.defined()) {
    // Varlen forward
    auto [dQuery, dKey, dValue, dSoftmax] = pytorch_flash::mha_varlen_bwd(
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
        -1, /*window_size_left*/
        -1, /*window_size_right*/
        determinisitic,
        philox_seed,
        philox_offset);
    return std::make_tuple(dQuery, dKey, dValue);
  } else {
    // Dense forward
    auto [dQuery, dKey, dValue, dSoftmax] = pytorch_flash::mha_bwd(
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
        -1, /*window_size_left*/
        -1, /*window_size_right*/
        determinisitic,
        philox_seed,
        philox_offset);
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& kernel_bias, // additive attention bias
    const at::Tensor& out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_k_dummy,
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
    const c10::optional<double> scale,
    c10::optional <int64_t> num_splits_key) {
  #if defined(USE_MEM_EFF_ATTENTION)
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  // This path is used when we directly call _efficient_attention_forward
  // from python.
  // This is needed because SaveVariable automatically converts
  // c10::optional to undefined tensor
  c10::optional<Tensor> bias, cu_seqlens_q, cu_seqlens_k;
  bias = kernel_bias.has_value() && !kernel_bias->defined() ? c10::nullopt : kernel_bias;
  cu_seqlens_q = cu_seqlens_q_dummy.has_value() && !cu_seqlens_q_dummy->defined() ? c10::nullopt : cu_seqlens_q_dummy;
  cu_seqlens_k = cu_seqlens_k_dummy.has_value() && !cu_seqlens_k_dummy->defined() ? c10::nullopt : cu_seqlens_k_dummy;

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
  grad_q = at::empty(query.sizes(), query.options());
  grad_k = at::empty(key.sizes(), key.options());
  grad_v = at::empty(value.sizes(), value.options());

  if (bias_requires_grad) {
    // force alignment for the last dim
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    grad_bias = at::empty(sz, bias->options())
                    .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
  }
  at::Tensor workspace;

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

  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = p->major * 10 + p->minor;

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
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = (typename Kernel::lse_scalar_t*)logsumexp.data_ptr();
    p.output_ptr = (scalar_t*)out.data_ptr();
    p.grad_output_ptr = (scalar_t*)grad_out.data_ptr();
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
    p.scale = sdp::calculate_scale(query, scale).as_float_unchecked();
    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
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
    // We removed the chunk/cat optimization and the multiplier is always 1
    p.gQKV_strideM_multiplier = 1;
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
    if (num_splits_key.has_value()) { // Skip heuristic, if user provided an explicit value
      p.num_splits_key = std::max<int64_t>(p.num_splits_key, num_splits_key.value());
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
            int(p.num_splits_key), 200 / (p.num_batches * p.num_heads));
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
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), std::move(grad_bias));
  #endif
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
    c10::optional<double> scale){
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
    c10::optional<double> scale) {

  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  auto grad_out = grad_out_.transpose(1, 2);
  auto out_t = out.transpose(1, 2);
  auto q_t = query.transpose(1, 2);
  auto k_t = key.transpose(1, 2);
  auto v_t = value.transpose(1, 2);

  Tensor grad_q, grad_k, grad_v, grad_bias;

  // This is needed because SaveVariable automatically converts
  // c10::optional to undefined tensor
  c10::optional<Tensor> kernel_bias;
  if (attn_bias.defined()) {
    kernel_bias = attn_bias;
  }
  // Will add with signauter changes for dropout and bias
  // We are only handling Dense inputs, but this should be passed
  // from forward to backward
  int64_t max_seqlen_q = q_t.size(1);
  int64_t max_seqlen_k = k_t.size(1);

  sdp::CustomMaskType custom_mask_type = causal
    ? sdp::CustomMaskType::CausalFromTopLeft
    : sdp::CustomMaskType::NoCustomMask;
  std::tie(grad_q, grad_k, grad_v, grad_bias) =
      at::_efficient_attention_backward(
          grad_out,
          q_t,
          k_t,
          v_t,
          kernel_bias,
          out_t,
          c10::nullopt,
          c10::nullopt,
          max_seqlen_q,
          max_seqlen_k,
          logsumexp,
          dropout_p,
          philox_seed,
          philox_offset,
          static_cast<int64_t>(custom_mask_type),
          grad_input_mask[3],
          scale,
          c10::nullopt);  // num_split_keys
  return std::make_tuple(
      grad_q.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2), grad_bias);
}

} // namespace at::native
