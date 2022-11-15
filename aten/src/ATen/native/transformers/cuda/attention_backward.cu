#include <type_traits>

#include <ATen/ATen.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>

#ifdef USE_FLASH_ATTENTION
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#endif

#define ASSIGN_CHECK_OVERFLOW(A, B)                                            \
  {                                                                            \
    A = B;                                                                     \
    TORCH_CHECK(B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

#define DISPATCH_MAXK(func)                                   \
  {                                                           \
    const auto maxK = std::max(query.size(3), value.size(3)); \
    if (maxK <= 64) {                                         \
      constexpr int kMaxK = 64;                               \
      func();                                                 \
    } else if (maxK <= 128) {                                 \
      constexpr int kMaxK = 128;                              \
      func();                                                 \
    } else {                                                  \
      constexpr int kMaxK = std::numeric_limits<int>::max();  \
      func();                                                 \
    }                                                         \
  }

#define DISPATCH_KERNEL(QUERY, KEY, VALUE, FUNC)                               \
  {                                                                            \
    cudaDeviceProp* properties =                                               \
        at::cuda::getDeviceProperties(QUERY.device().index());                 \
    const int computeCapability = properties->major * 10 + properties->minor;  \
    DISPATCH_MAXK(([&] {                                                       \
      DISPATCH_TYPES(                                                          \
          QUERY, ([&]() {                                                      \
            DISPATCH_ARCHTAG(                                                  \
                computeCapability, ([&]() {                                    \
                  using AlignedAK =                                            \
                      AttentionBackwardKernel<ArchTag, scalar_t, true, kMaxK>; \
                  bool isAligned =                                             \
                      (QUERY.stride(2) % AlignedAK::kOptimalAlignement == 0 && \
                       KEY.stride(2) % AlignedAK::kOptimalAlignement == 0 &&   \
                       VALUE.stride(2) % AlignedAK::kOptimalAlignement == 0);  \
                  DISPATCH_BOOL(isAligned, kIsAligned, ([&]() {                \
                                  using Kernel = AttentionBackwardKernel<      \
                                      ArchTag,                                 \
                                      scalar_t,                                \
                                      kIsAligned,                              \
                                      kMaxK>;                                  \
                                  FUNC();                                      \
                                }))                                            \
                }))                                                            \
          }))                                                                  \
    }));                                                                       \
  }

namespace at {

namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> _efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    bool causal) {
  #if defined(USE_FLASH_ATTENTION)
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
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

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);

  // It does not make sense to use that in practice,
  // but let's still make sure we are correct
  // As we iterate through keys first, we skip
  // keys with no query associated, so they are not
  // initialized
  bool grad_kv_needs_init = causal && N > M;
  at::Tensor grad_q, grad_k, grad_v;
  if (!grad_kv_needs_init && query.size(1) == key.size(1) &&
      query.size(3) == value.size(3) &&
      query.storage().is_alias_of(key.storage()) &&
      query.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else {
    grad_q = at::empty_like(query);
    grad_k = grad_kv_needs_init ? at::zeros_like(key) : at::empty_like(key);
    grad_v = grad_kv_needs_init ? at::zeros_like(value) : at::empty_like(value);
  }

  auto launchKernel = [&](auto _k, int computeCapability) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);

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
    p.num_queries = query.size(1);
    p.num_keys = key.size(1);
    p.num_batches = B;
    p.num_heads = nH;
    p.causal = causal;

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
    p.gQKV_strideM_multiplier = grad_q.is_contiguous() ? 1 : 3;
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

    Kernel::check_supported(p);

    constexpr auto kernel_fn = attention_kernel_backward_batched<Kernel>;

    if (smem_bytes > 0xc000) {
      TORCH_INTERNAL_ASSERT(
          computeCapability >= 70,
          "This kernel requires too much shared memory on this machine!");
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
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

  DISPATCH_KERNEL(
      query, key, value, ([&] { launchKernel(Kernel{}, computeCapability); }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_q, grad_k, grad_v);
  #endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
}

} // namespace native
} // namespace at
