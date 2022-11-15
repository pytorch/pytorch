#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>


#define DISPATCH_BLOCKSIZE(VALUE_HEAD_DIM, FN)        \
  {                                                   \
    if (VALUE_HEAD_DIM <= 64) {                       \
      constexpr bool kIs64x64 = true;                 \
      constexpr bool kSingleValueIteration = true;    \
      FN();                                           \
    } else {                                          \
      constexpr bool kIs64x64 = false;                \
      if (VALUE_HEAD_DIM <= 128) {                    \
        constexpr bool kSingleValueIteration = true;  \
        FN();                                         \
      } else {                                        \
        constexpr bool kSingleValueIteration = false; \
        FN();                                         \
      }                                               \
    }                                                 \
  }

#define DISPATCH_KERNEL(QUERY, KEY, VALUE, FUNC)                              \
  {                                                                           \
    cudaDeviceProp* properties =                                              \
        at::cuda::getDeviceProperties(QUERY.device().index());                \
    const int computeCapability = properties->major * 10 + properties->minor; \
    DISPATCH_BLOCKSIZE(                                                       \
        VALUE.size(-1), ([&]() {                                              \
          static constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;     \
          static constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;       \
          DISPATCH_TYPES(                                                     \
              QUERY, ([&]() {                                                 \
                DISPATCH_ARCHTAG(                                             \
                    computeCapability, ([&]() {                               \
                      using AlignedAK = AttentionKernel<                      \
                          scalar_t,                                           \
                          ArchTag,                                            \
                          true,                                               \
                          kQueriesPerBlock,                                   \
                          kKeysPerBlock,                                      \
                          kSingleValueIteration>;                             \
                      /* Run a more efficient kernel (with `isAligned=True`)  \
                      if memory is correctly aligned*/                        \
                      bool isAligned =                                        \
                          (QUERY.stride(2) % AlignedAK::kAlignmentQ == 0 &&   \
                           KEY.stride(2) % AlignedAK::kAlignmentK == 0 &&     \
                           VALUE.stride(2) % AlignedAK::kAlignmentV == 0);    \
                      /* TODO: Should we warn or log somewhere when we use a  \
                      less efficient kernel due to wrong alignment? */        \
                      DISPATCH_BOOL(isAligned, kIsAligned, ([&]() {           \
                                      using Kernel = AttentionKernel<         \
                                          scalar_t,                           \
                                          ArchTag,                            \
                                          kIsAligned,                         \
                                          kQueriesPerBlock,                   \
                                          kKeysPerBlock,                      \
                                          kSingleValueIteration>;             \
                                      FUNC();                                 \
                                    }))                                       \
                    }))                                                       \
              }));                                                            \
        }));                                                                  \
  }

namespace {
/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<at::Tensor, at::Tensor> efficient_attention_forward_cutlass(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_q,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    bool compute_logsumexp,
    bool causal) {
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q, max_seqlen_k;
  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k = 0; // Will be set inside the kernel
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor res;
  at::Tensor logsumexp;

  auto launchKernel = [&](auto _k, int computeCapability) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    res = at::empty(
        {B, M, num_heads, Kv},
        query.options().dtype(
            TypeTraits<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              TypeTraits<typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
    }

#define ASSIGN_CHECK_OVERFLOW(A, B)                                            \
  {                                                                            \
    A = B;                                                                     \
    TORCH_CHECK(B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = cu_seqlens_q.has_value() ? cu_seqlens_q->size(0) - 1 : B;
    p.causal = causal;

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));

    constexpr auto kernel_fn = attention_kernel_batched<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > 0xc000) {
      TORCH_INTERNAL_ASSERT(
          computeCapability >= 70,
          "This kernel requires too much shared memory on this machine!");
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    Kernel::check_supported(p);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
  };
  // Dispatch to the right kernel
  DISPATCH_KERNEL(query, key, value, ([&]() {
                    launchKernel(Kernel{}, computeCapability);
                  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
}
} // namespace

// TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
//   m.impl(
//       TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_cutlass"),
//       TORCH_FN(efficient_attention_forward_cutlass));
// }
