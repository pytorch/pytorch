#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>

#define DISPATCH_MAXK(func)                                   \
  {                                                           \
    const auto maxK = std::max(query.size(2), value.size(2)); \
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
                      (QUERY.stride(1) % AlignedAK::kOptimalAlignement == 0 && \
                       KEY.stride(1) % AlignedAK::kOptimalAlignement == 0 &&   \
                       VALUE.stride(1) % AlignedAK::kOptimalAlignement == 0);  \
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

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_cutlass(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    bool causal) {
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));
  TORCH_CHECK(value.size(2) == grad_out_.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();

  CHECK_NOSPARSE_CONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(value);
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(grad_out);

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  // It does not make sense to use that in practice,
  // but let's still make sure we are correct
  // As we iterate through keys first, we skip
  // keys with no query associated, so they are not
  // initialized
  bool grad_kv_needs_init = causal && N > M;
  at::Tensor grad_q = at::empty_like(query);
  at::Tensor grad_k =
      grad_kv_needs_init ? at::zeros_like(key) : at::empty_like(key);
  at::Tensor grad_v =
      grad_kv_needs_init ? at::zeros_like(value) : at::empty_like(value);

  auto launchKernel = [&](auto _k, int computeCapability) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);

    // TODO: Fuse this into a kernel?
    // This is a bottleneck for smaller sequences (M <= 128)
    auto delta = Kernel::kKernelComputesDelta
        ? at::empty({B, M}, query.options().dtype(at::ScalarType::Float))
        : (grad_out.to(at::kFloat) * out.to(at::kFloat)).sum(-1);
    TORCH_INTERNAL_ASSERT(delta.size(0) == B);
    TORCH_INTERNAL_ASSERT(delta.size(1) == M);

    typename Kernel::Params params;
    params.query_ptr = (scalar_t*)query.data_ptr();
    params.key_ptr = (scalar_t*)key.data_ptr();
    params.value_ptr = (scalar_t*)value.data_ptr();
    params.logsumexp_ptr = (typename Kernel::lse_scalar_t*)logsumexp.data_ptr();
    params.output_ptr = (scalar_t*)out.data_ptr();
    params.grad_output_ptr = (scalar_t*)grad_out.data_ptr();
    params.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
    params.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
    params.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
    params.delta_ptr = (float*)delta.data_ptr();
    params.head_dim = query.size(2);
    params.head_dim_value = value.size(2);
    params.num_queries = query.size(1);
    params.num_keys = key.size(1);
    params.num_batches = B;
    params.causal = causal;
    Kernel::check_supported(params);

    constexpr auto kernel_fn = attention_kernel_backward_batched<Kernel>;

    if (smem_bytes > 0xc000) {
      TORCH_INTERNAL_ASSERT(
          computeCapability >= 70,
          "This kernel requires too much shared memory on this machine!");
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    auto checkBinaryArchMatches = [&]() {
      cudaFuncAttributes attr;
      AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    };
    TORCH_INTERNAL_ASSERT(
        checkBinaryArchMatches(), "Something went wrong in the build process");

    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes>>>(
        params);
  };

  DISPATCH_KERNEL(
      query, key, value, ([&] { launchKernel(Kernel{}, computeCapability); }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_q, grad_k, grad_v);
} // namespace

} // namespace

// TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
//   m.impl(
//       TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_cutlass"),
//       TORCH_FN(mem_efficient_attention_backward_cutlass));
// }
