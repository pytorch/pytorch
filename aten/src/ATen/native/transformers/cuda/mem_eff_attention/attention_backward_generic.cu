#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_generic(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    int64_t rng_seed,
    int64_t rng_offset,
    bool causal) {
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(grad_out_.is_cuda(), "grad_out must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");
  TORCH_CHECK(!grad_out_.is_sparse(), "grad_out must be a dense tensor");

  // TODO drop this limitation in the future
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  // TODO: support other dtypes in the future
  // TORCH_CHECK(
  //     query.scalar_type() == at::ScalarType::Half,
  //     "Only f16 type is supported for now");

  at::cuda::CUDAGuard device_guard(query.device());

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor grad_q = at::zeros_like(query);
  at::Tensor grad_k = at::zeros_like(key);
  at::Tensor grad_v = at::zeros_like(value);

  cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = properties->major * 10 + properties->minor;

#define DISPATCH_ARCHTAG(func)                                            \
  {                                                                       \
    if (computeCapability >= 80) {                                        \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (computeCapability >= 75) {                                 \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (computeCapability >= 70) {                                 \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (computeCapability >= 50) {                                 \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      TORCH_CHECK(                                                        \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }

#define DISPATCH_TYPES(func)                                          \
  {                                                                   \
    if (query.scalar_type() == at::ScalarType::Float) {               \
      using scalar_t = float;                                         \
      func();                                                         \
    } else if (query.scalar_type() == at::ScalarType::Half) {         \
      using scalar_t = cutlass::half_t;                               \
      func();                                                         \
    } else {                                                          \
      TORCH_CHECK(false, "Only fp32 & half supported at the moment"); \
    }                                                                 \
  }

  DISPATCH_TYPES(([&]() {
    bool isAligned;
    DISPATCH_ARCHTAG(([&]() {
      using AlignedAK = AttentionBackwardKernel<scalar_t, true, ArchTag>;
      isAligned =
          (query.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           key.stride(1) % AlignedAK::kOptimalAlignement == 0 &&
           value.stride(1) % AlignedAK::kOptimalAlignement == 0);
      // TODO: Should we warn or log somewhere when we use a less efficient
      // kernel due to wrong alignment?

      DISPATCH_BOOL(
          isAligned, kIsAligned, ([&]() {
            using AK = AttentionBackwardKernel<scalar_t, kIsAligned, ArchTag>;
            size_t smem_bytes = sizeof(typename AK::SharedStorage);
            // Might happen on Sm80/half, where the minimum alignment is 32bits
            TORCH_CHECK(
                query.stride(1) % AK::kMinimumAlignment == 0,
                "query is not correctly aligned");
            TORCH_CHECK(
                key.stride(1) % AK::kMinimumAlignment == 0,
                "key is not correctly aligned");
            TORCH_CHECK(
                value.stride(1) % AK::kMinimumAlignment == 0,
                "value is not correctly aligned");

            AK::Params params;
            params.query_ptr = (scalar_t*)query.data_ptr();
            params.key_ptr = (scalar_t*)key.data_ptr();
            params.value_ptr = (scalar_t*)value.data_ptr();
            params.logsumexp_ptr =
                (typename AK::lse_scalar_t*)logsumexp.data_ptr();
            params.output_ptr = (scalar_t*)out.data_ptr();
            params.grad_output_ptr = (scalar_t*)grad_out.data_ptr();
            params.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
            params.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
            params.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
            params.head_dim = query.size(2);
            params.head_dim_value = value.size(2);
            params.num_queries = query.size(1);
            params.num_keys = key.size(1);
            params.num_batches = B;
            params.causal = causal;

            constexpr auto kernel_fn = attention_kernel_backward_batched<AK>;

            if (smem_bytes > 0xc000) {
              TORCH_INTERNAL_ASSERT(
                  computeCapability >= 70,
                  "This kernel requires too much shared memory on this machine!");
              cudaFuncSetAttribute(
                  kernel_fn,
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes);
            }

            auto checkBinaryArchMatches = [&]() {
              cudaFuncAttributes attr;
              AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
              return attr.binaryVersion >= ArchTag::kMinComputeCapability;
            };
            TORCH_INTERNAL_ASSERT(
                checkBinaryArchMatches(),
                "Something went wrong in the build process");

            kernel_fn<<<
                params.getBlocksGrid(),
                params.getThreadsGrid(),
                smem_bytes>>>(params);
            AT_CUDA_CHECK(cudaGetLastError());
          }));
    }));
  }));
  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace
