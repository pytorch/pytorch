#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

namespace at {
namespace native {
std::tuple<at::Tensor, at::Tensor>
efficient_attention_forward_generic(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    bool causal) {
  TORCH_CHECK(p == 0.0, "Dropout is not supported at the moment");
  TORCH_CHECK(
      !attn_bias_.has_value(), "attn_bias is not supported at the moment");

  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(key.dim() == 3);
  TORCH_CHECK(value.dim() == 3);

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res;
  at::Tensor logsumexp;

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
// Dispatch to the right kernel
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

#define DISPATCH_BLOCKSIZE(VALUE_HEAD_DIM, BLOCK_6464, SINGLE_VALUE_ITER, FN) \
  {                                                                           \
    if (VALUE_HEAD_DIM <= 64) {                                               \
      constexpr bool BLOCK_6464 = true;                                       \
      constexpr bool SINGLE_VALUE_ITER = true;                                \
      FN();                                                                   \
    } else {                                                                  \
      constexpr bool BLOCK_6464 = false;                                      \
      if (VALUE_HEAD_DIM <= 128) {                                            \
        constexpr bool SINGLE_VALUE_ITER = true;                              \
        FN();                                                                 \
      } else {                                                                \
        constexpr bool SINGLE_VALUE_ITER = false;                             \
        FN();                                                                 \
      }                                                                       \
    }                                                                         \
  }

  DISPATCH_BLOCKSIZE(
      value.size(2), kIs64x64, kSingleValueIteration, ([&]() {
        static constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;
        static constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;
        DISPATCH_TYPES(([&]() {
          DISPATCH_ARCHTAG(([&]() {
            // Run a more efficient kernel (with `isAligned=True`) if
            // memory is correctly aligned
            bool isAligned;
            using AlignedAK = AttentionKernel<
                scalar_t,
                ArchTag,
                true,
                kQueriesPerBlock,
                kKeysPerBlock,
                kSingleValueIteration>;
            isAligned =
                (query.stride(1) % AlignedAK::kAlignmentQ == 0 &&
                 key.stride(1) % AlignedAK::kAlignmentK == 0 &&
                 value.stride(1) % AlignedAK::kAlignmentV == 0);
            // TODO: Should we warn or log somewhere when we use a less
            // efficient kernel due to wrong alignment?
            DISPATCH_BOOL(
                isAligned, kIsAligned, ([&]() {
                  using Kernel = AttentionKernel<
                      scalar_t,
                      ArchTag,
                      kIsAligned,
                      kQueriesPerBlock,
                      kKeysPerBlock,
                      kSingleValueIteration>;
                  // Might happen on Sm80/half, where the minimum
                  // alignment is 32bits
                  TORCH_CHECK(
                      query.stride(1) % Kernel::kAlignmentQ == 0,
                      "query is not correctly aligned");
                  TORCH_CHECK(
                      key.stride(1) % Kernel::kAlignmentK == 0,
                      "key is not correctly aligned");
                  TORCH_CHECK(
                      value.stride(1) % Kernel::kAlignmentV == 0,
                      "value is not correctly aligned");

                  res = at::zeros(
                      {B, M, K},
                      query.options().dtype(
                          TypeTraits<
                              typename Kernel::output_t>::atScalarType()));
                  // NOTE: Should be aligned (by padding) in case M is not
                  // a good number for loading during backward
                  constexpr decltype(M) kAlignLSE =
                      32; // block size of backward
                  logsumexp = at::empty(
                      {B,
                       compute_logsumexp ? ceil_div(M, kAlignLSE) * kAlignLSE
                                         : 0},
                      query.options().dtype(at::ScalarType::Float));

                  constexpr auto kernel_fn = attention_kernel_batched<Kernel>;
                  size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
                  if (smem_bytes > 0xc000) {
                    TORCH_INTERNAL_ASSERT(
                        computeCapability >= 70,
                        "This kernel requires too much shared memory on this machine!");
                    AT_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel_fn,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        smem_bytes));
                  }

                  typename Kernel::Params p;
                  p.query_ptr = (scalar_t*)query.data_ptr();
                  p.key_ptr = (scalar_t*)key.data_ptr();
                  p.value_ptr = (scalar_t*)value.data_ptr();
                  p.logsumexp_ptr = compute_logsumexp
                      ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
                      : nullptr;
                  p.output_ptr = (typename Kernel::output_t*)res.data_ptr();
                  p.head_dim = query.size(2);
                  p.head_dim_value = value.size(2);
                  p.num_queries = query.size(1);
                  p.num_keys = key.size(1);
                  p.num_batches = B;
                  p.causal = causal;
                  kernel_fn<<<
                      p.getBlocksGrid(),
                      p.getThreadsGrid(),
                      smem_bytes>>>(p);
                }));
          }));
        }));
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
}

} // namespace native
} // namespace at
