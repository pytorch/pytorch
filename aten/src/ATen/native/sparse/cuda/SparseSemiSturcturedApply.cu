#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <ATen/native/sparse/cuda/SparseSemiStructuredPack.h>
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
template <typename KT>
__global__ void __launch_bounds__(32 /* num_threads */)
  sparse_semi_structured_apply_kernel(typename KT::Params p)
{
  KT::sparse_semi_structured_apply_kernel(p);
}

// Apply a 2:4 sparsify pattern computed with
// `_sparse_semi_structured_tile` to another Tensor
template <bool kIsMeta, typename Element>
std::tuple<Tensor, Tensor> _sparse_semi_structured_apply_typed(Tensor input, Tensor threads_masks)
{
  using KT = KernelTypes<Element>;
  // TODO: Technically we should be able to deal with that
  // by running on the transpose of `input` and swapping
  // `packed` & `packed_t`.
  // This would require to adapt the `threads_masks` a bit tho.
  if (input.stride(1) != 1) {
    input = input.contiguous();
  }
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(input.stride(0) % 8 == 0);
  TORCH_CHECK(input.size(1) % 32 == 0, "Wrong alignment shape[1]");

  auto roundedx = cutlass::round_up(input.size(0), kWarpX);
  auto roundedy = cutlass::round_up(input.size(1), kWarpY);
  at::Tensor packed =
      at::empty({roundedx, cutlass::ceil_div(roundedy, 2)}, input.options());
  at::Tensor packed_trans =
      at::empty({roundedy, cutlass::ceil_div(roundedx, 2)}, input.options());

  typename KT::Params p;
  p.input = (Element const*)input.data_ptr();
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);

  p.packed = (Element*)packed.data_ptr();
  p.packed_stride = packed.stride(0);
  p.packed_trans = (Element*)packed_trans.data_ptr();
  p.packed_trans_stride = packed_trans.stride(0);

  p.threads_masks = (uint64_t*)threads_masks.data_ptr();

  TORCH_CHECK(threads_masks.dim() == 3);
  TORCH_CHECK(
      threads_masks.size(0) == p.getBlocksGrid().x * p.getThreadsGrid().x);
  TORCH_CHECK(
      threads_masks.size(1) == p.getBlocksGrid().y * p.getThreadsGrid().y);
  TORCH_CHECK(threads_masks.stride(1) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.size(2) == sizeof(p.threads_masks[0]));
  TORCH_CHECK(threads_masks.stride(2) == 1);
  TORCH_CHECK(threads_masks.scalar_type() == at::ScalarType::Byte);

  if (!kIsMeta) {
    size_t smem_bytes = 0;
    sparse_semi_structured_apply_kernel<KT>
        <<<p.getBlocksGrid(),
           p.getThreadsGrid(),
           smem_bytes,
           at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return std::make_tuple(packed, packed_trans);
}
#endif

std::tuple<Tensor, Tensor> _sparse_semi_structured_apply(const Tensor& input, const Tensor& threads_masks) // Returned by `_sparse_semi_structured_tile`
{
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  AT_ERROR("_sparse_semi_structured_apply: not supported");
  return std::make_tuple(Tensor{}, Tensor{});
#else
  TORCH_CHECK(
    input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16,
    "Unsupported dtype - only `float16` and `bfloat16` are supported currently"
  );
  auto result = (input.scalar_type() == at::ScalarType::Half)
            ? _sparse_semi_structured_apply_typed<false, cutlass::half_t>(input, threads_masks)
            : _sparse_semi_structured_apply_typed<false, cutlass::bfloat16_t>(input, threads_masks);
  return result;
#endif
}

} // namespace
