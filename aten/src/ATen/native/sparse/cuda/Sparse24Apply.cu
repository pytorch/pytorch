#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <ATen/native/sparse/cuda/Sparse24Pack.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

using namespace torch::sparse;

namespace {

template <typename KT>
__global__ void __launch_bounds__(32 /* num_threads */)
  sparse24_apply_kernel(typename KT::Params p)
{
  KT::sparse24_apply_kernel(p);
}

// Apply a 2:4 sparsify pattern computed with
// `sparse24_sparsify_both_ways_kernel` to another Tensor
template <bool kIsMeta, typename Element>
std::tuple<at::Tensor, at::Tensor> sparse24_apply_typed(at::Tensor input, at::Tensor threads_masks) // Returned by `sparse24_sparsify_both_ways`
{
  using KT = KernelTypes<Element>;
  // TODO: Technically we should be able to deal with that
  // by running on the transpose of `input` and swapping
  // `packed` & `packed_t`.
  // This would require to adapt the `threads_masks` a bit tho.
  if (input.stride(1) != 1) {
    input = input.contiguous();
  }
  c10::optional<at::cuda::CUDAGuard> device_guard;
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
    sparse24_apply_kernel<KT>
        <<<p.getBlocksGrid(),
           p.getThreadsGrid(),
           smem_bytes,
           at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return std::make_tuple(packed, packed_trans);
}

template <bool kIsMeta>
std::tuple<at::Tensor, at::Tensor> sparse24_apply(at::Tensor input, at::Tensor threads_masks) // Returned by `sparse24_sparsify_both_ways`
{
  if (input.scalar_type() == at::ScalarType::Half)
  {
    return sparse24_apply_typed<kIsMeta, cutlass::half_t>(input, threads_masks);
  }
  else
  {
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16);
    return sparse24_apply_typed<kIsMeta, cutlass::bfloat16_t>(input, threads_masks);
  }
}

} // namespace

TORCH_LIBRARY_IMPL(sparse, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::sparse24_apply"),
      TORCH_FN(sparse24_apply<false>));
}

TORCH_LIBRARY_IMPL(sparse, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::sparse24_apply"),
      TORCH_FN(sparse24_apply<true>));
}
