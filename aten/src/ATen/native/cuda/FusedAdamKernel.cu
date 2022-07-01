#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/util/irange.h>


namespace at { namespace native {

namespace {
template <typename scalar_type, int Depth=4>
struct FusedAdamMathFunctor {
  using opmath_t = at::opmath_type<scalar_type>;
  C10_DEVICE __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<Depth>& tl,
    const bool amsgrad,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const bool maximize
  ) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    scalar_type* args[Depth];
    const bool all_aligned{init_args<Depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    n -= chunk_idx * chunk_size;
    scalar_type r_args[Depth][kILP];

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < Depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
      }
    }
  }
};
} // namespace

void _fused_adam_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const bool amsgrad,
    const bool maximize,
    const bool capturable
) {

}

}} // namespace at::native
