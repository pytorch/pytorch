#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAMathCompat.h>


namespace at { namespace native {

extern Tensor _amp_overflow_state(const Tensor &);

Tensor _amp_unscale_inf_check_cuda(const Tensor & scaled_grad, double scale) {
  TORCH_CHECK(scaled_grad.is_cuda());

  auto unscaled_grad = at::empty_like(scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scaled_grad.scalar_type(), "prelu_cuda", [&] {
    float rscale = 1.f/scale;
    // Not a huge fan of how this is architected, gotta move some things around.
    int* amp_overflow = at::native::_amp_overflow_state(Tensor{}).data<int>();
    at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      scaled_grad,
      unscaled_grad,
      [=] __device__ (
        const scalar_t& input_val,
        scalar_t& result_val) {
          float incoming_val = static_cast<float>(input_val);
          if(!std::isfinite(incoming_val))
            *amp_overflow = 1;
          result_val = static_cast<scalar_t>(incoming_val*rscale);
      });
  });

  return unscaled_grad;
}

}}  // namespace at::native
