#define _USE_MATH_DEFINES

#include <ATen/native/Amp.h>

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/amp_mode.h>
#include <ATen/native/cuda/Loops.cuh>


namespace at {
namespace native {
namespace {
  // Should this be mutexed?  Probably.
  at::Tensor amp_overflow_state;
  // This could avoid the use of is_variable(true) in the TensorOptions below, but apparently torch:: is not declared here
  // torch::Tensor amp_overflow_state;
}

Tensor _amp_overflow_state_cuda(const Tensor & new_state) {
  if(new_state.defined())
  {
    TORCH_CHECK(new_state.is_cuda(), "Overflow state must be a CUDA tensor.");
    TORCH_CHECK(new_state.numel() == 1, "Overflow state must be a 1-element tensor.");
    TORCH_CHECK(new_state.scalar_type() == at::ScalarType::Int, "Overflow state must be an int tensor.");
    amp_overflow_state = new_state;
  } else if(!amp_overflow_state.defined())
    amp_overflow_state = at::zeros({1}, at::device(kCUDA).dtype(kInt).is_variable(true));

  return amp_overflow_state;
}

// Lots of function calls to get here, maybe a performance issue
void _amp_unscale_inf_check_cuda_impl(TensorIterator& iter, double scale) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_unscale_inf_check_cuda_impl",
    [&] {
      float rscale = 1.f/scale;
      int* amp_overflow = amp_overflow_state.data<int>();
      gpu_kernel(iter, [rscale, amp_overflow] GPU_LAMBDA(scalar_t a) -> scalar_t {
          float incoming_val = static_cast<float>(a);
          if(!std::isfinite(incoming_val))
            *amp_overflow = 1;
          return static_cast<scalar_t>(incoming_val*rscale);
        });
    });
}

REGISTER_DISPATCH(amp_unscale_inf_check_stub, &_amp_unscale_inf_check_cuda_impl);

}}  // namespace at::native
