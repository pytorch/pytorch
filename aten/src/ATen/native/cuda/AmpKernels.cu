#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


namespace at {
namespace native {

// These functions have no CPU variants, so there's no need to bother with a .cpp file in native/, and
// also no need to impose the DECLARE/DEFINE/REGISTER_DISPATCH layer of indirection.

Tensor & _amp_unscale_inf_check_cuda(const Tensor & scaled_grad,
                                     const Tensor & current_scale,
                                     const Tensor & found_inf)
{
  // Do I really need all these checks?
  TORCH_CHECK(scaled_grad.is_cuda(), "scaled_grad must be a CUDA tensor.");
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  // Act on scaled_grad in place.
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_unscale_inf_check_kernel",
    [&] {
      float* found_inf_ptr = found_inf.data_ptr<float>();
      float* current_scale_ptr = current_scale.data_ptr<float>();

      gpu_kernel(iter, [found_inf_ptr, current_scale_ptr] GPU_LAMBDA(scalar_t val) -> scalar_t {
          float fval = static_cast<float>(val);
          if(!std::isfinite(fval))
            *found_inf_ptr = 1.f;
          return static_cast<scalar_t>(fval/(*current_scale_ptr));
        });
    });

  return scaled_grad;
}


// Update the scale factor Tensor on the device.
__global__ void(float* current_scale,
                float* found_inf,
                float* new_scale,
                float scale_growth_factor,
                float scale_backoff_factor)
{
  // This kernel should only ever be launched with one thread anyway, but just in case
  if(threadIdx.x == 0 && blockIdx.x == 0)
    if(*found_inf)
      *new_scale = (*current_scale)*scale_backoff_factor;
    else:
      *new_scale = (*current_scale)*scale_growth_factor;
}


Tensor _amp_update_scale_cuda(Tensor & current_scale,
                              Tensor & found_inf,
                              float scale_growth_factor,
                              float scale_backoff_factor)
{
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  auto new_scale = at::empty_like(current_scale);

  amp_update_scale_kernel<<<1,1>>>(current_scale.data_ptr<float>(),
                                   found_inf.data_ptr<float>(),
                                   scale_growth_factor,
                                   scale_backoff_factor);

  return new_scale;
}

}}  // namespace at::native
