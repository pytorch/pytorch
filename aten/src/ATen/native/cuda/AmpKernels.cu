#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


namespace at {
namespace native {
// These functions have no CPU variants, so there's no need to bother with a .cpp file in native/,
// and also no need to impose the DECLARE/DEFINE/REGISTER_DISPATCH layer of indirection.

// _amp_unscale_inf_check_cuda acts in-place on scaled_grad, multiplying each element by rscale's value.
// If any thread finds an inf/nan, it posts a 1. to found_inf.
//
// Args:
// scaled_grad:  An incoming scaled gradient tensor, which may contain infs/nans
// rscale:  The inverse of the scale factor by which scaled_grad is currently multiplied.
// found_inf:  A tensor to record whether scaled_grad contained any infs/nans
//
// Returns:
// A reference to the grad, which was unscaled in place.
Tensor & _amp_unscale_inf_check_cuda(Tensor & scaled_grad,
                                     const Tensor & rscale,
                                     const Tensor & found_inf)
{
  TORCH_CHECK(scaled_grad.is_cuda(), "scaled_grad must be a CUDA tensor.");
  TORCH_CHECK(rscale.is_cuda(), "rscale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(rscale.numel() == 1, "rscale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(rscale.scalar_type() == at::ScalarType::Float, "rscale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  // Act on scaled_grad in place.
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_unscale_inf_check_kernel",
    [&] {
      float* found_inf_ptr = found_inf.data_ptr<float>();
      float* rscale_ptr = rscale.data_ptr<float>();

      gpu_kernel(iter, [found_inf_ptr, rscale_ptr] GPU_LAMBDA(scalar_t val) -> scalar_t {
          float fval = static_cast<float>(val);
          if(!std::isfinite(fval))
            *found_inf_ptr = 1.f;
          float rscale = *rscale_ptr; // Every thread accesses rscale, but it will hit in cache.
          return static_cast<scalar_t>(rscale == 1.f ? fval : fval*(*rscale_ptr));
        });
    });

  return scaled_grad;
}


// amp_update_scale_kernel is meant to be launched with a single thread.
__global__ void amp_update_scale_kernel(float* current_scale,
                                        float* found_inf,
                                        float* new_scale,
                                        double scale_growth_factor,
                                        double scale_backoff_factor)
{
  if(*found_inf)
    *new_scale = (*current_scale)*scale_backoff_factor;
  else
    *new_scale = (*current_scale)*scale_growth_factor;
}


// amp_update_scale_kernel computes an updated scale factor Tensor, asynchronously.  It does not do any heavyweight
// work, but it's essential glue that gives optimizers the chance to implement sync-free dynamic loss scaling.
//
// Args:
// current_scale:  A one-element torch.cuda.FloatTensor containing the current scale value.
// found_inf:  A one-element torch.cuda.FloatTensor that contains > 0 if infs/nans were found by the relevant
//             prior _amp_unscale_inf_check_cuda call, and 0 if no infs/nans were found.
// scale_growth_factor:  The amount by which to multiply the scale if no infs/nans were found (typically slightly > 1).
// scale_backoff_factor:  The amount by which to multiply the scale if infs/nans were found (typically 0.5).
//
// Returns:
// new_scale:  A new one-element torch.cuda.FloatTensor containing the new recommended scale value.
Tensor _amp_update_scale_cuda(const Tensor & current_scale,
                              const Tensor & found_inf,
                              double scale_growth_factor,
                              double scale_backoff_factor)
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
                                   new_scale.data_ptr<float>(),
                                   scale_growth_factor,
                                   scale_backoff_factor);

  return new_scale;
}

}} // namespace at::native
