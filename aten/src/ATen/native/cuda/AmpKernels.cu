#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


namespace at {
namespace native {

// Multiplies scaled_grad in-place by inv_scale.  If an element of scaled_grad was inf or NaN sets found_inf to 1.
//
// Args:
// scaled_grad:  A (scaled) gradient tensor.  May contain infs or NaNs.
// inv_scale:  The inverse of the scale factor by which scaled_grad is currently multiplied.
// found_inf:  A single-element float tensor to which 1.0 will be written if any gradients contain infs/nans.
//             Pre-zeroing found_inf, if appropriate, is the responsibility of the caller.
// Returns:
// A reference to the grad, which was unscaled in place.
Tensor& _amp_unscale_inf_check_cuda(Tensor& scaled_grad,
                                     const Tensor& inv_scale,
                                     const Tensor& found_inf)
{
  TORCH_CHECK(scaled_grad.is_cuda(), "scaled_grad must be a CUDA tensor.");
  TORCH_CHECK(inv_scale.is_cuda(), "inv_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");
  TORCH_CHECK(scaled_grad.layout() == at::kStrided, "scaled_grad must be a strided (not sparse) Tensor.");

  // Act on scaled_grad in place.
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_unscale_inf_check_kernel",
    [&] {
      auto* found_inf_ptr = found_inf.data_ptr<float>();
      auto* inv_scale_ptr = inv_scale.data_ptr<float>();

      gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t val) -> scalar_t {
          auto fval = static_cast<float>(val);
          if (!isfinite(fval)) {
            *found_inf_ptr = 1.f;
          }
          const auto inv_scale = *inv_scale_ptr; // Every thread accesses inv_scale, but it will hit in cache.
          return static_cast<scalar_t>(inv_scale == 1.f ? fval : fval*inv_scale);
        });
    });

  return scaled_grad;
}


// amp_update_scale_kernel is launched with a single thread to compute the new scale.
__global__ void amp_update_scale_kernel(float* current_scale,
                                        float* found_inf,
                                        float* new_scale,
                                        double scale_growth_factor,
                                        double scale_backoff_factor)
{
  *new_scale = (*found_inf) ? (*current_scale)*scale_backoff_factor : (*current_scale)*scale_growth_factor;
}


// amp_update_scale_kernel asynchronously updates the scale factor.
//
// Args:
// current_scale:  A one-element torch.cuda.FloatTensor containing the current scale value.
// found_inf:  A one-element torch.cuda.FloatTensor. If > 0, indicates that infs/nans were found by the relevant
//             prior _amp_unscale_inf_check_cuda call, and 0 if no infs/nans were found.
// scale_growth_factor:  Multiplier if no infs/NaNs were found (typically slightly > 1).
// scale_backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
//
// Returns:
// new_scale:  A new one-element torch.cuda.FloatTensor containing the new recommended scale value.
Tensor _amp_update_scale_cuda(const Tensor& current_scale,
                              const Tensor& found_inf,
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

  amp_update_scale_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
    current_scale.data_ptr<float>(),
    found_inf.data_ptr<float>(),
    new_scale.data_ptr<float>(),
    scale_growth_factor,
    scale_backoff_factor);

  return new_scale;
}

}} // namespace at::native
