#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace {
// Thin wrapper around https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g57a3c8313f570282a1a7bcc78743b08e,
// to ensure the Cuda math library's isfinite is actually what gets called in
// _amp_non_finite_check_and_unscale_cuda_'s gpu_kernel lambda.
//
// isfinite_ensure_cuda_math is defined outside at::native because:
// - A bare call to "isfinite(val)" inside at::native causes nvcc to prefer the unrelated
//   Tensor at::native::isfinite(const Tensor&), resulting in an error:
//   "no suitable constructor exists to convert from "float" to "at::Tensor""
// - Unfortunately, the Cuda math library documentation doesn't say how (or if) you can provide a full namespace path
//   to ensure that its version of a particular function is invoked.  It only shows bare (not-namespaced)
//   calls to its routines inside kernel or device functions.
// - "std::isfinite(val)" in the gpu_kernel lambda causes an "unspecified launch failure" at runtime with cuda 9 on Windows.
//
// isfinite_ensure_cuda_math, declared at file scope outside the at::native region, uses isfinite as math library docs
// suggest and allows disambiguated usage in the lambda within the at::native region.
// GPU_LAMBDA is defined as __host__ __device__ (see Loops.cuh), so I need the __host__ keyword or else nvcc complains that
// "calling a __device__ function("isfinite_ensure_cuda_math") from a __host__ __device__ function("operator()") is not allowed."
static __host__ __device__ __forceinline__ int isfinite_ensure_cuda_math(float val) {
  return isfinite(val);
}
}

namespace at {
namespace native {

// Multiplies scaled_grad in-place by inv_scale.  If an element of scaled_grad was inf or NaN sets found_inf to 1.0.
//
// Args:
// scaled_grad:  A (scaled) gradient tensor.  May contain infs or NaNs.
// found_inf:  A single-element float tensor to which 1.0 will be written if any gradients contain infs/nans.
//             Pre-zeroing found_inf, if appropriate, is the responsibility of the caller.
// inv_scale:  The inverse of the scale factor by which scaled_grad is currently multiplied.
//
// Returns:
// A tuple with references to scaled_grad, which is now unscaled in place, and found_inf,
// which is now guaranteed to contain 1.0 if an inf or NaN was found in scaled_grad.
void _amp_non_finite_check_and_unscale_cuda_(Tensor& scaled_grad,
                                             Tensor& found_inf,
                                             const Tensor& inv_scale)
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
    "_amp_non_finite_check_and_unscale_cuda",
    [&iter, &found_inf, &inv_scale] {
      auto* found_inf_ptr = found_inf.data_ptr<float>();
      auto* inv_scale_ptr = inv_scale.data_ptr<float>();

      gpu_kernel(iter, [found_inf_ptr, inv_scale_ptr]GPU_LAMBDA(scalar_t val) -> scalar_t {
          float fval = static_cast<float>(val);
          // See isfinite_ensure_cuda_math above.
          if (!isfinite_ensure_cuda_math(fval)) {
            *found_inf_ptr = 1.f;
          }
          const auto inv_scale_val = *inv_scale_ptr; // Every thread accesses inv_scale, but it will hit in cache.
          return static_cast<scalar_t>(inv_scale_val == 1.f ? fval : fval*inv_scale_val);
        });
    });
}


// amp_update_scale_cuda_kernel is launched with a single thread to compute the new scale.
// The scale factor is maintained and updated on the GPU to avoid synchronization.
__global__ void amp_update_scale_cuda_kernel(int* growth_tracker,
                                             float* current_scale,
                                             float* found_inf,
                                             float* new_scale,
                                             double growth_factor,
                                             double backoff_factor,
                                             int growth_interval)
{
  if (*found_inf) {
    *new_scale = (*current_scale)*backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      *new_scale = (*current_scale)*growth_factor;
      *growth_tracker = 0;
    } else {
      *new_scale = *current_scale;
      *growth_tracker = successful;
    }
  }
}


// _amp_update_scale_cuda asynchronously updates the scale factor.
//
// Args:
// growth_tracker:  A one-element torch.cuda.IntTensor containing the number of recent consecutive unskipped steps.
// current_scale:  A one-element torch.cuda.FloatTensor containing the current scale value.
// found_inf:  A one-element torch.cuda.FloatTensor. If > 0, indicates that infs/nans were found by the relevant
//             prior _amp_non_finite_check_and_unscale_cuda call, and 0 if no infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly > 1).
// backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for current_scale to be multiplied by
//                   growth_factor.
//
// Returns:
// new_scale:  A new one-element torch.cuda.FloatTensor containing the new recommended scale value.
Tensor _amp_update_scale_cuda(Tensor& growth_tracker,
                              const Tensor& current_scale,
                              const Tensor& found_inf,
                              double growth_factor,
                              double backoff_factor,
                              int64_t growth_interval)
{
  TORCH_CHECK(growth_tracker.is_cuda(), "growth_tracker must be a CUDA tensor.");
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  auto new_scale = at::empty_like(current_scale);

  amp_update_scale_cuda_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
    growth_tracker.data_ptr<int>(),
    current_scale.data_ptr<float>(),
    found_inf.data_ptr<float>(),
    new_scale.data_ptr<float>(),
    growth_factor,
    backoff_factor,
    growth_interval);

  return new_scale;
}

}} // namespace at::native
