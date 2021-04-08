#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>


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

namespace {
// Single-tensor fallback for _amp_foreach_non_finite_check_and_unscale_cuda_.
// Handles individual tensors that are acceptable to unscale but not MTA-safe.
void _amp_non_finite_check_and_unscale_cuda_(Tensor& scaled_grad,
                                             Tensor& found_inf,
                                             const Tensor& inv_scale)
{
  // The only way we reach this function is through _amp_foreach_non_finite_check_and_unscale_cuda_, so no input checks.

  // It's not obvious gpu_kernel always guards onto its argument.  Guarding here just in case.
  const OptionalDeviceGuard device_guard(device_of(scaled_grad));

  // Acts on scaled_grad in place.
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_non_finite_check_and_unscale_cuda",
    [&iter, &found_inf, &inv_scale] {
      auto* found_inf_ptr = found_inf.data_ptr<float>();
      auto* inv_scale_ptr = inv_scale.data_ptr<float>();

      using opmath_t = get_opmath_t<scalar_t>::opmath_t;

      gpu_kernel(iter,
                 [found_inf_ptr, inv_scale_ptr] GPU_LAMBDA (scalar_t val_in) -> scalar_t {
                   auto val = static_cast<opmath_t>(val_in);
                   if (!isfinite_ensure_cuda_math(val)) {
                     *found_inf_ptr = 1.f;
                   }
                   // Every thread accesses inv_scale, but it will hit in cache.
                   const auto inv_scale_val = *inv_scale_ptr;
                   return static_cast<scalar_t>(inv_scale_val == 1.f ? val : val * inv_scale_val);
                 });
    });
}
} // anonymous namespace


// Multiplies each tensor in scaled_grads by inv_scale in-place.
// If any element of any tensor in scaled_grads is inf or NaN, sets found_inf to 1.0.
// Uses multi tensor apply (MTA) to process all MTA-safe tensors.
//
// Args:
// scaled_grads:  A TensorList of scaled gradient tensors.  May contain infs or NaNs.
// found_inf:  A single-element float tensor to which 1.0 will be written if any gradient contain infs/nans.
//             Pre-zeroing found_inf, if appropriate, is the responsibility of the caller.
// inv_scale:  The inverse of the scale factor by which scaled_grads are currently multiplied.
void _amp_foreach_non_finite_check_and_unscale_cuda_(TensorList scaled_grads,
                                                     Tensor& found_inf,
                                                     const Tensor& inv_scale)
{
  if (scaled_grads.size() == 0) {
    return;
  }

  TORCH_CHECK(inv_scale.is_cuda(), "inv_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  // Ensures client code (GradScaler) filtered scaled_grads by dtype.
  check_foreach_api_restrictions(scaled_grads);

  std::vector<std::vector<at::Tensor>> tensor_lists;

  // is_non_overlapping_and_dense() is not available in Python.
  // GradScaler can't filter for it. We need to filter here.
  if (can_use_fast_route(scaled_grads)) {
    // Hopefully common case.
    // can_use_fast_route is true, which confirms:
    //  - all scaled_grads are strided
    //  - all scaled_grads are non overlapping and dense
    //  - all scaled_grads are on the same device
    TORCH_CHECK(scaled_grads[0].is_cuda(), "scaled_grads must be CUDA tensors.");
    // Sets up MTA launch to use scaled_grads as-is.
    tensor_lists.emplace_back(scaled_grads.vec());
  } else {
    // Hopefully uncommon case.
    // can_use_fast_route is an all-or-nothing check.  In this path it was false,
    // so any of the above confirmations could have gone wrong.
    // We filter MTA-safe tensors into an MTA-able list.
    // If a tensor is acceptable but not MTA-safe, we fall back to the TensorIterator kernel.
    // If a tensor is unacceptable, we throw an error to blame GradScaler.
    tensor_lists.resize(1);
    tensor_lists[0].reserve(scaled_grads.size());
    auto expected_device = scaled_grads[0].device();
    for (const Tensor& t : scaled_grads) {
      // Ensures GradScaler filtered scaled_grads by device.
      TORCH_CHECK(t.is_cuda(), "one of scaled_grads was not a CUDA tensor.");
      TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device.");
      TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor.");
      if (!t.is_non_overlapping_and_dense()) {
        // t is acceptable but not MTA-safe.  Falls back to single-tensor TensorIterator kernel.
        _amp_non_finite_check_and_unscale_cuda_(const_cast<Tensor&>(t),
                                                found_inf,
                                                inv_scale);
      } else {
        tensor_lists[0].push_back(t);
      }
    }
    if (tensor_lists[0].size() == 0) {
      return;
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    tensor_lists[0][0].scalar_type(),
    "_amp_foreach_non_finite_check_and_unscale_cuda",
    [&tensor_lists, &found_inf, &inv_scale] {
      auto* found_inf_ptr = found_inf.data_ptr<float>();
      auto* inv_scale_ptr = inv_scale.data_ptr<float>();

      using opmath_t = get_opmath_t<scalar_t>::opmath_t;

      // multi_tensor_apply guards onto tensor_lists[0][0], no need to guard explicitly.
      multi_tensor_apply<1>(tensor_lists,
                            UnaryOpFunctor<scalar_t,
                                           /* depth */ 1,
                                           /* r_args_depth */ 1,
                                           /* res_arg_index */ 0>(),
                            [found_inf_ptr, inv_scale_ptr] GPU_LAMBDA (opmath_t val) -> opmath_t {
                              // There is a slight asymmetry here with the TensorIterator kernel above.
                              // MTA Functors ensure val comes in as opmath_t rather than scalar_t.
                              if (!isfinite_ensure_cuda_math(val)) {
                                *found_inf_ptr = 1.f;
                              }
                              // Every thread accesses inv_scale, but it will hit in cache.
                              const auto inv_scale_val = *inv_scale_ptr;
                              return static_cast<opmath_t>(inv_scale_val == 1.f ? val : val * inv_scale_val);
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return new_scale;
}

}} // namespace at::native
