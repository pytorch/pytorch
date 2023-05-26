#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/AmpKernels.h>
#include <math.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

namespace at::native {

namespace {
// Follow the implements of CUDA.
// Multiplies each tensor in scaled_grads by inv_scale in-place.
// If any element of any tensor in scaled_grads is inf or NaN, sets found_inf
// to 1.0.
//
// Args:
// scaled_grads:  A TensorList of scaled gradient tensors.  May contain infs or
// NaNs. found_inf:  A single-element float tensor to which 1.0 will be written
// if any gradient contain infs/nans.
//             Pre-zeroing found_inf, if appropriate, is the responsibility of
//             the caller.
// inv_scale:  The inverse of the scale factor by which scaled_grads are
// currently multiplied.
void _amp_foreach_non_finite_check_and_unscale_cpu_kernel(
    TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  if (scaled_grads.size() == 0) {
    return;
  }

  TORCH_CHECK(inv_scale.is_cpu(), "inv_scale must be a CPU tensor.");
  TORCH_CHECK(found_inf.is_cpu(), "found_inf must be a CPU tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(
      inv_scale.scalar_type() == at::ScalarType::Float,
      "inv_scale must be a float tensor.");
  TORCH_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");

  // Ensures client code (GradScaler) filtered scaled_grads by dtype.
  at::native::check_foreach_api_restrictions(scaled_grads);
  auto expected_device = scaled_grads[0].device();
  for (const at::Tensor& t : scaled_grads) {
    TORCH_CHECK(t.is_cpu(), "one of scaled_grads was not a CPU tensor.");
    TORCH_CHECK(
        t.device() == expected_device,
        "scaled_grads must be on the same device.");
    TORCH_CHECK(
        t.layout() == at::kStrided,
        "one of scaled_grads was not a strided tensor.");
    auto iter = at::TensorIterator::unary_op(
        const_cast<at::Tensor&>(t), const_cast<at::Tensor&>(t));
    if (at::isReducedFloatingType(iter.dtype())) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
      iter.dtype(),
      "_amp_foreach_non_finite_check_and_unscale_cpu",
      [&iter, &found_inf, &inv_scale] {
          auto* found_inf_ptr = found_inf.data_ptr<float>();
          auto* inv_scale_ptr = inv_scale.data_ptr<float>();

          using opmath_t = at::opmath_type<scalar_t>;

          at::native::cpu_kernel_vec(
              iter,
              [found_inf_ptr, inv_scale_ptr](scalar_t val_in) -> scalar_t {
                auto val = static_cast<opmath_t>(val_in);
                if (!std::isfinite(val)) {
                  *found_inf_ptr = 1.f;
                }
                // Every thread accesses inv_scale, but it will hit in cache.
                const auto inv_scale_val = *inv_scale_ptr;
                return static_cast<scalar_t>(
                    inv_scale_val == 1.f ? val : val * inv_scale_val);
              },
              [found_inf_ptr, inv_scale_ptr](Vectorized<scalar_t> val_vec) -> Vectorized<scalar_t>{
                Vectorized<opmath_t> val_vec0, val_vec1;
                std::tie(val_vec0, val_vec1) = convert_to_float<scalar_t>(val_vec);
                if (val_vec0.has_infinite() || val_vec1.has_infinite()) {
                  *found_inf_ptr = 1.f;
                }
                // Every thread accesses inv_scale, but it will hit in cache.
                const auto inv_scale_val = *inv_scale_ptr;
                val_vec0 = inv_scale_val == 1.f ? val_vec0 : val_vec0 * Vectorized<opmath_t>(inv_scale_val);
                val_vec1 = inv_scale_val == 1.f ? val_vec1 : val_vec1 * Vectorized<opmath_t>(inv_scale_val);
                return convert_from_float<scalar_t>(val_vec0, val_vec1);
              });
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(
        iter.dtype(),
        "_amp_foreach_non_finite_check_and_unscale_cpu",
        [&iter, &found_inf, &inv_scale] {
          auto* found_inf_ptr = found_inf.data_ptr<float>();
          auto* inv_scale_ptr = inv_scale.data_ptr<float>();
          at::native::cpu_kernel_vec(
              iter,
              [found_inf_ptr, inv_scale_ptr](scalar_t val_in) -> scalar_t {
                if (!std::isfinite(val_in)) {
                  *found_inf_ptr = 1.f;
                }
                // Every thread accesses inv_scale, but it will hit in cache.
                const auto inv_scale_val = *inv_scale_ptr;
                return static_cast<scalar_t>(
                    inv_scale_val == 1.f ? val_in : val_in * inv_scale_val);
              },
              [found_inf_ptr, inv_scale_ptr](Vectorized<scalar_t> val_vec) -> Vectorized<scalar_t>{
                if (val_vec.has_infinite()) {
                  *found_inf_ptr = 1.f;
                }
                // Every thread accesses inv_scale, but it will hit in cache.
                const auto inv_scale_val = *inv_scale_ptr;
                return inv_scale_val == 1.f ? val_vec : val_vec * Vectorized<scalar_t>(inv_scale_val);
              });
        });
    }
  }
}

// _amp_update_scale_cpu updates the scale tensor in place.
//
// Args:
// current_scale:  A one-element float tensor containing the scale value.
// growth_tracker:  A one-element IntTensor containing the number of recent
// consecutive unskipped steps. found_inf:  A one-element float tensor. If > 0,
// indicates that infs/nans were found by the relevant
//             prior _amp_non_finite_check_and_unscale_cpu call, and 0 if no
//             infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly >
// 1). backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for
// current_scale to be multiplied by
//                   growth_factor.
//
// Returns:
// current_scale
at::Tensor& _amp_update_scale_cpu_kernel(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  TORCH_CHECK(growth_tracker.is_cpu(), "growth_tracker must be a CPU tensor.");
  TORCH_CHECK(current_scale.is_cpu(), "current_scale must be a CPU tensor.");
  TORCH_CHECK(found_inf.is_cpu(), "found_inf must be a CPU tensor.");
  TORCH_CHECK(
      growth_tracker.numel() == 1,
      "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(
      current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(
      growth_tracker.scalar_type() == at::ScalarType::Int,
      "growth_tracker must be an int tensor.");
  TORCH_CHECK(
      current_scale.scalar_type() == at::ScalarType::Float,
      "current_scale must be a float tensor.");
  TORCH_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");

  float* current_scale_ptr = current_scale.data_ptr<float>();
  int* growth_tracker_ptr = growth_tracker.data_ptr<int>();
  float* found_inf_ptr = found_inf.data_ptr<float>();

  if (*found_inf_ptr) {
    *current_scale_ptr = (*current_scale_ptr) * backoff_factor;
    *growth_tracker_ptr = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker_ptr) + 1;
    if (successful == growth_interval) {
      auto new_scale = static_cast<float>((*current_scale_ptr) * growth_factor);
      // Do not grow the scale past fp32 bounds to inf.
      if (std::isfinite(new_scale)) {
        *current_scale_ptr = new_scale;
      }
      *growth_tracker_ptr = 0;
    } else {
      *growth_tracker_ptr = successful;
    }
  }

  return current_scale;
}

} // namepace

REGISTER_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu_stub, &_amp_foreach_non_finite_check_and_unscale_cpu_kernel);
REGISTER_DISPATCH(_amp_update_scale_cpu_stub, &_amp_update_scale_cpu_kernel);

} // namespace at::native
