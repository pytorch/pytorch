#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>

/* Fake quantization ops will be placed here. */

namespace at { namespace native {
namespace {
namespace utils {
// Borrowed from tensorflow/core/kernels/fake_quant_ops_functor.h.
// Gymnastics with nudged zero point is to ensure that real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
// Outputs nudged_min, nudged_max, nudged_scale.
void nudge(
    const double min,
    const double max,
    const long quant_min,
    const long quant_max,
    double* nudged_min,
    double* nudged_max,
    double* scale) {
  const double quant_min_double = static_cast<double>(quant_min);
  const double quant_max_double = static_cast<double>(quant_max);
  *scale = (max - min) / (quant_max_double - quant_min_double);
  const double zero_point_from_min = quant_min_double - min / *scale;

  const int32_t nudged_zero_point = [zero_point_from_min,
                                     quant_min,
                                     quant_min_double,
                                     quant_max,
                                     quant_max_double] {
    if (zero_point_from_min < quant_min_double) {
      return static_cast<int32_t>(quant_min);
    }
    if (zero_point_from_min > quant_max_double) {
      return static_cast<int32_t>(quant_max);
    }
    return static_cast<int32_t>(std::round(zero_point_from_min));
  }();
  *nudged_min = (quant_min_double - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max_double - nudged_zero_point) * (*scale);
}
}  // namespace utils

/* Fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  min, max: Minimum and maximum for the clamping range.
  num_bits: Number of quantization bits.
  quant_delay: Count of global steps for which to delay the quantization.
               See note below.
  iter: The current quantization iteration used for `quant_delay`.
  narrow_range: Use "narrow quantization range". See note below.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - narrow quantization range is [1, 2^bits - 1];
    wide quantization range [0, 2^bits - 1]
*/

/********* Begin MinMaxArgs ops (forward and backward). *********/
class FakeQuantizeMinMaxArgsOp_forward : public c10::OperatorKernel {
 public:
  at::Tensor operator()(
      at::Tensor X,
      double min,
      double max,
      int64_t num_bits = 8,
      int64_t quant_delay = 0,
      int64_t iter = 0,
      bool narrow_range = false
    ) {
    // Sanity checks.
    if (num_bits > 32 || num_bits < 1) {
      throw std::invalid_argument("`num_bits` should be in the [1, 32] range.");
    }
    if (min > 0.0f || max < 0.0f || min >= max) {
      throw std::invalid_argument("`min`/`max` are malformed.");
    }
    if (quant_delay < 0) {
      throw std::invalid_argument("`quant_delay` must be a positive integer.");
    }

    // Prepare args.
    long quant_min = int(narrow_range);
    long quant_max = (1 << num_bits) - 1;

    double nudge_min = 0.0, nudge_max = 0.0, nudge_scale;
    utils::nudge(min, max, quant_min, quant_max,
                 &nudge_min, &nudge_max, &nudge_scale);

    if (quant_delay != 0 && iter < 0) {
      throw std::invalid_argument(
        "`iter` must be >=0 for non-zero `quant_delay`");
    }

    auto Y = at::empty_like(X);

    if (quant_delay > 0 && iter <= quant_delay) {
      Y.copy_(X);  // We might want to just return the input here.
      return Y;
    }

    double inv_nudge_scale = 1.0f / nudge_scale;
    Y = ((X.clamp_min(nudge_min).clamp_max(nudge_max) - nudge_min) *
         inv_nudge_scale + 0.5f).floor() * nudge_scale + nudge_min;
    return Y;
  }
};

/* Backward path to fake-quantize the 'inputs' tensor.

Args:
  X: Forward input tensor.
  dY: Backward input tensor.
  min, max: Minimum and maximum for the clamping range.
  num_bits: Number of quantization bits.
  quant_delay: Count of global steps for which to delay the quantization.
               See note in forward.
  iter: The current quantization iteration used for `quant_delay`.
  narrow_range: Use "narrow quantization range". See notes in forward.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - narrow quantization range is [1, 2^bits - 1];
    wide quantization range [0, 2^bits - 1]
*/
class FakeQuantizeMinMaxArgsOp_backward : public c10::OperatorKernel {
 public:
  at::Tensor operator()(
      at::Tensor X,
      at::Tensor dY,
      double min,
      double max,
      int64_t num_bits = 8,
      int64_t quant_delay = 0,
      int64_t iter = 0,
      bool narrow_range = false) {
    // Sanity checks.
    if (num_bits > 32 || num_bits < 1) {
      throw std::invalid_argument("`num_bits` should be in the [1, 32] range.");
    }
    if (min > 0.0f || max < 0.0f || min >= max) {
      throw std::invalid_argument("`min`/`max` are malformed.");
    }
    if (quant_delay < 0) {
      throw std::invalid_argument("`quant_delay` must be a positive integer.");
    }
    if (X.numel() <= 0) {
      throw std::length_error("`X` is empty");
    }
    if (X.numel() != dY.numel()) {
      throw std::invalid_argument("`X` and `dY` are not the same size");
    }

    // Prepare args.
    const auto quant_min = int(narrow_range);
    const auto quant_max = (1 << num_bits) - 1;

    double nudge_min = 0.0, nudge_max = 0.0, nudge_scale;
    utils::nudge(min, max, quant_min, quant_max,
                 &nudge_min, &nudge_max, &nudge_scale);

    if (quant_delay != 0 && iter < 0) {
      throw std::invalid_argument(
        "`iter` must be >=0 for non-zero `quant_delay`");
    }

    auto dX = at::zeros_like(dY);
    if (quant_delay > 0 && iter <= quant_delay) {
      dX.copy_(dY);
      return dX;
    }

    at::Tensor mask_min = (X >= nudge_min);
    at::Tensor mask_max = (X <= nudge_max);
    at::Tensor mask = mask_min * mask_max;
    dX = mask.type_as(dY) * dY;

    return dX;
  }
};

static auto registry = c10::RegisterOperators()
.op(c10::FunctionSchema(
      "quantized::fake_quantize_minmax_forward",
      "",
      {{"X", TensorType::get()},
       {"min", FloatType::get()},
       {"max", FloatType::get()},
       {"num_bits", IntType::get(), /*N=*/c10::nullopt, /*defaul_valuet=*/8},
       {"quant_delay", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"iter", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"narrow_range", BoolType::get(), /*N=*/c10::nullopt,
                                         /*default_value=*/false}},
      {{"Y", TensorType::get()}}),
  c10::kernel<FakeQuantizeMinMaxArgsOp_forward>(),
  c10::dispatchKey(CPUTensorId()))
.op(c10::FunctionSchema(
      "quantized::fake_quantize_minmax_backward",
      "",
      {{"X", TensorType::get()},
       {"dY", TensorType::get()},
       {"min", FloatType::get()},
       {"max", FloatType::get()},
       {"num_bits", IntType::get(), /*N=*/c10::nullopt, /*defaul_valuet=*/8},
       {"quant_delay", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"iter", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"narrow_range", BoolType::get(), /*N=*/c10::nullopt,
                                         /*default_value=*/false}},
      {{"Y", TensorType::get()}}),
  c10::kernel<FakeQuantizeMinMaxArgsOp_backward>(),
  c10::dispatchKey(CPUTensorId()));
/********* End MinMaxArgs ops (forward and backward). *********/

}  // namespace
}}  // namespace at::native
