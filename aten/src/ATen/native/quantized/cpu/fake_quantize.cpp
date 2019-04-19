#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>

/* Fake quantization ops will be placed here. */

namespace at { namespace native {
namespace {
/* Fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  num_bits: Number of quantization bits.
  quant_delay: Count of global steps for which to delay the quantization.
               See note below.
  iter: The current quantization iteration used for `quant_delay`.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - quantization range [0, 2^bits - 1]
*/

/********* Begin FakeQuant ops (forward and backward). *********/
class FakeQuantizeOp_forward : public c10::OperatorKernel {
 public:
  at::Tensor operator()(
      at::Tensor X,
      double scale,
      int64_t zero_point,
      int64_t num_bits = 8,
      int64_t quant_delay = 0,
      int64_t iter = 0
    ) {
    // Sanity checks.
    if (num_bits > 32 || num_bits < 1) {
      throw std::invalid_argument("`num_bits` should be in the [1, 32] range.");
    }
    if (quant_delay < 0) {
      throw std::invalid_argument("`quant_delay` must be a positive integer.");
    }

    if (quant_delay != 0 && iter < 0) {
      throw std::invalid_argument(
        "`iter` must be >=0 for non-zero `quant_delay`");
    }

    auto Y = at::empty_like(X);

    if (quant_delay > 0 && iter <= quant_delay) {
      Y.copy_(X);  // We might want to just return the input here.
      return Y;
    }

    double inv_scale = 1.0f / scale;
    const auto quant_min = 0;
    const auto quant_max = (1 << num_bits) - 1;
    Y = (((X * inv_scale + 0.5f).floor() + zero_point)
      .clamp_min(quant_min).clamp_max(quant_max) - zero_point) * scale;
    return Y;
  }
};

/* Backward path to fake-quantize the 'inputs' tensor.

Args:
  X: Forward input tensor.
  dY: Backward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  num_bits: Number of quantization bits.
  quant_delay: Count of global steps for which to delay the quantization.
               See note in forward.
  iter: The current quantization iteration used for `quant_delay`.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - quantization range [0, 2^bits - 1]
*/
class FakeQuantizeOp_backward : public c10::OperatorKernel {
 public:
  at::Tensor operator()(
      at::Tensor X,
      at::Tensor dY,
      double scale,
      int64_t zero_point,
      int64_t num_bits = 8,
      int64_t quant_delay = 0,
      int64_t iter = 0) {
    // Sanity checks.
    if (num_bits > 32 || num_bits < 1) {
      throw std::invalid_argument("`num_bits` should be in the [1, 32] range.");
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

    if (quant_delay != 0 && iter < 0) {
      throw std::invalid_argument(
        "`iter` must be >=0 for non-zero `quant_delay`");
    }

    auto dX = at::zeros_like(dY);
    if (quant_delay > 0 && iter <= quant_delay) {
      dX.copy_(dY);
      return dX;
    }

    double inv_scale = 1.0f / scale;
    const auto quant_min = 0;
    const auto quant_max = (1 << num_bits) - 1;
    at::Tensor Xq = (X * inv_scale + 0.5).floor() + zero_point;
    at::Tensor mask_min = (Xq >= quant_min);
    at::Tensor mask_max = (Xq <= quant_max);
    at::Tensor mask = mask_min * mask_max;
    dX = mask.type_as(dY) * dY;
    return dX;
  }
};

static auto registry = c10::RegisterOperators()
.op(c10::FunctionSchema(
      "quantized::fake_quantize_forward",
      "",
      {{"X", TensorType::get()},
       {"scale", FloatType::get()},
       {"zero_point", IntType::get()},
       {"num_bits", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/8},
       {"quant_delay", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"iter", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0}},
      {{"Y", TensorType::get()}}),
  c10::kernel<FakeQuantizeOp_forward>(),
  c10::dispatchKey(CPUTensorId()))
.op(c10::FunctionSchema(
      "quantized::fake_quantize_backward",
      "",
      {{"X", TensorType::get()},
       {"dY", TensorType::get()},
       {"scale", FloatType::get()},
       {"zero_point", IntType::get()},
       {"num_bits", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/8},
       {"quant_delay", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0},
       {"iter", IntType::get(), /*N=*/c10::nullopt, /*default_value=*/0}},
      {{"Y", TensorType::get()}}),
  c10::kernel<FakeQuantizeOp_backward>(),
  c10::dispatchKey(CPUTensorId()));
/********* End FakeQuant ops (forward and backward). *********/

}  // namespace
}}  // namespace at::native
