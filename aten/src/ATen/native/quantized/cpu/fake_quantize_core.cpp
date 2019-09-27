#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

/* FakeQuantize Op for PerChannelAffine quantization scheme */
namespace at {
namespace native {

/* Per channel fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per channel affine quantization
  zero_point: zero_point of per channel affine quantization
  axis: int specifying the axis to be quantized
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (double dtype).

*/
void fake_quantize_slice(Tensor& output,
                        const Tensor& input,
                        float sc,
                        int64_t z_point,
                        int64_t quant_min,
                        int64_t quant_max)
                        {
  float inv_scale = 1.0f / sc;
  auto iter = TensorIterator::unary_op(output, input);
  cpu_kernel(iter, [&](float self) -> float {
    return (std::fmin(
                std::fmax(
                    static_cast<int64_t>(
                        std::nearbyint(self * inv_scale + z_point)),
                    quant_min),
                quant_max) -
            z_point) *
        sc;
  });
}

void fake_quantize_grad_slice(Tensor& input_grad,
                        const Tensor& input,
                        const Tensor& output_grad,
                        float sc,
                        int64_t z_point,
                        int64_t quant_min,
                        int64_t quant_max)
                        {
    float inv_scale = 1.0f / sc;
    auto iter = TensorIterator::binary_op(input_grad, input, output_grad);
    cpu_kernel(iter, [&](float x, float dy) -> float {
    int64_t xq =
        static_cast<int64_t>(std::nearbyint(x * inv_scale + z_point));
    return dy * (xq >= quant_min && xq <= quant_max);
  });
}

} // namespace native
} // namespace at
