#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/quantized/fake_quant_affine.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <thrust/tuple.h>
#include <cmath>

/* Fake quantize a tensor
Args:
  output: output tensor.
  input : input tensor.
  sc:  scale to quantize the input tensor to
  zero_point: zero_point
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (float dtype).
*/
namespace at {
namespace native {
void fake_quantize_tensor_cachemask_kernel_cuda(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float inv_scale = 1.0f / scale;
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_output(mask)
    .add_input(input)
    .build();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_types", [&] {
    gpu_kernel_multiple_outputs(
      iter,
      [=] GPU_LAMBDA (scalar_t input_val) -> thrust::tuple<scalar_t, bool> {
        const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + zero_point);
        return {
          // fake_quantized value
          (fminf(quant_max, fmaxf(quant_min, qval)) - zero_point) * scale,
          // mask for grad
          ((quant_min <= qval) && (qval <= quant_max))
        };
      }
    );
  });
}

void _fake_quantize_grad_learnable_tensor_kernel_cuda(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  float dscale_small = quant_min - zero_point;
  float dscale_big = quant_max - zero_point;
  gpu_kernel_multiple_outputs(
    iter, [=] GPU_LAMBDA (float XInput, float dYInput) -> thrust::tuple<float, float, float> {
      float dXOutput, dZeroPointOutput, dScaleOutput;
      int64_t xq = std::nearbyint(XInput * inv_scale) + zero_point;
      dXOutput = dYInput * (xq >= quant_min && xq <= quant_max);
      float xfq = static_cast<float>((std::max(std::min(xq, quant_max), quant_min) - zero_point) * scale);
      if (xq < quant_min || xq > quant_max) {
        dZeroPointOutput = (dYInput) * (-1) * scale * grad_factor;
        dScaleOutput = ((xq < quant_min) ? (dYInput * dscale_small) : (dYInput * dscale_big)) * grad_factor;
      } else {
        dZeroPointOutput = 0;
        dScaleOutput = (dYInput) * (xfq - (XInput)) * inv_scale * grad_factor;
      }
      return {dXOutput, dScaleOutput, dZeroPointOutput};
  });
}

REGISTER_DISPATCH(fake_quant_tensor_cachemask_stub, &fake_quantize_tensor_cachemask_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub, &_fake_quantize_grad_learnable_tensor_kernel_cuda);

// Fake quantize per channel

void fake_quant_per_channel_cachemask_cuda(
    TensorIterator &iter, TensorIterator &iter_mask, int64_t quant_min, int64_t quant_max) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  // write mask
  gpu_kernel(iter_mask,
    [=] GPU_LAMBDA (float input_val, float scale, int64_t zero_point) -> bool {
      float inv_scale = 1.0f / scale;
      const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + zero_point);
      return ((quant_min <= qval) && (qval <= quant_max));
    });

  // write fake_quant
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float input_val, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      return (fminf(
                  quant_max,
                  fmaxf(
                      quant_min,
                      static_cast<int64_t>(
                          std::nearbyint(input_val * inv_scale) +
                          zero_point))) -
              zero_point) *
          scale;
    });
}

void _fake_quantize_grad_learnable_channel_kernel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max, float grad_factor) {
  gpu_kernel_multiple_outputs(iter,
    [=] GPU_LAMBDA (float x_input, float dy_input, float scale_input, float zero_point_input) -> thrust::tuple<float, float, float> {
      float dx_output, dscale_output, dzero_point_output;
      float inv_scale = 1.0f / scale_input;
      float dscale_small = quant_min - zero_point_input;
      float dscale_big = quant_max - zero_point_input;
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint(x_input * inv_scale) + static_cast<int64_t>(zero_point_input);
      dx_output = dy_input * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      float xfqi = static_cast<float>((std::max(std::min(xqi, quant_max), quant_min) - zero_point_input) * scale_input);
      if (xqi < quant_min || xqi > quant_max) {
        dzero_point_output = dy_input * (-1) * scale_input * grad_factor;
        dscale_output = ((xqi < quant_min) ? (dy_input * dscale_small) : (dy_input * dscale_big)) * grad_factor;
      } else {
        dzero_point_output = 0;
        dscale_output = dy_input * (xfqi - x_input) * inv_scale * grad_factor;
      }
      return {dx_output, dscale_output, dzero_point_output};
    });
}

REGISTER_DISPATCH(fake_quant_per_channel_cachemask_stub, &fake_quant_per_channel_cachemask_cuda);
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub, &_fake_quantize_grad_learnable_channel_kernel_cuda);

} // namespace native
} // namespace at
