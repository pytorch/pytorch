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
void fake_quantize_tensor_kernel_cuda(
    Tensor& output,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // scalar type of this function is guaranteed to be float
  float inv_scale = 1.0f / scale;
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_input(input)
    .build();
  gpu_kernel(iter, [=] GPU_LAMBDA(float input_val) -> float {
    return (fminf(
                quant_max,
                fmaxf(
                    quant_min,
                    static_cast<int64_t>(
                        std::nearbyint(input_val * inv_scale) + zero_point))) -
            zero_point) *
        scale;
  });
}

void fake_quantize_grad_tensor_kernel_cuda(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // scalar type of this function is guaranteed to be float
  float inv_scale = 1.0f / scale;
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(input_grad)
    .add_input(output_grad)
    .add_input(input)
    .build();
  gpu_kernel(iter, [=] GPU_LAMBDA(float dy, float x) -> float {
    int64_t Xq = std::nearbyint(x * inv_scale) + zero_point;
    return (Xq >= quant_min && Xq <= quant_max) * dy;
  });
}

void _fake_quantize_grad_learnable_tensor_kernel_cuda(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float dscale_small = quant_min - zero_point;
  float dscale_big = quant_max - zero_point;
  gpu_kernel_multiple_outputs(
    iter, [=] GPU_LAMBDA (float XInput, float dYInput) -> thrust::tuple<float, float, float> {
      float dXOutput, dZeroPointOutput, dScaleOutput;
      int64_t xq = std::nearbyint(XInput * inv_scale) + zero_point;
      dXOutput = dYInput * (xq >= quant_min && xq <= quant_max);
      xq = std::max(std::min(xq, quant_max), quant_min);
      float xfq = static_cast<float>((xq - zero_point) * scale);
      if (xq == quant_min || xq == quant_max) {
        dZeroPointOutput = (dYInput) * (-1) * scale;
        dScaleOutput = (xq == quant_min) ? (dYInput * dscale_small) : (dYInput * dscale_big);
      } else {
        dZeroPointOutput = 0;
        dScaleOutput = (dYInput) * (xfq - (XInput)) * inv_scale;
      }
      return {dXOutput, dScaleOutput, dZeroPointOutput};
  });
}

REGISTER_DISPATCH(fake_quant_tensor_stub, &fake_quantize_tensor_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_tensor_stub, &fake_quantize_grad_tensor_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub, &_fake_quantize_grad_learnable_tensor_kernel_cuda);

// Fake quantize per channel

void fake_quant_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
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

void fake_quant_grad_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float x, float dy, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      int64_t Xq = std::nearbyint(x * inv_scale) + zero_point;
      return (Xq >= quant_min && Xq <= quant_max) * dy;
    });
}

void _fake_quantize_grad_learnable_channel_kernel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel_multiple_outputs(iter,
    [=] GPU_LAMBDA (float x_input, float dy_input, float scale_input, float zero_point_input) -> thrust::tuple<float, float, float> {
      float dx_output, dscale_output, dzero_point_output;
      float inv_scale = 1.0f / scale_input;
      float dscale_small = quant_min - zero_point_input;
      float dscale_big = quant_max - zero_point_input;
      // Calculate gradients for X.
      int64_t xqi = std::nearbyint(zero_point_input + x_input * inv_scale);
      dx_output = dy_input * (xqi >= quant_min && xqi <= quant_max);
      // Calculate gradients for scale and zero point.
      xqi = std::max(std::min(xqi, quant_max), quant_min);
      float xfqi = static_cast<float>((xqi - zero_point_input) * scale_input);
      if (xqi == quant_min || xqi == quant_max) {
        dzero_point_output = dy_input * (-1) * scale_input;
        dscale_output = (xqi == quant_min) ? (dy_input * dscale_small) : (dy_input * dscale_big);
      } else {
        dzero_point_output = 0;
        dscale_output = dy_input * (xfqi - x_input) * inv_scale;
      }
      return {dx_output, dscale_output, dzero_point_output};
    });
}

REGISTER_DISPATCH(fake_quant_per_channel_stub, &fake_quant_per_channel_cuda);
REGISTER_DISPATCH(fake_quant_grad_per_channel_stub, &fake_quant_grad_per_channel_cuda);
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub, &_fake_quantize_grad_learnable_channel_kernel_cuda);

} // namespace native
} // namespace at
