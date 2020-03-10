#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/quantized/fake_quant_affine.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
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
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.add_output(output);
  iter.add_input(input);
  iter.build();
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float input_val) -> float {
      return (fminf(
                quant_max,
                fmaxf(
                    quant_min,
                    static_cast<int64_t>(std::nearbyint(
                        input_val * inv_scale + zero_point)))) -
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
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.add_output(input_grad);
  iter.add_input(output_grad);
  iter.add_input(input);
  iter.build();
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float dy, float x) -> float {
      int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
      return (Xq >= quant_min && Xq <= quant_max) * dy;
    });
}

REGISTER_DISPATCH(fake_quant_tensor_stub, &fake_quantize_tensor_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_tensor_stub, &fake_quantize_grad_tensor_kernel_cuda);

// Fake quantize per channel

void fake_quant_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float input_val, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      return (fminf(
                quant_max,
                fmaxf(
                    quant_min,
                    static_cast<int64_t>(std::nearbyint(
                        input_val * inv_scale + zero_point)))) -
            zero_point) *
          scale;
    });
}

void fake_quant_grad_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float x, float dy, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
      return (Xq >= quant_min && Xq <= quant_max) * dy;
    });
}

REGISTER_DISPATCH(fake_quant_per_channel_stub, &fake_quant_per_channel_cuda);
REGISTER_DISPATCH(fake_quant_grad_per_channel_stub, &fake_quant_grad_per_channel_cuda);

} // namespace native
} // namespace at
