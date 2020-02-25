#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/quantized/fake_quant_affine.h>
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
  Fake quantized tensor (double dtype).
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
  float inv_scale = 1.0f / scale;
  at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
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
  float inv_scale = 1.0f / scale;
  at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
}

REGISTER_DISPATCH(fake_quant_tensor_stub, &fake_quantize_tensor_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_tensor_stub, &fake_quantize_grad_tensor_kernel_cuda);

// Fake quantize by channel

void fake_quant_by_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
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

void fake_quant_grad_by_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float x, float dy, float scale, int64_t zero_point) -> float {
      float inv_scale = 1.0f / scale;
      int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
      return (Xq >= quant_min && Xq <= quant_max) * dy;
    });
}

REGISTER_DISPATCH(fake_quant_by_channel_stub, &fake_quant_by_channel_cuda);
REGISTER_DISPATCH(fake_quant_grad_by_channel_stub, &fake_quant_grad_by_channel_cuda);

} // namespace native
} // namespace at
