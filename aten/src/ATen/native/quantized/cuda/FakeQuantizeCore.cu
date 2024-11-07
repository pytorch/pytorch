#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
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
namespace at::native {
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

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_types", [&] {
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
  } else {
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
}

void fake_quantize_tensor_cachemask_tensor_qparams_kernel_cuda(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();
  int64_t* fake_quant_on = fake_quant_enabled.data_ptr<int64_t>();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_output(mask)
    .add_input(input)
    .build();

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_types", [&] {
      gpu_kernel_multiple_outputs(
        iter,
        [=] GPU_LAMBDA (scalar_t input_val) -> thrust::tuple<scalar_t, bool> {
          if (*fake_quant_on == 0) {
            return {input_val, 1};
          }
          float inv_scale = 1.0f / (*scale_ptr);
          const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + (*zp_ptr));
          return {
            // fake_quantized value
            (fminf(quant_max, fmaxf(quant_min, qval)) - (*zp_ptr)) * (*scale_ptr),
            // mask for grad
            ((quant_min <= qval) && (qval <= quant_max))
          };
        }
      );
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_types", [&] {
      gpu_kernel_multiple_outputs(
        iter,
        [=] GPU_LAMBDA (scalar_t input_val) -> thrust::tuple<scalar_t, bool> {
          if (*fake_quant_on == 0) {
            return {input_val, 1};
          }
          float inv_scale = 1.0f / (*scale_ptr);
          const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + (*zp_ptr));
          return {
            // fake_quantized value
            (fminf(quant_max, fmaxf(quant_min, qval)) - (*zp_ptr)) * (*scale_ptr),
            // mask for grad
            ((quant_min <= qval) && (qval <= quant_max))
          };
        }
      );
    });
  }
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

REGISTER_DISPATCH(fake_quant_tensor_cachemask_stub, &fake_quantize_tensor_cachemask_kernel_cuda)
REGISTER_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub, &fake_quantize_tensor_cachemask_tensor_qparams_kernel_cuda)
REGISTER_DISPATCH(fake_quant_grad_learnable_tensor_stub, &_fake_quantize_grad_learnable_tensor_kernel_cuda)

// Fake quantize per channel

template<typename SelfType>
void _fake_quant_per_channel_cachemask_cuda_helper(
    TensorIterator & iter,
    TensorIterator & iter_mask,
    const int64_t quant_min,
    const int64_t quant_max
) {
  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.

  //
  const auto & zero_point_dtype = iter.input_dtype(2);

  //
  if (at::isFloatingType(zero_point_dtype)) {
    // When zero_point is float, quantize mirroring affine quantizer equation
    // Xq = Round(Xf * inv_scale + zero_point)
    // where zero_point is in float.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_point_dtype, "fake_quantize_channel_cachemask_cuda_mask_type_handling", [&] {
      // write mask
      gpu_kernel(iter_mask,
        [=] GPU_LAMBDA (const SelfType input_val, const float scale, const scalar_t zero_point) -> bool {
          const float inv_scale = 1.0f / scale;
          const auto qval = std::lrint(input_val * inv_scale + zero_point);
          return ((quant_min <= qval) && (qval <= quant_max));
      });

      // write fake_quant
      gpu_kernel(iter,
        [=] GPU_LAMBDA (const SelfType input_val, const float scale, const scalar_t zero_point) -> SelfType {
          const float inv_scale = 1.0f / scale;
          const auto qval = std::lrint(input_val * inv_scale + zero_point);
          const auto bounded_qval = fminf(quant_max, fmaxf(quant_min, qval));
          return (bounded_qval - zero_point) * scale;
      });
    });
  }
  else {
    // write mask
    gpu_kernel(iter_mask,
      [=] GPU_LAMBDA (const SelfType input_val, const float scale, const int64_t zero_point) -> bool {
        const float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale)) + zero_point;
        return ((quant_min <= qval) && (qval <= quant_max));
    });

    // write fake_quant
    gpu_kernel(iter,
      [=] GPU_LAMBDA (const SelfType input_val, const float scale, const int64_t zero_point) -> SelfType {
        const float inv_scale = 1.0f / scale;
        const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale)) + zero_point;
        const auto bounded_qval = std::min(quant_max, std::max(quant_min, qval));
        return (bounded_qval - zero_point) * scale;
    });
  }
}


void fake_quant_per_channel_cachemask_cuda(
    TensorIterator &iter, TensorIterator &iter_mask, int64_t quant_min, int64_t quant_max) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "fake_quantize_channel_cachemask_cuda_type_handling", [&] {
      _fake_quant_per_channel_cachemask_cuda_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cuda_type_handling", [&] {
      _fake_quant_per_channel_cachemask_cuda_helper<scalar_t>(iter, iter_mask, quant_min, quant_max);
    });
  }
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

REGISTER_DISPATCH(fake_quant_per_channel_cachemask_stub, &fake_quant_per_channel_cachemask_cuda)
REGISTER_DISPATCH(fake_quant_grad_learnable_channel_stub, &_fake_quantize_grad_learnable_channel_kernel_cuda)

} // namespace at::native
