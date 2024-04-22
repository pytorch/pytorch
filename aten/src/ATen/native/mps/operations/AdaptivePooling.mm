//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool2d_with_indices_backward.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#endif
namespace at::native {
namespace mps {
static void set_kernel_params(int64_t isizeH,
                              int64_t isizeW,
                              int64_t osizeH,
                              int64_t osizeW,
                              int64_t& strideH,
                              int64_t& strideW,
                              int64_t& kernel_sizeH,
                              int64_t& kernel_sizeW,
                              bool check_avg_pooling = false) {
  TORCH_CHECK((isizeH >= osizeH && isizeW >= osizeW) || (isizeH <= osizeH && isizeW <= osizeW),
              "Adaptive pool MPS: Input height and width must both be greater than, "
              "or equal to, or lesser than output height and width")

  if (isizeH >= osizeH) {
    if (check_avg_pooling) {
      TORCH_CHECK(
          (isizeH % osizeH == 0 && isizeW % osizeW == 0),
          "Adaptive pool MPS: input sizes must be divisible by output sizes. Non-divisible input sizes are not implemented on MPS device yet. For now, you can manually transfer tensor to cpu in this case. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/96056)");
    }
    strideH = (int64_t)(isizeH / osizeH);
    strideW = (int64_t)(isizeW / osizeW);
    kernel_sizeH = isizeH - (osizeH - 1) * strideH;
    kernel_sizeW = isizeW - (osizeW - 1) * strideW;
  } else {
    if (check_avg_pooling) {
      TORCH_CHECK(
          (osizeH % isizeH == 0 && osizeW % isizeW == 0),
          "Adaptive pool MPS: output sizes must be divisible by input sizes. Non-divisible input sizes are not implemented on MPS device yet. For now, you can manually transfer tensor to cpu in this case. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/96056)");
    }
    strideH = (int64_t)(osizeH / isizeH);
    strideW = (int64_t)(osizeW / isizeW);
    kernel_sizeH = osizeH - (isizeH - 1) * strideH;
    kernel_sizeW = osizeW - (isizeW - 1) * strideW;
  }
}
} // namespace mps

// Adaptive average pooling
Tensor& adaptive_avg_pool2d_out_mps(const Tensor& input, IntArrayRef output_size, Tensor& output) {
  for (int64_t i = 1; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
                "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                "but input has sizes ",
                input.sizes(),
                " with dimension ",
                i,
                " being empty");
  }

  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW, true);

  if (isizeH >= osizeH) {
    output = at::avg_pool2d(input,
                            IntArrayRef({kernel_sizeH, kernel_sizeW}),
                            IntArrayRef({strideH, strideW}),
                            IntArrayRef({0, 0}),
                            false,
                            true,
                            c10::nullopt);
  } else {
    Tensor phony_grad = at::ones_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto input_sizes = input.sizes();
    std::vector<int64_t> phony_shape{input_sizes.begin(), input_sizes.end() - 2};
    phony_shape.push_back(output_size[0]);
    phony_shape.push_back(output_size[1]);
    phony_grad.resize_(IntArrayRef(phony_shape));
    output = at::avg_pool2d_backward(input,
                                     phony_grad,
                                     IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                     IntArrayRef({strideH, strideW}),
                                     IntArrayRef({0, 0}),
                                     false,
                                     true,
                                     c10::nullopt);
    // Multiply output by kernel size
    output = at::mul(output, kernel_sizeH * kernel_sizeW);
  }

  return output;
}

Tensor adaptive_avg_pool2d_mps(at::Tensor const& input, IntArrayRef output_size) {
  IntArrayRef output_shape;

  auto osizeH = output_size[0];
  auto osizeW = output_size[1];

  std::vector<long long> out_dims = {};

  if (input.ndimension() == 4) {
    auto sizeB = input.size(0);
    auto sizeD = input.size(1);

    out_dims.push_back(sizeB);
    out_dims.push_back(sizeD);
    out_dims.push_back(osizeH);
    out_dims.push_back(osizeW);
    output_shape = IntArrayRef(out_dims);
  } else {
    auto sizeD = input.size(0);
    out_dims.push_back(sizeD);
    out_dims.push_back(osizeH);
    out_dims.push_back(osizeW);
    output_shape = IntArrayRef(out_dims);
  }

  const auto memory_format = input.suggest_memory_format();
  Tensor output = at::empty(output_shape, input.scalar_type(), c10::nullopt, kMPS, c10::nullopt, memory_format);
  return adaptive_avg_pool2d_out_mps(input, output_size, output);
}

Tensor adaptive_avg_pool2d_backward_mps(const Tensor& gradOutput, const Tensor& input) {
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW, true);

  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (gradInput.numel() != 0) {
    if (isizeH >= osizeH) {
      gradInput = at::avg_pool2d_backward(gradOutput,
                                          input,
                                          IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                          IntArrayRef({strideH, strideW}),
                                          IntArrayRef({0, 0}),
                                          false,
                                          true,
                                          c10::nullopt);
    } else {
      gradInput = at::avg_pool2d(gradOutput,
                                 IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                 IntArrayRef({strideH, strideW}),
                                 IntArrayRef({0, 0}),
                                 false,
                                 true,
                                 c10::nullopt);
      gradInput = at::mul(gradInput, kernel_sizeH * kernel_sizeW);
    }
  }

  return gradInput;
}

// Adaptive max pooling
TORCH_IMPL_FUNC(adaptive_max_pool2d_out_mps)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  for (int64_t i = 1; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
                "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                "but input has sizes ",
                input.sizes(),
                " with dimension ",
                i,
                " being "
                "empty");
  }

  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW);

  at::max_pool2d_with_indices_out(const_cast<Tensor&>(output),
                                  const_cast<Tensor&>(indices),
                                  input,
                                  IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                  IntArrayRef({strideH, strideW}),
                                  IntArrayRef({0, 0}),
                                  IntArrayRef({1, 1}),
                                  false);
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_mps)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices, const Tensor& gradInput) {
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  int64_t strideH = 0, strideW = 0;
  int64_t kernel_sizeH = 0, kernel_sizeW = 0;

  mps::set_kernel_params(isizeH, isizeW, osizeH, osizeW, strideH, strideW, kernel_sizeH, kernel_sizeW);

  at::max_pool2d_with_indices_backward_out(const_cast<Tensor&>(gradInput),
                                           gradOutput,
                                           input,
                                           IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                           IntArrayRef({strideH, strideW}),
                                           IntArrayRef({0, 0}),
                                           IntArrayRef({1, 1}),
                                           false,
                                           indices);
}

} // namespace at::native
