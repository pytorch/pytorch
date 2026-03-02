//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/AdaptivePooling.h>
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
#include <ATen/ops/mm.h>
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

namespace {

Tensor build_adaptive_avg_pool2d_weight(int64_t input_size, int64_t output_size, const Tensor& input) {
  TORCH_CHECK(output_size > 0, "adaptive_avg_pool2d(): output sizes must be positive");

  const auto weight_dtype = input.scalar_type();
  auto weight = at::zeros({output_size, input_size}, input.options().device(kCPU).dtype(weight_dtype));

  for (const auto out_idx : c10::irange(output_size)) {
    const auto start = start_index(out_idx, output_size, input_size);
    const auto end = end_index(out_idx, output_size, input_size);
    const double scale = 1.0 / static_cast<double>(end - start);
    weight[out_idx].narrow(0, start, end - start).fill_(scale);
  }

  return weight.to(kMPS);
}

} // namespace

// Adaptive average pooling
Tensor& adaptive_avg_pool2d_out_mps(const Tensor& input, IntArrayRef output_size, Tensor& output) {
  TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  TORCH_CHECK((input.dim() == 3 || input.dim() == 4),
              "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
              input.sizes());
  for (int64_t i = 1; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
                "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                "but input has sizes ",
                input.sizes(),
                " with dimension ",
                i,
                " being empty");
  }
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "expected dtype ",
              input.scalar_type(),
              " for `output` but got dtype ",
              output.scalar_type());

  const auto output_height = output_size[0];
  const auto output_width = output_size[1];
  TORCH_CHECK(output_height > 0 && output_width > 0, "adaptive_avg_pool2d(): output sizes must be positive");

  const bool has_batch = input.dim() == 4;
  const auto memory_format = input.suggest_memory_format();
  const bool restore_channels_last = has_batch && memory_format == MemoryFormat::ChannelsLast;

  if (has_batch) {
    output.resize_({input.size(0), input.size(1), output_height, output_width}, memory_format);
  } else {
    output.resize_({input.size(0), output_height, output_width});
  }
  if (output.numel() == 0) {
    return output;
  }

  Tensor contiguous_input = has_batch ? input : input.unsqueeze(0);
  contiguous_input = contiguous_input.contiguous(MemoryFormat::Contiguous);

  const auto batch = contiguous_input.size(0);
  const auto channels = contiguous_input.size(1);
  const auto input_height = contiguous_input.size(2);
  const auto input_width = contiguous_input.size(3);

  const auto weight_w = build_adaptive_avg_pool2d_weight(input_width, output_width, contiguous_input).t();
  const auto weight_h = build_adaptive_avg_pool2d_weight(input_height, output_height, contiguous_input).t();

  auto reshaped = contiguous_input.reshape({batch * channels * input_height, input_width});
  auto temp = at::mm(reshaped, weight_w).view({batch, channels, input_height, output_width});
  temp = temp.permute({0, 1, 3, 2}).contiguous();

  auto reduced = at::mm(temp.reshape({batch * channels * output_width, input_height}), weight_h);
  reduced = reduced.view({batch, channels, output_width, output_height}).permute({0, 1, 3, 2});

  if (!has_batch) {
    reduced = reduced.squeeze(0);
  }
  if (restore_channels_last && reduced.dim() == 4) {
    reduced = reduced.contiguous(MemoryFormat::ChannelsLast);
  } else {
    reduced = reduced.contiguous();
  }

  output.copy_(reduced);

  return output;
}

Tensor adaptive_avg_pool2d_mps(at::Tensor const& input, IntArrayRef output_size) {
  IntArrayRef output_shape;

  auto output_height = output_size[0];
  auto output_width = output_size[1];

  std::vector<long long> out_dims = {};

  if (input.ndimension() == 4) {
    auto sizeB = input.size(0);
    auto sizeD = input.size(1);

    out_dims.push_back(sizeB);
    out_dims.push_back(sizeD);
    out_dims.push_back(output_height);
    out_dims.push_back(output_width);
    output_shape = IntArrayRef(out_dims);
  } else {
    auto sizeD = input.size(0);
    out_dims.push_back(sizeD);
    out_dims.push_back(output_height);
    out_dims.push_back(output_width);
    output_shape = IntArrayRef(out_dims);
  }

  const auto memory_format = input.suggest_memory_format();
  Tensor output = at::empty(output_shape, input.scalar_type(), std::nullopt, kMPS, std::nullopt, memory_format);
  return adaptive_avg_pool2d_out_mps(input, output_size, output);
}

Tensor adaptive_avg_pool2d_backward_mps(const Tensor& gradOutput, const Tensor& input) {
  adaptive_pool_empty_output_check(gradOutput, "adaptive_avg_pool2d_backward");
  TORCH_CHECK(gradOutput.dim() == input.dim(),
              __func__,
              ": Expected dimensions ",
              input.dim(),
              " for `grad_output` but got dimensions ",
              gradOutput.dim());
  TORCH_CHECK((gradOutput.dim() == 3 || gradOutput.dim() == 4),
              __func__,
              ": Expected 3D or 4D tensor, but got ",
              input.sizes());
  TORCH_CHECK(input.scalar_type() == gradOutput.scalar_type(),
              __func__,
              ": Expected dtype ",
              input.scalar_type(),
              " for `grad_output` but got dtype ",
              gradOutput.scalar_type());

  const bool has_batch = gradOutput.dim() == 4;
  const auto memory_format = input.suggest_memory_format();
  const bool restore_channels_last = has_batch && memory_format == MemoryFormat::ChannelsLast;

  Tensor grad_output_contig = has_batch ? gradOutput : gradOutput.unsqueeze(0);
  grad_output_contig = grad_output_contig.contiguous(MemoryFormat::Contiguous);

  if (grad_output_contig.numel() == 0) {
    auto grad_input = at::zeros_like(input);
    if (restore_channels_last && grad_input.dim() == 4) {
      grad_input = grad_input.contiguous(MemoryFormat::ChannelsLast);
    }
    return grad_input;
  }

  const auto batch = grad_output_contig.size(0);
  const auto channels = grad_output_contig.size(1);
  const auto output_height = grad_output_contig.size(2);
  const auto output_width = grad_output_contig.size(3);
  const auto input_height = input.size(-2);
  const auto input_width = input.size(-1);

  const auto weight_w = build_adaptive_avg_pool2d_weight(input_width, output_width, grad_output_contig);
  const auto weight_h = build_adaptive_avg_pool2d_weight(input_height, output_height, grad_output_contig);

  auto grad_temp = grad_output_contig.permute({0, 1, 3, 2}).contiguous();
  grad_temp = at::mm(grad_temp.reshape({batch * channels * output_width, output_height}), weight_h)
                  .view({batch, channels, output_width, input_height})
                  .permute({0, 1, 3, 2})
                  .contiguous();

  auto grad_input = at::mm(grad_temp.reshape({batch * channels * input_height, output_width}), weight_w)
                        .view({batch, channels, input_height, input_width});

  if (!has_batch) {
    grad_input = grad_input.squeeze(0);
  }
  if (restore_channels_last && grad_input.dim() == 4) {
    grad_input = grad_input.contiguous(MemoryFormat::ChannelsLast);
  }

  return grad_input;
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
