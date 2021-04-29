#include <torch/nn/modules/_functions.h>

using namespace torch::autograd;

namespace torch {
namespace nn {
namespace functions {

Variable CrossMapLRN2d::forward(
    AutogradContext *ctx,
    const Variable& input,
    const CrossMapLRN2dOptions& options){
  ctx->saved_data["size"] = options.size();
  ctx->saved_data["alpha"] = options.alpha();
  ctx->saved_data["beta"] = options.beta();
  ctx->saved_data["k"] = options.k();
  ctx->saved_data["scale"] = torch::Tensor();

  TORCH_CHECK(input.dim() == 4);

  ctx->saved_data["scale"] = ctx->saved_data["scale"].toTensor().defined() ? ctx->saved_data["scale"] : torch::empty({0}, input.options());

  torch::Tensor output = torch::empty({0}, input.options());

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
  int64_t batch_size = input.size(0);
  int64_t channels = input.size(1);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
  int64_t input_height = input.size(2);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
  int64_t input_width = input.size(3);

  output.resize_as_(input);
  ctx->saved_data["scale"].toTensor().resize_as_(input);

  /// use output storage as temporary buffer
  auto input_square = output;
  torch::pow_out(input_square, input, 2);

  int64_t pre_pad = static_cast<int64_t>((ctx->saved_data["size"].toInt() - 1) / 2 + 1);
  int64_t pre_pad_crop = pre_pad > channels ? channels : pre_pad;

  auto scale_first = ctx->saved_data["scale"].toTensor().select(1, 0);
  scale_first.zero_();

  /// compute first feature map normalization
  for (int64_t c = 0; c < pre_pad_crop; ++c) {
    scale_first.add_(input_square.select(1, c));
  }

  /// reuse computations for next feature maps normalization
  /// by adding the next feature map and removing the previous
  torch::Tensor scale_previous, scale_current, square_next, square_previous;

  for (int64_t c = 1; c < channels; ++c) {
    scale_previous = ctx->saved_data["scale"].toTensor().select(1, c - 1);
    scale_current = ctx->saved_data["scale"].toTensor().select(1, c);
    scale_current.copy_(scale_previous);

    if (c < channels - pre_pad + 1) {
      square_next = input_square.select(1, c + pre_pad - 1);
      scale_current.add_(square_next, 1);
    }

    if (c > pre_pad) {
      square_previous = input_square.select(1, c - pre_pad);
      scale_current.add_(square_previous, -1);
    }
  }

  ctx->saved_data["scale"].toTensor()
      .mul_(ctx->saved_data["alpha"].toDouble() / ctx->saved_data["size"].toInt())
      .add_(ctx->saved_data["k"].toInt());

  torch::pow_out(output, ctx->saved_data["scale"].toTensor(), -ctx->saved_data["beta"].toDouble());
  output.mul_(input);

  ctx->save_for_backward({input, output});
  return output;
}

variable_list CrossMapLRN2d::backward(AutogradContext *ctx, variable_list grad_outputs) {
  auto grad_output = grad_outputs[0];
  auto input = ctx->get_saved_variables()[0];
  auto output = ctx->get_saved_variables()[1];
  auto grad_input = torch::empty({0}, grad_output.options());

  int64_t batch_size = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  auto padded_ratio = torch::empty({channels + ctx->saved_data["size"].toInt() - 1, input_height, input_width},
                                    input.options());
  auto accum_ratio = torch::empty({input_height, input_width},
                                    input.options());
  double cache_ratio_value = 2 * ctx->saved_data["alpha"].toDouble() * ctx->saved_data["beta"].toDouble() / ctx->saved_data["size"].toInt();
  int64_t inversePrePad = static_cast<int64_t>(ctx->saved_data["size"].toInt() - (ctx->saved_data["size"].toInt() - 1) / 2);

  grad_input.resize_as_(input);
  torch::pow_out(grad_input, ctx->saved_data["scale"].toTensor(), -ctx->saved_data["beta"].toDouble()).mul_(grad_output);

  padded_ratio.zero_();
  auto padded_ratio_center = padded_ratio.narrow(0, inversePrePad, channels);

  for (int64_t n = 0; n < batch_size; ++n) {
    torch::mul_out(padded_ratio_center, grad_output[n], output[n]);
    padded_ratio_center.div_(ctx->saved_data["scale"].toTensor()[n]);
    torch::sum_out(
        accum_ratio,
        padded_ratio.narrow(0, 0, ctx->saved_data["size"].toInt() - 1),
        0, /*keepdim=*/false);
    for (int64_t c = 0; c < channels; ++c) {
      accum_ratio.add_(padded_ratio[c + ctx->saved_data["size"].toInt() - 1]);
      grad_input[n][c].addcmul_(input[n][c], accum_ratio, -cache_ratio_value);
      accum_ratio.add_(padded_ratio[c], -1);
    }
  }

  return variable_list{grad_input, Variable(), Variable(), Variable(), Variable()};
}

}
} // namespace nn
} // namespace torch
