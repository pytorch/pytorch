//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/zeros.h>

namespace at::native {

using namespace mps;

Tensor _mps_linear(const Tensor& input, const Tensor& weight_arg, const std::optional<Tensor>& bias_opt) {
  // wT = transpose(weight);
  // y=x*wT+b

  TORCH_CHECK(supportedFloatingOrComplexType(input), "MPS device does not support linear for non-float inputs");
  TORCH_CHECK(input.is_mps(), "Tensor for argument input is on ", input.device(), " but expected on mps");
  TORCH_CHECK(supportedFloatingOrComplexType(weight_arg), "MPS device does not support linear for non-float weights");
  TORCH_CHECK(weight_arg.is_mps(), "Tensor for argument weight is on ", weight_arg.device(), " but expected on mps");

  const Tensor& bias = *(at::borrow_from_optional_tensor(bias_opt));
  const bool is_bias_defined = bias.defined();
  if (is_bias_defined) {
    TORCH_CHECK(bias.is_mps(), "Tensor for argument bias is on ", bias.device(), " but expected on mps");
    TORCH_CHECK(supportedFloatingOrComplexType(bias), "MPS device does not support linear for non-float bias");
  }

  auto weight = (weight_arg.dim() == 1) ? weight_arg.unsqueeze(0) : weight_arg;

  auto input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  TORCH_CHECK(input.size(-1) == weight_arg.size(-1),
              "linear(): input and weight.T shapes cannot be multiplied (",
              input.size(-2),
              "x",
              input.size(-1),
              " and ",
              weight_arg.size(-1),
              "x",
              weight_arg.size(-2),
              ")");

  if (is_bias_defined) {
    // Check bias and output shapes compatibility only.
    inferExpandGeometry_dimvector(bias.sizes(), bias.strides(), output_size);
  }

  Tensor output =
      at::empty(output_size, input.scalar_type(), std::nullopt, kMPS, std::nullopt, input.suggest_memory_format());

  if (output.numel() == 0) {
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }

  // An empty reduction dimension (in_features == 0) makes the matmul term
  // zero, so the result is just the broadcast bias. Neither the NDArray fast
  // path nor MPSGraph can take a zero-length dimension (MPSNDArray asserts
  // and aborts the process).
  if (input.size(-1) == 0) {
    if (is_bias_defined) {
      output.copy_(bias.expand(output.sizes()));
    } else {
      output.zero_();
    }
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }

  // See pytorch/pytorch#177116. Delegating to addmm/mm reaches the metal kernels, which
  // take the weight's real strides, so the transposed (column-major) operand costs nothing.
  const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;
  // addmm fuses the bias and routes rank-1 shapes to the GEMV kernels. A multi-dim bias
  // cannot broadcast against the 2D result, so it is added after the reshape instead.
  const bool fuse_bias = is_bias_defined && bias.dim() <= 1;
  auto result = (fuse_bias ? at::addmm(bias, input_2d, weight.t()) : at::mm(input_2d, weight.t())).view(output_size);
  if (is_bias_defined && !fuse_bias) {
    result.add_(bias);
  }
  result = result.contiguous(input.suggest_memory_format());
  // Squeeze last dim of 1D linear
  return weight_arg.dim() != 1 ? result : result.squeeze(-1);
}

static Tensor _mps_linear_backward_input(IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(grad_output.is_mps(), "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && supportedFloatingOrComplexType(weight),
              "mps_linear_backward: unsupported weights data type: ",
              weight.scalar_type());
  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  // An empty grad_output (out_features == 0) zeroes the grad-input; a zero-length
  // input_size dim (in_features == 0) makes it empty. Neither can go through mm.
  if (grad_output.numel() == 0 || c10::multiply_integers(input_size) == 0) {
    return at::zeros(input_size, grad_output.options());
  }

  const auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
  // A 1D weight is the out_features == 1 case with the trailing output dim squeezed
  // (see _mps_linear), so grad_output is missing it too. Restore both, otherwise mm
  // gets a vector for mat2 and rejects it.
  if (weight.dim() == 1) {
    return at::mm(grad_output.reshape({-1, 1}), weight_contig.unsqueeze(0).conj()).view(input_size);
  }
  const auto grad_output_2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  return at::mm(grad_output_2d, weight_contig.conj()).view(input_size);
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  // A 1D weight is the out_features == 1 case with the trailing output dim squeezed
  // (see _mps_linear), so grad_output is missing it too; flattening to {-1, 1}
  // restores it and grad_weight is viewed back to the weight's shape below.
  const auto out_features = weight.dim() == 1 ? 1 : grad_output.size(-1);

  // Guard before the reshapes below: for a 0-element input, reshape({-1, 0}) is
  // ambiguous and throws. The weight gradient is empty or zero here, but the bias
  // gradient is still the sum of grad_output over the leading dims.
  if (grad_output.numel() == 0 || input.numel() == 0) {
    auto grad_weight = at::zeros(weight.sizes(), grad_output.options());
    Tensor grad_bias;
    if (bias_defined) {
      grad_bias = at::zeros({out_features}, grad_output.options());
      if (grad_output.numel() != 0) {
        grad_bias.copy_(grad_output.reshape({-1, out_features}).sum(0));
      }
    }
    return {grad_weight, grad_bias};
  }

  const auto grad_output_2d = grad_output.reshape({-1, out_features});
  const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;

  // Route through at::mm so the dispatcher can pick the Metal fallback for K-dim
  // overflow on Apple7/8 (M1/M2). See pytorch/pytorch#177116.
  auto grad_weight = at::mm(grad_output_2d.t(), input_2d.contiguous().conj()).view(weight.sizes());
  // autocast promotes sum() to float32, but linear_backward's meta keeps grad_output's
  // dtype; cast back so inductor's baked-in dtype matches the runtime buffer.
  auto grad_bias = bias_defined ? grad_output_2d.sum(0).to(grad_output.scalar_type()) : Tensor();
  return {grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> mps_linear_backward(const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& weight,
                                                       std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = _mps_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = _mps_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace at::native
