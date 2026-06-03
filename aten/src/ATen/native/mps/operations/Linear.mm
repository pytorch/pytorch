//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/GemmMetal.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>

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
    return output;
  }

  const bool is_complex = input.is_complex() || weight.is_complex() || (is_bias_defined && bias.is_complex());

  // y = x @ weight.T (+ bias). The hand-written GEMM kernels handle the
  // transpose / strides; flatten leading dims to a 2-D (M, K) x (K, N) problem.
  if (!is_complex && gemm_supported_dtype(input.scalar_type())) {
    auto input2d = input.dim() == 2 ? input : input.reshape({-1, input.size(-1)});
    auto output2d = output.dim() == 2 ? output : output.reshape({-1, weight.size(0)});
    // A 1-D bias (the usual (out_features,) case) fuses into the GEMM epilogue
    // as a row broadcast. A higher-rank bias broadcasts over the input's leading
    // dims, which the 2-D flatten breaks, so add it separately on the full shape.
    if (is_bias_defined && bias.dim() <= 1) {
      mps_gemm(input2d, weight.t(), output2d, bias, /*alpha=*/1, /*beta=*/1,
               at_gemm::GemmEpilogue::AlphaBeta);
    } else {
      mps_gemm(input2d, weight.t(), output2d, std::nullopt, /*alpha=*/1, /*beta=*/0,
               at_gemm::GemmEpilogue::None);
      if (is_bias_defined) {
        output.add_(bias);
      }
    }
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }

  // Complex: y = x @ weight.T (+ bias) via the decomposed complex GEMM. A 1-D bias
  // fuses as a row-broadcast epilogue; a higher-rank bias is added on the full
  // shape (the 2-D flatten would otherwise break its broadcast over leading dims).
  TORCH_INTERNAL_ASSERT(is_complex, "linear: unsupported non-complex dtype reached fallback");
  auto input2d = input.dim() == 2 ? input : input.reshape({-1, input.size(-1)});
  auto output2d = output.dim() == 2 ? output : output.reshape({-1, weight.size(0)});
  if (is_bias_defined && bias.dim() <= 1) {
    mps_gemm_complex(input2d, weight.t(), output2d, bias, /*alpha=*/1, /*beta=*/1,
                     at_gemm::GemmEpilogue::AlphaBeta);
  } else {
    mps_gemm_complex(input2d, weight.t(), output2d, std::nullopt, /*alpha=*/1, /*beta=*/0,
                     at_gemm::GemmEpilogue::None);
    if (is_bias_defined) {
      output.add_(bias);
    }
  }
  // Squeeze last dim of 1D linear
  return weight_arg.dim() != 1 ? output : output.squeeze(-1);
}

static Tensor _mps_linear_backward_input(IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(grad_output.is_mps(), "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && supportedFloatingOrComplexType(weight),
              "mps_linear_backward: unsupported weights data type: ",
              weight.scalar_type());
  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  Tensor output = at::empty(input_size, grad_output.options());
  TORCH_CHECK(output.is_mps());
  if (grad_output.numel() == 0) {
    return output;
  }

  // grad_input = grad_output @ weight, flattened to a 2-D (M, out) x (out, in)
  // problem. mm routes through the MPSGraph-free GEMM kernels / complex path.
  auto go2d = grad_output.dim() == 2 ? grad_output : grad_output.reshape({-1, grad_output.size(-1)});
  output.view({-1, weight.size(-1)}).copy_(go2d.mm(weight));
  return output;
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");
  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  auto go2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  auto in2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;

  Tensor grad_weight = at::empty({go2d.size(1), in2d.size(1)}, grad_output.options());
  Tensor grad_bias = at::empty({go2d.size(1)}, grad_output.options());
  TORCH_CHECK(grad_weight.is_mps());
  TORCH_CHECK(grad_bias.is_mps());

  if (grad_output.numel() == 0) {
    grad_weight.zero_();
    grad_bias.zero_();
    return std::tuple<Tensor, Tensor>{grad_weight, grad_bias};
  }

  // grad_weight = grad_output.T @ input; grad_bias = sum_rows(grad_output). Both
  // run on the MPSGraph-free matmul / reduction kernels.
  grad_weight.copy_(go2d.t().mm(in2d));
  if (bias_defined) {
    grad_bias.copy_(go2d.sum(0));
  }
  return std::tuple<Tensor, Tensor>{grad_weight, grad_bias};
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
