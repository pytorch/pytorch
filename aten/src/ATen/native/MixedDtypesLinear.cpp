#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include <iostream>

namespace at {
namespace native {

// FIXME: add support for activation now!
template<typename ElementInputA,
         typename ElementInputB,
         typename ElementOutput,
         bool UseBias>
Tensor
_internal_mixed_dtypes_linear_cpu(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias) {
  const int length_m = input.size(0);
  const int length_k = input.size(1);
  const int length_n = scale.size(0);

  Tensor output = at::zeros({length_m, length_n}, input.options());

  // FIXME: all of the operands are expected in the row-major format!
  const auto* a = input.data_ptr<ElementInputA>();
  const auto* b = weight.data_ptr<ElementInputB>();
  const auto* s = scale.data_ptr<ElementInputA>();
  const auto* c = UseBias ? bias.data_ptr<ElementInputA>() : static_cast<ElementInputA*>(nullptr);
  auto* d = output.data_ptr<ElementOutput>();
  const auto lda = length_k;
  const auto ldb = length_k;
  const auto ldd = length_n;
  for (const auto i : c10::irange(length_m)) {
    for (const auto j : c10::irange(length_n)) {
      float acc = 0; // FIXME: set accumulator datatype according to template arguments!
      for (const auto k : c10::irange(length_k)) {
        acc += a[i * lda + k] * b[j * ldb + k];
      }
      acc *= s[j]; // FIXME: see if quantized values have zero-point, if so apply it beforehand!
      d[i * ldd + j] = acc;
      if constexpr (UseBias) {
        d[i * ldd + j] += c[j];
      }
    }
  }

  return output;
}

template<typename ElementInputA, typename ElementInputB>
Tensor
mixed_dtypes_linear_dispatch_bias_activation_cpu(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias, const c10::string_view& activation) {
    if (activation == "none") {
      if (bias.numel() == 0) {
        return _internal_mixed_dtypes_linear_cpu<
          ElementInputA,
          ElementInputB,
          ElementInputA,
          false>(input.contiguous(), weight.contiguous(), scale, bias);
      } else {
        return _internal_mixed_dtypes_linear_cpu<
          ElementInputA,
          ElementInputB,
          ElementInputA,
          true>(input.contiguous(), weight.contiguous(), scale, bias);
      }
    }
    else {
      AT_ERROR("mixed_dtypes_linear_dispatch_bias_activation: Activation \"",
               activation, "\" is not supported");
      return Tensor{};
    }
}

Tensor
_mixed_dtypes_linear_cpu(const Tensor& input, const Tensor& weight,
                     const Tensor& scale,
                     const c10::optional<Tensor>& bias_opt,
                     const c10::optional<c10::string_view> activation_opt) {

  const auto bias = bias_opt.has_value() ? *bias_opt : Tensor{};
  const auto activation = activation_opt.has_value() ? *activation_opt : "none";

  // FIXME: decide upon actual data-types of input and weight tensors to support!

  // Validate datatypes of input tensors.
  TORCH_CHECK(input.dtype() == at::kHalf ||
              input.dtype() == at::kBFloat16 ||
              input.dtype() == at::kFloat,
              "_mixed_dtypes_linear_cpu: The input datatype ", input.dtype(),
              " is not supported");
  TORCH_CHECK(weight.dtype() == at::kByte,
              "_mixed_dtypes_linear_cpu: The weight datatype ", weight.dtype(),
              " is not supported");
  TORCH_CHECK(scale.dtype() == input.dtype(),
              "_mixed_dtypes_linear_cpu: Expected scale datatype ", input.dtype(),
              " but got", scale.dtype());
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dtype() == input.dtype(),
                "_mixed_dtypes_linear_cpu: Expected bias datatype ", input.dtype(),
                " but got", bias.dtype());
  }

  // FIXME: add support for other types of layouts aside of row-major!

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(input_2d.layout() == Layout::Strided,
              "_mixed_dtypes_linear_cpu: Expected input argument to be strided, "
              "but got layout ", input_2d.layout());
  TORCH_CHECK(input_2d.dim() == 2,
              "_mixed_dtypes_linear_cpu: Expected input argument to be 2D tensor, "
              "got ", input_2d.dim(), " dims");
  const auto strides_input = input_2d.strides();
  TORCH_CHECK(strides_input[0] > 1 && strides_input[1] == 1,
              "_mixed_dtypes_linear_cpu: Invalid strides for input argument: row "
              "stride = ", strides_input[0], ", column stride = ",
              strides_input[1]);
  TORCH_CHECK(weight.layout() == Layout::Strided,
              "_mixed_dtypes_linear_cpu: Expected input argument to be strided, "
              "but got layout ", weight.layout());
  TORCH_CHECK(weight.dim() == 2,
              "_mixed_dtypes_linear_cpu: Expected weight argument to be 2D tensor, "
              "got ", weight.dim(), " dims");
  const auto strides_weight = weight.strides();
  TORCH_CHECK(strides_weight[0] > 1 && strides_weight[1] == 1,
              "_mixed_dtypes_linear_cpu: Invalid strides for weight argument: row "
              "stride = ", strides_weight[0], ", column stride = ",
              strides_weight[1]);
  TORCH_CHECK(scale.dim() == 1,
              "_mixed_dtypes_linear_cpu: Expected scale argument to be 1D tensor, "
              "got ", scale.dim(), " dims");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dim() == 1,
                "_mixed_dtypes_linear_cpu: Expected bias argument to be 1D ",
                "tensor, got ", bias.dim(), " dims");
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(input_2d.size(1) == weight.size(1),
              "_mixed_dtypes_linear_cpu: Expected input argument to have ",
              weight.size(1), " columns, but got ", input_2d.size(1));
  TORCH_CHECK(weight.size(0) == scale.size(0)  ||
              2 * weight.size(0) == scale.size(0),
              "_mixed_dtypes_linear_cpu: Expected weight argument to have either ",
              scale.size(0), " or ", scale.size(0) / 2.f, " columns, but got ",
              weight.size(0));
  if (bias.numel() != 0) {
      TORCH_CHECK(bias.size(0) == scale.size(0),
                  "_mixed_dtypes_linear_cpu: Expected bias argument to have ",
                  scale.size(0), " elements, but got ", bias.size(0));
  }

  Tensor output;
  auto scalar_type_quant = weight.scalar_type();
  // if (weight.size(1) != scale.size(0)) {
  //   scalar_type_quant = at::ScalarType::QUInt4x2;
  // }
    AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "_mixed_dtypes_linear",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation_cpu<
                              at::Half,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation_cpu<
                              at::BFloat16,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          }));



  auto output_sizes = input_sizes;
  output_sizes.back() = scale.size(0);
  return output.reshape(output_sizes);
}

}  // namespace native
}  // namespace at
