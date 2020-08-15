#pragma once

#include <algorithm>
#include <vector>

#include <ATen/div_rtn.h>
#include <ATen/ATen.h>

#define TORCH_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE) \
  TORCH_CHECK(                                       \
      T.dim() == DIM && T.size(DIM_SIZE) == SIZE,    \
      "Need " #T " of dimension ",                   \
      DIM,                                           \
      " and " #T ".size[",                           \
      DIM_SIZE,                                      \
      "] == ",                                       \
      SIZE,                                          \
      " but got input to be of shape ",              \
      T.sizes())

namespace at {
namespace native {
namespace internal {
namespace {
inline bool all_positive(IntArrayRef& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

inline bool all_nonnegative(std::vector<int64_t>& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item >= 0; });
}

} // namespace

// calculate the rear part of output tensor sizes
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  std::vector<int64_t> sizes;
  for (int index = 0; index < dim; index++) {
    sizes.push_back(
        div_rtn<int64_t>(
            input.size(index + input.dim() - dim) + 2 * pad_size[index] -
                (dilation_size[index] * (kernel_size[index] - 1) + 1),
            stride_size[index]) +
        1);
  }
  return sizes;
}

// calculate the sizes of output tensor
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto output_size = get_output_size<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);
  output_size.insert(output_size.begin(), weight.size(0));
  if (input.dim() == dim + 2) {
    output_size.insert(output_size.begin(), input.size(0));
  }
  return output_size;
}
/*
  slow_conv_dilated_shape_check - check user-input to dilated convolution
  forward and backward functions.
*/
template <int64_t dim>
void slow_conv_dilated_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  /*
    When the following tensors are defined:

    bias, grad_weight, grad_output

    then these are assumed to be contiguous without checking
    because of these tensors are made contiguous by calling
    .contiguous() method or by resizing of zero-sized tensors in
    forward/backward functions.

    When grad_weight is defined then it is assumed without
    checking to have the same shape as weight, see backward
    functions.
   */
  // Check size arguments
  TORCH_CHECK(
      kernel_size.size() == dim,
      "kernel sizes length should be ",
      dim,
      ", but got ",
      kernel_size.size());
  TORCH_CHECK(
      stride_size.size() == dim,
      "strides length should be ",
      dim,
      ", but got ",
      stride_size.size());
  TORCH_CHECK(
      dilation_size.size() == dim,
      "dilations length should be ",
      dim,
      ", but got ",
      dilation_size.size());
  TORCH_CHECK(
      pad_size.size() == dim,
      "pads length should be ",
      dim,
      ", but got ",
      pad_size.size());

  TORCH_CHECK(
      all_positive(kernel_size),
      "kernel size should be greater than zero, but got ",
      kernel_size);
  TORCH_CHECK(
      all_positive(stride_size),
      "stride should be greater than zero, but got ",
      stride_size);
  TORCH_CHECK(
      all_positive(dilation_size),
      "dilation should be greater than zero, but got ",
      dilation_size);

  // check input
  TORCH_CHECK(input.defined(), "input must be defined");
  bool is_batch = input.dim() == dim + 2;
  int64_t n = (is_batch ? 2 : 1);
  int64_t ndim = n + dim;
  if (!is_batch) {
    // input dim has to be dim + 1 if not batched
    TORCH_CHECK(
        input.dim() == dim + 1,
        "input must be 4D or 5D tensor but got ",
        input.dim(),
        "D tensor");
  }

  // check output sizes
  auto output_size = get_output_size<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);

  TORCH_CHECK(
      all_nonnegative(output_size),
      "calculated output size ",
      output_size,
      " is too small (all sizes must be non-negative)");

  // check weight
  TORCH_CHECK(weight.defined(), "weight must be defined");
  TORCH_CHECK(
      weight.dim() == dim + 2,
      "weight must be ",
      dim + 2,
      "D tensor but got ",
      weight.dim(),
      "D tensor dim=",
      dim);
  TORCH_CHECK(
      weight.sizes().slice(2) == kernel_size,
      "weight[2:] shape ",
      weight.sizes().slice(2),
      " must be equal to kernel_size ",
      kernel_size);

  TORCH_CHECK_DIM_SIZE(input, input.dim(), (is_batch ? 1 : 0), weight.size(1));

  // check bias when present
  if (bias.defined()) {
    TORCH_CHECK(
        bias.dim() == 1,
        "bias must be 1D tensor but got ",
        bias.dim(),
        "D tensor");
    TORCH_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
  }

  // check grad_output when present
  if (grad_output.defined()) {
    TORCH_CHECK(
        grad_output.dim() == ndim,
        "grad_output must be ",
        ndim,
        "D tensor but got ",
        grad_output.dim(),
        "D tensor");
    if (is_batch) {
      TORCH_CHECK(
          grad_output.size(0) == input.size(0),
          "grad_output.size(0)=",
          grad_output.size(0),
          " must be input.size(0)=",
          input.size(0));
    }
    TORCH_CHECK(
        grad_output.size(n - 1) == weight.size(0),
        "grad_output.size(",
        n - 1,
        ")=",
        grad_output.size(n - 1),
        " must be weight.size(0)=",
        weight.size(0));
    TORCH_CHECK(
        grad_output.sizes().slice(n) == output_size,
        "grad_output[",
        n,
        ":] shape",
        grad_output.sizes().slice(n),
        " must be equal to output size ",
        output_size);
  }
}

} // namespace internal
} // namespace native
} // namespace at
