#pragma once

#include "Descriptors.h"

#include <ATen/ScalarType.h>

#include "cudnn-wrapper.h"
#include <vector>

namespace at { namespace native {

// API overview:
// - We provide seperate functions for transposed convolution and
//   backwards convolution, even though they are algorithmically
//   the same.  This is because there are different conventions
//   for resolving the ambiguity in output sizing (due to the
//   floor in the convolution formula).  With backwards, it is
//   generally assumed that the desired output size is known;
//   with transposed convolution, the ambiguity is instead resolved
//   using an extra output_padding parameter.
// - The convention for desired output size is that it always
//   comes first in the argument list, before even the tensor arguments.
// - The convention for output_padding is that it always comes after
//   padding, when the function accepts it.
// - It's not necessary to provide a backward transposed convolution
//   distinct from forward convolution, as there is no ambiguity
//   to resolve here.

at::Tensor cudnn_convolution_forward(
    const at::Tensor& input, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_full_forward(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_backward(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_transpose_full_forward(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList output_padding, IntList stride, IntList dilation,
    int64_t groups, bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_transpose_backward(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_backward_weight(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_transpose_backward_weight(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    bool benchmark, bool deterministic);

at::Tensor cudnn_convolution_backward_bias(
    const at::Tensor& grad_output);

}}  // namespace at::cudnn
