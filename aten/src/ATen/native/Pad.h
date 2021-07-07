#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

// This function is used to process pad arguments that specify before- and
// after-padding qualities for each dimension, such as padding widths or
// constant values to fill the padding with. These arguments share a common
// format, even though they are used in different ways. This function checks
// for valid formats and then restrides to a common size, [ndim, 2]
//
// Arguments:
//
//  pad_spec        The pad specifier argument
//
//  arg_name        Name of the pad specifier, used for error messages
//
//  ndim            Number of dimensions in the tensor to be padded
//
// Returns: pad_spec restrided (if necessary) to the size [ndim, 2]
Tensor expand_pad_specifier_arg(const Tensor& pad_spec, const char* arg_name, int64_t ndim);

// Generate slices that can be used to index the original unpadded tensor
// elements from within a padded output
std::vector<at::indexing::TensorIndex> pad_width_to_inner_slices(const Tensor& pad_width, const Tensor& self);

}} // namespace at::native
