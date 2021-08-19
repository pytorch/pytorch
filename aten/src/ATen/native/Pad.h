#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

// Restrides a pad specifier argument to size(ndim, 2). Using a standard shape
// lets us simplify the logic required to use pad specifier arguments.
// This function also checks that the pad specifier has a legal format.
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
Tensor expand_pad_specifier(const Tensor& pad_spec, const char* arg_name, int64_t ndim);

// Generate slices that can be used to index the original unpadded tensor
// elements from within a padded output
std::vector<at::indexing::TensorIndex> pad_width_to_inner_slices(const Tensor& pad_width, const Tensor& self);

}} // namespace at::native
