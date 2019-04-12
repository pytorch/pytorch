#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace at { namespace native {

std::vector<int64_t> expand_param_if_needed(
    IntArrayRef param,
    int64_t expected_dim);

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation);
}}
