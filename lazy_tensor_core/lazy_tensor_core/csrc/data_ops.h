#pragma once

#include <c10/util/Optional.h>

#include <vector>

#include "lazy_tensors/computation_client/types.h"

// Collection of lowerings for operations which only involve some form of data
// movement and no computation.
namespace torch_lazy_tensors {

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<lazy_tensors::int64> GetCompleteShape(
    c10::ArrayRef<lazy_tensors::int64> output_sizes,
    c10::ArrayRef<lazy_tensors::int64> input_sizes);

std::vector<lazy_tensors::int64> BuildSqueezedDimensions(
    c10::ArrayRef<lazy_tensors::int64> dimensions,
    lazy_tensors::int64 squeeze_dim);

std::vector<lazy_tensors::int64> BuildUnsqueezeDimensions(
    c10::ArrayRef<lazy_tensors::int64> dimensions, lazy_tensors::int64 dim);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(lazy_tensors::int64 dim_size,
                         c10::ArrayRef<lazy_tensors::int64> split_sizes);

}  // namespace torch_lazy_tensors
