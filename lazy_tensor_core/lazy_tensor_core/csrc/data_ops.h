#pragma once

#include <c10/util/Optional.h>

#include <vector>

// Collection of lowerings for operations which only involve some form of data
// movement and no computation.
namespace torch_lazy_tensors {

// For input_sizes and a potentially incomplete output_sizes, return a complete
// output shape. The complete output shape has same total number of elements as
// input_sizes and matches output_sizes in all dimensions except for at most
// one, which can be inferred and stored as -1 in output_sizes.
std::vector<int64_t> GetCompleteShape(c10::ArrayRef<int64_t> output_sizes,
                                      c10::ArrayRef<int64_t> input_sizes);

std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim);

std::vector<int64_t> BuildUnsqueezeDimensions(c10::ArrayRef<int64_t> dimensions,
                                              int64_t dim);

// Computes the number of splits with a dimension size and the split sizes.
size_t ComputeSplitCount(int64_t dim_size, c10::ArrayRef<int64_t> split_sizes);

}  // namespace torch_lazy_tensors
