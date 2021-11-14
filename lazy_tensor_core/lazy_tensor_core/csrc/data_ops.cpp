#include "lazy_tensor_core/csrc/data_ops.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

std::vector<int64_t> GetCompleteShape(c10::ArrayRef<int64_t> output_sizes,
                                      c10::ArrayRef<int64_t> input_sizes) {
  c10::optional<size_t> incomplete_dim;
  int64_t incomplete_element_count = 1;
  for (size_t dim = 0; dim < output_sizes.size(); ++dim) {
    int64_t dim_size = output_sizes[dim];
    if (dim_size < 0) {
      CHECK(!incomplete_dim)
          << "More than one incomplete dimension found: " << *incomplete_dim
          << " and " << dim;
      incomplete_dim = dim;
    } else {
      incomplete_element_count *= dim_size;
    }
  }
  int64_t total_element_count =
      lazy_tensors::util::Multiply<int64_t>(input_sizes);
  if (!incomplete_dim) {
    CHECK_EQ(total_element_count,
             lazy_tensors::util::Multiply<int64_t>(output_sizes))
        << "(" << c10::Join(", ", output_sizes) << ") vs. ("
        << c10::Join(", ", input_sizes) << ")";
    return lazy_tensors::util::ToVector<int64_t>(output_sizes);
  }
  CHECK_GT(incomplete_element_count, 0)
      << "Cannot reshape tensor of 0 elements into shape "
      << "(" << c10::Join(", ", output_sizes)
      << ") because the unspecified dimension size -1 can be any value";
  CHECK_EQ(total_element_count % incomplete_element_count, 0)
      << "(" << c10::Join(", ", output_sizes) << ") vs. ("
      << c10::Join(", ", input_sizes) << ")";
  std::vector<int64_t> complete_output_sizes =
      lazy_tensors::util::ToVector<int64_t>(output_sizes);
  complete_output_sizes[*incomplete_dim] =
      total_element_count / incomplete_element_count;
  return complete_output_sizes;
}

std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions;
  for (int64_t i = 0; i < dimensions.size(); ++i) {
    int64_t dim = dimensions[i];
    if (dim != 1 || (i != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

std::vector<int64_t> BuildUnsqueezeDimensions(c10::ArrayRef<int64_t> dimensions,
                                              int64_t dim) {
  CHECK_LE(dim, dimensions.size());
  auto unsqueeze_dimensions = lazy_tensors::util::ToVector<int64_t>(dimensions);
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

size_t ComputeSplitCount(int64_t dim_size, c10::ArrayRef<int64_t> split_sizes) {
  size_t count = 0;
  for (auto size : split_sizes) {
    if (size > dim_size) {
      break;
    }
    dim_size -= size;
    ++count;
  }
  return count;
}

}  // namespace torch_lazy_tensors
