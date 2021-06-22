#include "lazy_tensor_core/csrc/data_ops.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {

std::vector<lazy_tensors::int64> GetCompleteShape(
    lazy_tensors::Span<const lazy_tensors::int64> output_sizes,
    lazy_tensors::Span<const lazy_tensors::int64> input_sizes) {
  c10::optional<size_t> incomplete_dim;
  lazy_tensors::int64 incomplete_element_count = 1;
  for (size_t dim = 0; dim < output_sizes.size(); ++dim) {
    lazy_tensors::int64 dim_size = output_sizes[dim];
    if (dim_size < 0) {
      LTC_CHECK(!incomplete_dim)
          << "More than one incomplete dimension found: " << *incomplete_dim
          << " and " << dim;
      incomplete_dim = dim;
    } else {
      incomplete_element_count *= dim_size;
    }
  }
  lazy_tensors::int64 total_element_count =
      lazy_tensors::util::Multiply<lazy_tensors::int64>(input_sizes);
  if (!incomplete_dim) {
    LTC_CHECK_EQ(
        total_element_count,
        lazy_tensors::util::Multiply<lazy_tensors::int64>(output_sizes))
        << "(" << lazy_tensors::StrJoin(output_sizes, ", ") << ") vs. ("
        << lazy_tensors::StrJoin(input_sizes, ", ") << ")";
    return lazy_tensors::util::ToVector<lazy_tensors::int64>(output_sizes);
  }
  LTC_CHECK_GT(incomplete_element_count, 0)
      << "Cannot reshape tensor of 0 elements into shape "
      << "(" << lazy_tensors::StrJoin(output_sizes, ", ")
      << ") because the unspecified dimension size -1 can be any value";
  LTC_CHECK_EQ(total_element_count % incomplete_element_count, 0)
      << "(" << lazy_tensors::StrJoin(output_sizes, ", ") << ") vs. ("
      << lazy_tensors::StrJoin(input_sizes, ", ") << ")";
  std::vector<lazy_tensors::int64> complete_output_sizes =
      lazy_tensors::util::ToVector<lazy_tensors::int64>(output_sizes);
  complete_output_sizes[*incomplete_dim] =
      total_element_count / incomplete_element_count;
  return complete_output_sizes;
}

std::vector<lazy_tensors::int64> BuildSqueezedDimensions(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::int64 squeeze_dim) {
  std::vector<lazy_tensors::int64> output_dimensions;
  for (lazy_tensors::int64 i = 0; i < dimensions.size(); ++i) {
    lazy_tensors::int64 dim = dimensions[i];
    if (dim != 1 || (i != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

std::vector<lazy_tensors::int64> BuildUnsqueezeDimensions(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::int64 dim) {
  LTC_CHECK_LE(dim, dimensions.size());
  auto unsqueeze_dimensions =
      lazy_tensors::util::ToVector<lazy_tensors::int64>(dimensions);
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

size_t ComputeSplitCount(
    lazy_tensors::int64 dim_size,
    lazy_tensors::Span<const lazy_tensors::int64> split_sizes) {
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
