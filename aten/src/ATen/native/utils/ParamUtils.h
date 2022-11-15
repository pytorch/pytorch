#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace at {
namespace native {

template <typename T>
inline std::vector<T> expand_param_if_needed(
    ArrayRef<T> list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<T>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

} // namespace native
} // namespace at
