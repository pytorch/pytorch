#include "ATen/ATen.h"

#include <ATen/Error.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

static inline void check_errors(int64_t total_dims, int64_t flip_dims_size, IntList dims) {
  // check if number of axis in dim is valid
  AT_CHECK(flip_dims_size > 0,
    "expected input tensor dims > 0, but got tensor dims size=", flip_dims_size);

  // check duplicates in dims
  auto flip_dims_v = std::vector<int64_t>(dims);
  flip_dims_v.erase(std::unique(flip_dims_v.begin(), flip_dims_v.end()), flip_dims_v.end());
  AT_CHECK((int64_t)flip_dims_v.size() == flip_dims_size,
    "dims has duplicates, original flip dims size=", flip_dims_size,
    ", but unique flip dims size=", flip_dims_v.size());

  // check len of dims
  AT_CHECK(flip_dims_size <= total_dims,
    "expected flip dims size <= tensor total dims, but got flip dims size=",
    flip_dims_size, " and tensor total dim=", total_dims);

  // check if dims axis within range
  auto min_max_d = std::minmax_element(flip_dims_v.begin(), flip_dims_v.end());

  AT_CHECK(*min_max_d.first >= 0,
    "expected flip dims axis >= 0, but got min flip dims=", *min_max_d.first);

  AT_CHECK(*min_max_d.second < total_dims,
    "expected flip dims axis < tensor total dims, but got max flip dims=",
    *min_max_d.second, " and tensor total dim=", total_dims);
}

}}  // namespace at::native
