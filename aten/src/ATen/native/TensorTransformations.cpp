#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include <functional>
#include <numeric>
#include <vector>


namespace at {
namespace native {

Tensor reverse_dim(const Tensor& t, int64_t dim) {
  Tensor index = at::arange(t.type().toScalarType(at::ScalarType::Long), t.size(dim) - 1, -1, -1);
  return t.index_select(dim, index);
}

Tensor flip_cpu(const Tensor& self, IntList dims) {

  int64_t total_dims = self.dim(), flip_dims_size = dims.size();

  // check if number of axis in dim is valid
  if (flip_dims_size == 0) {
    std::stringstream ss;
    ss << "expected input tensor dims not empty, "
       << "but got tensor dims size=" << flip_dims_size;
    throw std::runtime_error(ss.str());
  }

  // check duplicates in dims
  auto flip_dims_v = std::vector<int64_t>(dims);
  flip_dims_v.erase(std::unique(flip_dims_v.begin(), flip_dims_v.end()), flip_dims_v.end());
  if ((int64_t)flip_dims_v.size() < flip_dims_size) {
    std::stringstream ss;
    ss << "dims has duplicates, "
       << "original flip dims size=" << flip_dims_size << ", "
       << "but unique flip dims size= " << flip_dims_v.size();
    throw std::runtime_error(ss.str());
  }

  // check len of dims
  if (flip_dims_size > total_dims) {
    std::stringstream ss;
    ss << "expected flip dims size <= tensor total dims, "
       << "but got flip dims size=" << flip_dims_size << " and "
       << "tensor total dim=" << total_dims;
    throw std::runtime_error(ss.str());
  }

  // check if dims axis within range
  int64_t min_d = total_dims, max_d = 0;
  for (auto d : dims) {
    min_d = std::min(min_d, d);
    max_d = std::max(max_d, d);
  }

  if (min_d < 0) {
    std::stringstream ss;
    ss << "expected flip dims axis >= 0, "
       << "but got min flip dims=" << min_d;
    throw std::runtime_error(ss.str());
  }

  if (max_d >= total_dims) {
    std::stringstream ss;
    ss << "expected flip dims axis < tensor total dims, "
       << "but got max flip dims=" << max_d << " and "
       << "tensor total dim=" << total_dims;
    throw std::runtime_error(ss.str());
  }

  Tensor out_t = self.clone();
  for (auto d : dims) {
    out_t.copy_(reverse_dim(out_t, d));
  }
  return out_t;
}

}} // namespace at::native
