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

  int64_t shape_len = self.dim(), dims_len = dims.size();

  // check if number of axis in dim is valid
  if (dims_len == 0) {
    std::stringstream ss;
    ss << "expected dims not empty, "
       << "but got dims size=" << dims_len;
    throw std::runtime_error(ss.str());
  }

  // check duplicates in dims
  auto dims_v = std::vector<int64_t>(dims);
  dims_v.erase(std::unique(dims_v.begin(), dims_v.end()), dims_v.end());
  if (dims_v.size() < dims_len) {
    std::stringstream ss;
    ss << "dims has duplicates, "
       << "input dims size=" << dims_len << ", "
       << "but unique dims size= " << dims_v.size();
    throw std::runtime_error(ss.str());
  }

  if (dims_len > shape_len) {
    std::stringstream ss;
    ss << "expected dims to have size <= total tensor dims, "
       << "but got dims size=" << dims_len << " and "
       << "tensor dim=" << shape_len;
    throw std::runtime_error(ss.str());
  }

  // check if dims axis within range
  int64_t min_d = shape_len;
  int64_t max_d = 0;
  for (auto d : dims) {
    min_d = std::min(min_d, d);
    max_d = std::max(max_d, d);
  }

  if (min_d < 0) {
    std::stringstream ss;
    ss << "expected dims axis >= 0, "
       << "but got min dims=" << min_d;
    throw std::runtime_error(ss.str());
  }

  if (max_d >= shape_len) {
    std::stringstream ss;
    ss << "expected dims axis < total tensor dims, "
       << "but got max dims=" << max_d << " and "
       << "tensor dim=" << shape_len;
    throw std::runtime_error(ss.str());
  }

  Tensor res = self.clone();
  for (auto d : dims) {
    res.copy_(reverse_dim(res, d));
  }
  return res;
}

}} // namespace at::native
