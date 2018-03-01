#include <unordered_map>
#include <unordered_set>

#include "ATen/ATen.h"

namespace at {
namespace native{

std::tuple<Tensor, Tensor> unique_1d(
    const Tensor& self, const bool sorted, const bool return_inverse) {
  std::unordered_set<int64_t> set(
      self.data<int64_t>(), self.data<int64_t>() + self.numel());
  Tensor output = self.type().tensor({static_cast<long long>(set.size())});
  
  if (!sorted) {
    std::copy(set.begin(), set.end(), output.data<int64_t>());
  } else {
    std::vector<int64_t> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());
    std::copy(vec.begin(), vec.end(), output.data<int64_t>());
  }

  Tensor inverse_indices;
  if (!return_inverse) {
    inverse_indices = self.type().toScalarType(kLong).tensor({0});
    
  } else {
    inverse_indices = self.type().toScalarType(kLong).tensor({self.numel()});    
    std::unordered_map<int64_t, int64_t> inverse_map;
    inverse_map.reserve(output.numel());
    for (int i = 0; i < output.numel(); ++i) {
      inverse_map[output.data<int64_t>()[i]] = i;
    }
    for (int i = 0; i < self.numel(); ++i) {
      inverse_indices.data<int64_t>()[i] = inverse_map[self.data<int64_t>()[i]];
    }
  }
  
  return std::make_tuple(output, inverse_indices);
}

}  // namespace native
}  // namespace at
