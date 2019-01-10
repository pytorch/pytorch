// Returns unique elements of input tensor.

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"

#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace at {
namespace native{

namespace {

template <typename scalar_t>
std::tuple<Tensor, Tensor> _unique_cpu_template(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse) {
  const Tensor& input = self.contiguous();
  const scalar_t* input_data = input.data<scalar_t>();
  std::unordered_set<scalar_t> set(input_data, input_data + input.numel());
  Tensor output = input.type().tensor({static_cast<int64_t>(set.size())});
  scalar_t* output_data = output.data<scalar_t>();

  if (sorted) {
    std::vector<scalar_t> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());
    std::copy(vec.begin(), vec.end(), output_data);
  } else {
    std::copy(set.begin(), set.end(), output_data);
  }

  Tensor inverse_indices = self.type().toScalarType(kLong).tensor({0});
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
    int64_t* inverse_indices_data = inverse_indices.data<int64_t>();
    std::unordered_map<scalar_t, int64_t> inverse_map;
    inverse_map.reserve(output.numel());
    for (int i = 0; i < output.numel(); ++i) {
      inverse_map[output_data[i]] = i;
    }
    for (int i = 0; i < input.numel(); ++i) {
      inverse_indices_data[i] = inverse_map[input_data[i]];
    }
  }
  return std::make_tuple(output, inverse_indices);
}
} // namespace

std::tuple<Tensor, Tensor>
_unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    return _unique_cpu_template<scalar_t>(self, sorted, return_inverse);
  });
}

}  // namespace native
}  // namespace at
