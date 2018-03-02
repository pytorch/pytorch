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

template <template <class...> class set_type, typename scalar_t>
std::tuple<Tensor, Tensor> _unique_cpu_template(
    const Tensor& self,
    const bool return_inverse) {
  const Tensor& input = self.contiguous();
  set_type<scalar_t> set(
      input.data<scalar_t>(), input.data<scalar_t>() + input.numel());
  Tensor output = input.type().tensor({static_cast<long long>(set.size())});
  std::copy(set.begin(), set.end(), output.data<scalar_t>());

  Tensor inverse_indices = self.type().toScalarType(kLong).tensor({0});
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
    std::unordered_map<scalar_t, int64_t> inverse_map;
    inverse_map.reserve(output.numel());
    for (int i = 0; i < output.numel(); ++i) {
      inverse_map[output.data<scalar_t>()[i]] = i;
    }
    for (int i = 0; i < input.numel(); ++i) {
      inverse_indices.data<int64_t>()[i] =
          inverse_map[input.data<scalar_t>()[i]];
    }
  }
  return std::make_tuple(output, inverse_indices);
}
} // namespace

std::tuple<Tensor, Tensor>
unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  if (sorted) {
    return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      return _unique_cpu_template<std::set, scalar_t>(self, return_inverse);
    });
  } else {
    return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      return _unique_cpu_template<std::unordered_set, scalar_t>(
          self, return_inverse);
    });
  }
}

}  // namespace native
}  // namespace at
