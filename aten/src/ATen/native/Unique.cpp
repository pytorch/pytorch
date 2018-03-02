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
void _unique_cpu_template(
    const Tensor& self,
    const bool return_inverse,
    Tensor* output,
    Tensor* inverse_indices) {
  set_type<scalar_t> set(
      self.data<scalar_t>(), self.data<scalar_t>() + self.numel());
  output->resize_({static_cast<long long>(set.size())});
  std::copy(set.begin(), set.end(), output->data<scalar_t>());

  if (return_inverse) {
    inverse_indices->resize_(self.sizes());
    std::unordered_map<scalar_t, int64_t> inverse_map;
    inverse_map.reserve(output->numel());
    for (int i = 0; i < output->numel(); ++i) {
      inverse_map[output->data<scalar_t>()[i]] = i;
    }
    for (int i = 0; i < self.numel(); ++i) {
      inverse_indices->data<int64_t>()[i] =
          inverse_map[self.data<scalar_t>()[i]];
    }
  }
}
} // namespace

std::tuple<Tensor, Tensor>
unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  // output will be resized in unique_template once we know how big it is.
  // inverse_indices may also be resized depending on return_inverse.
  Tensor output = self.type().tensor({0});
  Tensor inverse_indices = self.type().toScalarType(kLong).tensor({0});

  if (sorted) {
    AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      _unique_cpu_template<std::set, scalar_t>(
          self, return_inverse, &output, &inverse_indices);
    });
  } else {
    AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      _unique_cpu_template<std::unordered_set, scalar_t>(
          self, return_inverse, &output, &inverse_indices);
    });
  }

  return std::make_tuple(output, inverse_indices);
}

}  // namespace native
}  // namespace at
