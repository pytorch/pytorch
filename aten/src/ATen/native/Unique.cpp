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
  Tensor output = at::empty({static_cast<int64_t>(set.size())}, input.type());
  scalar_t* output_data = output.data<scalar_t>();  

  if (sorted) {
    std::vector<scalar_t> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());
    std::copy(vec.begin(), vec.end(), output_data);
  } else {
    std::copy(set.begin(), set.end(), output_data);
  }

  Tensor inverse_indices = at::empty({0}, self.type().toScalarType(kLong));
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

template <typename scalar_t>
std::tuple<Tensor, Tensor> _unique_dim_cpu_template(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse) {
  // reshape tensor as [dim, -1]
  Tensor input_flat = self.transpose(dim, 0);
  std::vector<int64_t> orig_sizes(input_flat.sizes());

  input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

  // unbind to use in std::sort and sort
  std::vector<Tensor> input_unbind = at::unbind(input_flat, 0);
  std::sort(input_unbind.begin(), input_unbind.end(),
    [](const Tensor& lhs, const Tensor& rhs) -> bool {
      // comparable to lexicographical sort
      for (int i = 0; i < lhs.numel(); ++i) {
        if ((bool)at::lt(lhs[i], rhs[i]).toCByte()) {
          return true;
        }
        else if ((bool)at::gt(lhs[i], rhs[i]).toCByte()) {
          return false;
        }
      }
      return false;
    });

  auto last = std::unique(input_unbind.begin(), input_unbind.end(), [](Tensor& a, Tensor& b) {
    return at::equal(a, b);
  });
  input_unbind.erase(last, input_unbind.end());

  // reshape back
  auto output_dim = at::stack(input_unbind, 0);
  std::vector<int64_t> new_sizes(orig_sizes.begin(), orig_sizes.end());
  new_sizes[0] = -1;
  output_dim = output_dim.view(new_sizes);
  output_dim = output_dim.transpose(0, dim);

  Tensor inverse_indices_dim = at::empty({0}, self.type().toScalarType(kLong));
  int64_t size = self.size(dim);
  inverse_indices_dim.resize_(size);
  std::vector<Tensor> self_unbind = at::unbind(self, dim);
  std::vector<Tensor> output_unbind = at::unbind(output_dim, dim);
  for (int i = 0; i < self_unbind.size(); ++i) {
    for (int j = 0; j < output_unbind.size(); ++j) {
      if (at::equal(self_unbind[i], output_unbind[j])) {
        inverse_indices_dim[i] = j;
      }
    }
  }
  return std::make_tuple(output_dim, inverse_indices_dim);
}
} // namespace

std::tuple<Tensor, Tensor>
_unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    return _unique_cpu_template<scalar_t>(self, sorted, return_inverse);
  });
}

std::tuple<Tensor, Tensor>
_unique_dim_cpu(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique_dim", [&] {
    // The current implementation using `dim` always sorts due to unhashable tensors
    return _unique_dim_cpu_template<scalar_t>(self, dim, return_inverse);
  });
}

}  // namespace native
}  // namespace at
