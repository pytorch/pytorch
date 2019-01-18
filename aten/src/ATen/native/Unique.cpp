// Returns unique elements of input tensor.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

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
  Tensor output = at::empty({static_cast<int64_t>(set.size())}, input.options());
  scalar_t* output_data = output.data<scalar_t>();

  if (sorted) {
    std::vector<scalar_t> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());
    std::copy(vec.begin(), vec.end(), output_data);
  } else {
    std::copy(set.begin(), set.end(), output_data);
  }

  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
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

template<class ForwardIt>
ForwardIt _unique_dim_cpu_impl(ForwardIt first, ForwardIt last,
  std::vector<int64_t>& indices, Tensor inverse_indices_vec) {
    if (first == last) {
      return last;
    }
    // save to calculate distance to iterators
    ForwardIt begin = first;

    // set first inverse index
    inverse_indices_vec[indices[0]] = 0;

    ForwardIt result = first;
    while (++first != last) {
      if (!at::equal(*result, *first) && ++result != first) {
          *result = std::move(*first);
      }
      int64_t idx_result = std::distance(begin, result);
      int64_t idx_first = std::distance(begin, first);
      inverse_indices_vec[indices[idx_first]] = idx_result;
    }

    return ++result;
  }

template <typename scalar_t>
std::tuple<Tensor, Tensor> _unique_dim_cpu_template(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse) {
  // reshape tensor as [dim, -1]
  Tensor input_flat = self.transpose(dim, 0);
  auto orig_sizes = input_flat.sizes().vec();
  input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

  std::vector<int64_t> indices(input_flat.size(0));
  std::iota(indices.begin(), indices.end(), 0);
  int64_t numel = input_flat.size(1);
  scalar_t* input_flat_ptr = ((scalar_t*)input_flat.data_ptr());

  // sort indices using data
  std::sort(indices.begin(), indices.end(),
    [&](int64_t a, int64_t b) -> bool {
      for (int64_t i = 0; i < numel; ++i) {
        scalar_t lhs = input_flat_ptr[i + a * numel];
        scalar_t rhs = input_flat_ptr[i + b * numel];
        if (lhs < rhs) {
          return true;
        } else if (lhs > rhs) {
          return false;
        }
      }
      return false;
    });

  Tensor input_sorted = at::empty(input_flat.sizes(), input_flat.options());
  for (int i = 0; i < indices.size(); ++i) {
    input_sorted[i] = input_flat[indices[i]];
  }

  Tensor inverse_indices = at::empty(indices.size(), self.options().dtype(kLong));
  std::vector<Tensor> input_unbind = at::unbind(input_sorted, 0);
  auto last = _unique_dim_cpu_impl(
    input_unbind.begin(), input_unbind.end(), indices, inverse_indices);
  input_unbind.erase(last, input_unbind.end());

  // reshape back
  auto output = at::stack(input_unbind, 0);
  auto new_sizes = std::vector<int64_t>(orig_sizes);
  new_sizes[0] = -1;
  output = output.view(new_sizes);
  output = output.transpose(0, dim);

  return std::make_tuple(output, inverse_indices);
}
} // namespace

std::tuple<Tensor, Tensor>
_unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse, optional<int64_t> dim) {
  if (dim) {
    return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      // The current implementation using `dim` always sorts due to unhashable tensors
      return _unique_dim_cpu_template<scalar_t>(self, dim.value(), return_inverse);
    });
  }
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    return _unique_cpu_template<scalar_t>(self, sorted, return_inverse);
  });
}

std::tuple<Tensor, Tensor> unique_dim(const Tensor& self, int64_t dim, const bool sorted, const bool return_inverse) {
  return at::unique(self, sorted, return_inverse, dim);
}

std::tuple<Tensor, Tensor> _unique(const Tensor& self, const bool sorted, const bool return_inverse) {
  return at::unique(self, sorted, return_inverse);
}

std::tuple<Tensor, Tensor> _unique_dim(const Tensor& self, int64_t dim, const bool sorted, const bool return_inverse) {
  return at::unique(self, sorted, return_inverse, dim);
}

}  // namespace native
}  // namespace at
