// Returns unique elements of input tensor.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>
#include <c10/util/Load.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace std {
template <>
struct hash<at::BFloat16> {
  size_t operator()(const at::BFloat16& v) const noexcept {
    return std::hash<uint16_t>()(v.x);
  }
};

template <>
struct hash<at::Half> {
  size_t operator()(const at::Half& v) const noexcept {
    return std::hash<uint16_t>()(v.x);
  }
};
} // namespace std

namespace at {
namespace native{

namespace {

// Extract the unique elements from [begin, end) into a new Tensor
template <typename scalar_t>
Tensor unique_elements(const scalar_t* begin, const scalar_t* end,
                       bool sorted, const TensorOptions &options) {
  // Create unordered set of elements
  auto set = std::unordered_set<scalar_t>(begin, end);

  // Write the output tensor
  Tensor output = at::empty({static_cast<int64_t>(set.size())}, options);
  scalar_t *output_data = output.mutable_data_ptr<scalar_t>();
  std::copy(set.begin(), set.end(), output_data);
  if (sorted) {
    std::sort(output_data, output_data + set.size());
  }
  return output;
}

// Specialization for boolean inputs, since we can't construct a set
// directly from an array of bool as it won't handle invalid byte values.
// See NOTE [Loading boolean values]
Tensor unique_elements(const bool* begin, const bool* end,
                       bool /*sorted*/, const TensorOptions &options) {
  // Instead of a set, track whether a value has been seen
  std::array<bool, 2> seen;
  seen.fill(false);

  for (; begin != end; ++begin) {
    seen[c10::load(begin)] = true;
    if (seen[false] && seen[true]) {
      break;
    }
  }

  // Write the output tensor
  int64_t num_elem = seen[false] + seen[true];
  Tensor output = at::empty({num_elem}, options);
  bool *output_data = output.mutable_data_ptr<bool>();

  if (seen[false]) {
    *output_data++ = false;
  }
  if (seen[true]) {
    *output_data++ = true;
  }
  return output;
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cpu_template(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  const Tensor& input = self.contiguous();
  const scalar_t* input_data = input.data_ptr<scalar_t>();
  int64_t numel = input.numel();
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor counts = at::empty({0}, self.options().dtype(kLong));
  Tensor output = unique_elements(input_data, input_data + numel,
                                  sorted, input.options());
  const scalar_t *output_data = output.data_ptr<scalar_t>();

  if (return_inverse || return_counts) {
    inverse_indices.resize_(input.sizes());
    int64_t* inverse_indices_data = inverse_indices.data_ptr<int64_t>();
    std::unordered_map<scalar_t, int64_t> inverse_map;
    inverse_map.reserve(output.numel());
    for (const auto i : c10::irange(output.numel())) {
      inverse_map[output_data[i]] = i;
    }
    for (const auto i : c10::irange(numel)) {
      const auto val = c10::load(&input_data[i]);
      inverse_indices_data[i] = inverse_map[val];
    }
    if (return_counts) {
      std::unordered_map<scalar_t, int64_t> counts_map;
      counts_map.reserve(output.numel());
      for (const auto i : c10::irange(output.numel())) {
        counts_map[output_data[i]] = 0;
      }
      for (const auto i : c10::irange(numel)) {
        const auto val = c10::load(&input_data[i]);
        counts_map[val] += 1;
      }
      counts.resize_(output.sizes());
      counts.fill_(0);
      int64_t *counts_data = counts.data_ptr<int64_t>();
      for (const auto i : c10::irange(output.numel())) {
        counts_data[i] = counts_map[output_data[i]];
      }
    }
  }
  return std::make_tuple(output, inverse_indices, counts);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive_cpu_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  const Tensor& input = self.contiguous();
  const scalar_t* input_data = input.data_ptr<scalar_t>();
  int64_t numel = input.numel();
  Tensor output = at::empty({numel}, input.options());
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor counts = at::empty({0}, self.options().dtype(kLong));

  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
  }

  if (numel > 0) {
    scalar_t *output_data = output.data_ptr<scalar_t>();
    int64_t *inverse_data = inverse_indices.data_ptr<int64_t>();;
    int64_t *counts_data = nullptr;
    scalar_t last_value = c10::load(input_data);
    *output_data = last_value;

    if (return_counts) {
      counts.resize_({numel});
      counts_data = counts.data_ptr<int64_t>();
    }
    scalar_t *p = output_data;
    int64_t *q = counts_data;
    int64_t last = 0;
    if (return_inverse) {
      inverse_data[0] = 0;
    }
    for (const auto i : c10::irange(1, numel)) {
      const auto value = c10::load(&input_data[i]);
      if (value != last_value) {
        *(++p) = value;
        last_value = value;
        if (return_counts) {
          *(q++) = i - last;
          last = i;
        }
      }
      if (return_inverse) {
        inverse_data[i] = p - output_data;
      }
    }
    int64_t output_size = p - output_data + 1;
    if (return_counts) {
      *q = numel - last;
      counts.resize_({output_size});
    }
    output.resize_({output_size});
  }

  return std::make_tuple(output, inverse_indices, counts);
}

template<class ForwardIt>
ForwardIt _unique_dim_cpu_impl(ForwardIt first, ForwardIt last,
  std::vector<int64_t>& indices, Tensor inverse_indices_vec, Tensor counts) {
    if (first == last) {
      return last;
    }

    TORCH_INTERNAL_ASSERT(inverse_indices_vec.is_contiguous(),
        "_unique_dim_cpu_impl only support contiguous inverse_indices_vec");
    TORCH_INTERNAL_ASSERT(counts.is_contiguous(),
        "_unique_dim_cpu_impl only support contiguous counts");

    int64_t *indices_data = indices.data();
    int64_t *inverse_data = inverse_indices_vec.data_ptr<int64_t>();
    int64_t *counts_data = counts.data_ptr<int64_t>();

    ForwardIt result = first;
    ForwardIt previous = first;
    int64_t *current_counts = counts_data;
    inverse_data[*(indices_data++)] = 0;
    for (ForwardIt current = std::next(first); current != last; current++) {
      if (!at::equal(*current, *result)) {
        *(++result) = std::move(*current);
        *(current_counts++) = std::distance(previous, current);
        previous = current;
      }
      inverse_data[*(indices_data++)] = std::distance(first, result);
    }
    *current_counts = std::distance(previous, last);
    return ++result;
  }

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> _unique_dim_cpu_template(
    const Tensor& self,
    const int64_t dim,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {

    auto sizes = self.sizes().vec();
    // check how many zero dimensions exist
    auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);

    // tensor is not well formed as it has 0 sized dimensions
    if (self.size(dim) == 0){
      TORCH_CHECK(
          num_zero_dims == 1,
          "Number of zero sized dimensions is more than one, so unique cannot be applied ")
      Tensor output = at::empty(sizes, self.options());
      Tensor inverse_indices =
          at::empty({0}, self.options().dtype(kLong));
      Tensor counts = at::empty({0}, self.options().dtype(kLong));

      return std::make_tuple(output, inverse_indices, counts);
    }

    TORCH_CHECK(num_zero_dims == 0,
    "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  // reshape tensor as [dim, -1]
  Tensor input_flat = self.moveaxis(dim, 0);
  auto orig_sizes = input_flat.sizes().vec();
  input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

  std::vector<int64_t> indices(input_flat.size(0));
  std::iota(indices.begin(), indices.end(), 0);
  int64_t numel = input_flat.size(1);
  scalar_t* input_flat_ptr = ((scalar_t*)input_flat.data_ptr());

  // sort indices using data
  if (!consecutive) {
    std::sort(indices.begin(), indices.end(),
      [&](int64_t a, int64_t b) -> bool {
        for (const auto i : c10::irange(numel)) {
          scalar_t lhs = c10::load(&input_flat_ptr[i + a * numel]);
          scalar_t rhs = c10::load(&input_flat_ptr[i + b * numel]);
          if (lhs < rhs) {
            return true;
          } else if (lhs > rhs) {
            return false;
          }
        }
        return false;
      });
  }

  Tensor input_sorted;
  if (!consecutive) {
    input_sorted = at::empty(input_flat.sizes(), input_flat.options());
    for (const auto i : c10::irange(indices.size())) {
      input_sorted[i] = input_flat[indices[i]];
    }
  } else {
    input_sorted = input_flat;
  }

  Tensor inverse_indices = at::empty(indices.size(), self.options().dtype(kLong));
  Tensor counts = at::zeros(indices.size(), self.options().dtype(kLong));
  std::vector<Tensor> input_unbind = at::unbind(input_sorted, 0);
  auto last = _unique_dim_cpu_impl(
    input_unbind.begin(), input_unbind.end(), indices, inverse_indices, counts);
  input_unbind.erase(last, input_unbind.end());
  counts = at::narrow(counts, 0, 0, input_unbind.size());

  // reshape back
  auto output = at::stack(input_unbind, 0);
  auto new_sizes = std::vector<int64_t>(std::move(orig_sizes));
  new_sizes[0] = -1;
  output = output.view(new_sizes);
  output = output.moveaxis(0, dim);

  return std::make_tuple(output, inverse_indices, counts);
}

} // namespace


std::tuple<Tensor, Tensor>
_unique_cpu(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique", [&] {
    Tensor output, inverse;
    std::tie(output, inverse, std::ignore) = unique_cpu_template<scalar_t>(self, sorted, return_inverse, false);
    return std::make_tuple(output, inverse);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_cpu(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique", [&] {
    return unique_cpu_template<scalar_t>(self, sorted, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cpu(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    // The current implementation using `dim` always sorts due to unhashable tensors
    return _unique_dim_cpu_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_cpu(const Tensor& self, const int64_t dim, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    return _unique_dim_cpu_template<scalar_t>(self, dim, true, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_cpu(const Tensor& self, const bool return_inverse, const bool return_counts, c10::optional<int64_t> dim) {
  if (!dim.has_value() || (dim.value() == 0 && self.dim() == 1)) {
    return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique", [&] {
      return unique_consecutive_cpu_template<scalar_t>(self, return_inverse, return_counts);
    });
  }
  return unique_dim_consecutive_cpu(self, dim.value(), return_inverse, return_counts);
}

}  // namespace native
}  // namespace at
