// Returns unique elements of input tensor.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
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

namespace at {
namespace native{

namespace {

// check whether the element on index i is `unique`,
// in the sorted sequence, the 1st element is always true.
//
// NaN is propagated to the rear in a sorted sequence,
// consider a sorted sequence of
//   {1.0, 1.0, 2.0, 2.0, NaN, NaN, NaN}
//
// a. `equal_nan` == true will give:
//   {T,   F,   T,   F,   T,   F,   F  }
//
// b. `equal_nan` == false will give:
//   {T,   F,   T,   F,   T,   T,   T  }
//
template <typename scalar_t, bool equal_nan>
struct IsUnique {};

template <typename scalar_t>
struct IsUnique<scalar_t, false> {
  inline bool operator() (scalar_t* data_ptr, int64_t i) {
    if (i == 0) { return true; }
    return c10::load(&data_ptr[i]) != c10::load(&data_ptr[i - 1]);
  }
};

template <typename scalar_t>
struct IsUnique<scalar_t, true> {
  inline bool operator() (scalar_t* data_ptr, int64_t i) {
    if (i == 0) { return true; }
    return (c10::load(&data_ptr[i]) != c10::load(&data_ptr[i - 1]))
        && !(_isnan(data_ptr[i]) && _isnan(data_ptr[i - 1]));
  }
};

// NB: Unique implementation using sort
//
// The whole algo is taken from NumPy at numpy/lib/arraysetops.py
// which firstly do sort on the input sequence and then convert
// it to consecutive unique.
//
// Also improvement has been made upon the NumPy version: parallel
// `inverse_indices` and `counts` computation in a fused loop,
// which made this part almost a free launch.
//
// This kernel also implements a `equal_nan` flag which has same
// function as NumPy's unique. Currently this is always disabled.
//
// TODO: add `bool` specialization, use similar approach as UniqueCub
//
template <typename scalar_t, typename CompareOp>
std::tuple<Tensor, Tensor, Tensor> unique_cpu_sorted_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    CompareOp is_unique) {
  const Tensor& input = self.contiguous();

  int64_t numel = input.numel();
  Tensor output = at::empty({0}, self.options());
  Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor counts = at::empty({0}, self.options().dtype(kLong));

  if (numel == 0) {
    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

  // index of first unique in each consecutive section
  // this is used to compute counts for parallelization purpose
  Tensor unique_index = at::empty({0}, self.options().dtype(kLong));

  // original behavior with unique on scalar tensor
  // is to return a output size of ([1]), `flatten` here will do the job
  auto input_flattened = input.flatten();

  Tensor input_sorted, indices;
  std::tie(input_sorted, indices) = input_flattened.sort();

  scalar_t* input_sorted_data = input_sorted.data_ptr<scalar_t>();
  int64_t* indices_data = indices.data_ptr<int64_t>();

  int num_threads = at::get_num_threads();
  std::vector<int64_t> unique_count_thread(num_threads, 0);
  std::vector<int64_t> offset_thread(num_threads, 0);

  const int64_t grain_size = at::internal::GRAIN_SIZE;

  // calculate unique count from each thread
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for (const auto i : c10::irange(begin, end)) {
      if (is_unique(input_sorted_data, i)) {
        unique_count_thread[tid]++;
      }
    }
  });

  // calculate thread offset in output and
  // `unique_count` records total count of uniques at last
  int64_t unique_count = 0;
  for (const auto t : c10::irange(num_threads)) {
    offset_thread[t] = unique_count;
    unique_count += unique_count_thread[t];
  }

  output.resize_({unique_count});
  scalar_t* output_data = output.data_ptr<scalar_t>();

  int64_t* inverse_indices_data = nullptr;
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
    inverse_indices_data = inverse_indices.data_ptr<int64_t>();
  }

  int64_t* counts_data = nullptr;
  int64_t* unique_index_data = nullptr;
  if (return_counts) {
    counts.resize_({unique_count});
    counts_data = counts.data_ptr<int64_t>();

    unique_index.resize_({unique_count + 1});
    unique_index_data = unique_index.data_ptr<int64_t>();
    unique_index_data[unique_count] = numel;
  }

  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    int64_t offset = offset_thread[tid];

    for (const auto i : c10::irange(begin, end)) {
      if (is_unique(input_sorted_data, i)) {
        output_data[offset] = c10::load(&input_sorted_data[i]);
        if (return_counts) {
          unique_index_data[offset] = i;
        }
        offset++;
      }

      if (return_inverse) {
        int64_t inverse_index = offset - 1;
        int64_t perm = indices_data[i];
        inverse_indices_data[perm] = inverse_index;
      }
    }
  });

  if (return_counts) {
    // do diff to get count
    at::parallel_for(0, unique_count, grain_size, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        counts_data[i] = unique_index_data[i + 1] - unique_index_data[i];
      }
    });
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
    // The current CPU implementation of unique always sort due to
    // this is faster than hash table
    std::tie(output, inverse, std::ignore) = unique_cpu_sorted_template<scalar_t>(
        self, return_inverse, /* return_counts */false, IsUnique<scalar_t, /* equal_nan */false>());
    return std::make_tuple(output, inverse);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_cpu(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kBool, kHalf, self.scalar_type(), "unique", [&] {
    // The current CPU implementation of unique always sort due to
    // this is faster than hash table
    return unique_cpu_sorted_template<scalar_t>(
        self, return_inverse, return_counts, IsUnique<scalar_t, /* equal_nan */ false>());
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
