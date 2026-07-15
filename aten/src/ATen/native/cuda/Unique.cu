#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>

#include <c10/util/Load.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#endif

#include <tuple>
#include <iterator>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <ATen/native/cuda/UniqueCub.cuh>

namespace at::native {

namespace {

template <
  typename policy_t, typename scalar_t,
  typename equal_t, typename not_equal_t
>
std::tuple<Tensor, Tensor, int64_t> compute_unique(
  const policy_t &policy,
  scalar_t *data,
  int64_t num_inp,
  const Tensor &sorted_indices,
  const bool return_inverse,
  const bool return_counts,
  TensorOptions options,
  equal_t equal,
  not_equal_t not_equal
) {
  // inverse indices
  Tensor inverse_indices;
  if (!return_inverse || num_inp == 0) {
    inverse_indices = at::empty({0}, options);
  } else {
    TORCH_CHECK(sorted_indices.defined(),
      "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
    const int64_t *sorted_indices_ptr = sorted_indices.const_data_ptr<int64_t>();
    Tensor inv_loc = at::empty({num_inp}, options);
    inverse_indices = at::empty({num_inp}, options);
    int64_t* inv_loc_ptr = inv_loc.mutable_data_ptr<int64_t>();
    int64_t* inverse_indices_ptr = inverse_indices.mutable_data_ptr<int64_t>();
    thrust::adjacent_difference(policy, data, data + num_inp, inv_loc_ptr, not_equal);
    inv_loc[0] = 0;
    thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
    thrust::scatter(policy, inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
  }

  // unique and count
  Tensor counts = at::empty({0}, options);
  int64_t num_out;
  if (!return_counts) {
    num_out = thrust::unique(policy, data, data + num_inp, equal) - data;
  } else {
    Tensor range = at::arange(0, num_inp + 1, options);
    int64_t *range_ptr = range.mutable_data_ptr<int64_t>();
    num_out = thrust::unique_by_key(policy, data, data + num_inp, range_ptr, equal).first - data;
    range[num_out] = num_inp;
    counts.resize_(num_out);
    int64_t* counts_ptr = counts.mutable_data_ptr<int64_t>();
    thrust::adjacent_difference(policy, range_ptr + 1, range_ptr + num_out + 1, counts_ptr);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::tuple<Tensor, Tensor, int64_t>(inverse_indices, counts, num_out);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda_template(
  const Tensor& self,
  const int64_t dim,
  const bool consecutive,
  const bool return_inverse,
  const bool return_counts
) {

  /**
    * The idea for implementing this is basically the same as unique.
    * For unique_dim, we are taking the unique with respect to a index
    * tensor, but during the processes, we override the compare and equal
    * operator by checking the data underlying it instead. After the
    * algorithm, we would use index_select to map the resulting indices
    * to the result on the actual data.
    */

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::cuda::ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

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

  int64_t num_inp = self.size(dim);
  auto options = self.options().dtype(kLong);
  Tensor input_flat = self.moveaxis(dim, 0).contiguous().view({num_inp, -1});
  int64_t n = input_flat.size(1);
  const scalar_t *input_flat_ptr = input_flat.const_data_ptr<scalar_t>();

  Tensor indices = at::arange(0, num_inp, options);
  int64_t *indices_data = indices.mutable_data_ptr<int64_t>();
  if (!consecutive) {
    thrust::sort(policy, indices_data, indices_data + num_inp,
      [=] __device__ (int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < n; ++i) {
          scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
          scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
          if (lhs < rhs) {
            return true;
          } else if (lhs > rhs) {
            return false;
          }
        }
        return false;
      }
    );
  }

  auto [inverse_indices, counts, num_out] = compute_unique(
    policy, indices_data, num_inp, indices,
    return_inverse, return_counts, options,
    [=] __device__ (int64_t a, int64_t b) -> bool {
      for (int64_t i = 0; i < n; ++i) {
        scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
        scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
        if (lhs != rhs) {
          return false;
        }
      }
      return true;
    },
    [=] __device__ (int64_t a, int64_t b) -> int64_t {
      for (int64_t i = 0; i < n; ++i) {
        scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
        scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
        if (lhs != rhs) {
          return 1;
        }
      }
      return 0;
    }
  );
  indices.resize_(num_out);

  return std::tuple<Tensor, Tensor, Tensor>(self.index_select(dim, indices), inverse_indices, counts);
}

} // namespace


std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    auto [output, inverse, _] = internal::unique_cuda_template<scalar_t>(self, false, return_inverse, false);
    return std::make_tuple(output, inverse);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_cuda(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return internal::unique_cuda_template<scalar_t>(self, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_cuda(const Tensor& self, const int64_t dim, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, true, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_cuda(const Tensor& self, const bool return_inverse, const bool return_counts, std::optional<int64_t> dim) {
  if (!dim.has_value()) {
    return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
      // The current CUDA implementation of unique always sort due to the
      // lack of hashtable implementation in thrust
      return internal::unique_cuda_template<scalar_t>(self, true, return_inverse, return_counts);
    }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
  return unique_dim_consecutive_cuda(self, dim.value(), return_inverse, return_counts);
}

}  // namespace at::native
