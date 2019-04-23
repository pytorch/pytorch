#pragma once

#include <ATen/ATen.h>


namespace at {
namespace native{
namespace {

template <
  typename scalar_t, typename adjacent_not_equal_t,
  typename inclusive_scan_t, typename unique_t,
  typename unique_by_key_t, typename adjacent_difference_t
>
std::tuple<Tensor, Tensor> unique_consecutive_1d_(
  Tensor &self,
  const bool return_inverse,
  const bool return_counts,
  adjacent_not_equal_t adjacent_not_equal,
  inclusive_scan_t inclusive_scan,
  unique_t unique,
  unique_by_key_t unique_by_key,
  adjacent_difference_t adjacent_difference
) {
  AT_CHECK(self.dims() == 1,
    "unique_consecutive_1d_impl_ can only be used for 1D tensor");
  scalar_t *data = self.data<scalar_t>();
  int64_t num_inp = self.size(0);
  TensorOptions options = self.options().dtype(kLong);

  // inverse indices
  Tensor inverse_indices;
  if (return_inverse) {
    Tensor inv_loc = at::empty({num_inp}, options);
    inverse_indices = at::empty({num_inp}, options);
    int64_t* inv_loc_ptr = inv_loc.data<int64_t>();
    int64_t* inverse_indices_ptr = inverse_indices.data<int64_t>();
    adjacent_not_equal(data, data + num_inp, inv_loc_ptr);
    inv_loc[0] = 0;
    inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
  }

  // unique and count
  Tensor counts;
  int64_t num_out;
  if (!return_counts) {
    num_out = unique(policy, data, data + num_inp) - data;
  } else {
    counts = at::empty({0}, options);
    Tensor range = at::arange(0, num_inp + 1, options);
    int64_t *range_ptr = range.data<int64_t>();
    num_out = unique_by_key(data, data + num_inp, range_ptr).first - data;
    range[num_out] = num_inp;
    counts.resize_(num_out);
    int64_t* counts_ptr = counts.data<int64_t>();
    adjacent_difference(policy, range_ptr + 1, range_ptr + num_out + 1, counts_ptr);
  }

  self.resize_(num_out);
  return std::make_tuple(inverse_indices, counts);
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
    * algorithm, we would use index_select to map the resulting indicies
    * to the result on the actual data.
    */

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  int64_t num_inp = self.size(dim);
  auto options = self.options().dtype(kLong);
  Tensor input_flat = self.transpose(dim, 0).contiguous().view({num_inp, -1});
  int64_t n = input_flat.size(1);
  scalar_t *input_flat_ptr = input_flat.data<scalar_t>();

  Tensor indices = at::arange(0, num_inp, options);
  int64_t *indices_data = indices.data<int64_t>();
  if (!consecutive) {
    thrust::sort(policy, indices_data, indices_data + num_inp,
      [=] __device__ (int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < n; ++i) {
          scalar_t lhs = input_flat_ptr[i + a * n];
          scalar_t rhs = input_flat_ptr[i + b * n];
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

  Tensor inverse_indices, counts;
  int64_t num_out;
  std::tie(inverse_indices, counts, num_out) = compute_unique(
    policy, indices_data, num_inp, indices,
    return_inverse, return_counts, options,
    [=] __device__ (int64_t a, int64_t b) -> bool {
      for (int64_t i = 0; i < n; ++i) {
        scalar_t lhs = input_flat_ptr[i + a * n];
        scalar_t rhs = input_flat_ptr[i + b * n];
        if (lhs != rhs) {
          return false;
        }
      }
      return true;
    },
    [=] __device__ (int64_t a, int64_t b) -> int64_t {
      for (int64_t i = 0; i < n; ++i) {
        scalar_t lhs = input_flat_ptr[i + a * n];
        scalar_t rhs = input_flat_ptr[i + b * n];
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

}  // namespace
}  // namespace native
}  // namespace at