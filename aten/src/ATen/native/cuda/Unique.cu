#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <iterator>
#include <thrust/adjacent_difference.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/zip_iterator.h>

namespace at {
namespace native{

namespace {


template <
  typename policy_t, typename scalar_t,
  typename equal_t, typename not_equal_t
>
std::tuple<Tensor, Tensor, Tensor, int64_t> compute_unique(
  const policy_t &policy,
  scalar_t *data,
  int64_t num_inp,
  const Tensor &sorted_indices,
  const bool return_inverse,
  const bool return_counts,
  const bool return_indices,
  TensorOptions options,
  equal_t equal,
  not_equal_t not_equal
) {

  // inverse indices
  if (return_inverse || return_indices) {
    TORCH_CHECK(sorted_indices.defined(),
    "return_inverse or return_indices is set to true, but sorted_indices is undefined. Send a bug report!");
  }
  Tensor inverse_indices;
  if (!return_inverse || num_inp == 0) {
    inverse_indices = at::empty({0}, options);
  } else {
    const int64_t *sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
    Tensor inv_loc = at::empty({num_inp}, options);
    inverse_indices = at::empty({num_inp}, options);
    int64_t* inv_loc_ptr = inv_loc.data_ptr<int64_t>();
    int64_t* inverse_indices_ptr = inverse_indices.data_ptr<int64_t>();
    thrust::adjacent_difference(policy, data, data + num_inp, inv_loc_ptr, not_equal);
    inv_loc[0] = 0;
    thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
    thrust::scatter(policy, inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
  }

  // unique and count
  Tensor counts = at::empty({0}, options);
  Tensor indices;
  int64_t num_out;
  if (!return_counts && !return_indices) {
    num_out = thrust::unique(policy, data, data + num_inp, equal) - data;
  } else {
    if (return_counts) {
      Tensor range = at::arange(0, num_inp + 1, options);
      int64_t *range_ptr = range.data_ptr<int64_t>();
      if (!return_indices) {
        num_out = thrust::unique_by_key(policy, data, data + num_inp, range_ptr, equal).first - data;
      } else { //both r_c and r_i requested
        indices = at::empty({num_inp}, options).copy_(sorted_indices); // cannot operate directly on sorted indices, because
        //for unique_dim they are also used in unique key
        int64_t *indices_ptr = indices.data_ptr<int64_t>();
        num_out = thrust::unique_by_key(policy, data, data + num_inp,
        thrust::make_zip_iterator(thrust::make_tuple(range_ptr, indices_ptr)), equal).first - data;
        indices.resize_(num_out);
      }
      range[num_out] = num_inp;
      counts.resize_(num_out);
      int64_t* counts_ptr = counts.data_ptr<int64_t>();
      thrust::adjacent_difference(policy, range_ptr + 1, range_ptr + num_out + 1, counts_ptr);
    } else if (return_indices) { //r_c=false, r_i = true
      //we don't know exact number of future unique items, so unfortunately will have to allocate full size indices
      indices = at::empty({num_inp}, options).copy_(sorted_indices); // cannot operate directly on sorted indices, because
      //for unique_dim they are also used in unique key
      int64_t *indices_ptr = indices.data_ptr<int64_t>();
      num_out = thrust::unique_by_key(policy, data, data + num_inp, indices_ptr, equal).first - data;
      indices.resize_(num_out);
    }



  }


  AT_CUDA_CHECK(cudaGetLastError());
  return std::tuple<Tensor, Tensor, Tensor, int64_t>(inverse_indices, counts, indices, num_out);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> unique_cuda_template(
  const Tensor& self,
  const bool consecutive,
  const bool return_inverse,
  const bool return_counts,
  const bool return_indices
) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto options = self.options().dtype(kLong);
  Tensor output = self.clone(at::MemoryFormat::Contiguous).reshape(-1);
  int64_t num_inp = output.numel();
  scalar_t* output_data = output.data_ptr<scalar_t>();

  Tensor sorted_indices;
  if (!return_inverse && !return_indices) {
    if (!consecutive) {
      thrust::sort(policy, output_data, output_data + num_inp);
    }
  } else {
    sorted_indices = at::arange(0, num_inp, options);
    if (!consecutive) {
      int64_t *sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
      thrust::sort_by_key(policy, output_data, output_data + num_inp, sorted_indices_ptr);
    }
  }

  Tensor inverse_indices, counts, indices;
  int64_t num_out;
  std::tie(inverse_indices, counts, indices, num_out) = compute_unique(
    policy, output_data, num_inp, sorted_indices,
    return_inverse, return_counts, return_indices, options,
    thrust::equal_to<scalar_t>(),
    thrust::not_equal_to<scalar_t>()
  );
  output.resize_(num_out);

  if (return_inverse) {
      inverse_indices.resize_(self.sizes());
  }

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, inverse_indices, counts, indices);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> unique_dim_cuda_template(
  const Tensor& self,
  const int64_t dim,
  const bool consecutive,
  const bool return_inverse,
  const bool return_counts,
  const bool return_indices
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

  auto sizes = self.sizes().vec();
  // check how many zero dimensions exist
  auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);

  // tensor is not well formed as it has 0 sized dimensions
  if (self.size(dim) == 0){
    TORCH_CHECK(
        num_zero_dims == 1,
        "Number of zero sized dimensions is more than one, so unique cannot be applied ")
    Tensor output = at::empty({0}, self.options());
    Tensor inverse_indices =
        at::empty({0}, self.options().dtype(kLong));
    Tensor counts = at::empty({0}, self.options().dtype(kLong));
    Tensor indices = at::empty({0}, self.options().dtype(kLong));
    return std::make_tuple(output, inverse_indices, counts, indices);
  }

  TORCH_CHECK(num_zero_dims == 0,
    "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  int64_t num_inp = self.size(dim);
  auto options = self.options().dtype(kLong);
  Tensor input_flat = self.transpose(dim, 0).contiguous().view({num_inp, -1});
  int64_t n = input_flat.size(1);
  scalar_t *input_flat_ptr = input_flat.data_ptr<scalar_t>();

  Tensor indices = at::arange(0, num_inp, options);
  int64_t *indices_data = indices.data_ptr<int64_t>();
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

  Tensor inverse_indices, counts, unique_indices;
  int64_t num_out;
  std::tie(inverse_indices, counts, unique_indices, num_out) = compute_unique(
    policy, indices_data, num_inp, indices,
    return_inverse, return_counts, return_indices, options,
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

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(self.index_select(dim, indices), inverse_indices, counts, unique_indices);
}

} // namespace


std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    Tensor output, inverse;
    std::tie(output, inverse, std::ignore, std::ignore) = unique_cuda_template<scalar_t>(self, false, return_inverse, false, false);
    return std::make_tuple(output, inverse);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_cuda(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    Tensor output, inverse, counts;
    std::tie(output, inverse, counts, std::ignore) = unique_cuda_template<scalar_t>(self, false, return_inverse, return_counts, false);
    return std::make_tuple(output, inverse, counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    Tensor output, inverse, counts;
    std::tie(output, inverse, counts, std::ignore) = unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts, false);
    return std::make_tuple(output, inverse, counts);
    //return unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_cuda(const Tensor& self, const int64_t dim, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    Tensor output, inverse, counts;
    std::tie(output, inverse, counts, std::ignore) = unique_dim_cuda_template<scalar_t>(self, dim, true, return_inverse, return_counts, false);
    return std::make_tuple(output, inverse, counts);
    //return unique_dim_cuda_template<scalar_t>(self, dim, true, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_cuda(const Tensor& self, const bool return_inverse, const bool return_counts, c10::optional<int64_t> dim) {
  if (!dim.has_value()) {
    return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique", [&] {
      // The current CUDA implementation of unique always sort due to the
      // lack of hashtable implementation in thrust
      Tensor output, inverse, counts;
      std::tie(output, inverse, counts, std::ignore) = unique_cuda_template<scalar_t>(self, true, return_inverse, return_counts, false);
      return std::make_tuple(output, inverse, counts);
      //return unique_cuda_template<scalar_t>(self, true, return_inverse, return_counts);
    });
  }
  return unique_dim_consecutive_cuda(self, dim.value(), return_inverse, return_counts);
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
unique_good_dim_cuda(const Tensor & self, int64_t dim, bool return_inverse, bool return_indices, bool return_counts){
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    Tensor output, inverse, counts, indices;
    std::tie(output, inverse, counts, indices) = unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts, return_indices);
    return std::make_tuple(output, inverse, indices, counts);
  });
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
unique_good_cuda(const Tensor & self, bool return_inverse, bool return_indices, bool return_counts){
  return AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
    Tensor output, inverse, counts, indices;
    std::tie(output, inverse, counts, indices) = unique_cuda_template<scalar_t>(self, false, return_inverse, return_counts, return_indices);
    return std::make_tuple(output, inverse, indices, counts);
  });
}

}  // namespace native
}  // namespace at
