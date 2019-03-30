#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <iterator>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace at {
namespace native{

namespace {

template <
  typename scalar_t,
  typename less_t,
  typename equal_t,
  typename not_equal_t
>
std::tuple<Tensor, Tensor, int64_t> compute_unique(
  scalar_t *data,
  int64_t num_inp,
  const bool return_inverse,
  const bool return_counts,
  TensorOptions options,
  less_t less,
  equal_t equal,
  not_equal_t not_equal
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  //sort
  Tensor inverse_indices;
  if (!return_inverse) {
      inverse_indices = at::empty({0}, options);
      thrust::sort(policy, data, data + num_inp, less);
  } else {
      Tensor sorted_indices = at::arange(0, num_inp, options);
      int64_t* sorted_indices_ptr = sorted_indices.data<int64_t>();
      thrust::sort_by_key<decltype(policy), scalar_t *, int64_t *, less_t>(policy, data, data + num_inp, sorted_indices_ptr, less);
      Tensor inv_loc = at::empty({num_inp}, options);
      inverse_indices = at::empty({num_inp}, options);
      int64_t* inv_loc_ptr = inv_loc.data<int64_t>();
      int64_t* inverse_indices_ptr = inverse_indices.data<int64_t>();
      thrust::adjacent_difference(policy, data, data + num_inp, inv_loc_ptr, not_equal);
      inv_loc[0] = 0;
      thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
      thrust::scatter(policy, inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
  }

  // unique
  Tensor counts = at::empty({0}, options);
  int64_t num_out;
  if (!return_counts) {
      num_out = thrust::unique(policy, data, data + num_inp, equal) - data;
  } else {
      Tensor sorted_indices = at::arange(0, num_inp + 1, options);
      int64_t* sorted_indices_ptr = sorted_indices.data<int64_t>();
      num_out = thrust::unique_by_key(policy, data, data + num_inp, sorted_indices_ptr, equal).first - data;
      sorted_indices[num_out] = num_inp;
      counts.resize_(num_out);
      int64_t* counts_ptr = counts.data<int64_t>();
      thrust::adjacent_difference(policy, sorted_indices_ptr + 1, sorted_indices_ptr + num_out + 1, counts_ptr);
  }

  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor, Tensor, int64_t>(inverse_indices, counts, num_out);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cuda_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {

    Tensor output = self.clone().reshape(-1);
    int64_t num_inp = output.numel();
    scalar_t* output_data = output.data<scalar_t>();

    Tensor inverse_indices, counts;
    int64_t num_out;
    std::tie(inverse_indices, counts, num_out) =
    compute_unique<scalar_t, thrust::less<scalar_t>, thrust::equal_to<scalar_t>,
                   thrust::not_equal_to<scalar_t>>
      (
        output_data, num_inp, return_inverse, return_counts,
        self.options().dtype(kLong),
        thrust::less<scalar_t>(),
        thrust::equal_to<scalar_t>(),
        thrust::not_equal_to<scalar_t>()
      );
    output.resize_(num_out);

    if (return_inverse) {
        inverse_indices.resize_(self.sizes());
    }

    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
}

template <typename scalar_t>
class UniqueDimLess {
  scalar_t *data;
  int64_t n;
public:
  UniqueDimLess(scalar_t *data, int64_t n): data(data), n(n) {}
  __device__ bool operator()(int64_t a, int64_t b) {
    for (int64_t i = 0; i < n; ++i) {
      scalar_t lhs = data[i + a * n];
      scalar_t rhs = data[i + b * n];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
};

template <typename scalar_t>
class UniqueDimEqual {
  scalar_t *data;
  int64_t n;
public:
  UniqueDimEqual(scalar_t *data, int64_t n): data(data), n(n) {}
  __device__ bool operator()(int64_t a, int64_t b) {
    for (int64_t i = 0; i < n; ++i) {
      scalar_t lhs = data[i + a * n];
      scalar_t rhs = data[i + b * n];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

template <typename scalar_t>
class UniqueDimNotEqual {
  scalar_t *data;
  int64_t n;
public:
  UniqueDimNotEqual(scalar_t *data, int64_t n): data(data), n(n) {}
  __device__ int64_t operator()(int64_t a, int64_t b) {
    for (int64_t i = 0; i < n; ++i) {
      scalar_t lhs = data[i + a * n];
      scalar_t rhs = data[i + b * n];
      if (lhs != rhs) {
        return 1;
      }
    }
    return 0;
  }
};

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda_template(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts) {

    /**
     * The idea for implementing this is basically the same as unique.
     * For unique_dim, we are taking the unique with respect to a index
     * tensor, but during the processes, we override the compare and equal
     * operator by checking the data underlying it instead. After the
     * algorithm, we would use index_select to map the resulting indicies
     * to the result on the actual data.
     */

    int64_t num_inp = self.size(dim);
    Tensor input_flat = self.transpose(dim, 0).contiguous().view({num_inp, -1});
    int64_t numel = input_flat.size(1);
    scalar_t *input_flat_ptr = input_flat.data<scalar_t>();

    Tensor indices = at::arange(0, num_inp, self.options().dtype(kLong));
    int64_t *indices_data = indices.data<int64_t>();

    Tensor inverse_indices, counts;
    int64_t num_out;
    std::tie(inverse_indices, counts, num_out) =
    compute_unique<int64_t, UniqueDimLess<scalar_t>, UniqueDimEqual<scalar_t>,
                   UniqueDimNotEqual<scalar_t>>
    (
      indices_data, num_inp, return_inverse, return_counts,
      self.options().dtype(kLong),
      UniqueDimLess<scalar_t>(input_flat_ptr, numel),
      UniqueDimEqual<scalar_t>(input_flat_ptr, numel),
      UniqueDimNotEqual<scalar_t>(input_flat_ptr, numel)
    );
    indices.resize_(num_out);

    return std::tuple<Tensor, Tensor, Tensor>(self.index_select(dim, indices), inverse_indices, counts);
  }

} // namespace


std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    Tensor output, inverse;
    std::tie(output, inverse, std::ignore) = unique_cuda_template<scalar_t>(self, return_inverse, false);
    return std::make_tuple(output, inverse);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_cuda(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return unique_cuda_template<scalar_t>(self, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor>
_unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique_dim", [&] {
    Tensor output, inverse;
    std::tie(output, inverse, std::ignore) = unique_dim_cuda_template<scalar_t>(self, dim, return_inverse, false);
    return std::make_tuple(output, inverse);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique_dim2_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique_dim", [&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, return_inverse, return_counts);
  });
}

}  // namespace native
}  // namespace at
