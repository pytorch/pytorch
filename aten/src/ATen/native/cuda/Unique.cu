#include <ATen/ATen.h>
#include <ATen/native/Unique.h>
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

template <typename scalar_t, typename policy_t>
std::tuple<Tensor, Tensor> unique_consecutive_1d_cuda_(
  Tensor &self, const bool return_inverse,
  const bool return_counts, policy_t policy
) {
  return unique_consecutive_1d_(
    output, return_inverse, return_counts,
    std::bind(thrust::adjacent_difference, policy, _1, _2, _3, thrust::not_equal<scalar_t>()),
    std::bind(thrust::inclusive_scan, policy, _1, _2, _3),
    std::bind(thrust::unique, policy, _1, _2),
    std::bind(thrust::unique_by_key, policy, _1, _2, _3),
    std::bind(thrust::adjacent_difference, policy, _1, _2, _3)
  );
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive_template(
  const Tensor& self,
  const bool return_inverse,
  const bool return_counts
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  Tensor output = self.clone().reshape(-1);
  Tensor inverse_indices, counts;
  std::tie(inverse_indices, counts) = unique_consecutive_1d_cuda_(
    output, return_inverse, return_counts, policy
  );
  if (return_inverse) {
      inverse_indices.resize_(self.sizes());
  }
  return std::make_tuple(output, inverse_indices, counts);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cuda_template(
  const Tensor& self,
  const bool return_inverse,
  const bool return_counts
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  Tensor output = self.clone().reshape(-1);

  Tensor sorted_indices;
  if (!return_inverse) {
    thrust::sort(policy, output_data, output_data + num_inp);
  } else {
    sorted_indices = at::arange(0, num_inp, self.options().dtype(kLong));
    int64_t *sorted_indices_ptr = sorted_indices.data<int64_t>();
    thrust::sort_by_key(policy, output_data, output_data + num_inp, sorted_indices_ptr);
  }

  Tensor inverse_indices, counts;
  std::tie(inverse_indices, counts) = unique_consecutive_1d_cuda_(
    output, return_inverse, return_counts, policy
  );

  if (return_inverse) {
      inverse_indices = at::empty_like(inverse_indices).scatter_(0, sorted_indices, inverse_indices);
      inverse_indices.resize_(self.sizes());
  }

  return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
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

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique_dim", [&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_cuda(const Tensor& self, const int64_t dim, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique_dim", [&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, true, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_cuda(const Tensor& self, const bool return_inverse, const bool return_counts, c10::optional<int64_t> dim) {
  if (!dim.has_value()) {
    return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
      // The current CUDA implementation of unique always sort due to the
      // lack of hashtable implementation in thrust
      return unique_consecutive_template<scalar_t>(self, return_inverse, return_counts);
    });
  }
  return unique_dim_consecutive_cuda(self, dim.value(), return_inverse, return_counts);
}

}  // namespace native
}  // namespace at
