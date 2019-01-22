#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace at {
namespace native{

namespace {

template <typename scalar_t>
  std::tuple<Tensor, Tensor> _unique_cuda_template(
    const Tensor& self,
    const bool return_inverse) {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    const Tensor& input = self.contiguous();
    int64_t num_inp = input.numel();
    const scalar_t* input_data = input.data<scalar_t>();

    //sort & unique
    Tensor output = input.clone();
    output = output.view(-1);
    scalar_t* output_data = output.data<scalar_t>();
    Tensor inverse_indices;
    if (!return_inverse) {
        inverse_indices = at::empty({0},  self.type().toScalarType(kLong));
        thrust::sort(policy, output_data, output_data + num_inp);
    } else {
        Tensor sorted_indices = at::arange(0, num_inp, self.type().toScalarType(kLong));
        int64_t* sorted_indices_ptr = sorted_indices.data<int64_t>();
        thrust::sort_by_key(policy, output_data, output_data + num_inp, sorted_indices_ptr);
        Tensor inv_loc = at::empty({num_inp}, self.type().toScalarType(kLong));
        inverse_indices = at::empty({num_inp}, self.type().toScalarType(kLong));
        int64_t* inv_loc_ptr = inv_loc.data<int64_t>();
        int64_t* inverse_indices_ptr = inverse_indices.data<int64_t>();
        thrust::adjacent_difference(policy, output_data, output_data + num_inp, inv_loc_ptr, [=] __device__ (scalar_t a, scalar_t b) -> int64_t { if (a != b) {return 1;} else { return 0; }});
        inv_loc[0] = 0;
        thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
        thrust::scatter(policy,inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
        inverse_indices.resize_(input.sizes());
    }
    int64_t num_out = thrust::unique(policy, output_data, output_data + num_inp) - output_data;
    output.resize_(num_out);

    THCudaCheck(cudaGetLastError());
    return std::tuple<Tensor, Tensor>(output, inverse_indices);
  }

template <typename scalar_t>
  std::tuple<Tensor, Tensor> _unique_dim_cuda_template(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse) {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    Tensor input_flat = self.transpose(dim, 0);
    auto orig_sizes = input_flat.sizes().vec();
    input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

    scalar_t* input_flat_ptr = input_flat.data<scalar_t>();

    Tensor indices = at::arange(0, input_flat.size(0), self.type().toScalarType(kLong));
    int64_t* indices_ptr = indices.data<int64_t>();
    int64_t numel = input_flat.size(1);

    // sort indices using data
    thrust::sort(policy, indices_ptr, indices_ptr + indices.numel(),
      [=] __device__ (int64_t a, int64_t b) -> bool {
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

    Tensor input_sorted = input_flat.index_select(0, indices);

    // get unique tensors
    scalar_t* input_sorted_ptr = input_sorted.data<scalar_t>();
    Tensor input_sorted_indices = at::arange(0, input_sorted.size(0), self.type().toScalarType(kLong));
    int64_t* input_sorted_indices_ptr = input_sorted_indices.data<int64_t>();
    auto last = thrust::unique(policy, input_sorted_indices_ptr, input_sorted_indices_ptr + input_sorted_indices.numel(),
      [=] __device__ (int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < numel; ++i) {
          scalar_t lhs = input_sorted_ptr[i + a * numel];
          scalar_t rhs = input_sorted_ptr[i + b * numel];
          if (lhs != rhs) {
            return false;
          }
        }
        return true;
      });
    input_sorted_indices.resize_(last - input_sorted_indices_ptr);
    Tensor output = input_sorted.index_select(0, input_sorted_indices);

    // reshape back
    auto new_sizes = std::vector<int64_t>(orig_sizes);
    new_sizes[0] = -1;
    output = output.view(new_sizes);
    output = output.transpose(0, dim);

    // calculate inverse indices
    Tensor inverse_indices = at::empty({0}, self.type().toScalarType(kLong));
    if (return_inverse) {
      int64_t size = self.size(dim);
      inverse_indices.resize_(size);
      Tensor mask = at::empty(input_sorted.size(0), self.type().toScalarType(kLong));
      mask[0] = 1;
      for (int i = 0; i < input_sorted.size(0) - 1; ++i) {
        if (!at::equal(input_sorted[i], input_sorted[i+1])) {
          mask[i+1] = 1;
        } else {
          mask[i+1] = 0;
        }
      }

      Tensor imask = at::cumsum(mask, 0) - 1;
      for (int i = 0; i < indices.size(0); ++i) {
        inverse_indices[indices[i]] = imask[i];
      }
    }

    THCudaCheck(cudaGetLastError());
    return std::tuple<Tensor, Tensor>(output, inverse_indices);
  }
} // namespace

std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse, optional<int64_t> dim) {
  if (dim) {
    return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
      return _unique_dim_cuda_template<scalar_t>(self, dim.value(), return_inverse);
    });
  }
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return _unique_cuda_template<scalar_t>(self, return_inverse);
  });
}

}  // namespace native
}  // namespace at
