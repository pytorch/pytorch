#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <thrust/unique.h>
#include <thrust/sort.h>

namespace at {
namespace native{

namespace {
template <typename scalar_t>
__global__ void inverse_indices_kernel(
    const scalar_t* input_data,
    const scalar_t* output_data,
    int64_t* inverse_indices_data,
    int64_t num_inp,
    int64_t num_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < num_inp * num_out; i += stride) {
      if (input_data[i / num_out] == output_data[i % num_out]){
        inverse_indices_data[i / num_out] = i % num_out;   
      }
    }
  }


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
    thrust::sort(policy, output_data, output_data + num_inp);
    scalar_t* output_end = thrust::unique(policy, output_data, output_data + num_inp);
    int64_t num_out = output_end - output_data;
    output.resize_(num_out);

    Tensor inverse_indices = at::empty({0}, self.type().toScalarType(kLong));

    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
      int64_t* inverse_indices_data = inverse_indices.data<int64_t>();
      int block = 512;
      int grid = std::min<int64_t>((num_inp * num_out + block - 1) / block, 2048L);
      inverse_indices_kernel<<<grid, block, 0, stream>>>(
        input_data, output_data, inverse_indices_data, num_inp, num_out);
    }

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
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return _unique_cuda_template<scalar_t>(self, return_inverse);
  });
}

std::tuple<Tensor, Tensor>
_unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique_dim", [&] {
    return _unique_dim_cuda_template<scalar_t>(self, dim, return_inverse);
  });
}

}  // namespace native
}  // namespace at
