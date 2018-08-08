#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace at {
namespace native{

#ifndef __HIP_PLATFORM_HCC__

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
    std::vector<int64_t> orig_sizes(input_flat.sizes().begin(), input_flat.sizes().end());
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
          if ( lhs < rhs) {
            return true;
          }
          else if ( lhs > rhs ) {
            return false;
          }
        }
        return false;
      });

    Tensor input_sorted = at::empty(input_flat.sizes(), input_flat.type());
    for ( int i = 0; i < indices.size(0); ++i) {
      input_sorted[i] = input_flat[indices[i]];
    }

    // Why is this not working?
    //thrust::device_vector<Tensor> input_unbind = at::unbind(input_sorted, 0);
    // auto last = thrust::unique(policy, input_unbind.begin(), input_unbind.end(), 
    //   [=] __device__ (Tensor a, Tensor b) -> bool {
    //     return at::equal(a, b);
    //   });
    //input_unbind.erase(last, input_unbind.end());

    scalar_t* input_sorted_ptr = input_sorted.data<scalar_t>();
    thrust::device_vector<int64_t> input_sorted_indices(input_sorted.size(0));
    thrust::sequence(policy, input_sorted_indices.begin(), input_sorted_indices.end());
    auto last = thrust::unique(policy, input_sorted_indices.begin(), input_sorted_indices.end(),
      [=] __device__ (int64_t a, int64_t b) -> bool {
        bool eq = true;
        for (int64_t i = 0; i < numel; ++i) {
          scalar_t lhs = input_sorted_ptr[i + a * numel];
          scalar_t rhs = input_sorted_ptr[i + b * numel];
          if ( lhs != rhs ) {
            eq = false;
          }
        }
        return eq;
      });
    input_sorted_indices.erase(last, input_sorted_indices.end());
    
    Tensor output_dim = at::empty({0}, self.type());
    output_dim.resize_({(int64_t)input_sorted_indices.size(), numel});
    for (int i = 0; i < output_dim.size(0); ++i) {
      output_dim[i] = input_sorted[input_sorted_indices[i]];
    }
    
    // // reshape back
    std::vector<int64_t> new_sizes(orig_sizes.begin(), orig_sizes.end());
    new_sizes[0] = -1;
    output_dim = output_dim.view(new_sizes);
    output_dim = output_dim.transpose(0, dim);

    Tensor inverse_indices_dim = at::empty({0}, self.type().toScalarType(kLong));
    int64_t size = self.size(dim);
    inverse_indices_dim.resize_(size);
    Tensor mask = at::empty(input_sorted.size(0), self.type().toScalarType(kLong));
    mask[0] = 1;
    for (int i = 0; i < input_sorted.size(0) - 1; ++i) {
      if (!at::equal(input_sorted[i], input_sorted[i+1])) {
        mask[i+1] = 1; 
      }
      else {
        mask[i+1] = 0;
      }
    }

    Tensor imask = at::cumsum(mask, 0) - 1;
    for (int i = 0; i < indices.size(0); ++i) {
      inverse_indices_dim[indices[i]] = imask[i];
    }

    THCudaCheck(cudaGetLastError());  
    return std::tuple<Tensor, Tensor>(output_dim, inverse_indices_dim);
  }
} // namespace

#endif

std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
#ifndef __HIP_PLATFORM_HCC__
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return _unique_cuda_template<scalar_t>(self, return_inverse);
  });
#else
  AT_ERROR("unique_cuda: HIP not supported");
#endif
}

std::tuple<Tensor, Tensor>
_unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse) {
  #ifndef __HIP_PLATFORM_HCC__
    return AT_DISPATCH_ALL_TYPES(self.type(), "unique_dim", [&] {
      return _unique_dim_cuda_template<scalar_t>(self, dim, return_inverse);
    });
  #else
    AT_ERROR("unique_dim_cuda: HIP not supported");
  #endif
}

}  // namespace native
}  // namespace at
