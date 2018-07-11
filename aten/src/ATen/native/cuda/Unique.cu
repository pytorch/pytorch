#include "ATen/ATen.h"

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <thrust/unique.h>
#include <thrust/sort.h>

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

    cudaStream_t stream = globalContext().getCurrentCUDAStream();
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

}  // namespace native
}  // namespace at
