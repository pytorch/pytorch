#include "ATen/ATen.h"
#include "ATen/InitialTensorOptions.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAContext.h"
#include "c10/util/Exception.h"

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstddef>

namespace at {
namespace native {

Tensor& eye_out_cuda(Tensor& result, int64_t n) {
  return at::native::eye_out_cuda(result, n, /*m=*/-1);
}

Tensor& eye_out_cuda(Tensor& result, int64_t n, int64_t m) {
  AT_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if(m < 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

Tensor empty_cuda(IntList size, const TensorOptions& options) {
  AT_ASSERT(options.backend() == at::Backend::CUDA);
  AT_ASSERT(!options.is_variable());  // is_variable should have been 'unpacked'
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    options.dtype(), 0, cuda::getCUDADeviceAllocator(), true);

  auto tensor = detail::make_tensor<TensorImpl>(storage_impl, CUDATensorId(), false);
  resize_cuda_(tensor, size); // avoid dispatch overhead
  return tensor;
}

Tensor& randperm_out_cuda(Tensor& result, int64_t n, Generator* generator) {
  AT_CHECK(n >= 0, "n must be non-negative, got", n);
  AT_CHECK(result.type().scalarTensor(n).defined(),
  "n is too large for result tensor type: '", result.type().toString(), "'");

  result.resize_({n});

  if (result.type().scalarType() == at::ScalarType::Half) {
    auto result_float = at::empty({n}, initialTensorOptions().device(Device(DeviceType::CUDA)));
    result.copy_(randperm_out_cuda(result_float, n, generator));
  } else {
    if (n < 30000) {  // For small inputs, we offload it to CPU instead.
      auto result_cpu = at::empty({n}, result.options().device(kCPU));
      randperm_out(result_cpu, n, generator);
      result.copy_(result_cpu);
    } else {
      // Generate random values for the keys array
      AT_DISPATCH_ALL_TYPES(
        result.type(), "randperm_out_cuda", [&] {
          auto keys = at::empty(result.sizes(), result.options()).random_(generator);

          auto result_data = thrust::device_ptr<scalar_t>(result.data<scalar_t>());
          auto keys_data = thrust::device_ptr<scalar_t>(keys.data<scalar_t>());

          auto state = globalContext().getTHCState();
          THCThrustAllocator thrustAlloc(state);
          auto policy = thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state));

          thrust::sequence(policy, result_data, result_data + n);

          // Use the sorted order of keys to rearrange the result array
          thrust::sort_by_key(policy, keys_data, keys_data + n, result_data);
        }
      );
    }
  }

  return result;
}

}} // namespace at::native
