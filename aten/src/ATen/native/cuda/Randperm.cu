#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/cuda/CubUtils.cuh>

#include <limits>

namespace at {
namespace native {

Tensor& randperm_out_cuda(int64_t n, c10::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
  check_supported_max_int_with_precision(n, result);

  result.resize_({n});

  if (n < 30000) {  // For small inputs, we offload it to CPU instead.
    auto result_cpu = at::empty({n}, result.options().device(kCPU));
    randperm_out(result_cpu, n, generator);
    return result.copy_(result_cpu);
  }

#if 0
  // This if condition should never be true because if n >= 30000 and the tensor has a Half type,
  // check_supported_max_int_with_precision should have reported an error. This snippet is commented out but left here
  // for the sake of clarity, because Half in thrust is spotty, and we do not want future change unaware of this.
  if (result.scalar_type() == at::ScalarType::Half) {  // Half in thrust is spotty. Avoid.
    auto result_float = at::empty({n}, initialTensorOptions().device(Device(DeviceType::CUDA)));
    return result.copy_(randperm_out_cuda(result_float, n, generator));
  }
#endif

  // Generate random values for the keys array
  AT_DISPATCH_ALL_TYPES(
    result.scalar_type(), "randperm_out_cuda", [&] {
      TORCH_CHECK(n <= std::numeric_limits<int>::max(),
        "randperm of tensors larger than INT_MAX is not supported yet in pytorch");

      auto keys = at::empty(result.sizes(), result.options()).random_(generator);
      auto range = at::arange(n, result.options());
      auto keys_tmp = at::empty_like(keys);

      // shuffled_data points to the underlying data of the output tensor if the tensor is contiguous; otherwise it
      // points to a new tensor.
      Tensor shuffled;
      scalar_t *shuffled_data;
      if (result.is_contiguous()) {
        shuffled_data = result.data_ptr<scalar_t>();
      } else {
        shuffled = at::empty(n, result.options());
        shuffled_data = shuffled.data_ptr<scalar_t>();
      }

      // Use the sorted order of keys to rearrange the result array
      size_t temp_storage_bytes = 0;

      cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        keys.data_ptr<scalar_t>(), keys_tmp.data_ptr<scalar_t>(),
        range.data_ptr<scalar_t>(), shuffled_data, n,
        0, sizeof(scalar_t) * 8, at::cuda::getCurrentCUDAStream());
      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(temp_storage_bytes);
      cub::DeviceRadixSort::SortPairs(
        dataPtr.get(), temp_storage_bytes,
        keys.data_ptr<scalar_t>(), keys_tmp.data_ptr<scalar_t>(),
        range.data_ptr<scalar_t>(), shuffled_data, n,
        0, sizeof(scalar_t) * 8, at::cuda::getCurrentCUDAStream());

      if (!result.is_contiguous()) {
        result.copy_(shuffled);
      }
    }
  );

  return result;
}



}} // namespace at::native
