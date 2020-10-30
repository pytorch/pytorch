#include <ATen/Utils.h>
#include <stdarg.h>
#include <stdexcept>
#include <typeinfo>
#include <cstdlib>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Context.h>

namespace at {

int _crash_if_asan(int arg) {
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}

namespace detail {
// empty_cpu is used in ScalarOps.h, which can be referenced by other ATen files. Since we want to decouple direct referencing native symbols and only access native symbols through dispatching, we move its implementation here.
Tensor empty_cpu(
    IntArrayRef size,
    const TensorOptions& options_,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  TensorOptions options =
      options_.merge_in(TensorOptions().memory_format(optional_memory_format));

  AT_ASSERT(options.device().type() == DeviceType::CPU);
  check_size_nonnegative(size);

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = at::getCPUAllocator();
  }

  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      std::move(storage_impl), at::DispatchKey::CPU, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}
} // namespace detail

} // at
