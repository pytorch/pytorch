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
Tensor empty_cpu(IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt,
                 c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> memory_format_opt) {
  Device device = device_or_default(device_opt);

  TORCH_CHECK(device.type() == DeviceType::CPU);
  check_size_nonnegative(size);

  bool pin_memory = pinned_memory_or_default(pin_memory_opt);
  c10::Allocator* allocator;
  if (pin_memory) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = at::getCPUAllocator();
  }

  int64_t nelements = prod_intlist(size);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
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

  auto memory_format = memory_format_opt.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}
} // namespace detail

} // at
