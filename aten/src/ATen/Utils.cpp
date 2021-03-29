#include <ATen/Context.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>


#include <stdarg.h>
#include <cstdlib>
#include <stdexcept>
#include <typeinfo>

namespace at {

int _crash_if_asan(int arg) {
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}

namespace detail {
// empty_cpu is used in ScalarOps.h, which can be referenced by other ATen
// files. Since we want to decouple direct referencing native symbols and only
// access native symbols through dispatching, we move its implementation here.
Tensor empty_cpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {

  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  bool pin_memory = pinned_memory_or_default(pin_memory_opt);
  c10::Allocator* allocator;
  if (pin_memory) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = at::getCPUAllocator();
  }
  auto dtype = dtype_or_default(dtype_opt);

  return empty_generic(size, allocator, at::DispatchKey::CPU, dtype, device, memory_format_opt);
}

Tensor empty_generic(
  IntArrayRef size,
  c10::Allocator* allocator,
  // technically this can be inferred from the device, but usually the
  // correct setting is obvious from the call site so just make callers
  // pass it in
  c10::DispatchKey dispatch_key,
  ScalarType scalar_type,
  Device device,
  c10::optional<c10::MemoryFormat> memory_format_opt) {

  check_size_nonnegative(size);

  int64_t nelements = c10::multiply_integers(size);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      std::move(storage_impl), dispatch_key, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_cpu", [&] {
    std::copy(
        values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options) {
  auto cpu_tensor = tensor_cpu(values, options.device(DeviceType::CPU));
  return cpu_tensor.to(options.device());
}

template <typename T>
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_COMPLEX_TYPES(result.scalar_type(), "tensor_cpu", [&] {
    std::copy(
        values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_complex_backend(
    ArrayRef<T> values,
    const TensorOptions& options) {
  auto cpu_tensor = tensor_complex_cpu(values, options.device(DeviceType::CPU));
  return cpu_tensor.to(options.device());
}
} // namespace detail

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return at::detail::tensor_backend(values, options);           \
    } else {                                                        \
      return at::detail::tensor_cpu(values, options);               \
    }                                                               \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return at::detail::tensor_complex_backend(values, options);   \
    } else {                                                        \
      return at::detail::tensor_complex_cpu(values, options);       \
    }                                                               \
  }
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR
} // namespace at
