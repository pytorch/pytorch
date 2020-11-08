#include <ATen/Utils.h>
#include <stdarg.h>
#include <stdexcept>
#include <typeinfo>
#include <cstdlib>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/Dispatch.h>

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
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  const MemoryFormat memory_format =
    optional_memory_format.value_or(
      options.memory_format_opt().value_or(
        MemoryFormat::Contiguous));

  AT_ASSERT(options.device().type() == DeviceType::CPU);
  check_size_nonnegative(size);

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = at::getCPUAllocator();
  }

  int64_t nelements = prod_intlist(size);
  const caffe2::TypeMeta dtype = options.dtype();
  const int64_t size_bytes = nelements * dtype.itemsize();
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

  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_cpu", [&] {
    std::copy(values.begin(), values.end(), result.template data_ptr<scalar_t>());
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
    std::copy(values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_complex_backend(ArrayRef<T> values, const TensorOptions& options) {
  auto cpu_tensor = tensor_complex_cpu(values, options.device(DeviceType::CPU));
  return cpu_tensor.to(options.device());
}

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return tensor_backend(values, options);                       \
    } else {                                                        \
      return tensor_cpu(values, options);                           \
    }                                                               \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return tensor_complex_backend(values, options);               \
    } else {                                                        \
      return tensor_complex_cpu(values, options);                   \
    }                                                               \
  }
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR
} // namespace detail
} // at
