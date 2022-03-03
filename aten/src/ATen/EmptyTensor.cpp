#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/CPUAllocator.h>

// GCC has __builtin_mul_overflow from before it supported __has_builtin
#ifdef __GNUC__
#define HAS_BUILTIN_OVERFLOW() (1)
#elif defined(__has_builtin)
#define HAS_BUILTIN_OVERFLOW() (                    \
      __has_builtin(__builtin_mul_overflow) &&      \
      __has_builtin(__builtin_and_overflow))
#else
#define HAS_BUILTIN_OVERFLOW() (0)
#endif

namespace at {
namespace detail {

static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return at::detail::getCUDAHooks().getPinnedMemoryAllocator();
  }
  return c10::GetCPUAllocator();
}

static size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
#if HAS_BUILTIN_OVERFLOW()
    overflowed |= __builtin_mul_overflow(size, sizes[i], &size);
#else
    size *= sizes[i];
#endif
  }

#if HAS_BUILTIN_OVERFLOW()
  overflowed |= __builtin_mul_overflow(size, itemsize_bytes, &size);
#else
  size *= itemsize_bytes;
#endif

  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=", sizes);
  return size;
}

size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
    if(sizes[i] == 0) {
      return 0;
    }

#if HAS_BUILTIN_OVERFLOW()
    size_t strided_size;
    overflowed |= __builtin_mul_overflow(strides[i], sizes[i] - 1, &strided_size);
    overflowed |= __builtin_add_overflow(size, strided_size, &size);
#else
    size += strides[i] * (sizes[i] - 1);
#endif
  }

#if HAS_BUILTIN_OVERFLOW()
  overflowed |= __builtin_mul_overflow(size, itemsize_bytes, &size);
#else
  size *= itemsize_bytes;
#endif

  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=",
              sizes, " and strides=", strides);
  return size;
}

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  at::detail::check_size_nonnegative(size);

  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  size_t size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
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

TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  at::detail::check_size_nonnegative(size);

  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  size_t size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_cpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_cpu(IntArrayRef size, IntArrayRef stride,
                             ScalarType dtype, bool pin_memory) {
  auto allocator = at::detail::GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cpu_ks, dtype);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_cpu(size, stride, dtype, pin_memory);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_cpu(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

// The meta allocator ignores whatever allocation is requested and always
// gives you nullptr
struct MetaAllocator final : public at::Allocator {
  MetaAllocator() = default;
  ~MetaAllocator() override = default;
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  DataPtr allocate(const size_t nbytes) const override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
};

static MetaAllocator g_meta_alloc;

REGISTER_ALLOCATOR(kMeta, &g_meta_alloc);

TensorBase empty_meta(IntArrayRef size, ScalarType dtype,
                     c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_generic(
      size, allocator, meta_dks, dtype, memory_format_opt);
}

TensorBase empty_meta(
  IntArrayRef size,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt,
  c10::optional<c10::MemoryFormat> memory_format_opt
) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "non-strided meta tensors not supported yet"
  );

  auto dtype = dtype_or_default(dtype_opt);
  return empty_meta(size, dtype, memory_format_opt);
}

TensorBase empty_meta(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_meta(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_meta(IntArrayRef size, IntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_meta(size, stride, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

}} // namespace at::detail
