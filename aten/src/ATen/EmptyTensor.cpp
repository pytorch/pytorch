#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/safe_numerics.h>

#include <limits>

namespace at::detail {
namespace {
c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    // NB: This is not quite right, if you somehow had both CUDA and PrivateUse1 initialized
    // in the same PyTorch build, you would ONLY ever get the CUDA pinned memory allocator.
    // To properly support this, see https://github.com/pytorch/pytorch/issues/14560
    if (at::globalContext().hasCUDA()) {
      return at::detail::getCUDAHooks().getPinnedMemoryAllocator();
    } else if (at::globalContext().hasMTIA()) {
      return at::detail::getMTIAHooks().getPinnedMemoryAllocator();
    } else if (at::globalContext().hasXPU()) {
      return at::detail::getXPUHooks().getPinnedMemoryAllocator();
    } else if (at::globalContext().hasHPU()) {
      return at::detail::getHPUHooks().getPinnedMemoryAllocator();
    } else if(at::isPrivateUse1HooksRegistered()) {
      return at::detail::getPrivateUse1Hooks().getPinnedMemoryAllocator();
    } else {
      TORCH_CHECK(false, "Need to provide pin_memory allocator to use pin memory.")
    }
  }
  return c10::GetCPUAllocator();
}

#ifndef C10_MOBILE
constexpr uint64_t storage_max() {
  // int64_t and size_t are used somewhat inconsistently throughout ATen.
  // To be safe, storage size calculations must fit in both types.
  constexpr auto int64_max = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max());
  constexpr auto size_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max());
  return std::min(int64_max, size_max);
}
#endif

inline void raise_warning_for_complex_half(ScalarType dtype) {
  if (dtype == kComplexHalf) {
    TORCH_WARN_ONCE(
        "ComplexHalf support is experimental and many operators don't support it yet.");
  }
}

}  // namespace (anonymous)

size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  uint64_t size = 1;
  bool overflowed = c10::safe_multiplies_u64(sizes, &size);
  overflowed |= c10::add_overflows(size, storage_offset, &size);
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=", sizes);
  return static_cast<size_t>(size);
#else
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
#endif
}

size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = storage_offset + 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    uint64_t strided_size = 0;
    overflowed |= c10::mul_overflows(strides[i], sizes[i] - 1, &strided_size);
    overflowed |= c10::add_overflows(size, strided_size, &size);
  }
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=",
              sizes, " and strides=", strides);
  return static_cast<size_t>(size);
#else
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
#endif
}

SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
}

// not including mobile-only macros in this function,
// since mobile shouldn't be using symints.
SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  SymInt size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sizes[i].sym_eq(0))) {
      return 0;
    }

    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
}

template <typename T>
static TensorBase _empty_generic(
    ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (ks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

TensorBase empty_generic_symint(
    SymIntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

template <typename T>
static TensorBase _empty_strided_generic(
    T size,
    T stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<IntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<SymIntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
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
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
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
  DataPtr allocate(const size_t nbytes [[maybe_unused]]) override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {}
};

static MetaAllocator g_meta_alloc;

REGISTER_ALLOCATOR(kMeta, &g_meta_alloc)

TensorBase empty_meta(IntArrayRef size, ScalarType dtype,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_generic(
      size, allocator, meta_dks, dtype, memory_format_opt);
}

TensorBase empty_meta(
  IntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "non-strided meta tensors not supported yet"
  );

  auto dtype = dtype_or_default(dtype_opt);
  return empty_meta(size, dtype, memory_format_opt);
}

TensorBase empty_symint_meta(
  SymIntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::Meta);
  auto scalar_type = dtype_or_default(dtype_opt);
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
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
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
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

TensorBase empty_strided_symint_meta(SymIntArrayRef size, SymIntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_symint_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_symint_meta(size, stride, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_symint_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt());
}

} // namespace at::detail
