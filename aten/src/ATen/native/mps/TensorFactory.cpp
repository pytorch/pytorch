//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/TensorFactory.h>
namespace at { namespace native {

static inline void maybe_resize_storage_mps(TensorImpl* self, uint64_t new_size) {
  if (new_size == 0) {
    return;
  }

  auto storage = self->storage().unsafeGetStorageImpl();
  if (!storage) {
    TORCH_CHECK(false, "Tensor: invalid null storage");
  }
  uint64_t new_size_bytes = (new_size + self->storage_offset()) * self->dtype().itemsize();
  if (new_size_bytes > self->storage().nbytes()) {
    if (new_size_bytes == 0) {
      storage->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(at::DeviceType::MPS, 0)));
      storage->set_nbytes(0);
    } else {
      at::DataPtr new_data = storage->allocator()->allocate(new_size_bytes);
      size_t copy_capacity = std::min<size_t>(new_size_bytes, storage->nbytes());
      if (storage->data() && copy_capacity > 0) {
        at::native::mps::copy_blit_mps(new_data.get(), storage->data(), copy_capacity);
      }
      // Destructively overwrite data_ptr
      storage->set_data_ptr_noswap(std::move(new_data));
      storage->set_nbytes(new_size_bytes);
    }
  }
}

inline TensorImpl* resize_impl_mps_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_mps(self, storage_size);

  return self;
}

Tensor empty_mps(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {

  return at::detail::empty_mps(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

Tensor empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  // empty memory formatempty
  auto t = at::native::empty_mps(
      {0},
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt);
  resize_impl_mps_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

const Tensor& resize_mps_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  //std::cout << "resize mps  size " << size << std::endl;
  resize_impl_mps_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

Tensor& set_mps_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::mps::GetMPSAllocator(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& set_storage_mps_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);
  //std::cout << "set storage_mps " << storage_offset << " stride " << stride << std::endl;
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          c10::optional<IntArrayRef>(stride) : c10::nullopt;
  at::native::resize_impl_mps_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}
} // namespace native

namespace detail {
TensorBase empty_mps(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {

  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::MPS);

  TORCH_CHECK_NOT_IMPLEMENTED(
      layout_or_default(layout_opt) == Layout::Strided,
      "strided meta tensors not supported yet");
  check_size_nonnegative(size);

  auto* allocator = at::mps::GetMPSAllocator();
  int64_t nelements = c10::multiply_integers(size);
  auto dtype = dtype_or_default(dtype_opt);
  auto dtype_meta = scalarTypeToTypeMeta(dtype);
  int64_t size_bytes = nelements * dtype_meta.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::MPS, dtype_meta);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = memory_format_opt.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;

}

TensorBase empty_mps(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_mps(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mps());
  const DeviceGuard device_guard(device);
  auto* allocator = at::mps::GetMPSAllocator();
  constexpr c10::DispatchKeySet mps_dks(c10::DispatchKey::MPS);
  return at::detail::empty_strided_generic(
      size, stride, allocator, mps_dks, dtype);
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::native::empty_strided_mps(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

} // namespace detail
} // namespace at
