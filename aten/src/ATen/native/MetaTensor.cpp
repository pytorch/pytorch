#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

Tensor empty_meta(
  IntArrayRef size,
  c10::optional<ScalarType> dtype,
  c10::optional<Layout> layout,
  c10::optional<Device> device,
  c10::optional<bool> pin_memory,
  c10::optional<c10::MemoryFormat> memory_format
) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device).type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout) == Layout::Strided,
    "strided meta tensors not supported yet"
  );

  check_size_nonnegative(size);

  auto tensor = detail::make_tensor<TensorImpl>(
    DispatchKeySet{DispatchKey::Meta},
    scalarTypeToTypeMeta(dtype_or_default(dtype)),
    device
  );

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  auto memory_format_ = memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format_);

  tensor.unsafeGetTensorImpl()->set_storage_access_should_throw();

  return tensor;
}

Tensor empty_strided_meta(
  IntArrayRef size,
  IntArrayRef stride,
  c10::optional<ScalarType> dtype,
  c10::optional<Layout> layout,
  c10::optional<Device> device,
  c10::optional<bool> pin_memory
) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device).type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout) == Layout::Strided,
    "strided meta tensors not supported yet"
  );

  // NB: pin_memory intentionally ignored; it is a property of storage and
  // therefore meta does not track it  (this is not a forced choice, but it's
  // the choice we made)

  check_size_nonnegative(size);
  // TODO: check if strides are negative,
  // https://github.com/pytorch/pytorch/issues/53391
  // (bugged here to be consistent with CPU implementation)

  auto tensor = detail::make_tensor<TensorImpl>(
    DispatchKeySet{DispatchKey::Meta},
    scalarTypeToTypeMeta(dtype_or_default(dtype)),
    device
  );

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  tensor.unsafeGetTensorImpl()->set_storage_access_should_throw();

  return tensor;
}

} // namespace native
} // namespace at
