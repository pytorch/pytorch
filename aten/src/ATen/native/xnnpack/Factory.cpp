#ifdef USE_XNNPACK

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/xnnpack/Factory.h>
#include <ATen/native/utils/Allocator.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {

GuardingAllocator<0u, XNN_EXTRA_BYTES>* get_guarding_allocator() {
  static GuardingAllocator<0u, XNN_EXTRA_BYTES> allocator;
  return &allocator;
}

Tensor empty_with_tail_padding(
    const IntArrayRef size,
    const caffe2::TypeMeta dtype,
    const c10::MemoryFormat memory_format,
    const DimnameList maybe_names) {
  auto* const allocator_ptr = get_guarding_allocator();
  const int64_t nelements = prod_intlist(size);

  Tensor tensor(
      c10::make_intrusive<c10::TensorImpl>(
          c10::Storage{
              dtype,
              nelements,
              allocator_ptr->allocate(nelements * dtype.itemsize()),
              allocator_ptr,
              /*resizable=*/true,
          },
          DispatchKeySet{DispatchKey::CPU}));

  return namedinference::propagate_names_if_nonempty(
      tensor.resize_(size, memory_format),
      maybe_names);
}

Tensor allocate_padded_contiguous_if_needed(
    const Tensor& input,
    const c10::MemoryFormat memory_format) {
  const auto* const allocator = input.storage().allocator();
  const auto* const guarding_allocator = get_guarding_allocator();

  // If the allocators are the same and the memory is contiguous in the requested
  // format, then there is no need to reallocate the tensor.

  if ((allocator == guarding_allocator) && input.is_contiguous(memory_format)) {
    return input;
  }

  // If there is a need to reallocate the tensor on the other hand, either because
  // the allocators are not the same, or the allocators are the same but the input
  // is not contiguous in the requested format, then reallocate and directly copy
  // into destination.  There is no need to allocate a temporary contiguous memory
  // only to use it as the source of the copy operation onto our final destination.

  Tensor padded_input = empty_with_tail_padding(
      input.sizes(),
      input.options().dtype(),
      memory_format,
      input.names());

  return padded_input.copy_(input);
}

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
