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
          DispatchKeySet{DispatchKey::CPUTensorId}));

  return namedinference::propagate_names_if_nonempty(
      tensor.resize_(size, memory_format),
      maybe_names);
}

Tensor allocate_padded_if_needed(const Tensor& input_contig) {
  const auto* const allocator = input_contig.storage().allocator();
  const auto* const guarding_allocator = get_guarding_allocator();
  if (allocator == guarding_allocator) {
    return input_contig;
  }
  Tensor padded_input =
      empty_with_tail_padding(input_contig.sizes(), input_contig.options().dtype(),
          input_contig.suggest_memory_format(), input_contig.names());
  padded_input.copy_(input_contig);
  return padded_input;
}

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
