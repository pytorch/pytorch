#ifdef USE_XNNPACK

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
    const c10::MemoryFormat memory_format) {
  auto* allocator_ptr = get_guarding_allocator();

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

  return tensor.resize_(size, memory_format);
}

Tensor allocate_padded_if_needed(const Tensor& input_contig) {
  const auto* allocator = input_contig.storage().allocator();
  const auto* guarding_allocator = get_guarding_allocator();
  if (allocator == guarding_allocator) {
    return input_contig;
  }
  Tensor padded_input =
      empty_with_tail_padding(input_contig.sizes(), input_contig.options().dtype(),
          input_contig.suggest_memory_format());
  padded_input.copy_(input_contig);
  return padded_input;
}

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
