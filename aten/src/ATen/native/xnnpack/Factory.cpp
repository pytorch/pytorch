#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Factory.h>
#include <ATen/native/utils/Allocator.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {

Tensor empty_with_tail_padding(
    const IntArrayRef size,
    const caffe2::TypeMeta dtype,
    const c10::MemoryFormat memory_format) {
  static GuardingAllocator<0u, XNN_EXTRA_BYTES> allocator;

  const int64_t nelements = prod_intlist(size);

  Tensor tensor(
      c10::make_intrusive<c10::TensorImpl>(
          c10::Storage{
              dtype,
              nelements,
              allocator.allocate(nelements * dtype.itemsize()),
              &allocator,
              /*resizable=*/true,
          },
          DispatchKeySet{DispatchKey::CPUTensorId}));

  return tensor.resize_(size, memory_format);
}

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
