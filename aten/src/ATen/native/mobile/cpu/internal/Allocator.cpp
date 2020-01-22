#include <ATen/native/mobile/cpu/internal/Allocator.h>

#ifdef USE_XNNPACK

#include <ATen/native/TensorFactories.h>
#include <c10/core/CPUAllocator.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {

void Allocator::deleter(void* const memory) {
  c10::free_cpu(memory);
}

DataPtr Allocator::allocate(size_t nbytes) const {
  void* const memory = c10::alloc_cpu(nbytes + kGuard);

  return DataPtr{
      memory,
      memory,
      &deleter,
      Device(DeviceType::CPU),
  };
}

DeleterFnPtr Allocator::raw_deleter() const {
  return deleter;
}

Tensor new_tensor(
    const IntArrayRef size,
    const TensorOptions& options,
    const c10::MemoryFormat memory_format) {
  static Allocator allocator;

  AT_ASSERT(options.device().is_cpu());
  check_size_nonnegative(size);

  const int64_t nelements = prod_intlist(size);
  const caffe2::TypeMeta dtype = options.dtype();

  c10::intrusive_ptr<c10::TensorImpl> tensor_impl =
      c10::make_intrusive<c10::TensorImpl>(
          c10::Storage{
              dtype,
              nelements,
              allocator.allocate(nelements * dtype.itemsize()),
              &allocator,
              /*resizeable=*/true,
          },
          DispatchKeySet{DispatchKey::CPUTensorId});

  tensor_impl->set_sizes_contiguous(size);
  tensor_impl->empty_tensor_restride(memory_format);

  return Tensor(std::move(tensor_impl));
}

} // namespace internal
} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
