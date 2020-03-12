#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cuda_(Tensor& result) {
  Storage storage(result.dtype(), 0, at::cuda::getCUDADeviceAllocator(), true);
  return result.set_(storage, 0, {0}, {});
}

// unify with cuda implementation?  This is not done to avoid a dispatch in resize_impl_cpu_
Tensor& set_storage_cuda_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  // FIXME: stride should be optional
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

#ifdef DEBUG
  TORCH_CHECK(size_.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
#endif

  // storage: note this can't be replaced with result.set_(storage) as the semantics of that
  // function is to set the tensor size to be equal to the size of the storage.
  if (!result.storage().is_alias_of(storage)) {
    // Caffe2 might have tensors whose storages are null, but we
    // don't allow it in PyTorch.
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_INTERNAL_ASSERT(result.storage());

    // Caffe2 also has uninitialized dtype states, which we disallow here
    TORCH_INTERNAL_ASSERT(result.storage().dtype() == storage.dtype());

    // We used to allow this, but this breaks device caching.
    // Let's put an actual error message for this one.
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    result.unsafeGetTensorImpl()->set_storage(storage);
  }

  // storageOffset
  if (storage_offset < 0) {
    TORCH_CHECK("Tensor: invalid storage offset ", storage_offset);
  }

  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                              c10::optional<IntArrayRef>(stride) : c10::nullopt;
  at::native::resize_impl_cuda_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

}
}
