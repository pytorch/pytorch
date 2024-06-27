#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>

namespace at::native {

bool is_pinned_cuda(const Tensor& self, std::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  // TODO: unhook this
  return detail::getCUDAHooks().isPinnedPtr(self.storage().data());
}

bool is_pinned_sparse_coo(const Tensor& self, std::optional<Device> device) {
  return self._values().is_pinned(device);
}

bool is_pinned_sparse_compressed(const Tensor& self, std::optional<Device> device) {
  return self.values().is_pinned(device);
}

Tensor _pin_memory_cuda(const Tensor& self, std::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  auto* allocator = at::cuda::getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

Tensor _pin_memory_sparse_compressed(const Tensor& self, std::optional<Device> device) {
  at::sparse_csr::CheckSparseTensorInvariants _(false);
  TensorOptions options = self.options().pinned_memory(true);
  auto impl = at::sparse_csr::get_sparse_csr_impl(self);
  return at::_sparse_compressed_tensor_unsafe(
        impl->compressed_indices(),
        impl->plain_indices(),
        impl->values(),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
}

Tensor _pin_memory_sparse_coo(const Tensor& self, std::optional<Device> device) {
  TensorOptions options = self.options().pinned_memory(true);
  return at::_sparse_coo_tensor_with_dims_and_tensors(
      self.sparse_dim(),
      self.dense_dim(),
      self.sizes(),
      self._indices(),
      self._values(),
      options,
      self.is_coalesced());
}

} // namespace at::native
