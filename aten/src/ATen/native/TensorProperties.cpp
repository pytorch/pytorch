#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/op_registration/op_registration.h>

#include <ATen/Config.h>
namespace at {
namespace native {

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sizes().equals(other.sizes());
}

int64_t size(const Tensor& self, int64_t dim) {
  // false is passed to maybe_wrap_dim so behavior is identical to array access (but with wrapping)
  dim = maybe_wrap_dim(dim, self.dim(), false);
  return self.sizes()[dim];
}

int64_t stride(const Tensor& self, int64_t dim) {
  // false is passed to maybe_wrap_dim so behavior is identical to array access (but with wrapping)
  dim = maybe_wrap_dim(dim, self.dim(), false);
  return self.strides()[dim];
}

int64_t size(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.sizes()[pos_dim];
}

int64_t stride(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.strides()[pos_dim];
}

bool cudnn_is_acceptable(const Tensor& self) {
  if (!globalContext().userEnabledCuDNN()) return false;
  if (!self.is_cuda()) return false;
  auto st = self.scalar_type();
  if (!(st == kDouble || st == kFloat || st == kHalf)) return false;
  if (!detail::getCUDAHooks().compiledWithCuDNN()) return false;
  // cuDNN functions like grid_sampler returns CUDNN_STATUS_BAD_PARAM on empty
  // tensors. Maybe some cuDNN functions actually support empty tensors, but
  // native/THNN kernels shouldn't be much slower because the output is also
  // likely empty.
  if (self.numel() == 0) return false;
  // NB: In the old Python code, there was also a test to see if the
  // cuDNN library was actually dynamically linked or not.  I'm not
  // sure if we can actually test this.
  return true;
}

Tensor detach(const Tensor& self) {
#ifndef USE_STATIC_DISPATCH
  // this just exists to give us a hook in VariableType and an entry in Declarations.yaml
  //AT_ERROR("detach is not implemented for Tensor");
#endif
  // this is no-op for USE_STATIC_DISPATCH mode
  return self;
}

Tensor & detach_(Tensor & self) {
#ifndef USE_STATIC_DISPATCH
  // this just exists to give us a hook in VariableType and an entry in Declarations.yaml
  //AT_ERROR("detach_ is not implemented for Tensor");
#endif
  // this is no-op for USE_STATIC_DISPATCH mode
  return self;
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::detach(Tensor self) -> Tensor")
    .catchAllKernel<decltype(detach), &detach>())
  .op(torch::RegisterOperators::options()
    .schema("aten::detach_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyCatchAllKernel<decltype(detach_), &detach_>())
  ;

Tensor contiguous(const Tensor & self) {
  return contiguous(self, MemoryFormat::Contiguous);
}

Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  auto result = at::empty_like(self, self.options().memory_format(memory_format));
  return result.copy_(self);
}

bool is_set_to(const Tensor& self, const Tensor& src) {
  if (self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() &&
      self.dim() == src.dim()) {
    for (int64_t d = 0; d < self.dim(); ++d) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace native
} // namespace at
