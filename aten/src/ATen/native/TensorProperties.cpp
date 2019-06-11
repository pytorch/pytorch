#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>

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
  // this just exists to give us a hook in VariableType and an entry in Declarations.yaml
  AT_ERROR("detach is not implemented for Tensor");
  return self;
}

Tensor & detach_(Tensor & self) {
  // this just exists to give us a hook in VariableType and an entry in Declarations.yaml
  AT_ERROR("detach_ is not implemented for Tensor");
  return self;
}

Tensor contiguous(const Tensor & self) {
  return contiguous(self, MemoryFormat::Contiguous);
}

Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  auto  result = at::empty_like(self);
  switch (memory_format) {
    case MemoryFormat::Any: // Back compatibility with old defaults
    case MemoryFormat::Contiguous: {
      break;
    }
    case MemoryFormat::ChannelsLast: {
      TORCH_CHECK(
          result.dim() == 4,
          " required rank 4 tensor to use channels_last format");
      std::vector<int64_t> newStrides(self.dim());
      auto sizes = result.sizes();
      newStrides[1] = 1;
      newStrides[3] = sizes[1];
      newStrides[2] = newStrides[3] * sizes[3];
      newStrides[0] = newStrides[2] * sizes[2];
      result = result.as_strided(sizes, newStrides);
      break;
    }
    default: {
      TORCH_CHECK(false, " unsupported memory format");
    }
  }
  return result.copy_(self);
}
} // namespace native
}
