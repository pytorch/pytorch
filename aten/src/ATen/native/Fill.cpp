// Functions that fill Tensors with constants.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/Fill.h>
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fill_diagonal_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zero_native.h>
#endif

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor& fill_out(Tensor& self, const Scalar& value) {
  if (self.device() == at::kCPU && self.numel() == 1) {
    return at::detail::scalar_fill(self, value);
  }
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
    .check_all_same_dtype(false)
    .add_output(self)
    .resize_outputs(false)
    .build();
  fill_stub(iter.device_type(), iter, value);
  return self;
}

static Tensor& fill_out_quantized(Tensor& self, const Scalar& value) {
  at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
  out = out.to(self.device()).to(self.suggest_memory_format());
  // Trust the `copy_` to handle the quantization and the boundary checks.
  self.copy_(out);
  return self;
}

Tensor& fill_(Tensor& self, const Scalar& value) {
  return fill_out(self, value);
}

Tensor& fill_quantized_(Tensor& self, const Scalar& value) {
  return fill_out_quantized(self, value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  if (self.device() != value.device()){
    return fill_out(self, value.item());
  }
  // Check if value is a view of self and if it is we clone
  // it to avoid overwriting self prematurely
  if(self.is_alias_of(value)) {
    self.copy_(value.clone());
  } else{
    self.copy_(value);
  }
  return self;
}

Tensor& fill_quantized_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill_out_quantized(self, value.item());
}

Tensor& fill_meta_(Tensor& self, const Scalar& value) {
  return self;
}

Tensor& fill_meta_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return self;
}

Tensor fill(const Tensor& self, const Scalar& value) {
  return at::empty_like(self).fill_(value);
}

Tensor fill(const Tensor& self, const Tensor& value) {
  return at::empty_like(self).fill_(value);
}

DEFINE_DISPATCH(fill_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill_diagonal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& fill_diagonal_(Tensor& self, const Scalar& fill_value, bool wrap) {
  int64_t nDims = self.dim();
  TORCH_CHECK(nDims >= 2, "dimensions must larger than 1");

  int64_t height = self.size(0);
  int64_t width = self.size(1);

  if (nDims > 2) {
    int64_t dim1 = height;
    for (const auto i : c10::irange(1, nDims)) {
      if (self.size(i) != dim1) {
        AT_ERROR("all dimensions of input must be of equal length");
      }
    }
  }

  int64_t storage_offset = self.storage_offset();
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t size = std::min(height, width);

  int64_t stride = 0;
  for (const auto i : c10::irange(nDims)) {
    stride += self.stride(i);
  }
  strides.push_back(stride);
  sizes.push_back(size);

  auto main_diag = self.as_strided(sizes, strides, storage_offset);
  main_diag.fill_(fill_value);

  if (wrap && nDims == 2 && height > width + 1) {
    std::vector<int64_t> wrap_sizes;

    int64_t step = width + 1;
    int64_t wrap_size = ((self.numel() + step - 1) / step) - size;
    wrap_sizes.push_back(wrap_size);

    int64_t offset = self.stride(0) * (width + 1);

    auto wrap_diag = self.as_strided(wrap_sizes, strides, storage_offset + offset);
    wrap_diag.fill_(fill_value);
  }

  return self;
}

static Tensor& zero_cpu_(Tensor &self, int64_t nelements) {
  void* ptr = self.data_ptr();
  if (nullptr == ptr) {
    return self.fill_(0);
  }
  int64_t size_bytes = nelements * self.dtype().itemsize();
  if (size_bytes > 0) {
    std::memset(ptr, 0, size_bytes);
  }
  return self;
}

Tensor& zero_(Tensor &self) {
  int64_t nelements = c10::multiply_integers(self.sizes());
  if (self.device() == at::kCPU &&
      self.is_non_overlapping_and_dense() &&
      nelements < internal::GRAIN_SIZE) {
    return zero_cpu_(self, nelements);
  }
  return self.fill_(0);
}

Tensor& zero_meta_(Tensor& self) {
  return self;
}

} // namespace native
} // namespace at
