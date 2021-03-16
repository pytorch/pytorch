// Functions that fill Tensors with constants.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor& fill_out(Tensor& self, const Scalar& value) {
  if (self.is_quantized()) {
    at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
    out = out.to(self.device());
    // Trust the `copy_` to handle the quantization and the boundary chacks.
    self.copy_(out);
    return self;
  }
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

Tensor& fill_(Tensor& self, const Scalar& value) {
  return fill_out(self, value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill_out(self, value.item());
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
    for (int64_t i = 1; i < nDims; i++) {
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
  for (int64_t i = 0; i < nDims; i++) {
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

Tensor& zero_cpu_(Tensor &self, int64_t nelements) {
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

} // namespace native
} // namespace at
