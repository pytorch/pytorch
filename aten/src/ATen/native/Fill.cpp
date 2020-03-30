// Functions that fill Tensors with constants.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace {
  template <typename scalar_t>
  inline void fill_fast(Tensor& self, Scalar value_scalar) {
    auto value = value_scalar.to<scalar_t>();
    scalar_t * dptr = static_cast<scalar_t *>(self.data_ptr());
    *dptr = value;
  }
} // namspace

Tensor& fill_out(Tensor& self, Scalar value) {
  // When filling a number to 1-element CPU tensor, we want to skip
  // everything but manipulate data ptr directly.
  // Ideally this fast pass should be implemented in TensorIterator,
  // but we also want to skip compute_types which in not avoidable
  // in TensorIterator for now.
  if (self.device() == at::kCPU && self.numel() == 1 && !self.is_complex() && !value.isComplex()) {
     AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "fill_out", [&]() {
        fill_fast<scalar_t>(self, value);});
     return self;
  }
  auto iter = TensorIterator::nullary_op(self);
  fill_stub(iter.device_type(), iter, value);
  return self;
}

Tensor& fill_(Tensor& self, Scalar value) {
  return fill_out(self, value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill_out(self, value.item());
}

DEFINE_DISPATCH(fill_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill_diagonal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& fill_diagonal_(Tensor& self, Scalar fill_value, bool wrap) {
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

Tensor& zero_(Tensor &self) {
  return self.fill_(0);
}

} // namespace native
} // namespace at
