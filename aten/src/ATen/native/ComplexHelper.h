#pragma once

#include <ATen/ATen.h>
#include <c10/util/irange.h>

// WARNING: this header contains non-inline functions and should be only
// included from ONE cpp file

namespace at { namespace native {

// View tensor with new dtype, storage offset, sizes and strides
inline Tensor view_tensor(
    const Tensor &tensor, ScalarType dtype,
    int64_t offset, IntArrayRef sizes, IntArrayRef strides) {
  Storage storage = tensor.storage();
  auto key_set = tensor.key_set().remove(DispatchKey::Conjugate);
  auto new_tensor = detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage), key_set, scalarTypeToTypeMeta(dtype));
  auto * impl = new_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(offset);
  impl->set_sizes_and_strides(sizes, strides);
  return new_tensor;
}

inline DimVector computeStrideForViewAsReal(IntArrayRef oldstride) {
  DimVector res(oldstride.size() + 1);
  for(size_t i = 0; i < oldstride.size(); i++) {
    res[i] = oldstride[i] * 2;
  }
  res.back() = 1;
  return res;
}

Tensor _view_as_real_physical(const Tensor& self) {
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  auto old_sizes = self.sizes();
  DimVector new_sizes(old_sizes.size() + 1);
  std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.back() = 2;
  auto new_strides = computeStrideForViewAsReal(self.strides());
  auto new_storage_offset = 2 * self.storage_offset();
  const auto float_type = c10::toValueType(self.scalar_type());
  auto real_tensor = view_tensor(self, float_type, new_storage_offset, new_sizes, new_strides);
  return real_tensor;
}

// expects as input a complex tensor and returns back a tensor
// with corresponding real dtype containing the complex values
// in the last two dimensions
Tensor view_as_real(const Tensor& self) {
  TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  To resolve the conjugate tensor so you can view it as real, use self.resolve_conj(); however, be warned that the resulting tensor will NOT alias the original.");
  return _view_as_real_physical(self);
}

inline DimVector computeStrideForViewAsComplex(IntArrayRef oldstride) {
  const int64_t dim = oldstride.size();
  TORCH_CHECK(oldstride[dim-1] == 1, "Tensor must have a last dimension with stride 1");

  DimVector res(dim - 1);
  for (const auto i : c10::irange(res.size())) {
    TORCH_CHECK(oldstride[i] % 2 == 0, "Tensor must have a stride divisible by 2 for all but last dimension");
    res[i] = oldstride[i] / 2;
  }
  return res;
}

// expects as input a float or double tensor with last dimension of size 2
// and returns back a tensor with corresponding complex dtype
Tensor view_as_complex(const Tensor& self) {
  TORCH_CHECK(
    self.scalar_type() == kFloat || self.scalar_type() == kDouble || self.scalar_type() == kHalf,
    "view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: ", self.scalar_type());

  auto old_sizes = self.sizes();
  TORCH_CHECK(old_sizes.size() != 0, "Input tensor must have one or more dimensions");
  TORCH_CHECK(old_sizes[old_sizes.size()-1] == 2, "Tensor must have a last dimension of size 2");
  DimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);

  const auto new_strides = computeStrideForViewAsComplex(self.strides());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_CHECK(self.storage_offset() % 2 == 0, "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.storage_offset() / 2;

  return view_tensor(self, complex_type, new_storage_offset, new_sizes, new_strides);
}

}} // namespace at::native
