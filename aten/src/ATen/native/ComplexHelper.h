#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>

#include <utility>
#endif

// WARNING: this header contains non-inline functions and should be only
// included from ONE cpp file

namespace at::native {

// View tensor with new dtype, storage offset, sizes and strides
inline Tensor view_tensor(
    const Tensor &tensor, ScalarType dtype,
    c10::SymInt offset, SymIntArrayRef sizes, SymIntArrayRef strides) {
  Storage storage = tensor.storage();
  auto key_set = tensor.key_set().remove(DispatchKey::Conjugate);
  auto new_tensor = detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage), key_set, scalarTypeToTypeMeta(dtype));
  auto * impl = new_tensor.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides, offset);
  return new_tensor;
}

inline SymDimVector computeStrideForViewAsReal(SymIntArrayRef oldstride) {
  SymDimVector res(oldstride.size() + 1);
  for (const auto i : c10::irange(oldstride.size())) {
    res[i] = oldstride[i] * 2;
  }
  res.back() = 1;
  return res;
}

inline Tensor _view_as_real_physical(const Tensor& self) {
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  auto old_sizes = self.sym_sizes();
  SymDimVector new_sizes(old_sizes.size() + 1);
  std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.back() = 2;
  auto new_strides = computeStrideForViewAsReal(self.sym_strides());
  auto new_storage_offset = self.sym_storage_offset() * 2;
  const auto float_type = c10::toRealValueType(self.scalar_type());
  auto real_tensor = view_tensor(self, float_type, std::move(new_storage_offset), new_sizes, new_strides);
  return real_tensor;
}

// expects as input a complex tensor and returns back a tensor
// with corresponding real dtype containing the complex values
// in the last two dimensions
Tensor view_as_real(const Tensor& self) {
  TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  To resolve the conjugate tensor so you can view it as real, use self.resolve_conj(); however, be warned that the resulting tensor will NOT alias the original.");
  return _view_as_real_physical(self);
}

inline SymDimVector computeStrideForViewAsComplex(
    SymIntArrayRef oldstride,
    SymIntArrayRef oldsizes) {
  const auto dim = oldstride.size();
  TORCH_CHECK(dim > 0 && oldstride[dim - 1] == 1, "Tensor must have a last dimension with stride 1");

  SymDimVector res(dim - 1);
  for (const auto i : c10::irange(res.size())) {
    // Skip divisibility check for singleton dimensions
    if (oldsizes[i] != 1) {
      TORCH_CHECK(oldstride[i] % 2 == 0, "Tensor must have a stride divisible by 2 for all but last dimension");
    }
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

  auto old_sizes = self.sym_sizes();
  TORCH_CHECK(!old_sizes.empty(), "Input tensor must have one or more dimensions");
  TORCH_CHECK(old_sizes[old_sizes.size()-1] == 2, "Tensor must have a last dimension of size 2");
  SymDimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);

  const auto new_strides = computeStrideForViewAsComplex(self.sym_strides(), self.sym_sizes());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_CHECK(self.sym_storage_offset() % 2 == 0, "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.sym_storage_offset() / 2;

  return view_tensor(self, complex_type, new_storage_offset, new_sizes, new_strides);
}

} // namespace at::native
