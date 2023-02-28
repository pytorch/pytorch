#pragma once

#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <torch/csrc/Export.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TORCH_API TensorDesc {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::ScalarType scalar_type;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<bool> contiguity;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TensorDesc(const at::ScalarType& type, const std::vector<bool>& contiguity)
      : scalar_type{type}, contiguity{contiguity} {
    if (contiguity.empty()) {
      nDim_ = 0;
    } else {
      nDim_ = std::count(contiguity.begin(), contiguity.end(), false) +
          (lastIsContiguous() ? 1 : 0);
    }
  }

  // Delegating constructors
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TensorDesc(
      const at::ScalarType& type,
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides)
      : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}

  TensorDesc(const at::Tensor& t)
      : TensorDesc(t.scalar_type(), t.sizes(), t.strides()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  TensorDesc(const c10::TensorTypePtr& type)
      : TensorDesc(
            type->scalarType().value(),
            type->sizes().concrete_sizes().value(),
            type->strides().concrete_sizes().value()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }

  // True iff innermost stride is 1
  bool lastIsContiguous() const {
    return (contiguity.empty() || contiguity.back());
  }

  static std::vector<bool> findContiguous(
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides) {
    AT_ASSERT(sizes.size() == strides.size());
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<bool> cont(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      const auto expected_stride =
          (i + 1 < sizes.size()) ? sizes[i + 1] * strides[i + 1] : 1;
      cont[i] = (strides[i] == expected_stride);
    }
    return cont;
  }

  bool operator==(const TensorDesc& desc) const {
    return scalar_type == desc.scalar_type && contiguity == desc.contiguity;
  }

  bool operator!=(const TensorDesc& desc) const {
    return !(*this == desc);
  }

  static size_t hash(const TensorDesc& spec) {
    return c10::get_hash(
        spec.scalar_type,
        spec.nDim_,
        std::hash<std::vector<bool>>{}(spec.contiguity));
  }

 private:
  size_t nDim_;
};

inline std::ostream& operator<<(std::ostream& out, const TensorDesc& d) {
  out << d.scalar_type << "[";
  for (const auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

} // namespace fuser
} // namespace jit
} // namespace torch
