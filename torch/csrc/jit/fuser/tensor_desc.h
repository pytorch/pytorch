#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/assertions.h"

#include <vector>
#include <iostream>
#include <algorithm>

namespace torch { namespace jit { namespace fuser {

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TORCH_API TensorDesc {
  at::ScalarType scalar_type;
  std::vector<bool> contiguity;

  TensorDesc(
    const at::ScalarType& type
  , const std::vector<bool>& contiguity)
  : scalar_type{type}
  , contiguity{contiguity} {
    if (contiguity.size() == 0) {
      nDim_ = 0;
    } else {
      nDim_ = std::count(contiguity.begin(), contiguity.end(), false) + (lastIsContiguous() ? 1 : 0);
    }
  }

  // Delegating constructors
  TensorDesc(
    const at::ScalarType& type
  , const at::IntList& sizes
  , const at::IntList& strides)
  : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}

  TensorDesc(const at::Tensor& t)
  : TensorDesc(t.type().scalarType(), t.sizes(), t.strides()) {}

  TensorDesc(CompleteTensorTypePtr type)
  : TensorDesc(type->scalarType(), type->sizes(), type->strides()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const { return nDim_; }

  // True iff innermost stride is 1
  bool lastIsContiguous() const {
    return (contiguity.size() == 0 || contiguity.back());
  }

  static std::vector<bool> findContiguous(
    const at::IntList& sizes
  , const at::IntList& strides) {
    JIT_ASSERT(sizes.size() == strides.size());
    std::vector<bool> cont(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      const auto expected_stride = (i + 1 < sizes.size()) ? sizes[i+1]*strides[i+1] : 1;
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
    return torch::get_hash(spec.scalar_type, spec.nDim_, std::hash<std::vector<bool>>{}(spec.contiguity));
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

#endif // USE_CUDA_FUSER || USE_CPU_FUSER
