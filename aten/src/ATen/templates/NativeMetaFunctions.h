#pragma once

// ${generated_comment}

#include <ATen/TensorMeta.h>
#include <ATen/TensorIterator.h>

namespace at {

namespace meta {

${declarations}

} // namespace meta

// From build/aten/src/ATen/NativeMetaFunctions.h
namespace meta {
  struct TORCH_API structured_add_Tensor : public TensorIteratorBase {
    void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
  };
} //namespace meta

} // namespace at
