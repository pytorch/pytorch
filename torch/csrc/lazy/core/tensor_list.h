#pragma once

#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

// Note: this OpKind is separate from ltc_ops.h since it would be a circular import otherwise, I like leaving TensorList
// in this file, and I think most of ltc_ops special cases will be deleted anyway
const OpKind tensor_list_opkind = OpKind::Get("lazy_tensors::tensor_list");

// TensorList represents an at::TensorList which is a vector[Tensor] but is also a first-class IValue
// and can be fed as a single input to a program. It is much easier to handle TensorLists in Lazy Tensor code
// if they are represented as a single Node so there can be more than one TensorList and more than one Tensor
// side-by-side as operands to an op.
//
// Note: shape is undefined for TensorList.  We assert in some places that #shapes matches #outputs and this stems from
//       the fact that currently all IR nodes represent tensors (there is no type system for this IR).  Becuase of this,
//       TensorList is a bit of a hack.
struct TORCH_API TensorList : public Node {
  TensorList() = delete;
  TensorList(OpList values);

  const Shape& shape(size_t output_index = 0) const override;
};

}  // namespace lazy
}  // namespace torch
