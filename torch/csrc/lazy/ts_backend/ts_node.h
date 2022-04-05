#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;

class TORCH_API TsNode : public lazy::Node {
 public:
  using Node::Node;

  ~TsNode() override = default;

  const std::string& getPythonStacktrace() const { return python_stacktrace_; }

  // Lower is a backend-specific method since it returns a backend specific
  // type. hence, it is convenient to define it differently per-backend rather
  // than at Node API
  virtual TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                           TSLoweringContext* loctx) const;

 private:
  std::string python_stacktrace_;
};

// Note: this OpKind is separate from ltc_ops.h since it would be a circular import otherwise, I like leaving TensorList
// in this file, and I think most of ltc_ops special cases will be deleted anyway
const OpKind tensor_list_opkind = OpKind::Get("lazy_tensors::tensor_list");

// TensorList represents an at::TensorList which is a vector[Tensor] but is also a first-class IValue
// and can be fed as a single input to a TS program.  It is much easier to handle TensorLists in Lazy Tensor code
// if they are represented as a single Node so there can be more than one TensorList and more than one Tensor
// side-by-side as operands to an op.
//
// Note: shape is undefined for TensorList.  We assert in some places that #shapes matches #outputs and this stems from
//       the fact that currently all IR nodes represent tensors (there is no type system for this IR).  Becuase of this,
//       TensorList is a bit of a hack.
//
// TODO(whc) once Shape() API is moved to Node base, also make it virtual, and then implement it as NotImplemented for
// TensorList, also fixing the assertion that would fail.
struct TORCH_API TensorList : public TsNode {
  TensorList() = delete;
  TensorList(OpList values);

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   TSLoweringContext* loctx) const override;
};

}  // namespace lazy
}  // namespace torch
