#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Cast : public Node {
 public:
  Cast(const Value& input, lazy_tensors::PrimitiveType type);
  Cast(const Value& input, at::ScalarType dtype,
       c10::optional<at::ScalarType> stype = c10::nullopt);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::PrimitiveType type() const { return type_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; };

  const c10::optional<at::ScalarType>& stype() const { return stype_; };

 private:
  lazy_tensors::PrimitiveType type_;
  c10::optional<at::ScalarType> dtype_;
  c10::optional<at::ScalarType> stype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
