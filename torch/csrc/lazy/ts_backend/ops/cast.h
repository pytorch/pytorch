#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Cast : public TsNode {
 public:
  static const OpKind class_op_kind;

  Cast(
      const Value& input,
      at::ScalarType dtype,
      c10::optional<at::ScalarType> stype = c10::nullopt);

  std::string ToString() const override;

  at::ScalarType dtype() const {
    return dtype_;
  }

  const c10::optional<at::ScalarType>& stype() const {
    return stype_;
  }

 private:
  at::ScalarType dtype_;
  c10::optional<at::ScalarType> stype_;
};

} // namespace lazy
} // namespace torch
