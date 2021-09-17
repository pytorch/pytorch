#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Var : public Node {
 public:
  Var(const Value& input, std::vector<lazy_tensors::int64> dimensions,
      lazy_tensors::int64 correction, bool keep_reduced_dimensions);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& dimensions() const {
    return dimensions_;
  }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  lazy_tensors::int64 correction() const { return correction_; }

 private:
  std::vector<lazy_tensors::int64> dimensions_;
  lazy_tensors::int64 correction_;
  bool keep_reduced_dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
