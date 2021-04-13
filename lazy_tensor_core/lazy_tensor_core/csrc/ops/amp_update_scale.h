#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AmpUpdateScale : public Node {
 public:
  AmpUpdateScale(const Value& growth_tracker, const Value& current_scale,
                 const Value& found_inf, double scale_growth_factor,
                 double scale_backoff_factor, int growth_interval);

  NodePtr Clone(OpList operands) const override;

 private:
  double scale_growth_factor_;
  double scale_backoff_factor_;
  int growth_interval_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
