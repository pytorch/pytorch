#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AmpForachNonFiniteCheckAndUnscale : public Node {
 public:
  // found_inf is set if infinite gradients are found during unscale. More
  // details here:
  // https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
  AmpForachNonFiniteCheckAndUnscale(const OpList& inputs,
                                    const Value& found_inf,
                                    const Value& inv_scale);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
