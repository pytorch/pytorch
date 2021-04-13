#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

RreluWithNoiseBackward::RreluWithNoiseBackward(
    const Value& grad_output, const Value& input, const Value& noise,
    const at::Scalar& lower, const at::Scalar& upper, bool training)
    : Node(ir::OpKind(at::aten::rrelu_with_noise_backward),
           {grad_output, input, noise}, input.shape(),
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(ScalarHash(lower), ScalarHash(upper),
                                     training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

NodePtr RreluWithNoiseBackward::Clone(OpList operands) const {
  return MakeNode<RreluWithNoiseBackward>(operands.at(0), operands.at(1),
                                          operands.at(2), lower_, upper_,
                                          training_);
}

std::string RreluWithNoiseBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
