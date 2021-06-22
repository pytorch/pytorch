#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const Value& input, std::vector<lazy_tensors::int64> output_size,
    std::vector<lazy_tensors::int64> input_size, bool align_corners)
    : Node(ir::OpKind(at::aten::upsample_bilinear2d_backward), {input},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpsampleBilinearBackward::Clone(OpList operands) const {
  return MakeNode<UpsampleBilinearBackward>(operands.at(0), output_size_,
                                            input_size_, align_corners_);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << lazy_tensors::StrJoin(output_size_, ", ") << "), input_size=("
     << lazy_tensors::StrJoin(input_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
