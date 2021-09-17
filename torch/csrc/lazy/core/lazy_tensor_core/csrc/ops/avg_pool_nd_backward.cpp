#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

c10::Symbol AvgNdBackwardSymbol(lazy_tensors::int64 spatial_dim_count) {
  switch (spatial_dim_count) {
    case 2:
      return at::aten::avg_pool2d_backward;
    case 3:
      return at::aten::avg_pool3d_backward;
    default:
      LTC_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

AvgPoolNdBackward::AvgPoolNdBackward(
    const Value& grad_output, const Value& input,
    lazy_tensors::int64 spatial_dim_count,
    std::vector<lazy_tensors::int64> kernel_size,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding, bool ceil_mode,
    bool count_include_pad)
    : Node(OpKind(AvgNdBackwardSymbol(spatial_dim_count)), {grad_output, input},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(spatial_dim_count, kernel_size, stride,
                                     padding, ceil_mode, count_include_pad)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode),
      count_include_pad_(count_include_pad) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AvgPoolNdBackward::Clone(OpList operands) const {
  return MakeNode<AvgPoolNdBackward>(operands.at(0), operands.at(1),
                                     spatial_dim_count_, kernel_size_, stride_,
                                     padding_, ceil_mode_, count_include_pad_);
}

std::string AvgPoolNdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << lazy_tensors::StrJoin(kernel_size_, ", ")
     << "), stride=(" << lazy_tensors::StrJoin(stride_, ", ") << "), padding=("
     << lazy_tensors::StrJoin(padding_, ", ")
     << "), count_include_pad=" << count_include_pad_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
