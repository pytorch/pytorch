#include "lazy_tensor_core/csrc/ops/max_pool_nd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

c10::Symbol MaxPoolNdSymbol(int64_t spatial_dim_count) {
  switch (spatial_dim_count) {
    case 1:
      return at::aten::max_pool1d;
    case 2:
      return at::aten::max_pool2d;
    case 3:
      return at::aten::max_pool3d;
    default:
      LOG(ERROR) << "Invalid number of spatial dimensions: "
                 << spatial_dim_count;
  }
}

}  // namespace

MaxPoolNd::MaxPoolNd(const torch::lazy::Value& input, int64_t spatial_dim_count,
                     std::vector<int64_t> kernel_size,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     bool ceil_mode)
    : TsNode(torch::lazy::OpKind(MaxPoolNdSymbol(spatial_dim_count)), {input},
             /*num_outputs=*/2,
             torch::lazy::MHash(spatial_dim_count, kernel_size, stride, padding,
                                ceil_mode)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MaxPoolNd::Clone(OpList operands) const {
  return torch::lazy::MakeNode<MaxPoolNd>(operands.at(0), spatial_dim_count_, kernel_size_,
                             stride_, padding_, ceil_mode_);
}

std::string MaxPoolNd::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << c10::Join(", ", kernel_size_) << "), stride=("
     << c10::Join(", ", stride_) << "), padding=(" << c10::Join(", ", padding_)
     << "), ceil_mode=" << ceil_mode_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
