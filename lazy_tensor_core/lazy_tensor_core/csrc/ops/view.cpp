#include "lazy_tensor_core/csrc/ops/view.h"

#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/shape.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const torch::lazy::Value& input,
                                    c10::ArrayRef<int64_t> output_sizes) {
  const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsValue(input);
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return lazy_tensors::ShapeUtil::MakeShape(input_shape.at_element_type(),
                                            complete_output_sizes);
}

}  // namespace

View::View(const torch::lazy::Value& input, std::vector<int64_t> output_size)
    : TsNode(torch::lazy::OpKind(at::aten::view), {input},
             NodeOutputShape(input, output_size),
             /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

std::string View::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
