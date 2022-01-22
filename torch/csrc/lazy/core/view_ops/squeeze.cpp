#include <torch/csrc/lazy/core/view_ops/squeeze.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

// This is almost like at::inferSqueezeGeometry, but that requires a Tensor input
// and also computes new strides.  This logic seems correct.
std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions;
  for (int64_t i = 0; i < dimensions.size(); ++i) {
    int64_t dim = dimensions[i];
    if (dim != 1 || (i != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

Squeeze::Squeeze(const torch::lazy::Value& input, int dim)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::squeeze), {input},
                          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() {
        auto input_shape = GetShapeFromTsValue(input);
        return torch::lazy::Shape(input_shape.scalar_type(),
          BuildSqueezedDimensions(input_shape.sizes(), dim));
      });
}

std::string Squeeze::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
