#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/ts_backend/view_ops/permute.h>

#include <torch/csrc/lazy/core/helpers.h>

namespace torch {
namespace lazy {

const OpKind Permute::class_op_kind(at::aten::permute);

Permute::Permute(const Value& input, std::vector<int64_t> dims)
    : TsNode(
          OpKind(at::aten::permute),
          {input},
          /*num_outputs=*/1,
          MHash(dims)),
      dims_(std::move(dims)) {
  addComputedShape([&]() {
    return MakePermuteShape(operand(0).shape(), dims_);
  });
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dims=(" << c10::Join(", ", dims_) << ")";
  return ss.str();
}

} // namespace lazy
} // namespace torch
