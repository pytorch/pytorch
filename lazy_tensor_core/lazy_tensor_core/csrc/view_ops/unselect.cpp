#include "lazy_tensor_core/csrc/view_ops/unselect.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/view_ops/select.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Unselect::Unselect(const torch::lazy::Value& target,
                   const torch::lazy::Value& source, int64_t dim, int64_t start,
                   int64_t end, int64_t stride)
    : torch::lazy::TsNode(ltc_unselect, {target, source},
                          {torch::lazy::GetShapeFromTsValue(target)},
                          /*num_outputs=*/1,
                          torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

std::string Unselect::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_
     << ", start=" << start_ << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
