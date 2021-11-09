#include "lazy_tensor_core/csrc/ops/dynamic_expand.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DynamicExpand2::DynamicExpand2(torch::lazy::Value lhs, torch::lazy::Value sz)
    : TsNode(torch::lazy::OpKind(c10::Symbol::prim("_dynamic_expand2")), {lhs, sz},
             {ir::GetShapeFromTsValue(sz)}) {}


}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors

