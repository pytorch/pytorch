#include "lazy_tensor_core/csrc/ops/dynamic_size.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DynamicSize2::DynamicSize2(torch::lazy::Value lhs)
    : TsNode(torch::lazy::OpKind(c10::Symbol::prim("_dynamic_size2")), lhs,
             {ir::GetShapeFromTsValue(lhs)}) {}


}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors

