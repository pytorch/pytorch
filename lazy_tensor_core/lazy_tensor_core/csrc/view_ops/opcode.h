#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// TODO: Allow other backend to define its own BaseNode, or
// promote TsNode to Node.
using BaseNode = torch::lazy::TsNode;

const static torch::lazy::OpKind as_strided_reverse =
    torch::lazy::OpKind::Get("lazy::as_strided_reverse");
const static torch::lazy::OpKind diagonal_reverse =
    torch::lazy::OpKind::Get("lazy::diagonal_reverse");
const static torch::lazy::OpKind generic_slice =
    torch::lazy::OpKind::Get("lazy::generic_slice");
const static torch::lazy::OpKind generic_slice_reverse =
    torch::lazy::OpKind::Get("lazy::generic_slice_reverse");
const static torch::lazy::OpKind select =
    torch::lazy::OpKind::Get("lazy::select");
const static torch::lazy::OpKind select_reverse =
    torch::lazy::OpKind::Get("lazy::select_reverse");

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
