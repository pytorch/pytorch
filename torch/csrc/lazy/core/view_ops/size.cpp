#include <torch/csrc/lazy/core/view_ops/size.h>

#include <ATen/InferSize.h>
#include "ATen/core/functional.h"
#include "ATen/core/interned_strings.h"
#include "lazy/core/ir.h"

namespace torch {
namespace lazy {

Size::Size(Value input, int64_t dim)
    : TsNode(
          OpKind(at::aten::size),
          {input},
          //NodeOutputShape(input, output_size)
          std::vector<Shape>{Shape{}}, // (const Value& input, OpList dims)
          /*num_outputs=*/1),
      dim_(dim)
      {
        // TODO: shape functions?
      }

  std::string Size::ToString() const {
    std::stringstream ss;
    ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
    return ss.str();
  }

  TSOpVector Size::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {

      auto graph = function->graph();
      auto dim_val = graph->insertConstant(dim_);
      auto inp_val = loctx->GetOutputOp(operand(0));
      auto size_out = graph->insert(op().op, {inp_val, dim_val}, {});
      return {size_out};
    }

} // namespace lazy
} // namespace torch
