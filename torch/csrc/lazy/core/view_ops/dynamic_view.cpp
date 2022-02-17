#include <torch/csrc/lazy/core/view_ops/dynamic_view.h>

#include <ATen/InferSize.h>
#include "ATen/core/functional.h"
#include "c10/util/ArrayRef.h"
#include "lazy/core/ir.h"

namespace torch {
namespace lazy {

namespace {

std::vector<Value> chain(c10::ArrayRef<Value> a, c10::ArrayRef<Value> b) {
  std::vector<Value> result;
  result.reserve(a.size() + b.size());
  result.insert(result.end(), a.begin(), a.end());
  result.insert(result.end(), b.begin(), b.end());
  return result;
}

} // namespace





DynamicView::DynamicView(Value input, OpList dims)    
    : TsNode(
          OpKind(at::aten::view),
          chain({input}, dims),
          //NodeOutputShape(input, output_size)
          std::vector<Shape>{Shape(GetShapeFromTsValue(input).scalar_type(), {1})}, // (const Value& input, OpList dims)
          /*num_outputs=*/1)
      {

      }

  TSOpVector DynamicView::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {

      auto graph = function->graph();
      auto pos_args = c10::fmap(operands(), [&loctx](const Output& o) { 
        return loctx->GetOutputOp(o);
      });

      auto input = pos_args.at(0);
      
      // we will slice off the first element which is an input tensor
      c10::ArrayRef<torch::jit::Value*> input_and_dims (pos_args);

        
      auto dim_vals = graph
            ->insertNode(graph->createList(c10::IntType::get(), input_and_dims.slice(1)))
            ->output();

      auto view_out = graph->insert(op().op, {input_and_dims.at(0), dim_vals});
      return {view_out};
    }

} // namespace lazy
} // namespace torch
