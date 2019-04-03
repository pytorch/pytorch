#include <torch/csrc/jit/passes/verify_shape_analysis.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>

namespace torch {
namespace jit {

void checkDimTensor(
    const DimensionedTensorTypePtr& type,
    const at::Tensor& ten,
    const Node* n) {
  // guarding in if makes it easier to debug
  if (type->dim() != ten.dim()) {
    AT_ASSERTM(
        type->dim() == ten.dim(), "DIM_TENSOR_FAIL", *n->getSourceLocation());
  }
  if (type->scalarType() != ten.scalar_type()) {
    AT_ASSERTM(
        type->scalarType() == ten.scalar_type(),
        "DIM_TENSOR_FAIL",
        *n->getSourceLocation());
  }
  if (type->device() != ten.device()) {
    AT_ASSERTM(
        type->device() == ten.device(),
        "DIM_TENSOR_FAIL",
        *n->getSourceLocation());
  }
}

void checkCompleteTensor(
    const CompleteTensorTypePtr& type,
    const at::Tensor& ten,
    const Node* n) {
  // guarding in if makes it easier to debug
  if (type->dim() != ten.dim()) {
    AT_ASSERTM(
        type->dim() == ten.dim(), "COMPLETE_FAIL", n->getSourceLocation());
  }
  if (type->scalarType() != ten.scalar_type()) {
    AT_ASSERTM(
        type->scalarType() == ten.scalar_type(),
        "COMPLETE_FAIL",
        n->getSourceLocation());
  }
  if (type->device() != ten.device()) {
    AT_ASSERTM(
        type->device() == ten.device(),
        "COMPLETE_FAIL",
        n->getSourceLocation());
  }
  if (type->sizes() != ten.sizes()) {
    AT_ASSERTM(
        type->sizes() == ten.sizes(), "COMPLETE_FAIL", n->getSourceLocation());
  }
}

RegisterOperators reg_shape({
    // Mutate input so that the op isn't removed
    Operator(
        "aten::check_dim_tensor(Tensor(z!) a) -> ()",
        [](const Node* node) {
          auto dim_type =
              node->input()->type()->expect<DimensionedTensorType>();
          return [dim_type, node](Stack& stack) {
            auto tensor = pop(stack).toTensor();
            checkDimTensor(dim_type, tensor, node);
            return 0;
          };
        }),
    // Mutate input so that the op isn't removed
    Operator(
        "aten::check_complete_tensor(Tensor(z!) a) -> ()",
        [](const Node* node) {
          auto complete = node->input()->type()->expect<CompleteTensorType>();
          return [complete, node](Stack& stack) {
            auto tensor = pop(stack).toTensor();
            checkDimTensor(complete, tensor, node);
            return 0;
          };
        }),
});

// NOT FOR LANDING YET THUS COPY PASTA
void VerifyShapeAnalysis(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      VerifyShapeAnalysis(b);
    }
    {
      std::unordered_set<Value*> dims;
      std::unordered_set<Value*> complete;
      for (auto input : n->inputs()) {
        if (auto complete_ten = input->type()->cast<CompleteTensorType>()) {
          complete.insert(input);
        } else if (
            auto dim_ten = input->type()->cast<DimensionedTensorType>()) {
          dims.insert(input);
        }
      }
      WithInsertPoint guard(n);
      for (Value* v : dims) {
        n->owningGraph()->insert(
            Symbol::fromQualString("aten::check_dim_tensor"),
            {
                v,
            });
      }
      for (Value* v : complete) {
        n->owningGraph()->insert(
            Symbol::fromQualString("aten::check_complete_tensor"),
            {
                v,
            });
      }
    }
    {
      std::unordered_set<Value*> dims;
      std::unordered_set<Value*> complete;
      for (auto output : n->outputs()) {
        if (auto complete_ten = output->type()->cast<CompleteTensorType>()) {
          complete.insert(output);
        } else if (
            auto dim_ten = output->type()->cast<DimensionedTensorType>()) {
          dims.insert(output);
        }
      }
      for (Value* v : dims) {
        auto node = n->owningGraph()->create(
            Symbol::fromQualString("aten::check_dim_tensor"), {v}, 0);
        node->insertAfter(n);
      }
      for (Value* v : complete) {
        auto node = n->owningGraph()->create(
            Symbol::fromQualString("aten::check_complete_tensor"), {v}, 0);
        node->insertAfter(n);
      }
    }
  }
}

void VerifyShapeAnalysis(const std::shared_ptr<Graph>& graph) {
  const auto& inputs = graph->inputs();
  WithInsertPoint g(graph->block());
  {
    std::unordered_set<Value*> dims;
    std::unordered_set<Value*> complete;
    for (auto input : inputs) {
      if (auto complete_ten = input->type()->cast<CompleteTensorType>()) {
        complete.insert(input);
      } else if (auto dim_ten = input->type()->cast<DimensionedTensorType>()) {
        dims.insert(input);
      }
    }
    for (Value* v : dims) {
      graph->insert(
          Symbol::fromQualString("aten::check_dim_tensor"),
          {
              v,
          });
    }
    for (Value* v : complete) {
      graph->insert(
          Symbol::fromQualString("aten::check_complete_tensor"),
          {
              v,
          });
    }
  }

  VerifyShapeAnalysis(graph->block());
}

} // namespace jit
} // namespace torch
