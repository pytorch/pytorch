#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/peephole_non_tensor.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/utils/memory.h>
#include <exception>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

// TODO: better registration mechanism
std::mutex lock;
std::unordered_map<std::string, std::shared_ptr<Graph>> operator_functions;

c10::optional<size_t> normIndex(int64_t index, size_t len) {
  if (index < 0) {
    index = index + len;
  }
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    return c10::nullopt;
  }
}

void replaceWithIValue(Value* v, IValue val) {
  WithInsertPoint guard(*v->node()->owningBlock()->nodes().begin());
  v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
}

// Symbolic Shape Analysis works through iteratively partially evaluating
// a TorchScript shape compute graph by inputting properties from input
// Tensors. We can substitute in properties like `len(x)` and `x[1]`
// if they are statically on the input Tensors. We can also use
// assertions like `assert len(x) == 4` in order to refine the input
// length and unroll loops over its elements. We iterativley optimize and
// substitute in properties until we are unable to make any further
// optimizations. Finally, we try to extract Tensor properties from the output.
// For instance `return [1, 2, inp[2] + 1, inp[3]]` we know that the ouptut
// will be length 4 with first two dimensions equal to 1 and 2.
// It is not implemented yet but in the future we will also be able to
// infer that the 4th dimension will have the same symbolic shape as inp[3]

struct SymbolicShapeAnalyzer {
  SymbolicShapeAnalyzer(std::shared_ptr<Graph> graph, Node* n)
      : graph_(graph->copy()), node_(n) {
    for (size_t i = 0; i < node_->inputs().size(); i++) {
      auto type = node_->input(i)->type();
      if (auto tt = type->castRaw<TensorType>()) {
        c10::SymbolicShape symbolic_shapes = tt->symbolic_sizes();
        if (symbolic_shapes.isComplete()) {
          replaceWithIValue(
              graph_->inputs().at(i), *tt->sizes().concrete_sizes());
          continue;
        }
        tensor_indices.push_back(i);
      } else if (
          type->cast<ListType>() &&
          type->cast<ListType>()->getElementType()->cast<TensorType>()) {
        TORCH_INTERNAL_ASSERT(false); // not handled yet
      } else {
        if (auto ival = toIValue(node_->input(i))) {
          replaceWithIValue(graph_->inputs().at(i), *ival);
        }
      }
    }
  }

  c10::SymbolicShape run() {
    // TODO: only run while the last iteration has made a change
    for (size_t i = 0; i < 6; i++) {
      // XXX: we cannot substitute symbolic dims before passes like constant
      // propagation, or we might inadvertently use them in arithmetic or
      // other operators
      substituteInputTensorProperties(/*substitute_symbolic_dims*/ false);
      LowerSimpleTuples(graph_);
      RemoveListMutation(graph_);
      UnrollConstantLoops(graph_);
      ConstantPropagation(graph_);
      PeepholeOptimizeNonTensor(graph_);
      PeepholeOptimizeListIdioms(graph_, /*refine_list_len*/ true);
      ConstantPropagation(graph_);
      EliminateCommonSubexpression(graph_);
    }
    substituteInputTensorProperties(/*substitute_symbolic_dims*/ true);
    // XXX: do not run any passes after we have substituted in symbolic
    // dimension value, we do it so they can be easily extracted into the output
    // shape
    return extractOutputShape();
  }

 private:
  void substituteInputTensorProperties(bool substitute_symbolic_dims) {
    std::unordered_map<int64_t, std::vector<Value*>> symbolic_shape_map;

    for (auto tensor_index : tensor_indices) {
      auto tensor_value = node_->inputs().at(tensor_index);
      auto shape = tensor_value->type()->expect<TensorType>()->symbolic_sizes();
      if (!shape.rank().has_value()) {
        return;
      }

      for (const auto& use : graph_->inputs().at(tensor_index)->uses()) {
        // TODO: either decompose composite ops like slice or add handling here
        switch (use.user->kind()) {
          case aten::len: {
            size_t len = shape.rank().value();
            replaceWithIValue(use.user->output(), static_cast<int64_t>(len));
          } break;
          case aten::__getitem__: {
            auto index = constant_as<int64_t>(use.user->inputs().at(1));
            if (index) {
              auto norm_index = normIndex(*index, *shape.rank());
              if (norm_index &&
                  (shape[*norm_index].is_static() ||
                   substitute_symbolic_dims)) {
                replaceWithIValue(
                    use.user->output(), shape[*norm_index].value());
              } else if (norm_index) {
                int64_t symbolic_index = shape[*norm_index].value();
                if (!symbolic_shape_map.count(symbolic_index)) {
                  symbolic_shape_map[symbolic_index] = {};
                }
                symbolic_shape_map[symbolic_index].push_back(
                    use.user->output());
              }
            }
          }
        }
      }

      for (const auto& symbolic_set : symbolic_shape_map) {
        mergeSymbolicShapeSets(symbolic_set.second);
      }
    }
  }

  void mergeSymbolicShapeSets(const std::vector<Value*>& symbolic_set) {
    // resolve all symbolic values to values which they are dominated by
    // there are ways to compute this more efficiently but typically number of
    // Values for each symbolic set is low and this is cheap to run
    for (size_t i = 0; i < symbolic_set.size(); ++i) {
      Value* v = symbolic_set[i];
      Value* dominating_value = v;
      for (size_t j = 0; j < symbolic_set.size(); ++j) {
        if (dominating_value->node()->isDominatedBy(symbolic_set[j]->node())) {
          dominating_value = symbolic_set[j];
        }
      }
      if (dominating_value != v) {
        v->replaceAllUsesWith(dominating_value);
      }
    }
  }

  c10::SymbolicShape extractOutputShape() {
    TORCH_INTERNAL_ASSERT(graph_->outputs().size() == 1);
    auto output = graph_->outputs().at(0);
    TORCH_INTERNAL_ASSERT(
        output->type()->cast<ListType>() &&
        output->type()->cast<ListType>()->getElementType()->cast<IntType>());
    if (output->node()->kind() == prim::Constant) {
      auto int_list = toIValue(output)->toIntVector();
      return c10::SymbolicShape(int_list);
    }
    // If it is not a single list construct or constant, bail,
    // otherwise we cannot analyze its output and it might be modified
    if (output->node()->kind() != prim::ListConstruct ||
        output->uses().size() != 1) {
      return c10::SymbolicShape();
    }
    Node* list_construct = output->node();
    std::vector<c10::optional<int64_t>> output_shape;
    for (Value* input : list_construct->inputs()) {
      output_shape.push_back(constant_as<int64_t>(input));
    }
    return c10::SymbolicShape(output_shape);
  }

  std::vector<int64_t> tensor_indices;
  std::shared_ptr<Graph> graph_;
  Node* node_;
};

void PropagateShapesWithShapeFunction(
    Node* n,
    const std::shared_ptr<Graph>& graph) {
  c10::SymbolicShape out = SymbolicShapeAnalyzer(graph, n).run();
  n->output()->setType(
      n->output()->type()->expect<TensorType>()->withSymbolicShapes(out));
}

void RegisterOperatorShapeFunction(
    Node* n,
    const std::shared_ptr<Graph>& graph) {
  std::lock_guard<std::mutex> guard(lock);
  if (!n->maybeSchema()) {
    return;
  }
  if (operator_functions.count(toString(n->schema()))) {
    return;
  }
  operator_functions[toString(n->schema())] = graph;
}

void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph) {
  std::lock_guard<std::mutex> guard(lock);
  for (Node* n : graph->nodes()) {
    if (n->maybeSchema()) {
      if (operator_functions.count(toString(n->schema()))) {
        PropagateShapesWithShapeFunction(
            n, operator_functions[toString(n->schema())]);
      }
    }
  }
}

} // namespace jit
} // namespace torch
