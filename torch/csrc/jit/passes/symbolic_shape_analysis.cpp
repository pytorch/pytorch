#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/peephole_non_tensor.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/utils/memory.h>
#include <memory>
#include <unordered_map>
#include <vector>

/*
XXX: this is still in prototype phase and has much work left to do, including
but not limited to:
- Refactor APIs
- Only iteratively optimize shape function while a change has been made
- Add decent coverage of common ops
- Add shape analysis pass on Graph that handles Ifs and Loops
- Allow concurrent reads to the operator map
- Successive applications of same inputs to same shape function (e.g. series of
pointwise ops)
- Better support for Symbolic Shapes (additional optimizations, etc)
- Supporting returning partially evaluated shape compute graph
*/

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool symbolic_shape_analysis_test_mode = false;

namespace torch {
namespace jit {

bool setSymbolicShapeAnalysisTestMode(bool value) {
  bool old_value = symbolic_shape_analysis_test_mode;
  symbolic_shape_analysis_test_mode = value;
  return old_value;
}

bool symbolicShapeAnalysisTestModeEnabled() {
  return symbolic_shape_analysis_test_mode;
}

// TODO: better registration mechanism
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex lock;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
// length and unroll loops over its elements. We iteratively optimize and
// substitute in properties until we are unable to make any further
// optimizations. Finally, we try to extract Tensor properties from the output.
// For instance `return [1, 2, inp[2] + 1, inp[3]]` we know that the ouptut
// will be length 4 with first two dimensions equal to 1 and 2.
// It is not implemented yet but in the future we will also be able to
// infer that the 4th dimension will have the same symbolic shape as inp[3]

struct SymbolicShapeAnalyzer {
  SymbolicShapeAnalyzer(Node* n, std::shared_ptr<Graph> shape_compute_graph)
      : graph_(shape_compute_graph->copy()), node_(n) {
    for (size_t i = 0; i < node_->inputs().size(); i++) {
      auto type = node_->input(i)->type();
      if (auto tt = type->castRaw<TensorType>()) {
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        c10::SymbolicShape symbolic_shapes = tt->symbolic_sizes();

        // for testing, we don't insert complete tensor shapes and rely on our
        // partial evaluation pipeline to propagate information.
        // this is a good proxy for our ability to propagate non-complete shape
        // information.

        if (symbolic_shapes.isComplete() &&
            !symbolic_shape_analysis_test_mode) {
          replaceWithIValue(
              graph_->inputs().at(i), *tt->sizes().concrete_sizes());
          continue;
        }
        // we can't optimize a tensor without fixed rank
        if (symbolic_shapes.rank()) {
          node_input_tensor_indices.push_back(i);
        }
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
    size_t num_optimization_iters = 6;
    for (size_t i = 0; i < num_optimization_iters; i++) {
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
      RefineIntegerValues(graph_);
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
    // here we iteratively substitute properties of the node's input tensors
    // into the shape compute graph. in addition to direct constants we can
    // substitute, like len(inp) or inp[0] if the tensor has fixed length
    // or first dimension, we also try to resolve symbolic shapes of the same
    // symbolic value to the same Value * in the shape compute graph.
    // for the shape logic:
    // dim1 = inp1[0];
    // dim2 = inp2[0];
    // return dim1 if dim2 == 1 else dim2;
    // if we see that inp1[0] and inp2[0] both have the same symbolic shape
    // value, then it is a valid transformation to replace dim2 with dim1 or
    // vice versa. to do this we collect  all Value * for a particular symbolic
    // dimension value and then Value * with their dominator of the same
    // symbolic dimension value in the example above, this allows us to infer
    // that the output will be the symbolic dimension value of dim1
    // if `substitute_symbolic_dims` is true, then we insert list accesses
    // which resolve to symbolic dimension values as constants in the graph
    // because symbolic dimensions are represented as negative numbers and
    // are not real values, this is only safe to do if you are not running
    // any further optimizations. representing them as constants in the graph
    // makes extracting output shapes with symbolic dimensions possible.

    std::unordered_map<int64_t, std::vector<Value*>> symbolic_shape_map;

    for (auto tensor_index : node_input_tensor_indices) {
      auto tensor_value = node_->inputs().at(tensor_index);
      auto tensor_shape =
          tensor_value->type()->expect<TensorType>()->symbolic_sizes();
      TORCH_INTERNAL_ASSERT(tensor_shape.rank().has_value());

      for (const auto& use : graph_->inputs().at(tensor_index)->uses()) {
        // TODO: either decompose composite ops like slice or add handling here
        switch (use.user->kind()) {
          case aten::len: {
            size_t len = tensor_shape.rank().value();
            replaceWithIValue(use.user->output(), static_cast<int64_t>(len));
          } break;
          case aten::__getitem__: {
            auto index = constant_as<int64_t>(use.user->inputs().at(1));
            if (!index) {
              continue;
            }
            auto norm_index = normIndex(*index, *tensor_shape.rank());
            if (!norm_index) {
              continue;
            }
            if (tensor_shape[*norm_index].is_static() ||
                substitute_symbolic_dims) {
              replaceWithIValue(
                  use.user->output(), tensor_shape[*norm_index].value());
            } else {
              int64_t symbolic_index = tensor_shape[*norm_index].value();
              symbolic_shape_map[symbolic_index].push_back(use.user->output());
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
    // `symbolic_set` represents a set of Value * which are all equal
    // to each other. Here, we optimize the graph by replacing values
    // in the set with other dominating values.
    // in the following example, where a, b and c are all in the same
    // symbolic set:
    // if cond:
    //    a = li[0]
    //    b = li[1]
    //    return [a, b]
    // else:
    //    c = li[0]
    //    return [c, c]
    // we can replace `b` with `a` because it is dominated by `a`,
    // but we cannot replace `c` with another dominating value

    // there are ways to compute this more efficiently but typically number of
    // Values for each symbolic set is low and this is cheap to run
    for (size_t i = 0; i < symbolic_set.size(); ++i) {
      Value* v = symbolic_set[i];
      Value* dominating_value = v;
      // NOLINTNEXTLINE(modernize-loop-convert)
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

  // node input indices that are TensorType and we need to iteratively
  // substitute properties of. We only substitute properties
  // of TensorTypes with a fixed dimension but not a complete shape,
  // because a complete shape we can completely replace with a constant
  // and non-fixed dimensions we cannot reason about at all
  // TODO: might be cleaner to store as a pair of index -> symbolic shape
  // but there were weird lifetime issues
  std::vector<int64_t> node_input_tensor_indices;
  std::shared_ptr<Graph> graph_;
  Node* node_;
};

void PropagateShapesWithShapeFunction(
    Node* n,
    std::shared_ptr<Graph>& shape_compute_graph) {
  c10::SymbolicShape out = SymbolicShapeAnalyzer(n, shape_compute_graph).run();
  n->output()->setType(
      n->output()->type()->expect<TensorType>()->withSymbolicShapes(out));
}

void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph) {
  std::lock_guard<std::mutex> guard(lock);
  for (Node* n : graph->nodes()) {
    if (n->maybeSchema()) {
      if (auto maybe_graph = shapeComputeGraphForSchema(n->schema())) {
        PropagateShapesWithShapeFunction(n, *maybe_graph);
      }
    }
  }
}

} // namespace jit
} // namespace torch
