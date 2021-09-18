#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
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
#include <torch/csrc/jit/passes/shape_analysis.h>
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
- Add decent coverage of common ops
- Add shape analysis pass on Graph that handles Ifs and Loops
- Allow concurrent reads to the operator map
- Successive applications of same inputs to same shape function (e.g. series of
pointwise ops)
- Supporting returning partially evaluated shape compute graph
*/

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
// a TorchScript shape compute graph by inputing properties from input
// Tensors. We can substitute in properties like `len(x)` and `x[1]`
// if they are statically on the input Tensors. We can also use
// assertions like `assert len(x) == 4` in order to refine the input
// length and unroll loops over its elements. We iteratively optimize and
// substitute in properties until we are unable to make any further
// optimizations. Finally, we try to extract Tensor properties from the output.
// For instance `return [1, 2, inp[2] + 1, inp[3]]` we know that the ouptut
// will be length 4 with first two dimensions equal to 1 and 2. We can also
// deduce that the 4th dimension has the same symbolic shape as inp[3], which
// means that we do know its concrete value statically but we can asssign sets
// of tensor dimensions which must be equal at runtime.

struct SymbolicShapeAnalyzer {
  SymbolicShapeAnalyzer(
      Node* n,
      std::shared_ptr<Graph> shape_compute_graph,
      const AliasDb& db)
      : graph_(shape_compute_graph->copy()), node_(n) {
    for (size_t i = 0; i < node_->inputs().size(); i++) {
      auto type = node_->input(i)->type();

      if (auto opt_type =
              graph_->inputs().at(i)->type()->cast<OptionalType>()) {
        // None will get handled with constant substitution later
        if (!type->cast<OptionalType>() &&
            !NoneType::get()->isSubtypeOf(type)) {
          graph_->inputs().at(i)->setType(opt_type->getElementType());
        }
      } else if (graph_->inputs().at(i)->type()->cast<NumberType>()) {
        graph_->inputs().at(i)->setType(type);
      }

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
        // TODO: remove, all constant tensors should have typed sizes
        if (toIValue(node_->input(i)) && !symbolic_shape_analysis_test_mode) {
          replaceWithIValue(
              graph_->inputs().at(i),
              constant_as<at::Tensor>(node_->input(i))->sizes());
          continue;
        }

        // we can't optimize a tensor without fixed rank
        if (symbolic_shapes.rank()) {
          node_symbolic_input_indices.emplace_back(i, symbolic_shapes);
        }
      } else if (
          type->cast<ListType>() &&
          type->cast<ListType>()->getElementType()->cast<TensorType>()) {
        TORCH_INTERNAL_ASSERT(false); // not handled yet
      } else if (auto ival = toIValue(node_->input(i))) {
        replaceWithIValue(graph_->inputs().at(i), *ival);
      } else if (
          type->cast<ListType>() &&
          type->cast<ListType>()->getElementType()->cast<IntType>()) {
        if (node_->input(i)->node()->kind() == prim::ListConstruct &&
            !db.hasWriters(node_->input(i))) {
          // it is a very common in graphs to see patterns like:
          // z = x.view(y.size())
          // or:
          // z = x.view(1, 10, y.size(0), y.size(1))
          // We want to propagate symbolic dimensions and concrete sizes
          // from y to z. To do this we try to associate symbolic dimensions
          // or concrete sizes with the integer list inputs that have a
          // constructor taken from constants or y.size() or y.size(0)
          auto list_construct = node_->input(i)->node();
          std::vector<c10::optional<int64_t>> shape;
          for (Value* v : list_construct->inputs()) {
            if (auto constant = constant_as<int64_t>(v)) {
              shape.emplace_back(*constant);
            } else if (
                v->node()->kind() == aten::size &&
                v->node()->input(1)->node()->kind() == prim::Constant) {
              auto tt = v->node()->input(0)->type()->expect<TensorType>();
              auto ss = tt->symbolic_sizes();
              if (!ss.rank()) {
                shape.emplace_back(c10::nullopt);
                continue;
              }
              auto norm_index = normIndex(
                  *constant_as<int64_t>(v->node()->input(1)), *ss.rank());
              if (!norm_index) {
                shape.emplace_back(c10::nullopt);
                continue;
              }
              shape.emplace_back(ss[*norm_index].value());
            } else {
              shape.emplace_back(c10::nullopt);
            }
          }
          node_symbolic_input_indices.emplace_back(
              i, c10::SymbolicShape(shape));
        } else if (
            node_->input(i)->node()->kind() == aten::size &&
            node_->input(i)->uses().size() == 1) {
          auto ten_inp = node_->input(i)->node()->input();
          auto ss = ten_inp->type()->expect<TensorType>()->symbolic_sizes();
          node_symbolic_input_indices.emplace_back(i, ss);
        }
      }
    }
  }

  c10::SymbolicShape run() {
    bool made_change = true;
    constexpr size_t MAX_ATTEMPTS = 8;
    size_t curr_attempt = 0;
    while (made_change && curr_attempt < MAX_ATTEMPTS) {
      curr_attempt++;
      made_change = false;
      // symbolic shape concrete values are only used in final shape extraction
      substituteInputTensorProperties(/*symbolic_shape_values*/ nullptr);
      // TODO: lower simple tuples ?
      made_change |= RemoveListMutation(graph_);
      made_change |= UnrollConstantLoops(graph_);
      made_change |= ConstantPropagation(graph_);
      made_change |= PeepholeOptimizeNonTensor(graph_);
      made_change |=
          PeepholeOptimizeListIdioms(graph_, /*refine_list_len*/ true);
      made_change |= RefineIntegerValues(graph_);
      made_change |= ConstantPropagation(graph_);
      made_change |= EliminateCommonSubexpression(graph_);
      EliminateDeadCode(graph_);
    }
    std::unordered_map<Value*, int64_t> symbolic_shape_values;
    substituteInputTensorProperties(&symbolic_shape_values);
    GRAPH_DUMP("Done with partial evaluation", graph_);

    return extractOutputShape(symbolic_shape_values);
  }

 private:
  void substituteInputTensorProperties(
      std::unordered_map<Value*, int64_t>* symbolic_shape_values) {
    // clang-format off
    // here we iteratively substitute properties of the node's input tensors
    // into the shape compute graph. we can substitute constants into the
    // like len(inp) or inp[0] if the tensor has a fixed length or a fixed
    // first dimension. we also try to resolve symbolic shapes of the same
    // symbolic value to the same Value * in the shape compute graph.
    // for the shape logic:
    // dim1 = inp1[0]
    // dim2 = inp2[0]
    // return dim1 if dim2 == 1 else dim2
    // if we see that inp1[0] and inp2[0] both have the same symbolic shape
    // value, then it is a valid transformation to replace dim2 with dim1 or
    // vice versa. to do this we collect all Value * for a particular symbolic
    // shape. Then, we replace all Value * within that set with their dominator.
    // In the example above, this allows us to infer  that the output will be the
    // symbolic dimension value of dim1.

    // if `symbolic_shape_values` is not null, record list accesses
    // which resolve to symbolic dimension values with their concrete symbolic
    // shape value. Because symbolic dimensions are represented as negative numbers and
    // are not real values, inserting them as constants in the graph would invalidate
    // the graph for further use. Instead, we keep track of what their value would be
    // for extracting output shapes.
    // clang-format on

    std::unordered_map<int64_t, std::vector<Value*>> symbolic_shape_map;

    for (const auto& index_symbolic_shape : node_symbolic_input_indices) {
      auto index = index_symbolic_shape.first;
      auto tensor_shape = index_symbolic_shape.second;
      TORCH_INTERNAL_ASSERT(tensor_shape.rank().has_value());

      for (const auto& use : graph_->inputs().at(index)->uses()) {
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
            if (tensor_shape[*norm_index].is_static()) {
              replaceWithIValue(
                  use.user->output(), tensor_shape[*norm_index].value());
            } else if (symbolic_shape_values) {
              symbolic_shape_values->emplace(
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
    for (const auto i : c10::irange(symbolic_set.size())) {
      Value* v = symbolic_set[i];
      Value* dominating_value = v;
      for (const auto& sym_set : symbolic_set) {
        if (dominating_value->node()->isDominatedBy(sym_set->node())) {
          dominating_value = sym_set;
        }
      }
      if (dominating_value != v) {
        v->replaceAllUsesWith(dominating_value);
      }
    }
  }

  c10::SymbolicShape extractOutputShape(
      std::unordered_map<Value*, int64_t>& symbolic_shape_values) {
    TORCH_INTERNAL_ASSERT(graph_->outputs().size() == 1);
    auto output = graph_->outputs().at(0);
    TORCH_INTERNAL_ASSERT(
        output->type()->cast<ListType>() &&
        output->type()->cast<ListType>()->getElementType()->cast<IntType>());
    if (output->node()->kind() == prim::Constant) {
      auto int_list = toIValue(output)->toIntVector();
      return c10::SymbolicShape(int_list);
    }
    // TODO: would be nice if there were easy facility to look at uses and see
    // if they are all pure instead of instanting db.
    AliasDb db(graph_);
    // If it is not a single list construct or constant, bail,
    // otherwise we cannot analyze its output and it might be modified
    if (output->node()->kind() != prim::ListConstruct ||
        db.hasWriters(output)) {
      GRAPH_DEBUG("Could not extract shape ", getHeader(node_));
      return c10::SymbolicShape();
    }
    Node* list_construct = output->node();
    std::vector<c10::optional<int64_t>> output_shape;
    for (Value* input : list_construct->inputs()) {
      if (symbolic_shape_values.count(input)) {
        output_shape.push_back(symbolic_shape_values[input]);
      } else {
        output_shape.push_back(constant_as<int64_t>(input));
      }
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
  std::vector<std::pair<int64_t, c10::SymbolicShape>>
      node_symbolic_input_indices;
  std::shared_ptr<Graph> graph_;
  Node* node_;
};

void PropagateShapesWithShapeFunction(
    Node* n,
    std::shared_ptr<Graph>& shape_compute_graph,
    const AliasDb& db) {
  c10::SymbolicShape out =
      SymbolicShapeAnalyzer(n, shape_compute_graph, db).run();
  n->output()->setType(
      n->output()->type()->expect<TensorType>()->withSymbolicShapes(out));
}

void PropagateShapesOnBlock(Block* b, const AliasDb& db) {
  for (Node* n : b->nodes()) {
    // TODO: handle loop
    if (n->kind() == prim::If) {
      IfView if_v(n);
      PropagateShapesOnBlock(if_v.thenBlock(), db);
      PropagateShapesOnBlock(if_v.elseBlock(), db);
      mergeTypes(if_v.thenOutputs(), if_v.elseOutputs(), if_v.outputs());
    } else if (n->maybeSchema()) {
      if (auto maybe_graph = shapeComputeGraphForSchema(n->schema())) {
        PropagateShapesWithShapeFunction(n, *maybe_graph, db);
      }
    }
  }
}

void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  PropagateShapesOnBlock(graph->block(), db);
}

} // namespace jit
} // namespace torch
