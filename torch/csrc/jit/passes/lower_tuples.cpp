#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_tuples.h>

namespace torch {
namespace jit {

namespace {

// operators where we expect to find tuples as inputs/outputs
// this is to assert we are only doing modifications when we know
// we can flatten tuples
std::unordered_set<Symbol> white_list = {
    prim::If,
    prim::Loop,
    prim::TupleUnpack,
    prim::TupleConstruct,
    prim::TupleIndex,
    prim::TupleSlice,
    prim::Param,
    prim::Return,
};

void removeTupleNodes(Node* n, bool must_remove_tuples) {
  if (n->kind() != prim::TupleUnpack && n->kind() != prim::TupleIndex &&
      n->kind() != prim::TupleSlice) {
    return;
  }
  // tuple index has two inputs, tuple and index
  auto construct = n->inputs().at(0)->node();
  if (construct->kind() != prim::TupleConstruct) {
    if (must_remove_tuples) {
      AT_ERROR(n->kind().toQualString(), " not matched to tuple construct");
    }
    return;
  }
  if (n->kind() == prim::TupleUnpack) {
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs()[i]->replaceAllUsesWith(construct->inputs().at(i));
    }
  } else if (n->kind() == prim::TupleIndex) {
    auto idx = n->inputs().at(1);
    auto maybe_int = constant_as<int64_t>(idx);
    if (!maybe_int) {
      if (must_remove_tuples) {
        AT_ERROR(n->sourceRange(), "tuple index with non-constant index");
      }
      return;
    }
    auto int_idx = *maybe_int;
    auto len = construct->output()->type()->containedTypes().size();
    if (int_idx < 0) {
      int_idx += len;
    }
    // currently, we allow non-constant tuple index if the tuple is of one type.
    // so we need to check bounds here
    if (int_idx >= 0 && static_cast<size_t>(int_idx) < len) {
      n->output()->replaceAllUsesWith(construct->inputs().at(int_idx));
    }
  } else if (n->kind() == prim::TupleSlice) {
    std::vector<Value*> values;
    int64_t beg = n->i(attr::beg);
    int64_t end = n->i(attr::end);
    for (int64_t i = beg; i < end; i += 1) {
      values.push_back(construct->inputs().at(i));
    }
    auto graph = n->owningGraph();
    auto tuple_out = graph->createTuple(values);
    WithInsertPoint insert(n);
    graph->insertNode(tuple_out);
    n->output()->replaceAllUsesWith(tuple_out->output());
  }
}
} // anonymous namespace

static void LowerAllTuples(Block* block);

static void VisitNode(Node* n, Node* insert_point) {
  auto& graph = *n->owningGraph();

  // tuple construction operators will become dead when the unpacks are replaced
  if (n->kind() == prim::TupleConstruct) {
    return;
  }

  // note: changing the second argument to false changes this pass from a
  // complete lowering pass to one that removes tuples when possible. When
  // tuples are first-class in the interpreter, we should still run this pass to
  // remove extraneous uses

  if (n->kind() == prim::TupleUnpack || n->kind() == prim::TupleIndex ||
      n->kind() == prim::TupleSlice) {
    removeTupleNodes(n, /*must_remove_tuples*/ true);
    return;
  }

  // flatten the input list  op(a, tup, b) --> op(a, t0, t1, b)
  for (size_t i = 0; i < n->inputs().size();) {
    auto input = n->inputs()[i];
    if (TupleTypePtr tt = input->type()->cast<TupleType>()) {
      TORCH_CHECK(
          white_list.count(n->kind()) > 0,
          "tuple appears in op that does not forward tuples, ",
          "unsupported kind: ", n->kind().toQualString());
      TORCH_CHECK(
          input->node()->kind() == prim::TupleConstruct,
          "tuple use not matched to tuple construct");
      for (size_t j = 0; j < tt->elements().size(); ++j) {
        n->insertInput(i + 1 + j, input->node()->inputs().at(j));
      }
      n->removeInput(i);
      // note: no update to i
      // since tuples might be nested we need to recursively scan
      // the new flattened inputs
    } else {
      ++i;
    }
  }
  for (auto b : n->blocks()) {
    LowerAllTuples(b);
  }

  // flatten the outputs list
  for (size_t i = 0; i < n->outputs().size();) {
    Value* output = n->outputs()[i];
    // (a, b, tup, c) -> (a, b, t0, t1, c)
    // and:
    //    tup = (t0, t1)
    // is placed at the current insertion point
    if (TupleTypePtr tt = output->type()->cast<TupleType>()) {
      TORCH_CHECK(
          white_list.count(n->kind()) > 0,
          "tuple appears in op that does not forward tuples, ",
          "unsupported kind: ", n->kind().toQualString());
      for (size_t j = 0; j < tt->elements().size(); j++) {
        n->insertOutput(i + 1 + j)->setType(tt->elements()[j]);
      }
      auto new_tup =
          graph.createTuple(n->outputs().slice(i + 1, tt->elements().size()));
      new_tup->insertBefore(insert_point);
      insert_point = new_tup;
      output->replaceAllUsesWith(new_tup->output());
      n->eraseOutput(i);
      // note: no update to i to handle nested tuples
    } else {
      ++i;
    }
  }
}

static void LowerAllTuples(Block* block) {
  // tuples in parameter lists of a block behave exactly the same as
  // _outputs_ of normal instructions, since the param_node represents the
  // parameters as outputs, we can handle it by simply visiting the node
  VisitNode(block->param_node(), *block->nodes().begin());
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    auto n = *it++;
    VisitNode(n, *it);
  }
  // tuples in return lists of blocks behave exactly the same as
  // _inputs_ of normal instructions, so we can use VisitNode here as well
  // insert_point is null because it will never be used since return nodes
  // have no outputs
  VisitNode(block->return_node(), nullptr);
}

static void EnsureNoTuples(ArrayRef<Value*> values) {
  for (Value* v : values) {
    TORCH_CHECK(
        v->type()->kind() != TypeKind::TupleType, "Couldn't lower all tuples.");
  }
}

static void EnsureNoTuples(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      EnsureNoTuples(b);
    }
    EnsureNoTuples(n->outputs());
  }
}

void LowerAllTuples(const std::shared_ptr<Graph>& graph) {
  LowerAllTuples(graph->block());
  EliminateDeadCode(graph->block());
  EnsureNoTuples(graph->block());
}

void LowerSimpleTuples(Block* block) {
  for (auto n : block->nodes()) {
    removeTupleNodes(n, /*must_remove_tuples*/ false);
    for (auto b : n->blocks()) {
      LowerSimpleTuples(b);
    }
  }
}

void LowerSimpleTuples(const std::shared_ptr<Graph>& graph) {
  LowerSimpleTuples(graph->block());
  EliminateDeadCode(graph);
}
} // namespace jit
} // namespace torch
