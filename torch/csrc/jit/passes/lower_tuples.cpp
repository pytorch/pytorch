#include <torch/csrc/jit/passes/lower_tuples.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <utility>

namespace torch {
namespace jit {

namespace {

// operators where we expect to find tuples as inputs/outputs
// this is to assert we are only doing modifications when we know
// we can flatten tuples
std::unordered_set<Symbol> supported_ops = {
    prim::If,
    prim::Loop,
    prim::Uninitialized,
    prim::TupleUnpack,
    prim::TupleConstruct,
    prim::TupleIndex,
    prim::TupleSlice,
    prim::Param,
    prim::Return,
    prim::PythonOp,
    aten::format,
    prim::Uninitialized,
    aten::__getitem__};

// Flatten block inputs and insert a tuple construct in the block
static void flattenTupleInLoopParams(Node* n, size_t index) {
  auto input = n->inputs().at(index);
  TupleTypePtr tt = input->type()->cast<TupleType>();
  TORCH_INTERNAL_ASSERT(tt);

  Block* block = n->blocks().at(0);
  Node* block_node = n;

  std::vector<Value*> new_node_inputs = {};
  auto new_construct_node =
      block->prependNode(block->owningGraph()->create(prim::TupleConstruct));
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    auto new_block_in = block->insertInput(index + j);
    new_construct_node->addInput(new_block_in);
    block_node->insertInput(index + j + 1, input->node()->inputs().at(j));
  }
  new_construct_node->output()->setType(block->inputs().at(index - 1)->type());
  new_construct_node->copyMetadata(n);
  block->inputs().at(index - 1)->replaceAllUsesWith(
      new_construct_node->output());
  block->eraseInput(index - 1);
  block_node->removeInput(index);
}

// Flatten tuple outputs of the block node and append a TupleConstruct
// node after the block node if there is an outer block.
static void flattenTupleInBlockReturn(Node* n, size_t index) {
  auto input = n->inputs().at(index);
  Block* block = n->owningBlock();
  Node* block_node = block->owningNode();
  Node* new_construct_node = nullptr;
  TupleTypePtr tt = input->type()->cast<TupleType>();
  TORCH_INTERNAL_ASSERT(tt);

  // 1- Add flattened tuple to block outputs
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    block->insertOutput(index + j + 1, input->node()->inputs().at(j));
  }
  block->eraseOutput(index);

  if (block_node == nullptr)
    return;
  // 2- For uses of the block node in the outer block,
  // flatten the blocknode outputs and insert a tuple construct
  // to replace that.
  // Loop block has an extra element (iter counter)
  if (block_node->kind() == prim::Loop)
    index = index - 1;
  auto tuple_output = block_node->outputs().at(index);
  // When node has multiple blocks, do not flatten outputs on the second block
  // again
  if (!(tuple_output->type()->cast<TupleType>()))
    return;

  new_construct_node = block->owningGraph()->create(prim::TupleConstruct);
  new_construct_node->insertAfter(block_node);
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    auto new_block_out = block_node->insertOutput(index + j + 1);
    new_construct_node->addInput(new_block_out);
  }
  // Replace the block node with the new TupleConstruct node
  new_construct_node->output()->setType(tuple_output->type());
  new_construct_node->copyMetadata(block_node);
  tuple_output->replaceAllUsesWith(new_construct_node->output());
  block_node->eraseOutput(index);
}

void removeTupleNodes(Node* n, bool must_remove_tuples) {
  if (n->kind() != prim::TupleUnpack && n->kind() != prim::TupleIndex &&
      n->kind() != prim::TupleSlice) {
    return;
  }
  // tuple index has two inputs, tuple and index
  auto construct_node = n->inputs().at(0)->node();
  if (construct_node->kind() != prim::TupleConstruct) {
    if (must_remove_tuples) {
      AT_ERROR(n->kind().toQualString(), " not matched to tuple construct");
    }
    return;
  }
  if (n->kind() == prim::TupleUnpack) {
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs()[i]->replaceAllUsesWith(construct_node->inputs().at(i));
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
    size_t len = construct_node->output()->type()->containedTypes().size();
    if (int_idx < 0) {
      int_idx += len;
    }
    // currently, we allow non-constant tuple index if the tuple is of one type.
    // so we need to check bounds here
    if (int_idx >= 0 && static_cast<size_t>(int_idx) < len) {
      n->output()->replaceAllUsesWith(construct_node->inputs().at(int_idx));
    }
  } else if (n->kind() == prim::TupleSlice) {
    std::vector<Value*> values;
    int64_t beg = n->i(attr::beg);
    int64_t end = n->i(attr::end);
    for (int64_t i = beg; i < end; i += 1) {
      values.push_back(construct_node->inputs().at(i));
    }
    auto graph = n->owningGraph();
    auto tuple_out = graph->createTuple(values);
    tuple_out->copyMetadata(n);
    WithInsertPoint insert(n);
    graph->insertNode(tuple_out);
    n->output()->replaceAllUsesWith(tuple_out->output());
  }
}
} // anonymous namespace

static void LowerAllTuples(Block* block);

static void RemoveTupleConstants(Node* n) {
  if (!(n->kind() == prim::Constant &&
        n->output()->type()->cast<TupleType>())) {
    return;
  }

  auto g = n->owningGraph();
  auto tuple = toIValue(n->output()).value().toTuple();
  const auto& tuple_elements = tuple->elements();
  WithInsertPoint insert(n);
  std::vector<Value*> elements;
  for (const auto& elem : tuple_elements) {
    auto constant = insertConstant(*n->owningGraph(), elem);
    elements.push_back(constant);
  }
  auto tuple_type = n->output()->type()->expect<TupleType>();
  auto tuple_construct = g->insertNode(n->owningGraph()->createTuple(
      elements, tuple_type->schema() ? std::move(tuple_type) : nullptr));
  tuple_construct->copyMetadata(n);

  // insert the tuple first before recursing on its elements, so that its
  // elements will have a use
  for (Value* elem : elements) {
    RemoveTupleConstants(elem->node());
  }

  n->replaceAllUsesWith(tuple_construct);
}

static void flattenInputs(Node* n, Node* insert_point) {
  // flatten the input list  op(a, tup, b) --> op(a, t0, t1, b)
  for (size_t i = 0; i < n->inputs().size();) {
    auto input = n->inputs()[i];
    if (TupleTypePtr tt = input->type()->cast<TupleType>()) {
      TORCH_CHECK(
          (input->node()->kind() == prim::TupleConstruct),
          "tuple use not matched to tuple construct. Instead found: ",
          n->kind().toQualString());
      if (supported_ops.count(n->kind()) > 0) {
        if (n->kind() == prim::Loop) {
          // This function supports all node types with blocks that take tuple
          // inputs.
          flattenTupleInLoopParams(n, i);
        } else if (n->kind() == prim::Return) {
          flattenTupleInBlockReturn(n, i);
        } else {
          for (size_t j = 0; j < tt->elements().size(); ++j) {
            n->insertInput(i + 1 + j, input->node()->inputs().at(j));
          }
          n->removeInput(i);
        }
        // note: no update to i
        // since tuples might be nested we need to recursively scan
        // the new flattened inputs
      } else {
        TORCH_WARN(
            "tuple appears in op inputs, but this op does not forward tuples, ",
            "unsupported kind: ",
            n->kind().toQualString());
        ++i;
      }
    } else {
      ++i;
    }
  }
}

static void flattenOutputs(Node* n, Node* insert_point) {
  // flatten the outputs list
  auto& graph = *n->owningGraph();
  for (size_t i = 0; i < n->outputs().size();) {
    Value* output = n->outputs()[i];
    if (!output->hasUses()) {
      ++i;
      continue;
    }

    // (a, b, tup, c) -> (a, b, t0, t1, c)
    // and:
    //    tup = (t0, t1)
    // is placed at the current insertion point
    if (TupleTypePtr tt = output->type()->cast<TupleType>()) {
      if (supported_ops.count(n->kind()) > 0) {
        for (const auto j : c10::irange(tt->elements().size())) {
          n->insertOutput(i + 1 + j)->setType(tt->elements()[j]);
        }
        auto new_tup =
            graph.createTuple(n->outputs().slice(i + 1, tt->elements().size()));
        new_tup->copyMetadata(n);
        new_tup->insertBefore(insert_point);
        insert_point = new_tup;
        output->replaceAllUsesWith(new_tup->output());
        n->eraseOutput(i);
        // note: no update to i to handle nested tuples
      } else {
        TORCH_WARN(
            "tuple appears in the op outputs, but this op does not forward tuples, ",
            "unsupported kind: ",
            n->kind().toQualString());
        ++i;
      }
    } else {
      ++i;
    }
  }
}

static void VisitNode(Node* n, Node* insert_point) {
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
  flattenInputs(n, insert_point);
  for (auto b : n->blocks()) {
    LowerAllTuples(b);
  }
  flattenOutputs(n, insert_point);
}

static void LowerAllTuples(Block* block) {
  // tuples in parameter lists of a block behave exactly the same as
  // _outputs_ of normal instructions, since the param_node represents the
  // parameters as outputs, we can handle it by simply visiting the node
  VisitNode(block->param_node(), *block->nodes().begin());
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    auto n = *it++;
    RemoveTupleConstants(n);
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
  GRAPH_DUMP("After LowerAllTuples: ", graph);
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
  GRAPH_DUMP("After LowerSimpleTuples: ", graph);
  EliminateDeadCode(graph);
}
} // namespace jit
} // namespace torch
