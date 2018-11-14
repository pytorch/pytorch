#include "torch/csrc/jit/passes/loop_unrolling.h"

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/symbolic_variable.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/constants.h"

namespace torch { namespace jit {

namespace {

static constexpr int64_t kUnrollFactor = 8;
static constexpr int64_t kMaxBodySize = 32;
static constexpr int64_t kMaxBodyRepeats = 64;

bool isTrueConstant(Value *val) {
  c10::optional<bool> maybe_value = constant_as<bool>(val);
  return maybe_value && *maybe_value;
}

bool isForLoop(Node* node) {
  if (node->kind() != prim::Loop)
    return false;
  Value *start_cond = node->inputs().at(1);
  Value *continue_cond = node->blocks().at(0)->outputs().at(0);
  return isTrueConstant(start_cond) && isTrueConstant(continue_cond);
}

// Counts the size of this block, stopping and returning once reaches limit instructions.
int64_t limitedBlockSize(Block *body, int64_t limit) {
  auto it = body->nodes().begin();
  auto end = body->nodes().end();
  for (int64_t i = 0; i < limit; ++i, ++it) {
    for (Block *subblock : it->blocks()) {
      i += limitedBlockSize(subblock, limit - i);
    }
    if (it == end) {
      return i;
    }
  }
  return limit;
}

bool isSmallBlock(Block *body) {
  return limitedBlockSize(body, kMaxBodySize + 1) <= kMaxBodySize;
}

// XXX: This function can only be called with a loop that is guaranteed to execute EXACTLY ONCE.
void inlineBody(Node *loop) {
  auto graph = loop->owningGraph();
  auto body = loop->blocks().at(0);
  WithInsertPoint insert_point_guard { loop };

  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value *v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };

  // Loop node has extra (max_iters, initial_cond) inputs,
  // body has an extra (loop_counter) input.
  for (size_t i = 2; i < loop->inputs().size(); ++i) {
    value_map[body->inputs()[i - 1]] = loop->inputs()[i];
  }

  for (Node *orig : body->nodes()) {
    Node *clone = graph->insertNode(graph->createClone(orig, get_value));
    for (size_t i = 0; i < orig->outputs().size(); ++i) {
      value_map[orig->outputs()[i]] = clone->outputs()[i];
    }
  }
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs().at(i)->replaceAllUsesWith(get_value(body->outputs().at(i + 1)));
  }
  // XXX: it is extremely important to destroy the loop in here. DCE might not be able
  // to conclude that it's safe, because the loop might contain side effects.
  loop->destroy();
}

void repeatBody(Block *body, int64_t times) {
  // We will be adding nodes to the body, so cache the initial start and end.
  // XXX: they are both inclusive, because the exclusive body_end would point to
  //      return_node, which would move further away if we were to add nodes, and we
  //      would enter an infinite loop.
  auto body_start = body->nodes().begin();
  auto body_end = std::prev(body->nodes().end());
  auto graph = body->owningGraph();
  WithInsertPoint insert_point_guard { body };

  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value *v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };

  for (int64_t i = 1; i < times; ++i) {
    // Update loop-carried values
    // NB: note that we don't need to worry about the loop counter, because we've
    //     replaced it with a loop-carried variable
    JIT_ASSERT(body->inputs().size() == body->outputs().size());
    for (size_t i = 1; i < body->inputs().size(); ++i) {
      value_map[body->inputs()[i]] = get_value(body->outputs()[i]);
    }

    // Clone the nodes
    for (auto it = body_start; it != std::next(body_end); ++it) {
      Node *orig = *it;
      Node *clone = graph->insertNode(graph->createClone(orig, get_value));
      for (size_t i = 0; i < orig->outputs().size(); ++i) {
        value_map[orig->outputs()[i]] = clone->outputs()[i];
      }
    }
  }

  // Update outputs of the body
  const std::vector<Value*> new_outputs = fmap(body->outputs(), get_value);
  for (int64_t i = new_outputs.size() - 1; i >= 0; --i) {
    body->eraseOutput(i);
  }
  for (Value *output : new_outputs) {
    body->registerOutput(output);
  }

  // It's likely that we have some dead nodes now - for example the "true" constant
  // that prevents the loop from breaking. We shouldn't wait too long before removing
  // them because they might artificially increase the loop size and prevent outer loop
  // unrolling.
  EliminateDeadCode(body, false);
}

// Replaces the builtin loop counter with a "mutable" variable outside of the loop.
void replaceLoopCounter(Node *loop) {
  Graph *graph = loop->owningGraph();
  Block *body = loop->blocks().at(0);
  WithInsertPoint guard(loop);
  Value* init_counter = graph->insertConstant(0);

  loop->insertInput(2, init_counter);
  loop->insertOutput(0)->setType(IntType::get());

  Value * internal_counter = body->insertInput(1)->setType(init_counter->type());
  body->inputs()[0]->replaceAllUsesWith(internal_counter);

  WithInsertPoint insertPointGuard{ body->return_node() };
  Value* result = graph->insert(aten::add, {internal_counter, 1});
  body->insertOutput(1, result);
}

void unroll(Node *loop) {
  Graph *graph = loop->owningGraph();
  Block *body = loop->blocks().at(0);
  if (!isSmallBlock(body))
    return;

  // We will be using a "mutable" counter outside of the loop instead of the default
  // one, because this will allow us to share it between the unrolled loop and its epilogue.
  // This is necessary only if the loop counter is actually used in the body.
  if (body->inputs()[0]->uses().size() > 0)
    replaceLoopCounter(loop);

  // Some optimization for constant-length loops. If we know they won't run too many
  // times, then we can unroll them entirely.
  Value *trip_count = loop->inputs().at(0);
  int64_t const_len = constant_as<int64_t>(trip_count).value_or(-1);
  if (const_len != -1 && const_len < kMaxBodyRepeats) {
    repeatBody(body, const_len);
    inlineBody(loop);
    return;
  }

  WithInsertPoint insert_point_guard { loop };

  // Clone the loop before we unroll it. The clone will become the epilogue.
  Node *loop_epilogue = graph->createClone(loop, [](Value *v) { return v; })
                             ->insertAfter(loop);
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs()[i]->replaceAllUsesWith(loop_epilogue->outputs()[i]);
    loop_epilogue->replaceInput(i + 2, loop->outputs()[i]);
  }

  repeatBody(body, kUnrollFactor);

  // Change the iteration counts of both loops
  Value* iter_count = loop->inputs().at(0);
  Value* unrolled_iter_count = graph->insert(aten::__round_to_zero_floordiv, {iter_count, kUnrollFactor});
  loop->replaceInput(0, unrolled_iter_count);
  loop_epilogue->replaceInput(0, graph->insert(aten::sub, {iter_count, graph->insert(aten::mul,{unrolled_iter_count , kUnrollFactor})}));
}

void UnrollLoops(Block *block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // XXX: unroll might destroy the current node, so we need to pre-increment the iterator
    Node *node = *it; ++it;
    for (Block *subblock : node->blocks()) {
      UnrollLoops(subblock);
    }
    if (isForLoop(node)) {
      unroll(node);
    }
  }
}

} // anonymous namespace

void UnrollLoops(std::shared_ptr<Graph>& graph) {
  UnrollLoops(graph->block());
  EliminateDeadCode(graph);
}

}} // namespace torch::jit
