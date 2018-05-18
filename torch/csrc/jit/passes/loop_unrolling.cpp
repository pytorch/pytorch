#include "torch/csrc/jit/passes/loop_unrolling.h"

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

namespace {

static constexpr int64_t kUnrollFactor = 8;
static constexpr int64_t kMaxBodySize = 16;
static constexpr int64_t kMaxBodyRepeats = 64;

// XXX: only valid for for loops that will execute exactly once! Ignores the condition entirely!
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

  // Loop node has extra (max_iters, initial_cond) inputs, body has an extra (loop_counter) input
  for (size_t i = 2; i < loop->inputs().size(); ++i) {
    value_map[body->inputs()[i - 1]] = loop->inputs()[i];
  }
  Node *init_counter_node = graph->insertNode(
      graph->createConstant(at::CPU(at::kLong).scalarTensor(0)));
  value_map[body->inputs()[0]] = init_counter_node->output();

  for (Node *orig : body->nodes()) {
    Node *clone = graph->insertNode(graph->createClone(orig, get_value));
    for (auto orig_it = orig->outputs().begin(), clone_it = clone->outputs().begin();
          orig_it != orig->outputs().end();
          ++orig_it) {
      value_map[*orig_it] = *clone_it;
    }
  }
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs().at(i)->replaceAllUsesWith(value_map.at(body->outputs().at(i + 1)));
  }
  // XXX: it is extremely important to destroy the loop in here. DCE might not be able
  // to conclude that it's safe, because the loop might contain side effects.
  loop->destroy();
}

void repeatBody(Block *body, int64_t times) {
  // We will be adding nodes to it, so cache the initial start and end.
  // XXX: they are both inclusive
  auto body_start = body->nodes().begin();
  auto body_end = std::prev(body->nodes().end());
  auto graph = body->owningGraph();
  WithInsertPoint insert_point_guard { body->return_node() };

  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value *v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };

  for (int64_t i = 1; i < times; ++i) {
    // Update loop-carried values
    JIT_ASSERT(body->inputs().size() == body->outputs().size());
    for (auto input_it = std::next(body->inputs().begin()), output_it = std::next(body->outputs().begin());
         input_it != body->inputs().end();
         ++input_it) {
      value_map[*input_it] = get_value(*output_it);
    }
    // Update the loop counter
    Value *orig_loop_counter = body->inputs().at(0);
    // XXX: this needs to happen before the next line, because value_map[orig_loop_counter]
    // will insert a nullptr the first time this is run, and get_value will return nullptr in
    // that case.
    Value *cur_loop_counter = get_value(orig_loop_counter);
    value_map[orig_loop_counter] = SymbolicVariable(cur_loop_counter) + at::Scalar(1);

    // Clone the nodes
    for (auto it = body_start; it != std::next(body_end); ++it) {
      Node *orig = *it;
      Node *clone = graph->insertNode(graph->createClone(orig, get_value));
      for (auto orig_it = orig->outputs().begin(), clone_it = clone->outputs().begin();
           orig_it != orig->outputs().end();
           ++orig_it) {
        value_map[*orig_it] = *clone_it;
      }
    }
  }

  const std::vector<Value*> orig_outputs = body->outputs();
  for (int64_t i = orig_outputs.size() - 1; i >= 0; --i) {
    body->eraseOutput(i);
  }
  for (Value *output : orig_outputs) {
    body->registerOutput(value_map.at(output));
  }

  //EliminateDeadCode(body, false);
}

void multiplyTripCountBy(Block *body, int64_t factor) {
  WithInsertPoint insert_point_guard(*body->nodes().begin());
  Value *old_trip_count = body->inputs().at(0);
  Value *new_trip_count = SymbolicVariable(old_trip_count) * at::Scalar(factor);
  old_trip_count->replaceAllUsesWith(new_trip_count);
  // Replace has changed this use as well, so we need to fix it.
  new_trip_count->node()->replaceInput(0, old_trip_count);
}

void offsetTripCountBy(Block *body, Value *factor) {
  WithInsertPoint insert_point_guard(*body->nodes().begin());
  Value *old_trip_count = body->inputs().at(0);
  Value *new_trip_count = SymbolicVariable(old_trip_count) + SymbolicVariable(factor);
  old_trip_count->replaceAllUsesWith(new_trip_count);
  // Replace has changed this use as well, so we need to fix it.
  new_trip_count->node()->replaceInput(0, old_trip_count);
}

bool isTrueConstant(Value *val) {
  Node *producer = val->node();
  if (producer->kind() != prim::Constant)
    return false;
  auto value = producer->t(attr::value);
  return value.type() == at::CPU(at::kByte) && value.dim() == 0 && value.toCLong() == 1;
}

bool isForLoop(Node* node) {
  if (node->kind() != prim::Loop)
    return false;
  Value *start_cond = node->inputs().at(1);
  Value *continue_cond = node->blocks().at(0)->outputs().at(0);
  return isTrueConstant(start_cond) && isTrueConstant(continue_cond);
}

int64_t limitedBlockSize(Block *body, int64_t limit) {
  auto it = body->nodes().begin();
  auto end = body->nodes().end();
  for (int64_t i = 0; i < limit; ++i, ++it) {
    for (Block *subblock : it->blocks()) {
      i += limitedBlockSize(subblock, limit);
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

at::optional<int64_t> getConstantLength(Node *loop) {
  Value *trip_count = loop->inputs().at(0);
  if (trip_count->node()->kind() != prim::Constant)
    return at::nullopt;
  return {trip_count->node()->t(attr::value).toCLong()};
}

void unroll(Node *loop) {
  Graph *graph = loop->owningGraph();
  Block *body = loop->blocks().at(0);
  if (!isSmallBlock(body))
    return;

  int64_t const_len = getConstantLength(loop).value_or(-1);
  if (const_len != -1 && const_len < kMaxBodyRepeats) {
    repeatBody(body, const_len);
    inlineBody(loop);
    return;
  }

  WithInsertPoint insert_point_guard { loop };
  Node *loop_epilogue = graph->createClone(loop, [](Value *v) { return v; })
                             ->insertAfter(loop);
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs()[i]->replaceAllUsesWith(loop_epilogue->outputs()[i]);
    loop_epilogue->replaceInput(i + 2, loop->outputs()[i]);
  }

  repeatBody(body, kUnrollFactor);

  Value *iter_count = loop->inputs().at(0);
  loop_epilogue->replaceInput(0, SymbolicVariable(iter_count) % at::Scalar(kUnrollFactor));
  loop->replaceInput(0, SymbolicVariable(iter_count) / at::Scalar(kUnrollFactor));

  Value *epilogue_offset = SymbolicVariable(iter_count) - SymbolicVariable(loop_epilogue->inputs()[0]);
  multiplyTripCountBy(body, kUnrollFactor);
  offsetTripCountBy(loop_epilogue->blocks().at(0), epilogue_offset);
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
