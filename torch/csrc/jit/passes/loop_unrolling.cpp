#include <torch/csrc/jit/passes/loop_unrolling.h>

#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

namespace {

static constexpr int64_t kUnrollFactor = 8;
static constexpr int64_t kMaxBodySize = 32;
static constexpr int64_t kMaxBodyRepeats = 64;

bool isTrueConstant(Value* val) {
  std::optional<bool> maybe_value = constant_as<bool>(val);
  return maybe_value && *maybe_value;
}

bool isForLoop(Node* node) {
  if (node->kind() != prim::Loop)
    return false;
  Value* start_cond = node->inputs().at(1);
  Value* continue_cond = node->blocks().at(0)->outputs().at(0);
  return isTrueConstant(start_cond) && isTrueConstant(continue_cond);
}

// Counts the size of this block, stopping and returning once reaches limit
// instructions.
int64_t limitedBlockSize(Block* body, int64_t limit) {
  auto it = body->nodes().begin();
  auto end = body->nodes().end();
  for (int64_t i = 0; i < limit; ++it) {
    for (Block* subblock : it->blocks()) {
      i += limitedBlockSize(subblock, limit - i);
    }
    if (!it->notExecutedOp()) {
      ++i;
    }
    if (it == end) {
      return i;
    }
  }
  return limit;
}

bool isSmallBlock(Block* body) {
  return limitedBlockSize(body, kMaxBodySize + 1) <= kMaxBodySize;
}

// XXX: This function can only be called with a loop that is guaranteed to
// execute EXACTLY ONCE.
void inlineBody(Node* loop) {
  auto graph = loop->owningGraph();
  auto body = loop->blocks().at(0);
  WithInsertPoint insert_point_guard{loop};

  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value* v) {
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

  for (Node* orig : body->nodes()) {
    Node* clone = graph->insertNode(graph->createClone(orig, get_value));
    for (size_t i = 0; i < orig->outputs().size(); ++i) {
      value_map[orig->outputs()[i]] = clone->outputs()[i];
    }
  }
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs().at(i)->replaceAllUsesWith(
        get_value(body->outputs().at(i + 1)));
  }
  // XXX: it is extremely important to destroy the loop in here. DCE might not
  // be able to conclude that it's safe, because the loop might contain side
  // effects.
  loop->destroy();
}

// inserts a copy of body, passing inputs to the inputs of the block
// it returns the a list of the Values for the output of the block
std::vector<Value*> insertBlockCopy(
    Graph& graph,
    Block* body,
    at::ArrayRef<Value*> inputs) {
  TORCH_INTERNAL_ASSERT(inputs.size() == body->inputs().size());
  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value* v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };
  auto inputs_it = inputs.begin();
  for (Value* input : body->inputs()) {
    value_map[input] = *inputs_it++;
  }
  for (Node* node : body->nodes()) {
    Node* new_node = graph.insertNode(graph.createClone(node, get_value));
    auto outputs_it = new_node->outputs().begin();
    for (Value* output : node->outputs()) {
      value_map[output] = *outputs_it++;
    }
  }
  return fmap(body->outputs(), get_value);
}

void repeatBody(Block* body, size_t times, Block* dest) {
  auto graph = body->owningGraph();
  WithInsertPoint insert_point_guard(dest);
  for (Value* input : body->inputs()) {
    dest->addInput()->copyMetadata(input);
  }

  std::vector<Value*> io = dest->inputs().vec();
  TORCH_INTERNAL_ASSERT(
      !body->inputs().at(0)->hasUses(), "loop counter should be unused");
  for (const auto i : c10::irange(times)) {
    (void)i; // Suppress unused variable warning
    io[0] = body->inputs().at(0);
    io = insertBlockCopy(*graph, body, io);
  }
  for (Value* output : io) {
    dest->registerOutput(output);
  }

  // It's likely that we have some dead nodes now - for example the "true"
  // constant that prevents the loop from breaking. We shouldn't wait too long
  // before removing them because they might artificially increase the loop size
  // and prevent outer loop unrolling.
  EliminateDeadCode(dest, false);
}

// Replaces the builtin loop counter with a "mutable" variable outside of the
// loop.
void replaceLoopCounter(Node* loop) {
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);
  WithInsertPoint guard(loop);
  Value* init_counter = graph->insertConstant(0);

  loop->insertInput(2, init_counter);
  loop->insertOutput(0)->setType(IntType::get());

  Value* internal_counter = body->insertInput(1)->setType(init_counter->type());
  body->inputs()[0]->replaceAllUsesWith(internal_counter);

  WithInsertPoint insertPointGuard{body->return_node()};
  Value* result = graph->insert(aten::add, {internal_counter, 1});
  body->insertOutput(1, result);
}

void unroll(Node* loop) {
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);

  // We will be using a "mutable" counter outside of the loop instead of the
  // default one, because this will allow us to share it between the unrolled
  // loop and its epilogue. This is necessary only if the loop counter is
  // actually used in the body.
  if (!body->inputs()[0]->uses().empty())
    replaceLoopCounter(loop);

  // Some optimization for constant-length loops. If we know they won't run too
  // many times, then we can unroll them entirely.
  Value* trip_count = loop->inputs().at(0);
  std::optional<int64_t> const_len = constant_as<int64_t>(trip_count);
  if (const_len && *const_len < kMaxBodyRepeats) {
    Block* dest = loop->addBlock();
    repeatBody(body, *const_len, dest);
    loop->eraseBlock(0);
    inlineBody(loop);
    return;
  }

  WithInsertPoint insert_point_guard{loop};

  // Clone the loop before we unroll it. The clone will become the epilogue.
  Node* loop_epilogue =
      graph->createClone(loop, [](Value* v) { return v; })->insertAfter(loop);
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs()[i]->replaceAllUsesWith(loop_epilogue->outputs()[i]);
    loop_epilogue->replaceInput(i + 2, loop->outputs()[i]);
  }

  Block* dest = loop->addBlock();
  repeatBody(body, kUnrollFactor, dest);
  loop->eraseBlock(0);

  // Change the iteration counts of both loops
  Value* iter_count = loop->inputs().at(0);
  Value* unrolled_iter_count = graph->insert(
      aten::__round_to_zero_floordiv, {iter_count, kUnrollFactor});
  loop->replaceInput(0, unrolled_iter_count);
  loop_epilogue->replaceInput(
      0,
      graph->insert(
          aten::sub,
          {iter_count,
           graph->insert(aten::mul, {unrolled_iter_count, kUnrollFactor})}));
}

bool UnrollLoops(Block* block, bool constant_only) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // XXX: unroll might destroy the current node, so we need to pre-increment
    // the iterator
    Node* node = *it;
    ++it;
    for (Block* subblock : node->blocks()) {
      changed |= UnrollLoops(subblock, constant_only);
    }
    if (!isForLoop(node)) {
      continue;
    }
    if (constant_only) {
      if (node->inputs().at(0)->node()->kind() != prim::Constant) {
        continue;
      }
    } else if (!isSmallBlock(node->blocks().at(0))) {
      continue;
    }

    unroll(node);
    changed = true;
  }
  return changed;
}

} // anonymous namespace

static void addCondAsOutput(Node* loop) {
  LoopView loop_view(loop);
  loop->addInput(loop_view.inputCond());
  auto block_cond_input = loop_view.bodyBlock()->addInput();
  block_cond_input->copyMetadata(loop_view.inputCond());
  auto cond_output_index =
      loop_view.bodyBlock()->registerOutput(loop_view.nextCond());
  loop_view.bodyBlock()->outputs()[cond_output_index]->copyMetadata(
      loop_view.nextCond());
  auto cond_output = loop->addOutput();
  cond_output->copyMetadata(loop_view.nextCond());
}

bool LoopsPeeler::run(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before LoopsPeeler", graph);
  collectLoops(graph->block());
  peelLoops();
  GRAPH_DUMP("After LoopsPeeler", graph);
  return true;
}

void LoopsPeeler::collectLoop(Node* n) {
  if (callback_(n)) {
    if (in_loop_) {
      GRAPH_DEBUG("Loop ", getHeader(in_loop_), " will be unrolled");
      loops_to_peel_.push_back(in_loop_);
      in_loop_ = nullptr;
    }
  }
}

void LoopsPeeler::collectLoops(Block* block) {
  // we do a pre-order traversal to reduce the number
  // of peeled loops.
  for (auto n : block->nodes()) {
    collectLoop(n);
  }
  collectLoop(block->return_node());

  // process child blocks
  for (auto n : block->nodes()) {
    auto old_in_loop_ = in_loop_;
    if (n->kind() == prim::Loop) {
      in_loop_ = n;
    }
    for (auto b : n->blocks()) {
      collectLoops(b);
    }
    in_loop_ = old_in_loop_;
  }
}

void LoopsPeeler::peelLoops() {
  for (auto loop : loops_to_peel_) {
    PeelLoop(loop, num_iterations_);
  }
}

bool PeelProfilingLoops(const std::shared_ptr<Graph>& graph) {
  auto peel_predicate = [](Node* n) {
    for (auto i : n->inputs()) {
      if (i->type()->isSubtypeOf(*TensorType::get())) {
        return true;
      }
    }

    return false;
  };

  LoopsPeeler lp(peel_predicate);
  return lp.run(graph);
}

Node* PeelLoop(Node* n, size_t times) {
  GRAPH_DEBUG("Peeling the loop ", getHeader(n), " ", times, " times");

  auto graph = n->owningGraph();
  auto orig_loop = LoopView(n);

  WithInsertPoint wip(n);
  auto times_const = graph->insertConstant(static_cast<int64_t>(times));
  // N.B. even though a caller may request to peel `times` iterations
  // `maxTripCount` of the original loop might be less than that
  // so we should take the minimum of the two
  auto min_trip_count =
      graph->insert(prim::min, {orig_loop.maxTripCount(), times_const});

  // make the peeled clone
  auto peeled_copy = graph->createClone(n, [](Value* v) { return v; });
  addCondAsOutput(peeled_copy);

  LoopView new_lv(peeled_copy);
  graph->insertNode(peeled_copy);
  // only run until the peeled count
  new_lv.replaceMaxTripCount(min_trip_count);

  // subtract `maxTripCount` of the original loop by the number iterations
  // the peeled loop runs
  auto new_max_trip_count =
      graph->insert(aten::sub, {orig_loop.maxTripCount(), min_trip_count});
  orig_loop.replaceMaxTripCount(new_max_trip_count);
  // update the termination condition
  auto cond_index = peeled_copy->outputs().size() - 1;
  orig_loop.replaceInputCondition(peeled_copy->output(cond_index));

  static const size_t LOOP_DEPS_WITH_COND_OFFSET = 2;
  for (size_t i = 0; i < peeled_copy->outputs().size() -
           1 /* leave off the termination condition */;
       i++) {
    n->replaceInput(LOOP_DEPS_WITH_COND_OFFSET + i, peeled_copy->output(i));
  }

  // the induction variable also needs to be adjusted by the number of
  // iterations the peeled loop runs
  {
    WithInsertPoint peeled_wip(*orig_loop.bodyBlock()->nodes().begin());
    // we can't create the expression: `new_counter` = `old_counter` + 1 yet
    // because when we
    // run `old_counter->replaceAllUsesWith(new_counter)`, we will get
    // `new_counter = new_counter + 1`
    auto adjusted_iter_counter =
        graph->insert(aten::add, {min_trip_count, min_trip_count});
    orig_loop.currentTripCount()->replaceAllUsesWith(adjusted_iter_counter);
    adjusted_iter_counter->node()->replaceInput(
        0, orig_loop.currentTripCount());
  }

  return peeled_copy;
}

bool UnrollLoops(std::shared_ptr<Graph>& graph) {
  bool changed = UnrollLoops(graph->block(), false);
  if (changed) {
    EliminateDeadCode(graph);
  }
  return changed;
}

bool UnrollConstantLoops(std::shared_ptr<Graph>& graph) {
  bool changed = UnrollLoops(graph->block(), true);
  if (changed) {
    EliminateDeadCode(graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
