
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <unordered_map>

namespace torch {

namespace jit {

namespace {

ProfileIValueOp* createProfileIValueNode(
    std::shared_ptr<Graph>& graph,
    ArrayRef<Value*> inputs) {
  auto pn = new ProfileIValueOp(graph.get(), nullptr);
  for (auto inp : inputs) {
    pn->addInput(inp);
  }
  return pn;
}

// A helper structure to mantain the mappings
// between values from a scripted graph and
// a traced graph
struct TracingData {
  std::unordered_map<Value*, Value*> old_to_new_;
  std::shared_ptr<Graph> traced_graph_ = nullptr;
  std::mutex mutex_;

  TracingData() {
    traced_graph_ = std::make_shared<Graph>();
  }
};

// create a node in the traced graph that corresponds to `node`
// in the scripted graph. Similar to how `cloneNode` works
Node* traceNode(Node* node, TracingData& td, Stack& stack) {
  GRAPH_DEBUG("Tracing node ", getHeader(node));
  auto* block = td.traced_graph_->block();
  auto env = [&td](Value* v) { return td.old_to_new_.at(v); };

  auto new_node = block->appendNode(td.traced_graph_->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    no->copyMetadata(oo);
    td.old_to_new_[oo] = no;
    GRAPH_DEBUG(
        "Mapping ",
        oo->debugName(),
        " to ",
        no->debugName()); // old to new outputs
  }
  return new_node;
}

void insertTracingNodes(Block*, std::shared_ptr<Graph>&, TracingData&);

// The subtlety in `createPropNodeForIfBlock` is that we need to create
// a "propagate" node that will propagate the mapping between the outputs
// of a then/else block and the outputs in the traced graph onto the outputs
// of the if node in the scripted node. Note, if nodes will disappear in the
// the traced graph but they are still used in the scripted graph.
void createPropNodeForIfBlock(
    Block* b,
    Node* n,
    std::shared_ptr<Graph>& graph,
    TracingData& td) {
  std::vector<Value*> empty_values{};
  auto opt_pn = createProfileIValueNode(graph, empty_values);
  insertTracingNodes(b, graph, td);
  b->appendNode(opt_pn);
  std::function<void(Stack&)> optional_profiler = [n, b, &td](Stack& stack) {
    std::lock_guard<std::mutex> lock(td.mutex_);

    // frame_id is unused
    int64_t frame_id = 0;
    pop(stack, frame_id);

    for (size_t i = 0; i < b->outputs().size(); i++) {
      // propagate a then-block or else-output to an if-output
      auto nbo = td.old_to_new_.at(b->outputs()[i]);
      td.old_to_new_[n->outputs()[i]] = nbo;
      GRAPH_DEBUG(
          "Map ",
          td.old_to_new_[n->outputs()[i]]->debugName(),
          " to ",
          nbo->debugName());
    }
  };

  // uncomment for debugging
  // opt_pn->i_(Symbol::attr("propagate"), 1);
  opt_pn->setCallback(optional_profiler);
}

// loop counter is implicit in the loop body outputs, we need to make
// it explicit so it can used in 2+ iterations
void traceLoopCounter(Node* n, std::shared_ptr<Graph>& graph, TracingData& td) {
  LoopView lv(n);
  auto opt_pn = createProfileIValueNode(graph, lv.currentTripCount());
  lv.bodyBlock()->prependNode(opt_pn);
  std::function<void(Stack&)> optional_profiler = [n, &td](Stack& stack) {
    std::lock_guard<std::mutex> lock(td.mutex_);
    // frame_id is unused
    int64_t frame_id = 0;
    pop(stack, frame_id);
    int64_t loop_counter = 0;
    pop(stack, loop_counter);
    WithInsertPoint wip(td.traced_graph_->block());
    auto lc = td.traced_graph_->insertConstant(loop_counter);
    LoopView lv(n);
    td.old_to_new_[lv.currentTripCount()] = lc;
  };

  // uncomment for debugging
  // opt_pn->i_(Symbol::attr("loop_counter"), 1);
  opt_pn->setCallback(optional_profiler);
}

// Similar to how we propagate the mappings for If nodes, we need to propagate
// the mappings from the loop body to the beginning of the block in case we
// run another iteration and to the outputs of the Loop node, for any logic
// downstream that uses the output values of the loop node
static void traceLoop(Node* n, std::shared_ptr<Graph>& graph, TracingData& td) {
  std::vector<Value*> empty_values{};

  // this is a propagation node for block inputs (phi values)
  // these come from either `prim::Loop` inputs or loop body outputs
  {
    auto opt_pn = createProfileIValueNode(graph, empty_values);
    opt_pn->insertBefore(n);
    LoopView lv(n);
    std::function<void(Stack&)> optional_profiler = [n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(td.mutex_);

      // frame_id is unused
      int64_t frame_id = 0;
      pop(stack, frame_id);

      LoopView lv(n);
      TORCH_INTERNAL_ASSERT(
          lv.bodyCarriedInputs().size() == lv.carriedInputs().size());
      for (size_t i = 0; i < lv.bodyCarriedInputs().size(); i++) {
        auto bno = td.old_to_new_.at(lv.carriedInputs()[i]);
        td.old_to_new_[lv.bodyCarriedInputs()[i]] = bno;
        GRAPH_DEBUG(
            "Map ",
            td.old_to_new_[lv.bodyCarriedInputs()[i]]->debugName(),
            " to ",
            bno->debugName());
      }
    };

    // uncomment for debugging
    // opt_pn->i_(Symbol::attr("loop_entry"), 1);
    opt_pn->setCallback(optional_profiler);
  }

  {
    insertTracingNodes(LoopView(n).bodyBlock(), graph, td);
    traceLoopCounter(n, graph, td);
  }

  // this is a propagation node for loop outputs
  {
    auto opt_pn = createProfileIValueNode(graph, empty_values);
    LoopView(n).bodyBlock()->appendNode(opt_pn);

    // opt_pn->i_(Symbol::attr("loop_propagate"), 1);

    std::function<void(Stack&)> optional_profiler = [n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(td.mutex_);

      // frame_id is unused
      int64_t frame_id = 0;
      pop(stack, frame_id);

      LoopView lv(n);

      TORCH_INTERNAL_ASSERT(
          lv.bodyCarriedOutputs().size() == lv.carriedOutputs().size());
      for (size_t i = 0; i < lv.bodyCarriedOutputs().size(); i++) {
        auto bno = td.old_to_new_.at(lv.bodyCarriedOutputs()[i]);
        td.old_to_new_[lv.carriedOutputs()[i]] = bno;
        GRAPH_DEBUG(
            "Map ",
            td.old_to_new_[lv.bodyCarriedOutputs()[i]]->debugName(),
            " to ",
            bno->debugName());
      }
    };

    // uncomment for debugging
    // opt_pn->i_(Symbol::attr("loop_exit"), 1);
    opt_pn->setCallback(optional_profiler);
  }
}

void insertTracingNodes(
    Block* block,
    std::shared_ptr<Graph>& graph,
    TracingData& td) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto n = *it;
    it++;

    GRAPH_DEBUG("Inserting trace for ", getHeader(n));
    if (n->kind() == prim::If) {
      IfView ifv(n);
      createPropNodeForIfBlock(ifv.thenBlock(), n, graph, td);
      createPropNodeForIfBlock(ifv.elseBlock(), n, graph, td);
      continue;
    }

    if (n->kind() == prim::Loop) {
      traceLoop(n, graph, td);
      continue;
    }

    TORCH_INTERNAL_ASSERT(n->blocks().empty());
    auto opt_pn = createProfileIValueNode(graph, n->outputs());
    opt_pn->insertAfter(n);

    // we only use the `opt_pn->node()` to trigger the handler
    // we still capture the actual scripted node `n` we want to trace
    // we look at its inputs, map them to the inputs in the traced graph
    // and create a new node with `traceNode`
    std::function<void(Stack&)> optional_profiler = [n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(td.mutex_);

      // frame_id is unused
      int64_t frame_id = 0;
      pop(stack, frame_id);

      GRAPH_DEBUG("Tracing ", getHeader(n));
      auto tracer = traceNode(n, td, stack);
      auto outputs_size = n->outputs().size();
      auto iivs = pop(stack, outputs_size);
      for (size_t j = 0; j < outputs_size; j++) {
        auto& iiv = iivs[j];
        if (iiv.isTensor()) {
          auto t = iiv.toTensor();
          auto type = t.defined() ? tensorTypeInCurrentExecutionContext(t)
                                  : TensorType::get();
          tracer->outputs().at(j)->setType(type);
        }
      }
    };

    opt_pn->setCallback(optional_profiler);
  }
}
} // namespace

// To trace graph we create a profile node for every one
// in a scripted graph. When a profiled node handler runs
// we insert a new traced node in a trace graph
// If the profiled node handler is called in a loop
// we will have multiple nodes.
// We also maintain the mapping between the outputs of traced
// nodes and the outputs of the node in the scripted graph.
// There are a few subtleties with tracing Ifs and Loops
// discussed above
std::shared_ptr<Graph> TraceGraph(std::shared_ptr<Graph> graph, Stack& stack) {
  TracingData td;
  GRAPH_DUMP("Before Inline:", graph);
  Inline(*graph.get());
  EliminateDeadCode(graph);
  GRAPH_DUMP("After Inline:", graph);
  auto new_g = graph->copy();

  for (auto& inp : new_g->inputs()) {
    auto ni = td.traced_graph_->addInput();
    ni->copyMetadata(inp);
    ni->setType(ni->type());
    td.old_to_new_[inp] = ni;
  }

  // Set type of the graph inputs using the inputs from the stack.
  // This needs to be done before running the interpreter because the stack
  // will only have the outputs after the run.
  for (auto i : c10::irange(stack.size())) {
    if (stack[i].isTensor()) {
      td.traced_graph_->inputs().at(i)->setType(
          tensorTypeInCurrentExecutionContext(stack[i].toTensor()));
    }
  }

  insertTracingNodes(new_g->block(), new_g, td);
  GRAPH_DUMP("Profiling Graph:", new_g);
  Code cd(new_g, "");
  InterpreterState is{cd};
  is.run(stack);
  for (auto out : new_g->outputs()) {
    td.traced_graph_->block()->registerOutput(td.old_to_new_.at(out));
  }

  GRAPH_DUMP("Traced graph:", td.traced_graph_);
  return td.traced_graph_;
}
} // namespace jit
} // namespace torch
