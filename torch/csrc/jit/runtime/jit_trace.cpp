
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/jit_trace.h>

#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <unordered_map>
#include <jit/passes/dead_code_elimination.h>

namespace torch {

namespace jit {

namespace {

struct TracingData {
  std::unordered_map<Value*, Value*> old_to_new_;
  std::shared_ptr<Graph> traced_graph_ = nullptr;

  TracingData() {
    traced_graph_ = std::make_shared<Graph>();
  }
};

Node* traceNode(Node* node, TracingData& td, Stack& stack) {
  GRAPH_DEBUG("Tracing node ", getHeader(node));
  auto* block = td.traced_graph_->block();
  auto env = [&td](Value* v) {
    return td.old_to_new_.at(v);
  };

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

void eraseAllOutputs(Node* opt_pn) {
  for (int i = opt_pn->outputs().size() - 1; i >= 0; i--) {
    opt_pn->eraseOutput(i);
  }
}

void insertTracingNodes(Block*, ProfilingRecord* , TracingData&);

void createPropNodeForIfBlock(
    Block* b,
    Node* n,
    ProfilingRecord* pr,
    TracingData& td) {
  std::vector<Value*> empty_values{};
  auto opt_pn = pr->createProfileIValueNode(empty_values);
  eraseAllOutputs(opt_pn);
  insertTracingNodes(b, pr, td);
  b->appendNode(opt_pn);
  std::function<void(Stack&)> optional_profiler =
      [pr, n, b, &td](Stack& stack) {
        std::lock_guard<std::mutex> lock(pr->mutex_);

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
  //opt_pn->i_(Symbol::attr("propagate"), 1);
  opt_pn->setCallback(optional_profiler);
}

void traceLoopCounter(Node* n, ProfilingRecord* pr, TracingData& td) {
  LoopView lv(n);
  auto opt_pn = pr->createProfileIValueNode(lv.currentTripCount());
  eraseAllOutputs(opt_pn);
  lv.bodyBlock()->prependNode(opt_pn);
  std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);
    // frame_id is unused
    int64_t frame_id = 0;
    pop(stack, frame_id);
    int64_t loop_counter;
    pop(stack, loop_counter);
    WithInsertPoint wip(td.traced_graph_->block());
    auto lc = td.traced_graph_->insertConstant(loop_counter);
    LoopView lv(n);
    td.old_to_new_[lv.currentTripCount()] = lc;
  };

  // uncomment for debugging
  //opt_pn->i_(Symbol::attr("loop_counter"), 1);
  opt_pn->setCallback(optional_profiler);
}

static void traceLoop(Node* n, ProfilingRecord* pr, TracingData& td) {
  std::vector<Value*> empty_values{};
  {
    auto opt_pn = pr->createProfileIValueNode(empty_values);
    eraseAllOutputs(opt_pn);
    opt_pn->insertBefore(n);
    LoopView lv(n);
    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(pr->mutex_);

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
    //opt_pn->i_(Symbol::attr("loop_entry"), 1);
    opt_pn->setCallback(optional_profiler);
  }

  {
    insertTracingNodes(LoopView(n).bodyBlock(), pr, td);
    traceLoopCounter(n, pr, td);
  }

  {
    auto opt_pn = pr->createProfileIValueNode(empty_values);
    eraseAllOutputs(opt_pn);
    LoopView(n).bodyBlock()->appendNode(opt_pn);

    opt_pn->i_(Symbol::attr("loop_propagate"), 1);
    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(pr->mutex_);

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
    //opt_pn->i_(Symbol::attr("loop_exit"), 1);
    opt_pn->setCallback(optional_profiler);
  }
}

void insertTracingNodes(Block* block, ProfilingRecord* pr, TracingData& td) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto n = *it;
    it++;

    GRAPH_DEBUG("Inserting trace for ", getHeader(n));
    if (n->kind() == prim::If) {
      IfView ifv(n);
      createPropNodeForIfBlock(ifv.thenBlock(), n, pr, td);
      createPropNodeForIfBlock(ifv.elseBlock(), n, pr, td);
      continue;
    }

    if (n->kind() == prim::Loop) {
      traceLoop(n, pr, td);
      continue;
    }

    TORCH_INTERNAL_ASSERT(n->blocks().empty());
    auto opt_pn = pr->createProfileIValueNode(n->outputs());
    eraseAllOutputs(opt_pn);
    opt_pn->insertAfter(n);

    std::function<void(Stack&)> optional_profiler = [pr, n, &td](Stack& stack) {
      std::lock_guard<std::mutex> lock(pr->mutex_);

      // frame_id is unused
      int64_t frame_id = 0;
      pop(stack, frame_id);

      GRAPH_DEBUG("Tracing ", getHeader(n));
      auto tracer = traceNode(n, td, stack);
      auto ouputs_size = n->outputs().size();
      auto iivs = pop(stack, ouputs_size);
      for (size_t j = 0; j < ouputs_size; j++) {
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

std::shared_ptr<Graph> TraceGraph(
    std::shared_ptr<Graph> graph,
    Stack& stack) {
  TracingData td;
  GRAPH_DUMP("Before Inline:", graph);
  Inline(*graph.get());
  EliminateDeadCode(graph);
  GRAPH_DUMP("After Inline:", graph);
  auto pr = ProfilingRecord::instrumentGraph(graph);
  for (auto inp : pr->profiled_graph_->inputs()) {
    auto ni = td.traced_graph_->addInput();
    ni->copyMetadata(inp);
    ni->setType(ni->type());
    td.old_to_new_[inp] = ni;
  }
  ProfilingRecord::removeProfileCounter(pr->profiled_graph_->block());
  RemoveProfilingNodes(pr->profiled_graph_);
  insertTracingNodes(pr->profiled_graph_->block(), pr.get(), td);
  GRAPH_DUMP("Profiling Graph:", pr->profiled_graph_);
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);
  for (auto out : pr->profiled_graph_->outputs()) {
    td.traced_graph_->block()->registerOutput(td.old_to_new_.at(out));
  }
  GRAPH_DUMP("Traced graph:", td.traced_graph_);
  return td.traced_graph_;
}
} // namespace jit
} // namespace torch