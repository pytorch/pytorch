#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(getNumProfiledRuns()) {}

ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

void ProfilingRecord::insertShapeProfile(Node* n, Value* i) {
  auto pn = createProfileNode(nullptr, {i});
  auto pno = pn->addOutput();
  pno->setType(TensorType::get());
  std::function<void(Stack&)> shape_profiler = [this,
                                                pno](Stack& stack) mutable {
    GRAPH_DEBUG("Profiling %", pno->debugName(), " addr = ", pno);
    int64_t frame_id;
    pop(stack, frame_id);
    IValue t;
    pop(stack, t);
    if (t.isTensor()) {
      if (t.toTensor().defined()) {
        auto pttp = tensorTypeInCurrentExecutionContext(t.toTensor());
        GRAPH_DEBUG("pttp = ", *pttp);
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (auto type = pno->type()->cast<TensorType>()) {
          auto result = this->seen_.insert(pno);
          if (!result.second) {
            GRAPH_DEBUG("Merging ", *pttp, " with ", *type);
            pttp = pttp->merge(type);
          }
          pno->setType(pttp);
          GRAPH_DEBUG("Setting a new type ", *pttp);
        }
      } else {
        pno->setType(TensorType::get()->withUndefined());
      }
    }

    // passing t through
    push(stack, t);
  };

  pn->setCallback(shape_profiler);
  pn->insertBefore(n);
  n->replaceInputWith(i, pn->output());
}

void ProfilingRecord::instrumentBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (auto i : n->inputs()) {
      if (!i->type()->isSubtypeOf(TensorType::get()) ||
          i->node()->kind() == prim::profile) {
        continue;
      }

      insertShapeProfile(n, i);
    }

    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }

  // inserting profile nodes on block outputs
  // allows us to eliminate more guards as
  // the use of a guard is now in the same
  // block as opposed to being separated from
  // the definition by block boundaries
  for (auto i : block->return_node()->inputs()) {
    if (i->type()->isSubtypeOf(TensorType::get())) {
      insertShapeProfile(block->return_node(), i);
    }
  }
}

std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  auto new_g = graph->copy();
  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  auto raw_pr = pr.get();
  ClearProfilingInformation(new_g);
  pr->instrumentBlock(new_g->block());

  std::function<void(Stack&)> counter = [raw_pr](Stack& stack) {
    int64_t frame_id;
    pop(stack, frame_id);
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);
    if (raw_pr->profiling_count_ > 0) {
      raw_pr->profiling_count_--;
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  return pr;
}

} // namespace jit
} // namespace torch
