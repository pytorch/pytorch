#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

namespace torch {
namespace jit {

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(3) {}

ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

void ProfilingRecord::instrumentBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (auto i : n->inputs()) {
      if (!i->type()->isSubtypeOf(TensorType::get()) ||
          i->node()->kind() == prim::profile) {
        continue;
      }

      auto pn = createProfileNode(nullptr, {i});
      auto pno = pn->addOutput();
      pno->setType(TensorType::get());
      std::function<void(Stack&)> shape_profiler = [this, pno](Stack& stack) {
        IValue t;
        pop(stack, t);
        if (t.isTensor()) {
          auto pttp = TensorType::create(t.toTensor());
          std::lock_guard<std::mutex> lock(this->mutex_);
          if (auto type = pno->type()->cast<TensorType>()) {
            pno->setType(type->merge(pttp));
          } else {
            pno->setType(pttp);
          }
        }
        // passing t through
        push(stack, t);
      };

      pn->setCallback(shape_profiler);
      pn->insertBefore(n);
      n->replaceInputWith(i, pn->output());
    }

    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }
}

std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  auto new_g = graph->copy();
  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  auto raw_pr = pr.get();

  pr->instrumentBlock(new_g->block());
  std::function<void(Stack&)> counter = [raw_pr](Stack&) {
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);
    if (raw_pr->profiling_count_ > 0)
    {
        raw_pr->profiling_count_--;
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  return pr;
}

TensorTypePtr ProfilingRecord::toTensorTypePtr(const IValue& ival) {
  if (ival.isTensor()) {
    auto tensor = ival.toTensor();
    return TensorType::create(tensor);
  }

  return {nullptr};
}

} // namespace jit
} // namespace torch
