#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

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

static void unprofileGraphInputs(const std::shared_ptr<Graph> &graph) {
  for (auto i : graph->inputs()) {
    if (i->type()->isSubtypeOf(TensorType::get())) {
      i->setType(unshapedType(i->type()));
    }
  }
}

static void unprofileBlock(Block* start_block) {
  std::vector<Block*> stack;
  stack.push_back(start_block);

  while (!stack.empty()) {
    Block* block = stack.back();
    stack.pop_back();

    for (auto n : block->nodes()) {
      for (auto o : n->outputs()) {
        if (o->type()->isSubtypeOf(TensorType::get())) {
          o->setType(unshapedType(o->type()));
        }
      }
      stack.insert(stack.end(), n->blocks().begin(), n->blocks().end());
    }
  }
}

void ProfilingRecord::insertShapeProfile(Node *n, Value *i) {

  auto pn = createProfileNode(nullptr, {i});
  auto pno = pn->addOutput();
  bool first = true;
  pno->setType(TensorType::get());
  std::function<void(Stack &)> shape_profiler = [this, pno,
                                                 first](Stack &stack) mutable {
    IValue t;
    pop(stack, t);
    if (t.isTensor()) {

      if (t.toTensor().defined()) {
        auto pttp = tensorTypeInCurrentExecutionContext(t.toTensor());
        std::lock_guard<std::mutex> lock(this->mutex_);
        if (auto type = pno->type()->cast<TensorType>()) {
          if (!first) {
            pttp = pttp->merge(type);
          }
          pno->setType(pttp);
          first = false;
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

void ProfilingRecord::instrumentBlock(Block *block) {
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
  unprofileGraphInputs(new_g);
  unprofileBlock(new_g->block());
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

} // namespace jit
} // namespace torch
