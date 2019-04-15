#include <torch/csrc/jit/profiling_record.h>

namespace torch {
namespace jit {

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(3) {}

Node* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = profiled_graph_->create(prim::profile, 0);
  for (auto in : inputs) {
    pn->addInput(in);
  }

  callbacks_.push_back(fp);
  auto& stored_fp = callbacks_.back();
  pn->i_(attr::data, reinterpret_cast<int64_t>(&stored_fp));
  return pn;
}

void ProfilingRecord::instrumentBlock(Block* block) {

  // iterating backwards allows us to easily insert profile nodes
  // without affecting an iterator
  for (auto it = block->nodes().rend(); it != block->nodes().rbegin(); --it) {
    auto n = *it;
    for (auto o : n->outputs()) {
      if (!o->type()->isSubclass(TypeKind::TensorType)) {
        continue;
      }

      std::function<void(Stack&)> shape_profiler = [this, o](Stack& stack) {
        IValue t;
        pop(stack, t);
        if (t.isTensor()) {
          auto pttp = ProfiledTensorType::create(t.toTensor());
          std::lock_guard<std::mutex> lock(this->mutex_);
          if (o->type()->isSubclass(TypeKind::ProfiledTensorType)) {
            auto type = o->type()->cast<ProfiledTensorType>();
            o->setType(type->merge(pttp));
          } else {
            o->setType(pttp);
          }
        }
      };

      auto pn = createProfileNode(shape_profiler, {o});
      pn->insertAfter(n);
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
    raw_pr->profiling_count_--;
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  return pr;
}

ProfiledTensorTypePtr ProfilingRecord::toProfiledTensorTypePtr(const IValue& ival)
{
  if (ival.isTensor())
  {
    auto tensor = ival.toTensor();
    return ProfiledTensorType::create(tensor);
  }

  return {nullptr};
}

} //jit
} //torch
