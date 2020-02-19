#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/jit_log.h>

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

static void insertExpand(Value* input, Value* target, Node* parent, size_t i) {
  auto ea = parent->owningGraph()->create(prim::inflate, {input, target});
  ea->insertBefore(parent);
  parent->replaceInput(i, ea->output());
}

static void insertExpands(Block* b) {
  for (auto n : b->nodes()) {
    switch (n->kind()) {
      case aten::add:
      case aten::sub:
      case aten::mul:
      case aten::div: {
        auto x = n->input(0);
        auto y = n->input(1);
        insertExpand(x, y, n, 0);
        insertExpand(y, x, n, 1);
        break;
      }
      default:
        break;
    }

    for (auto ib : n->blocks()) {
      insertExpands(b);
    }
  }
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

int64_t ProfilingRecord::toSymbol(size_t val) {
  if (dims2symbols_.count(val) == 0 /*|| val == 1*/) {
    int64_t new_sym = -dims2symbols_.size() - 1;
    dims2symbols_[val] = new_sym;
    return new_sym;
  }

  return dims2symbols_[val];
}

void ProfilingRecord::convertToStaticShapes(Block* b) {
  for (auto n : b->nodes()) {
    for (auto o : n->outputs()) {
      if (auto tt = o->type()->cast<TensorType>()) {
        if (tt->sizes().size().has_value()) {
          std::vector<c10::optional<int64_t>> symbolWithStaticShapes;
          for (size_t i = 0; i < tt->sizes().size(); i++) {
            auto dim = tt->sizes()[i];
            if (!dim.has_value()) {
              symbolWithStaticShapes.push_back(c10::nullopt);
              continue;
            }
            auto static_size = static_sizes_[*dim];
            symbolWithStaticShapes.push_back(
                static_size.has_value() ? c10::optional<int64_t>(*static_size)
                                        : dim);
          }
          auto symbolStaticType =
              tt->withSymbolicShapes(c10::VaryingShape{symbolWithStaticShapes});
          o->setType(symbolStaticType);
        }
      }
    }
    for (auto ib : n->blocks()) {
      convertToStaticShapes(ib);
    }
  }
}

/*
size_t ProfilingRecord::toDimension(int64_t symbol, size_t new_val) {

  if (symbols2dims_.count(symbol) == 0) {
    symbols2dims_[symbol] = new_val;
    return new_val;
  }

  return symbols2dims_[symbol];

}

std::vector<size_t> ProfilingRecord::mergeSymbolicShapes(VaryingShape& vs,
at::IntArrayRef sizes) { std::vector<c10::optional<int64_t>> new_symbols; for
(auto s : vs) { if (!s.has_value()) { new_symbols.push_back(c10::nullopt);
    }
    else {
      auto dim = toDimension(s.value(), sizes[i]);
      // consider creating a new dim
      new_symbols.push_back() (dim == sizes[i] ? s : c10::nullopt);
    }
  }
}
*/

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
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (t.toTensor().defined()) {
        if (first) {
          // a bit ugly
          auto pttp = tensorTypeInCurrentExecutionContext(t.toTensor());
          auto symbols = fmap(t.toTensor().sizes(), [this](size_t dim) {
            return this->toSymbol(dim);
          });
          GRAPH_DEBUG("pttp = ", *pttp);
          pttp = pttp->withSymbolicShapes(c10::VaryingShape{symbols});
          first = false;
          pno->setType(pttp);
        } else {
          auto type = pno->type()->cast<TensorType>();
          auto pttp = type->merge(t.toTensor(), symbols2dims_);
          pno->setType(pttp);
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
}

void ProfilingRecord::updateStaticSizes(int64_t symbol, size_t dim) {
  if (static_sizes_.count(symbol) == 0) {
    static_sizes_.insert({symbol, c10::optional<size_t>{dim}});
  } else {
    auto prev_size = static_sizes_[symbol];
    if (prev_size.has_value() && *prev_size != dim) {
      static_sizes_[symbol] = c10::nullopt;
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
  static auto const INSERT_EXPANDS = std::getenv("PYTORCH_EXPANDS");
  if (INSERT_EXPANDS) {
    insertExpands(new_g->block());
  }
  pr->instrumentBlock(new_g->block());

  for (auto i : new_g->return_node()->inputs()) {
    if (i->type()->isSubtypeOf(TensorType::get())) {
      pr->insertShapeProfile(new_g->return_node(), i);
    }
  }
  std::function<void(Stack&)> counter = [raw_pr](Stack&) {
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);

    for (auto e : raw_pr->dims2symbols_) {
      raw_pr->updateStaticSizes(e.second, e.first);
    }
    //
    for (auto e : raw_pr->symbols2dims_) {
      raw_pr->updateStaticSizes(e.first, e.second);
    }
    raw_pr->symbols2dims_.clear();
    raw_pr->dims2symbols_.clear();
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
