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

int64_t ProfilingRecord::toSymbol(
    int64_t val,
    std::map<int64_t, int64_t>& dims2symbols) {
  if (dims2symbols.count(val) == 0 /*|| val == 1*/) {
    int64_t new_sym = getNewSymbol();
    dims2symbols[val] = new_sym;
    return new_sym;
  }

  return dims2symbols[val];
}

std::vector<int64_t> ProfilingRecord::mergeSymbolicShapes(
    at::IntArrayRef new_sizes,
    c10::VaryingShape sym_shapes,
    std::map<int64_t, int64_t>& dims2symbols,
    std::map<int64_t, int64_t>& symbols2dims,
    std::map<int64_t, std::map<int64_t, int64_t>> split_symbols) {
  std::vector<int64_t> new_symbols;
  if (new_sizes.size() == sym_shapes.size()) {
    for (size_t i = 0; i < new_sizes.size(); i++) {
      auto symbol = sym_shapes[i];
      TORCH_INTERNAL_ASSERT(
          symbol.has_value(), "should always have some symbol");
      if (*symbol >= 0) {
        if (*symbol == new_sizes[i]) {
          new_symbols.push_back(new_sizes[i]);
        } else {
          // also works for symbols from previous runs
          // in which case we are renaming a symbol from previous run
          // to a symbol in the current run
          auto new_sym = toSymbol(new_sizes[i], dims2symbols);
          new_symbols.push_back(new_sym);
        }
      } else if (symbols2dims.count(symbol.value()) == 0) {
        symbols2dims[symbol.value()] = new_sizes[i];
        new_symbols.push_back(*symbol);
      } else {
        if (symbols2dims[symbol.value()] == new_sizes[i]) {
          new_symbols.push_back(*symbol);
        } else {
          auto& symbol_subsets = split_symbols[symbol.value()];
          if (symbol_subsets.count(new_sizes[i])) {
            new_symbols.push_back(symbol_subsets[new_sizes[i]]);
          } else {
            int64_t new_sym = getNewSymbol();
            symbol_subsets.insert({new_sizes[i], new_sym});
            new_symbols.push_back(new_sym);
          }
        }
      }
    }
  }

  return new_symbols;
}

void ProfilingRecord::insertShapeProfile(Node *n, Value *i) {

  auto pn = createProfileNode(nullptr, {i});
  auto pno = pn->addOutput();
  bool first = true;
  pno->setType(TensorType::get());
  std::function<void(Stack&)> shape_profiler = [this, pno, first](
                                                   Stack& stack) mutable {
    int64_t frame_id;
    pop(stack, frame_id);
    IValue t;
    pop(stack, t);
    if (t.isTensor()) {
      std::lock_guard<std::mutex> lock(this->mutex_);
      auto& record = profiling_records_[frame_id];
      if (t.toTensor().defined()) {
        auto pttp = tensorTypeInCurrentExecutionContext(t.toTensor());
        if (first) {
          first = false;
          record.symbolic_shapes_.insert({pno, pttp});
          GRAPH_DEBUG(
              "In run ",
              frame_id,
              " annotating %",
              pno->debugName(),
              " with ",
              *pttp);
        } else {
          auto type = record.symbolic_shapes_.at(pno);
          auto new_sym_shapes = this->mergeSymbolicShapes(
              t.toTensor().sizes(),
              type->sizes(),
              record.dims2symbols_,
              record.symbols2dims_,
              record.split_symbols_);
          pttp = type->merge(pttp)->withSymbolicShapes(
              c10::VaryingShape{new_sym_shapes});

          GRAPH_DEBUG(
              "In run ",
              frame_id,
              "merging for %",
              pno->debugName(),
              " into ",
              *pttp);
          record.symbolic_shapes_[pno] = pttp;
        }
      } else {
        record.symbolic_shapes_[pno] = TensorType::get()->withUndefined();
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
  std::function<void(Stack&)> counter = [raw_pr](Stack& stack) {
    int64_t frame_id;
    pop(stack, frame_id);

    // const auto& it_dims2symbols = *it.dims2symbols_;
    // const auto& it_symbols2dims = *it.symbols_2dims;
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);

    if (raw_pr->profiling_count_ > 0) {
      raw_pr->profiling_count_--;
    }

    if (raw_pr->profiling_count_ == 0) {
      // merge profiling information from all runs

      GRAPH_DEBUG("Collected ", raw_pr->profiling_records_.size(), " records");
      auto it = raw_pr->profiling_records_.begin();
      auto first_run = it->second;
      it++;
      // merge profiling information from next runs into the first one
      for (; it != raw_pr->profiling_records_.end(); it++) {
        first_run.symbols2dims_.clear();
        first_run.dims2symbols_.clear();
        first_run.split_symbols_.clear();
        for (auto e : it->second.symbolic_shapes_) {
          if (first_run.symbolic_shapes_.count(e.first) == 0) {
            first_run.symbolic_shapes_[e.first] = e.second;
            auto type = first_run.symbolic_shapes_[e.first];
            if (e.second->sizes().size().has_value()) {
              std::vector<int64_t> new_symbols;
              for (size_t i = 0; i < e.second->sizes().size(); i++) {
                auto old_symbol = type->sizes()[i];
                new_symbols.push_back(
                    raw_pr->toSymbol(*old_symbol, first_run.dims2symbols_));
              }
              GRAPH_DEBUG(
                  "Merging ",
                  *e.second,
                  " of run ",
                  it->first,
                  " into ",
                  *type);
              auto new_type =
                  type->withSymbolicShapes(c10::VaryingShape{new_symbols});
              GRAPH_DEBUG("Result (if type absent) : ", *new_type);
              first_run.symbolic_shapes_[e.first] = new_type;
            }
          } else {
            auto concrete_sizes = e.second->sizes().concrete_sizes();
            if (concrete_sizes.has_value()) {
              auto type = first_run.symbolic_shapes_[e.first];
              auto new_shape = raw_pr->mergeSymbolicShapes(
                  *concrete_sizes,
                  type->sizes(),
                  first_run.dims2symbols_,
                  first_run.symbols2dims_,
                  first_run.split_symbols_);
              GRAPH_DEBUG(
                  "Merging ",
                  *e.second,
                  " of run ",
                  it->first,
                  " into ",
                  *type);
              type = type->merge(e.second);
              type = type->withSymbolicShapes(c10::VaryingShape(new_shape));
              GRAPH_DEBUG("Result : ", *type);
              first_run.symbolic_shapes_[e.first] = type;
            } else {
              TORCH_INTERNAL_ASSERT(false, "NYI");
            }
          }
        }
      }

      // update types in the graph
      for (auto e : first_run.symbolic_shapes_) {
        e.first->setType(e.second);
      }
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  return pr;
}

} // namespace jit
} // namespace torch
