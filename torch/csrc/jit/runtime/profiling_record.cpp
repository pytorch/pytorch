#include <torch/csrc/jit/runtime/profiling_record.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include "jit/ir/ir.h"

namespace torch {
namespace jit {

bool ShapeSymbolTable::bindSymbolicShapes(
    at::IntArrayRef new_sizes,
    const c10::SymbolicShape& sym_shapes) {
  if (!sym_shapes.rank().has_value()) {
    return true;
  }
  if (*sym_shapes.rank() != new_sizes.size()) {
    return false;
  }
  for (size_t i = 0; i < new_sizes.size(); i++) {
    auto symbol = (*sym_shapes.sizes())[i];
    if (!symbol.is_static()) {
      continue;
    }

    if (!isBound(symbol)) {
      assign(symbol, new_sizes[i]);
      continue;
    }

    if (getValue(symbol) != new_sizes[i]) {
      return false;
    }
  }
  return true;
}

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

ProfileOptionalOp* ProfilingRecord::createProfileOptionalNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOptionalOp(profiled_graph_.get(), fp);
  pn->i_(attr::num_present, 0);
  pn->i_(attr::num_none, 0);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

static void unprofileGraphInputs(const std::shared_ptr<Graph>& graph) {
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

c10::SymbolicShape ProfilingRecord::mergeSymbolicShapes(
    const c10::SymbolicShape& new_sizes,
    const c10::SymbolicShape& sym_shapes,
    SetPartitioningHelper& partition_helper) {
  std::vector<c10::ShapeSymbol> new_symbols;
  TORCH_INTERNAL_ASSERT(
      new_sizes.rank().has_value() && sym_shapes.rank().has_value() &&
      *new_sizes.rank() == *sym_shapes.rank());

  for (size_t i = 0; i < *new_sizes.rank(); i++) {
    if (!(*sym_shapes.sizes())[i].is_static() ||
        !(*new_sizes.sizes())[i].is_static()) {
      new_symbols.emplace_back();
      continue;
    }
    auto symbol = (*sym_shapes.sizes())[i];
    Dimension new_size = (*new_sizes.sizes())[i].static_size();
    GRAPH_DEBUG("Merging symbol ", symbol);
    auto new_sym = partition_helper.partitionSetByDimension(new_size, symbol);
    new_symbols.emplace_back(new_sym);
  }

  return c10::SymbolicShape(new_symbols);
}

void ProfilingRecord::profileOptionalValue(Value* none_output) {
  c10::Dict<std::string, int64_t> noneCountsDict;
  noneCountsDict.insert("num_none", 0);
  noneCountsDict.insert("num_present", 0);
  IValue init_val(noneCountsDict);

  auto combine = [](const IValue& acc, const IValue& val) {
    auto noneCounts =
        c10::impl::toTypedDict<std::string, int64_t>(acc.toGenericDict());
    if (val.isNone()) {
      noneCounts.insert_or_assign("num_none", noneCounts.at("num_none") + 1);
    } else {
      noneCounts.insert_or_assign(
          "num_present", noneCounts.at("num_present") + 1);
    }
    return IValue{noneCounts};
  };
  insertProfileIValueOp(none_output, combine, init_val, "none_counts");
}

void ProfilingRecord::profileListValue(Value* none_output) {

  IValue init_val;

  auto combine = [](const IValue& acc, const IValue& val) {
    if (val.isNone()) {
      return val;
    } else if (acc == val) {
        return acc;
    } else {
        return IValue{std::vector<int64_t>{}};
    }
  };
  insertProfileIValueOp(none_output, combine, init_val, "axes_or_shape");
}

void ProfilingRecord::insertProfileIValueOp(
    Value* none_output,
    std::function<IValue(const IValue& acc, const IValue& val)> combine,
    const IValue& init,
    const std::string& attr_name) {
  auto pn = new ProfileIValueOp(none_output->node()->owningGraph(), {nullptr});
  pn->addInput(none_output);
  auto pno = pn->addOutput();
  pn->insertAfter(none_output->node());
  none_output->replaceAllUsesAfterNodeWith(pn, pno);
  pn->ival_(Symbol::attr(attr_name), init);
  pno->setType(none_output->type());

  std::function<void(Stack&)> wrapper =
      [this, pn, combine, attr_name](Stack& stack) {
        int64_t frame_id = 0;
        pop(stack, frame_id);
        IValue val;
        pop(stack, val);
        std::lock_guard<std::mutex> lock(this->mutex_);
        auto old_val = pn->ival(Symbol::attr(attr_name));
        auto new_val = combine(old_val, val);
        pn->ival_(Symbol::attr(attr_name), new_val);
        GRAPH_DEBUG(
            "old_val = ",
            old_val,
            " val = ",
            val,
            "Combined value = ",
            new_val);
        push(stack, val);
      };

  pn->setCallback(wrapper);
}

void ProfilingRecord::insertShapeProfile(Node* n, size_t offset) {
  Value* i = n->input(offset);
  auto pn = createProfileNode(nullptr, {i});
  auto pno = pn->addOutput();
  pn->ty_(attr::profiled_type, TensorType::get());
  pno->setType(TensorType::get());
  std::function<void(Stack&)> shape_profiler = [this, pno](Stack& stack) {
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue v;
    pop(stack, v);
    if (v.isTensor()) {
      std::lock_guard<std::mutex> lock(this->mutex_);
      auto& profiled_types = profiled_types_per_frame_[frame_id];
      auto t = v.toTensor();
      if (t.defined()) {
        auto pttp = tensorTypeInCurrentExecutionContext(t);
        GRAPH_DEBUG(
            "In run ",
            frame_id,
            " annotating %",
            pno->debugName(),
            " with ",
            *pttp);
        if (profiled_types.count(pno) == 0) {
          profiled_types.insert({pno, pttp});
        } else {
          auto type = profiled_types.at(pno);
          GRAPH_DEBUG("Existing type for %", pno->debugName(), " ", *type);
          pttp = type->merge(pttp);
          GRAPH_DEBUG("Result for %", pno->debugName(), " ", *pttp);
          profiled_types[pno] = pttp;
        }
      } else {
        profiled_types[pno] = TensorType::get()->withUndefined();
      }
    }
    // passing t through
    push(stack, v);
  };

  pn->setCallback(shape_profiler);
  pn->insertBefore(n);
  n->replaceInput(offset, pn->output());
}

bool needsProfiledInputs(Node* n) {
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    // specialize_autogradzero
    case prim::AutogradAdd:
    case prim::AutogradAnyNonZero:
    case prim::AutogradAllNonZero:
    case prim::AutogradAllZero:
    case prim::AutogradZero:
    // peephole
    case aten::dim:
    case aten::size:
    case aten::expand:
    case prim::dtype:
    case prim::device:
    case prim::is_cuda:
    case aten::is_floating_point:
    case aten::type_as:
    // TODO: hack to make `test_lstm_gates_permutations_cuda`
    // pass.
    case aten::t:
    case aten::mm:
      return true;
    default:
      return false;
  }
}

bool needsProfiledOutput(Node* n) {
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    case prim::AutogradAdd:
    case prim::AutogradZero:
      return true;
    default:
      return false;
  }
}

void ProfilingRecord::removeProfileCounter(Block* b) {
  for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
    auto n = *it;
    if (n->kind() == prim::profile && n->inputs().size() == 0) {
      it.destroyCurrent();
      // there is only one counter node
      return;
    } else {
      it++;
    }
  }
}

bool hasKindUses(Value* v, NodeKind kind) {
  return std::any_of(v->uses().begin(), v->uses().end(), [kind](const Use& use) {
    return use.user->kind() == kind;
  });
}

bool hasGradSumToSizeUses(Value* v) {
  return hasKindUses(v, aten::_grad_sum_to_size);
}

bool hasSumUses(Value* v) {
  return hasKindUses(v, aten::sum);
}


void ProfilingRecord::instrumentBlock(Block* block) {

  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (size_t offset = 0; offset < n->inputs().size(); offset++) {
      auto i = n->input(offset);
      if (i->type()->kind() == c10::TypeKind::TensorType &&
          (needsProfiledInputs(n) || needsProfiledOutput(i->node()))) {
        insertShapeProfile(n, offset);
      }

      if (i->type()->cast<OptionalType>() && hasGradSumToSizeUses(i)) {
        // here we are profile the definition instead of the use,
        // because we are only optimizing in the case of a None value which is
        // immutable
        profileOptionalValue(i);
      }

      GRAPH_DEBUG("before %", i->debugName(), " isList ", (i->type()->cast<OptionalType>() || i->type()->cast<ListType>()), ", hasSumUses ", hasSumUses(i));
      i = n->input(offset);
      GRAPH_DEBUG("after %", i->debugName(), " isList ", (i->type()->cast<OptionalType>() || i->type()->cast<ListType>()), ", hasSumUses ", hasSumUses(i));
      if ((i->type()->cast<OptionalType>() || i->type()->cast<ListType>()) && (hasSumUses(i) || hasGradSumToSizeUses(i))) {
        // get a profiled optional
        profileListValue(i);
      }
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
  for (size_t offset = 0; offset < block->return_node()->inputs().size();
       offset++) {
    auto i = block->return_node()->input(offset);
    if (i->type()->isSubtypeOf(TensorType::get())) {
      insertShapeProfile(block->return_node(), offset);
    }
  }
}

void ProfilingRecord::removeProfilingNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile || it->kind() == prim::profile_optional ||
        it->kind() == prim::profile_ivalue) {
      it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes(ib);
      }
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

  std::function<void(Stack&)> counter = [raw_pr](Stack& stack) {
    int64_t frame_id = 0;
    pop(stack, frame_id);

    std::lock_guard<std::mutex> lock(raw_pr->mutex_);

    if (raw_pr->profiling_count_ > 0) {
      raw_pr->profiling_count_--;
    }

    // merge profiling information from all runs
    if (raw_pr->profiling_count_ == 0) {
      GRAPH_DEBUG(
          "Collected ",
          raw_pr->profiled_types_per_frame_.size(),
          " records for run ",
          frame_id);

      if (raw_pr->profiled_types_per_frame_.size() == 0) {
        return;
      }

      // the key is a frame id
      // the value is a mapping from a Value in a graph
      // to a profiled TensorType
      // we make a copy of profiling information from the very first run
      // and use it for building the symbol sets
      auto profiled_types_iter = raw_pr->profiled_types_per_frame_.begin();
      auto merged_profiled_types = profiled_types_iter->second;
      profiled_types_iter++;

      // merge profiling information from next runs into the first one
      for (; profiled_types_iter != raw_pr->profiled_types_per_frame_.end();
           profiled_types_iter++) {
        SetPartitioningHelper partition_helper;
        for (const auto& val_type_pair : profiled_types_iter->second) {
          if (merged_profiled_types.count(val_type_pair.first) == 0) {
            merged_profiled_types[val_type_pair.first] = val_type_pair.second;
          } else {
            auto type = merged_profiled_types[val_type_pair.first];
            auto merged_type = type->merge(val_type_pair.second);
            if (merged_type->sizes().size().has_value()) {
              auto new_shape = raw_pr->mergeSymbolicShapes(
                  val_type_pair.second->symbolic_sizes(),
                  type->symbolic_sizes(),
                  partition_helper);
              GRAPH_DEBUG(
                  "Merging ",
                  *val_type_pair.second,
                  " of run ",
                  profiled_types_iter->first,
                  " into ",
                  *type);
              merged_type = type->withSymbolicShapes(new_shape);
              GRAPH_DEBUG("Result : ", *merged_type);
              merged_profiled_types[val_type_pair.first] = merged_type;
            } else {
              // reset symbolic shapes when ranks are different
              type = type->merge(val_type_pair.second);
              merged_profiled_types[val_type_pair.first] = type;
            }
          }
        }
      }

      // update types in the graph
      for (auto val_type_pair : merged_profiled_types) {
        val_type_pair.first->node()->ty_(
            attr::profiled_type, val_type_pair.second);
      }
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  GRAPH_DUMP("Instrumented Graph: ", new_g);
  return pr;
}

} // namespace jit
} // namespace torch
