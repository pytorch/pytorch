#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

kir::Val* getPredicatePerParallelType(
    ParallelType pt,
    const ThreadPredicateMap::SourceMap& source_map) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (pt == ParallelType::BIDx || pt == ParallelType::BIDy ||
      pt == ParallelType::BIDz) {
    auto source = source_map.at(pt);
    TORCH_INTERNAL_ASSERT(!source.empty(), "No predicate source found");
    TORCH_INTERNAL_ASSERT(source.size() == 1, "Multiple sources detected");
    auto src = *source.begin();
    auto flag_name = kir::GridReduction::getPredicateFlagName(src);
    return ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool);
  } else {
    return ir_builder.eqExpr(
        kir::NamedScalar::getParallelIndex(pt), ir_builder.create<kir::Int>(0));
  }
}

kir::Bool* getPredicate(
    const ParallelTypeBitmap& bits,
    const ThreadPredicateMap::SourceMap& source_map) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (bits.none()) {
    return ir_builder.create<kir::Bool>(true);
  }

  kir::Val* pred = nullptr;

  for (const auto& pt_bool : bits.getMap()) {
    if (pt_bool.second) {
      const auto tp = getPredicatePerParallelType(pt_bool.first, source_map);
      pred = (pred == nullptr) ? tp : ir_builder.andExpr(pred, tp);
    }
  }

  TORCH_INTERNAL_ASSERT(pred != nullptr);
  TORCH_INTERNAL_ASSERT(pred->dtype() == DataType::Bool);

  return pred->as<kir::Bool>();
}

void mergeSourceMap(
    ThreadPredicateMap::SourceMap& dst,
    const ThreadPredicateMap::SourceMap& src) {
  for (const auto& kv : src) {
    const auto& src_key = kv.first;
    const auto& src_value = kv.second;
    auto& dst_set = dst[src_key];
    for (const auto& src_tensor : src_value) {
      dst_set.insert(src_tensor);
    }
  }
}

void addToSouceMap(
    ThreadPredicateMap::SourceMap& dst,
    const TensorView* tv,
    const ParallelTypeBitmap& reducton_pred) {
  for (const auto& kv : reducton_pred.getMap()) {
    if (kv.second) {
      ParallelType ptype = kv.first;
      dst[ptype].insert(tv);
    }
  }
}

void maskSouceMap(
    ThreadPredicateMap::SourceMap& src_map,
    const ParallelTypeBitmap& mask) {
  for (const auto& kv : mask.getMap()) {
    if (!kv.second) {
      ParallelType ptype = kv.first;
      src_map[ptype].clear();
    }
  }
}

// A bit of a hack for now for GEMM tiling so we don't fetch tiles multiple
// times. It's safe to do, there may simply be a better place to do it.
ParallelTypeBitmap avoidRedundantWritesToSmem(
    const TensorView* out_tv,
    const ParallelTypeBitmap& pred) {
  auto new_pred = pred;
  if (out_tv->getMemoryType() == MemoryType::Shared) {
    for (size_t i = 0; i < out_tv->nDims(); i++) {
      auto id = out_tv->getComputeAtAxis(i).first;
      if (out_tv->axis(i)->isBroadcast() && id->isThreadDim()) {
        new_pred.set(id->getParallelType(), true);
      }
    }
  }
  return new_pred;
}

} // namespace

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicateMap::updateBitSet(const Expr* expr) {
  FUSER_PERF_SCOPE("ThreadPredicateMap::updateBitSet");

  // Which predicates were set for the inputs
  ParallelTypeBitmap input_preds;

  // Which dims are reductions in inputs
  ParallelTypeBitmap input_reductions;

  // Which dims are bcast in inputs
  ParallelTypeBitmap input_bcasts;

  SourceMap src_map;

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = inp->as<TensorView>();
    TORCH_INTERNAL_ASSERT(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    const auto& pred_and_src = at(tv_inp);

    input_preds |= pred_and_src.pred;

    mergeSourceMap(src_map, pred_and_src.source_map);

    ParallelTypeBitmap id_reductions;
    ParallelTypeBitmap id_bcasts;
    ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        id_ptypes.set(id->getParallelType(), true);
        if (id->isReduction())
          id_reductions.set(id->getParallelType(), true);
        if (id->isBroadcast())
          id_bcasts.set(id->getParallelType(), true);
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (size_t i = 0; i < ParallelTypeBitmap::num_p_type; i++) {
      if (input_reductions[i]) {
        if (id_ptypes[i]) {
          TORCH_INTERNAL_ASSERT(
              id_reductions[i],
              "Mismatched parallelized reductions found on inputs of epxr: ",
              expr);
          TORCH_CHECK(
              !id_bcasts[i],
              "Invalid broadcast and reduction combination, tried to parallelize both with the same thread dim: ",
              inp);
        }
      }
    }

    // Accumulate
    input_reductions |= id_reductions;
    input_bcasts |= id_bcasts;

    if (id_reductions.any()) {
      // add tv_inp as a source
      addToSouceMap(src_map, tv_inp, id_reductions);
    }
  }

  // Update map for this tv, before accumulating to other inputs
  // Add any reductions this id has to any input predicates
  auto output_preds = input_preds | input_reductions;

  // Figure out which dims bcast wants to reset
  const auto bcast_reset_mask = ~(output_preds & input_bcasts);

  // Get rid of any reductions which are bcasted
  output_preds &= bcast_reset_mask;

  // Similarly, drop non-relevant source tensors
  maskSouceMap(src_map, bcast_reset_mask);

  // Run through outputs and set bitset predicates
  for (auto* out : expr->outputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(out)) {
      TORCH_INTERNAL_ASSERT(find(tv) == end());
      insert(tv, avoidRedundantWritesToSmem(tv, output_preds), src_map);
    }
  }
}

ThreadPredicateMap::ThreadPredicateMap(Fusion* fusion) : fusion_(fusion) {
  FUSER_PERF_SCOPE("ThreadPredicateMap");

  // Initialize mapping for input tensors
  for (auto inp : fusion_->inputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(inp)) {
      insert(tv, ParallelTypeBitmap(), SourceMap());
    }
  }
  for (auto expr : fusion_->exprs(true)) {
    updateBitSet(expr);
  }
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::find(
    const TensorView* tv) const {
  return thread_predicates_.find(tv);
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::end() const {
  return thread_predicates_.end();
}

const ThreadPredicateMap::PredAndSource& ThreadPredicateMap::at(
    const TensorView* tv) const {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::PredAndSource& ThreadPredicateMap::at(
    const TensorView* tv) {
  return thread_predicates_.at(tv);
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const ParallelTypeBitmap& pred,
    const SourceMap& src_map) {
  insert(tv, {pred, src_map});
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const PredAndSource& pred_and_src) {
  thread_predicates_.insert({tv, pred_and_src});
}

kir::Bool* ThreadPredicateMap::getExpr(const TensorView* out_tv) const {
  TORCH_INTERNAL_ASSERT(find(out_tv) != end(), "Couldn't find ", out_tv);
  const auto& pred_and_src = at(out_tv);
  return getPredicate(pred_and_src.pred, pred_and_src.source_map);
}

void ThreadPredicateMap::print() const {
  std::cout << "\nThreadPredicateMap\n";
  std::cout << "--------------------------------\n";
  for (const auto& kv : thread_predicates_) {
    std::cout << "T" << kv.first->name() << " {";
    // ParallelTypeBitmap
    for (auto ptkv : kv.second.pred.getMap()) {
      if (ptkv.second) {
        std::cout << " " << ptkv.first;
      }
    }
    std::cout << " }\n";
    // SourceMap
    for (const auto& pkv : kv.second.source_map) {
      std::cout << "  " << pkv.first << " : [";
      for (auto tv : pkv.second) {
        std::cout << " T" << tv->name();
      }
      std::cout << " ]\n";
    }
  }
  std::cout << "--------------------------------\n\n";
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
