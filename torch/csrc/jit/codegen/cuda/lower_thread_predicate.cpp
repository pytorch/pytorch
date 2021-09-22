#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

kir::Bool* getPredicatePerParallelType(
    ParallelType pt,
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  kir::SimplifyingIrBuilder ir_builder(GpuLower::current()->kernel());

  auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);

  // If pt is not used or is proven to be one, no need to predicate.
  if (pt_dim == nullptr || pt_dim->isOneInt()) {
    return ir_builder.trueVal();
  }

  // When BID needs to be predicated, it means either BID == 1, or if
  // there's a corresponding source_map entry, that means it's an
  // output of a grid reduction and the predicate flag is stored in
  // the special variable for each grid reduction expression.
  if (isParallelTypeBlockDim(pt) && pred_info.limited_types.get(pt)) {
    auto source_it = pred_info.source_map.find(pt);
    TORCH_INTERNAL_ASSERT(
        source_it != pred_info.source_map.end(),
        "Source map not found for ",
        pt);
    const auto& source = source_it->second;
    TORCH_INTERNAL_ASSERT(!source.empty(), "No predicate source found");
    kir::Val* pred = ir_builder.trueVal();
    for (auto src : source) {
      auto flag_name = kir::GridReduction::getPredicateFlagName(src);
      auto src_pred =
          ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool);
      pred = ir_builder.andExpr(pred, src_pred);
    }
    // pred can be just a NamedScalar because of the simplification by
    // the simplifying IR build. To return Bool always, create a set
    // op to Bool and return its output.
    if (pred->isA<kir::NamedScalar>()) {
      return ir_builder.setExpr(pred)->as<kir::Bool>();
    } else {
      return pred->as<kir::Bool>();
    }
  }

  // By default, only thread/block of index 0 executes the computation
  return ir_builder
      .eqExpr(
          kir::NamedScalar::getParallelIndex(pt),
          ir_builder.create<kir::Int>(0))
      ->as<kir::Bool>();
}

kir::Bool* getPredicateFromPredicateInfo(
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  kir::SimplifyingIrBuilder ir_builder(GpuLower::current()->kernel());

  const auto pred_types = pred_info.limited_types | pred_info.redundant_types;

  if (pred_types.none()) {
    return ir_builder.trueVal();
  }

  kir::Bool* pred = nullptr;

  for (const auto pt : pred_types) {
    const auto tp = getPredicatePerParallelType(pt, pred_info);
    pred = ir_builder.andExpr(pred, tp)->as<kir::Bool>();
  }

  TORCH_INTERNAL_ASSERT(pred != nullptr);

  return pred;
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
  for (const auto pt : reducton_pred) {
    dst[pt].insert(tv);
  }
}

void maskSouceMap(
    ThreadPredicateMap::SourceMap& src_map,
    const ParallelTypeBitmap& mask) {
  for (const auto pt : kParallelTypeThreads) {
    if (!mask.get(pt)) {
      src_map[pt].clear();
    }
  }
}

// Build redundant predicate flags. Will be stored as
// PredicateInfo.redundant_types for the given tensor.
ParallelTypeBitmap avoidRedundantWrites(const TensorView* out_tv) {
  // If the memory type is Local, it's fine to write into it always as
  // it's thread local. If it's Global, it's also fine to let each
  // thread do its own write, unless out_tv is an output of a
  // reduction. Reduction reads from and writes to the tensor, so the
  // result would be incorrect if the buffer is shared by redundant
  // threads.
  const bool is_reduction = out_tv->definition()->isA<ReductionOp>() ||
      out_tv->definition()->isA<WelfordOp>();
  if (!(out_tv->getMemoryType() == MemoryType::Shared ||
        (out_tv->getMemoryType() == MemoryType::Global && is_reduction))) {
    return ParallelTypeBitmap();
  }
  ParallelTypeBitmap pred;
  // Track which TID types are not used to find redundant parallel
  // types. Only TID types are checked as the tensor is on shared
  // memory.
  ParallelTypeBitmap unused_types;
  // Initially all types are conservatively assumed to be used.
  unused_types = ~unused_types;
  for (auto out_tv_id : out_tv->domain()->domain()) {
    auto pt = out_tv_id->getParallelType();
    if (!isParallelTypeThread(pt)) {
      continue;
    }
    // If the axis is a broadcast domain and is parallelized by TID,
    // it is sufficient to use just one thread since the tensor is on
    // shared memory.
    if (out_tv->getMemoryType() == MemoryType::Shared &&
        out_tv_id->isBroadcast() && isParallelTypeThreadDim(pt)) {
      pred.set(pt);
    }
    unused_types.clear(pt);
  }

  const auto& par_dim_map = GpuLower::current()->parallelDimensionMap();

  for (const auto pt : unused_types) {
    // For shared memory tensors, unused BID isn't redundant
    if (isParallelTypeBlockDim(pt) &&
        out_tv->getMemoryType() == MemoryType::Shared) {
      continue;
    }
    // If the pt is not used or is proven to be one, it is not
    // really redundant.
    auto pt_dim = par_dim_map.get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    pred.set(pt);
  }

  return pred;
}

// If tv is an output of a reduction with unused parallel types, those
// unused parallel types need to be predicated if the tensor is on
// global memory.
ParallelTypeBitmap getReductionPredicateForUnusedParallelTypes(
    const TensorView* tv,
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  auto tv_def = tv->definition();
  if (!(tv_def && (tv_def->isA<ReductionOp>() || tv_def->isA<WelfordOp>()) &&
        tv->getMemoryType() == MemoryType::Global)) {
    return {};
  }

  // Unused types are set as redundant types of tv
  return pred_info.redundant_types;
}

} // namespace

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicateMap::updateBitSet(const Expr* expr) {
  FUSER_PERF_SCOPE("GpuLower::Lower::ThreadPredicateMap::updateBitSet");

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

    // Change for welford Op, we want the users of all outputs of welfordOp
    //  to use a single predicate name.
    if (auto tv_def = tv_inp->definition()) {
      if (auto wop = dynamic_cast<WelfordOp*>(tv_def)) {
        tv_inp = wop->out()->as<TensorView>();
      }
    }

    TORCH_INTERNAL_ASSERT(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    const auto& pred_and_src = at(tv_inp);

    input_preds |= pred_and_src.limited_types;

    mergeSourceMap(src_map, pred_and_src.source_map);

    ParallelTypeBitmap id_reductions;
    ParallelTypeBitmap id_bcasts;
    ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        id_ptypes.set(id->getParallelType());
        if (id->isReduction())
          id_reductions.set(id->getParallelType());
        if (id->isBroadcast())
          id_bcasts.set(id->getParallelType());
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (const auto i : c10::irange(ParallelTypeBitmap::kNumParallelTypes)) {
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

    id_reductions |=
        getReductionPredicateForUnusedParallelTypes(tv_inp, at(tv_inp));

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
  for (auto* out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    TORCH_INTERNAL_ASSERT(find(out_tv) == end());
    auto redundant_types = avoidRedundantWrites(out_tv);
    insert(out_tv, output_preds, src_map, redundant_types);
  }
}

void ThreadPredicateMap::build(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::ThreadPredicateMap");

  // Initialize mapping for input tensors
  for (auto inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(inp)) {
      insert(tv, ParallelTypeBitmap(), SourceMap(), ParallelTypeBitmap());
    }
  }
  for (auto expr : fusion->exprs()) {
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

const ThreadPredicateMap::PredicateInfo& ThreadPredicateMap::at(
    const TensorView* tv) const {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::PredicateInfo& ThreadPredicateMap::at(
    const TensorView* tv) {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::PredicateInfo ThreadPredicateMap::getPredicateInfo(
    const TensorView* tv) const {
  auto pred_info = thread_predicates_.at(tv);
  // Do not predicate a paralell type if it is a parallel bcast domain
  if (auto bop = dynamic_cast<BroadcastOp*>(tv->definition())) {
    auto parallel_bcast = getParallelBroadcastDomains(tv);
    pred_info.limited_types ^= parallel_bcast;
  }
  return pred_info;
}

ParallelTypeBitmap ThreadPredicateMap::getPredicatedParallelTypes(
    const TensorView* tv) const {
  auto pred_info = getPredicateInfo(tv);
  return pred_info.limited_types | pred_info.redundant_types;
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const ParallelTypeBitmap& valid_types,
    const SourceMap& src_map,
    const ParallelTypeBitmap& redundant_types) {
  insert(tv, {valid_types, src_map, redundant_types});
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const PredicateInfo& pred_and_src) {
  thread_predicates_.insert({tv, pred_and_src});
}

kir::Bool* ThreadPredicateMap::getPredicate(const TensorView* tv) const {
  TORCH_INTERNAL_ASSERT(find(tv) != end(), "Couldn't find ", tv);
  auto pred_info = getPredicateInfo(tv);
  return getPredicateFromPredicateInfo(pred_info);
}

ParallelTypeBitmap ThreadPredicateMap::getParallelBroadcastDomains(
    const TensorView* tv) const {
  // If no pred is found for tv, no predicate is necessary
  if (find(tv) == end()) {
    return ParallelTypeBitmap();
  }

  ParallelTypeBitmap parallel_broadcast;

  const auto& iter_domains = tv->domain()->domain();

  // If the output is on shared memory, assume that all subsequent
  // reads from all threads in its CTA can be done with no parallel
  // broadcast. Only one thread will write to shared memory followed
  // by a proper _syncthreads.
  const bool output_smem = tv->getMemoryType() == MemoryType::Shared;

  for (auto id : iter_domains) {
    if (!id->isBroadcast()) {
      continue;
    }
    if (id->isBlockDim() || (!output_smem && id->isThreadDim())) {
      parallel_broadcast.set(id->getParallelType());
    }
  }

  return parallel_broadcast & at(tv).limited_types;
}

void ThreadPredicateMap::print() const {
  std::cout << "\nThreadPredicateMap\n";
  std::cout << "--------------------------------\n";
  for (const auto& kv : thread_predicates_) {
    std::cout << "T" << kv.first->name();
    std::cout << " {" << kv.second.limited_types.toString() << "}\n";
    for (const auto& pkv : kv.second.source_map) {
      std::cout << "  " << pkv.first << " : [";
      for (auto tv : pkv.second) {
        std::cout << " T" << tv->name();
      }
      std::cout << " ]\n";
    }
    std::cout << "{" << kv.second.redundant_types.toString() << "}\n";
  }
  std::cout << "--------------------------------\n\n";
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
