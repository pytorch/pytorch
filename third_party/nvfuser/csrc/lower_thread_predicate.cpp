#include <lower_thread_predicate.h>

#include <arith.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <lower_utils.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

Bool* getPredicatePerParallelType(
    ParallelType pt,
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);

  // If pt is not used or is proven to be one, no need to predicate.
  if (pt_dim == nullptr || pt_dim->isOneInt()) {
    return GpuLower::current()->kernel()->trueVal();
  }
  // When BID needs to be predicated, that means it's an output of a grid
  // reduction and only the last block index in that dimension has the right
  // value from the grid reduce.
  if (isParallelTypeBlockDim(pt) && pred_info.limited_types.get(pt)) {
    return SimplifyingIrBuilder::eqExpr(
               NamedScalar::getParallelIndex(pt),
               SimplifyingIrBuilder::subExpr(
                   NamedScalar::getParallelDim(pt),
                   GpuLower::current()->kernel()->oneVal()))
        ->as<Bool>();
  }

  // Otherwise, only thread of index 0 executes the computation
  return SimplifyingIrBuilder::eqExpr(
             NamedScalar::getParallelIndex(pt),
             GpuLower::current()->kernel()->zeroVal())
      ->as<Bool>();
}

} // namespace

Bool* ThreadPredicateMap::getPredicateFromPredicateInfo(
    const ThreadPredicateMap::PredicateInfo& pred_info) {
  const auto pred_types = pred_info.limited_types | pred_info.redundant_types;

  if (pred_types.none()) {
    return GpuLower::current()->kernel()->trueVal();
  }

  Bool* pred = nullptr;
  for (const auto pt : pred_types) {
    const auto tp = getPredicatePerParallelType(pt, pred_info);
    pred = SimplifyingIrBuilder::andExpr(pred, tp)->as<Bool>();
  }
  TORCH_INTERNAL_ASSERT(pred != nullptr);

  return pred;
}

namespace {

// Build redundant predicate flags. Will be stored as
// PredicateInfo.redundant_types for the given tensor.
ParallelTypeBitmap avoidRedundantWrites(const TensorView* out_tv) {
  // If the memory type is Local, it's fine to write into it always as
  // it's thread local. If it's Global, it's also fine to let each
  // thread do its own write, unless out_tv is an output of a
  // reduction. Standard reductions (forget gridReduce for the sake of this
  // argument) directly into global memory buffers accumulate into the global
  // memory buffer. If this is done redundantly then it could lead to incorrect
  // results. Correctness issues here can come from smem aliasing, smem
  // reductions or gmem reductions because the reduction itself performs an
  // update to a value, not just a set. For performance it's safe to ommit the
  // redundant writes to gmem or smem, this comment is just specifying it's not
  // always just a performance optimization, but can also be a correctness
  // requirement.
  //
  // For now this is enabled for shared memory buffers, global memory buffers
  // undergoing a reduction, and global memory buffers with terminating outputs.
  // This could be extended to all global memory buffer transactions, but in the
  // test AdvancedIndexing11 there's a case where an intermediate global buffer
  // is set and used to perform a broadcast. At the moment a grid sync is not
  // being inserted here, and it's generally safe since it's just a set. We
  // could enable this more generally for global memory buffers, but would have
  // to insert a sync or a grid broadcast in that example. For now the
  // approach is to only do this on a grid buffer (not undergoing a reduction)
  // if there are no other uses in the kernel.
  //
  // TODO: Revisit if something like AdvancedIndexing11 could be happening at
  // the same time of a global reduction in a way that could produce an
  // incorrect result.
  const bool is_reduction = ir_utils::isReductionOp(out_tv->definition());
  if (!(out_tv->getMemoryType() == MemoryType::Shared ||
        (out_tv->getMemoryType() == MemoryType::Global && is_reduction) ||
        (out_tv->getMemoryType() == MemoryType::Global &&
         out_tv->uses().empty()))) {
    return ParallelTypeBitmap();
  }

  ParallelTypeBitmap pred;
  // Track which TID types are not used to find redundant parallel
  // types. Only TID types are checked if the tensor is on shared
  // memory otherwise on global memory all TID and BID types are checked.
  ParallelTypeBitmap unused_types;
  // Initially all types are conservatively assumed to not be used.
  unused_types = ~unused_types;
  for (auto out_tv_id : out_tv->domain()->domain()) {
    auto pt = out_tv_id->getParallelType();
    if (!isParallelTypeThread(pt)) {
      continue;
    }
    // If the axis is a broadcast domain and is parallelized by TID,
    // it is sufficient to use just one thread since the tensor is on
    // shared memory.
    if ((out_tv->getMemoryType() == MemoryType::Shared &&
         out_tv_id->isBroadcast() && isParallelTypeThreadDim(pt)) ||
        // Protect against global memory and is_reduction as we don't want to
        // predicate grid dimensions as codegen will complain predication on
        // block dimensions is not allowed in grid reductions. The old
        // grid reduction runtime kernel does not differentiate
        // non-reduction and predicated parallel types, so the sync
        // integer buffer would need to be expanded even for
        // predicated parallel types, which is not what
        // getGridSyncBufferSize does. The right thing here is either:
        // retire the old grid reduction kernel, or update the kernel
        // to propertly ignore predicated types. The new kernel is
        // significantly complex and has not been tested, so the
        // latter option seems more reasonable for now. See #1671.
        (!is_reduction && out_tv->getMemoryType() == MemoryType::Global &&
         out_tv_id->isBroadcast() && isParallelTypeThread(pt))) {
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
  if (!(tv_def && ir_utils::isReductionOp(tv_def) &&
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

  // If all of the inputs are not updated and all of the outputs have
  // already mappings, don't do anything
  if (std::all_of(
          ir_utils::filterByType<TensorView>(expr->inputs()).begin(),
          ir_utils::filterByType<TensorView>(expr->inputs()).end(),
          [this](TensorView* tv) {
            return updated_tvs_.find(tv) == updated_tvs_.end();
          }) &&
      std::all_of(
          ir_utils::filterByType<TensorView>(expr->outputs()).begin(),
          ir_utils::filterByType<TensorView>(expr->outputs()).end(),
          [this](TensorView* tv) { return find(tv) != end(); })) {
    return;
  }

  // Which predicates were set for the inputs
  ParallelTypeBitmap input_preds;

  // Which dims are reductions in inputs
  ParallelTypeBitmap input_reductions;

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = inp->as<TensorView>();

    // If tv_inp was an output of a multi-output expression, just change it to a
    // consistent sibling to use a single predicate name.
    if (auto tv_def = tv_inp->definition()) {
      if (tv_def->outputs().size() > 1) {
        tv_inp = ir_utils::getTvOutput(tv_def);
      }
    }

    TORCH_INTERNAL_ASSERT(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    const auto& pred_info = at(tv_inp);

    ParallelTypeBitmap id_reductions;
    ParallelTypeBitmap id_bcasts;
    ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        id_ptypes.set(id->getParallelType());
        if (id->isReduction() &&
            !GpuLower::current()->fusedReductionInfo().isAllreduce(id)) {
          id_reductions.set(id->getParallelType());
        }
        if (id->isBroadcast() &&
            GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
                id)) {
          id_bcasts.set(id->getParallelType());
        }
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

    // Figure out which dims bcast wants to reset
    auto this_input_preds = pred_info.limited_types;
    const auto bcast_reset_mask = ~(this_input_preds & id_bcasts);
    this_input_preds &= bcast_reset_mask;

    input_preds |= this_input_preds;

    id_reductions |=
        getReductionPredicateForUnusedParallelTypes(tv_inp, at(tv_inp));

    // Accumulate
    input_reductions |= id_reductions;
  }

  // Update map for this tv, before accumulating to other inputs
  // Add any reductions this id has to any input predicates
  auto output_preds = input_preds | input_reductions;

  // Run through outputs and set bitset predicates
  for (auto* out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    auto redundant_types = avoidRedundantWrites(out_tv);
    update(out_tv, output_preds, redundant_types);
  }
}

namespace {

//! A simple backward data flow pass:
//!  This pass propagates information backward to annotate "redundant use
//!  chain"'s.
//! The reason this is needed is that, say for example, if we have a chain
//! of register-to-register ops that begins with a redundant shared mem write
//! and ends with an op that non-redundantly uses the result, we'd need to
//! insert a sync at the begining of the register-to-register chain.
//!
//! The same mechanism also applies in the case of a register/sharedmem chain
//! that starts and ends with global memory read/write.
//!
//! The propagation rule is summarized as follows:
//!
//!   Shared TV val:
//!      Reset all block redundant info to its own redundant write info
//!      Backpropagate grid redundant info
//!   Global TV val:
//!      Reset all redundant info to its own redundant write info
//!   Local Tv val:
//!      Backpropagate all redundant info
//!   Exprs:
//!      Propagate redundant info backwards from outputs to inputs:
//!        For each parallel type,
//!          The parallel type is redundantly used in the expr input
//!          only if all of the outputs redundantly use the same type.
class RedundantUseAnalysis : BackwardVisitor {
 public:
  RedundantUseAnalysis(Fusion* fusion, const ThreadPredicateMap& pred_map)
      : fusion_(fusion), pred_map_(pred_map) {
    traverseTo(fusion, fusion->terminatingMathVals());
  }

  //! Returns a bit map signifying the parallel dimensions
  //!  on which the given tv is redundantly used. On these
  //!  dimensions not all threads/blocks are required to
  //!  hold valid value for their dependent computations.
  ParallelTypeBitmap getRedundantUseBitMap(const TensorView* tv) {
    // Since all tv's consumers are visited at this point, we
    //  can aggregate the final redundant use info for this tv.
    if (fusion_->unordered_uses(tv).empty()) {
      // Base case, un-used is also not redundantly used
      return ParallelTypeBitmap();
    } else {
      // Aggregate redundant use as a conjunction of all
      //  consumer's redundant consumer info propagated
      //  backward from their consumer chains.
      ParallelTypeBitmap redundant_use;
      redundant_use.setAllBID();
      redundant_use.setAllTID();
      for (auto expr : fusion_->unordered_uses(tv)) {
        redundant_use &= redundant_expr_use_map_.at(expr);
      }

      return redundant_use;
    }
  }

 private:
  using BackwardVisitor::handle;

  void handle(TensorView* tv) final {
    auto redundant_tv_map = pred_map_.getPredicateInfo(tv).redundant_types;

    // Setup the info to propagate backward for the producer tv's and
    //  expressions.
    ParallelTypeBitmap& redundant_consumer_map =
        redundant_consumer_parallel_type_map_[tv];

    // Initialize the use map to the redundant pred result
    redundant_consumer_map = redundant_tv_map;

    if (tv->getMemoryType() == MemoryType::Shared) {
      backPropagateRedundantUse(
          redundant_consumer_map,
          tv,
          false, // no propagate TID redundant use for shared tv
          true //  propagate BID redundant use
      );

    } else if (tv->getMemoryType() == MemoryType::Local) {
      backPropagateRedundantUse(
          redundant_consumer_map,
          tv,
          true, // propagate TID redundant use
          true // propagate BID redundant use
      );
    }
  }

  void backPropagateRedundantUse(
      ParallelTypeBitmap& use_map,
      TensorView* tv,
      bool propagate_tid,
      bool propagate_bid) {
    // Clear the propagated part of the original result
    if (propagate_bid) {
      use_map.setAllBID();
    }
    if (propagate_tid) {
      use_map.setAllTID();
    }

    for (auto expr : fusion_->unordered_uses(tv)) {
      // Assuming all consumer expressions have been
      //  visited at this point since we are traversing
      //  backward.
      auto expr_use_map = redundant_expr_use_map_.at(expr);
      // Clear the part of expression use map that does not
      //  need to be propagated.
      if (!propagate_bid) {
        expr_use_map.setAllBID();
      }
      if (!propagate_tid) {
        expr_use_map.setAllTID();
      }

      // Accumulate expression redundant usage
      //  This implements the `only if all` part in
      //   the discussion above.
      use_map &= expr_use_map;
    }
  }

  void handle(Expr* expr) final {
    if (ir_utils::isTvOp(expr)) {
      // Initialize redundant info for current expr
      c10::optional<ParallelTypeBitmap> maybe_expr_pred_map;

      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto tv_redundant_bitmap =
            redundant_consumer_parallel_type_map_.at(consumer_tv);

        if (maybe_expr_pred_map.has_value()) {
          // Accumulate redundant info of this tv output.
          maybe_expr_pred_map.value() &= tv_redundant_bitmap;
        } else {
          // Copy the tv's redundant info as the first valid case.
          maybe_expr_pred_map = tv_redundant_bitmap;
        }
      }

      TORCH_INTERNAL_ASSERT(
          maybe_expr_pred_map.has_value(), "TV op not having a tv output");
      redundant_expr_use_map_[expr] = maybe_expr_pred_map.value();
    }
  }

 private:
  // Populated redundant use information on the used tv's
  //  This map provides information on if the given tv does not require
  // valid data from its producer on any parallel dimensions.
  // For example:
  //  T1_local = T0_shared[...]
  //  if(tid.x == 0)
  //    T2_shared[...] = T1_local[...]
  // Then tidx would be redundant consumer parallel type
  //  for T1, as T1 is local tensor, and only threads satisfying
  //  tidx == 0 would need to provide a valid data.
  // In this case, not all threads would need to read correct data
  //  from T0_shared, which would help remove some sync's.
  std::unordered_map<const TensorView*, ParallelTypeBitmap>
      redundant_consumer_parallel_type_map_;

  // Populated redundant use information on the used tv expressions.
  std::unordered_map<const Expr*, ParallelTypeBitmap> redundant_expr_use_map_;

  // Short cut to the owning fusion of this analysis.
  Fusion* fusion_ = nullptr;

  // Short cut to the active pred map analysis this pass is running as part of.
  const ThreadPredicateMap& pred_map_;
};

} // namespace

void ThreadPredicateMap::build(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::ThreadPredicateMap");

  // Initialize mapping for input tensors
  for (auto inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(inp)) {
      update(tv, ParallelTypeBitmap(), ParallelTypeBitmap());
    }
  }
  for (auto expr : fusion->exprs()) {
    updateBitSet(expr);
  }
  updated_tvs_.clear();
  populateRedundantUseMap(fusion);
}

void ThreadPredicateMap::populateRedundantUseMap(Fusion* fusion) {
  RedundantUseAnalysis redundant_use(fusion, *this);
  for (auto& it : thread_predicates_) {
    it.second.redundant_use_types =
        redundant_use.getRedundantUseBitMap(it.first);
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
  if (dynamic_cast<BroadcastOp*>(tv->definition())) {
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

bool ThreadPredicateMap::update(
    const TensorView* tv,
    const ParallelTypeBitmap& limited_types,
    const ParallelTypeBitmap& redundant_types) {
  return update(tv, {limited_types, redundant_types});
}

bool ThreadPredicateMap::update(
    const TensorView* tv,
    const PredicateInfo& pred_info) {
  auto existing_mapping_it = thread_predicates_.find(tv);
  if (existing_mapping_it != end()) {
    PredicateInfo& existing_info = existing_mapping_it->second;
    if (existing_info == pred_info) {
      return false;
    } else {
      existing_info = pred_info;
      markAsUpdated(tv);
      return true;
    }
  } else {
    thread_predicates_.insert({tv, pred_info});
    markAsUpdated(tv);
    return true;
  }
}

Bool* ThreadPredicateMap::getPredicate(const TensorView* tv) const {
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
    if (!id->isBroadcast() ||
        !GpuLower::current()->concretizedBroadcastDomains()->isConcretized(
            id)) {
      continue;
    }
    if (id->isBlockDim() || (!output_smem && id->isThreadDim())) {
      parallel_broadcast.set(id->getParallelType());
    }
  }

  return parallel_broadcast & at(tv).limited_types;
}

ParallelTypeBitmap ThreadPredicateMap::getRedundantConsumerType(
    Expr* expr) const {
  c10::optional<ParallelTypeBitmap> result;
  for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    auto out_tv_redundant_map = getPredicateInfo(out_tv).redundant_use_types;
    if (!result.has_value()) {
      result = out_tv_redundant_map;
    } else {
      result.value() &= out_tv_redundant_map;
    }
  }

  TORCH_INTERNAL_ASSERT(
      result.has_value(), "ThreadPredicateMap : TV op assumed");
  return result.value();
}

void ThreadPredicateMap::markAsUpdated(const TensorView* tv) {
  updated_tvs_.insert(tv);
}

void ThreadPredicateMap::print() const {
  std::cout << "\nThreadPredicateMap\n";
  std::cout << "--------------------------------\n";
  for (const auto& kv : thread_predicates_) {
    std::cout << "T" << kv.first->name();
    std::cout << " {" << kv.second.limited_types.toString() << "}\n";
    std::cout << "{" << kv.second.redundant_types.toString() << "}\n";
    std::cout << "{" << kv.second.redundant_use_types.toString() << "}\n";
  }
  std::cout << "--------------------------------\n\n";
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
