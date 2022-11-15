#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_index_compute.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IndexFromIdGraph::IndexFromIdGraph(
    IndexCompute index_,
    IndexCompute concrete_index_,
    std::unordered_map<IterDomain*, Val*> initial_concrete_index_map_,
    std::vector<IterDomain*> loop_domains_)
    : index(index_),
      concrete_index(concrete_index_),
      initial_concrete_index_map(initial_concrete_index_map_),
      resolved_loop_domains(loop_domains_) {}

namespace {

// Maps all producer domains to consumer with broadcast
// forwarding. Used to find the allocation position.
// TODO: should this be an ir_util ? Didn't seem to be
//  used too much though.
std::unordered_map<IterDomain*, IterDomain*> mapAllProducerDomainsToConsumer(
    const TensorView* producer_tv,
    const TensorView* consumer_tv) {
  // This map has forwarded broadcast axes, it should only be used to compute
  // the allocation position of the producer
  std::unordered_map<IterDomain*, IterDomain*> p2c_alloc_map;

  //  We want to replay producer as consumer instead of the other way around
  //  since consumer may have some broadcasted axes producer doesn't have
  //  merged into loops producer may use. If we did consumer as producer we
  //  wouldn't have this information in the mapping.
  auto replay_PasC = BestEffortReplay::replayPasC(
      producer_tv,
      consumer_tv,
      -1,
      PairwiseRootDomainMap(producer_tv, consumer_tv));

  // Grab consumer domain entries and reverse replay map. TODO: Maybe
  // TransformReplay::replayPasC could return this map
  for (auto id : consumer_tv->domain()->domain()) {
    const auto& c2p_map = replay_PasC.getReplay();
    auto c2p_it = c2p_map.find(id);
    if (c2p_it != c2p_map.end()) {
      auto c_id = c2p_it->first;
      auto p_id = c2p_it->second;
      p2c_alloc_map[p_id] = c_id;
    }
  }

  return p2c_alloc_map;
}

std::unordered_map<IterDomain*, IterDomain*> invertOneToOneMap(
    const std::unordered_map<IterDomain*, IterDomain*>& map) {
  std::unordered_map<IterDomain*, IterDomain*> inverted;
  for (const auto& kv : map) {
    bool inserted = inverted.emplace(kv.second, kv.first).second;
    TORCH_INTERNAL_ASSERT(
        inserted,
        "Multiple mappings to the same value detected: ",
        kv.second->toString());
  }
  return inverted;
}

//! A struct to keep track of necessary parameters used in
//!  configuring index compute pass.
//! These parameters are needed to propagate the indexing from the leaf nodes of
//! the TVs and loop nests to the TVs rfactor domain during
//! index_compute.cpp::IndexCompute passes.
//! TODO:
//!   Would expect this list to become shorter over time,
//!  as more info can be determined holistically.
struct IndexingParameters {
  //! Initial binding of index math to concrete iterdomain ids,
  //!  from the loop nest analysis.
  std::unordered_map<IterDomain*, Val*> initial_concrete_id_index;

  //! (Used in non-global indexing) the concrete iterdomains that
  //!  we want to skip or merge into contiguous indexing paths.
  std::unordered_set<IterDomain*> zero_domains;

  //! (Used in non-global indexing) the preferred path we would
  //!  be propagating contiguously merged indices backward.
  std::unordered_set<IterDomain*> preferred_concrete_ids;

  //! The inferred halo padded extents of the concrete iterdomains.
  std::unordered_map<IterDomain*, Val*> concrete_id_to_halo_extent;
};

// Initial loop index map for global producer or consumer case.
IndexingParameters getLinearIndexParameters(
    const LoopIndexing& loop_indexing,
    bool index_producer = false) {
  IndexingParameters index_parameters;

  auto& loops = loop_indexing.loops();
  auto& loop_domain = loop_indexing.loopDomains();
  auto& loop_index_map = index_parameters.initial_concrete_id_index;

  for (auto loop_idx : c10::irange(loops.size())) {
    auto loop = loops[loop_idx];
    auto index_domain = GpuLower::current()->caMap()->getConcreteMappedID(
        loop_domain[loop_idx], IdMappingMode::EXACT);
    if (loop->isTrivial()) {
      // This is useful information in the case of
      //  MisalignedVectorize and double buffer epilog, etc.
      loop_index_map[index_domain] = loop->start();
    } else {
      // Default use pre-allocated integers for index
      loop_index_map[index_domain] = loop->index();
    }
  }

  // Derive the halo extents from the loop indexing result.
  index_parameters.concrete_id_to_halo_extent =
      GpuLower::current()->haloInfo()->buildConcreteHaloExtentMap(
          loop_indexing);

  protectNonPredicateIndexWithMagicZero(
      loops,
      loop_indexing.loopDomains(),
      index_parameters.initial_concrete_id_index);

  // Setup double buffer increment for producer case:
  // TODO: could unify these double buffer index calculation
  //  in follow ups.
  if (index_producer) {
    auto double_buffer_loop =
        GpuLower::current()->doubleBufferInfo().getDoubleBufferLoop(
            loop_indexing.consumerTv(), loops, true);

    for (auto loop_idx : c10::irange(loops.size())) {
      auto loop = loops[loop_idx];
      if (loop == double_buffer_loop) {
        TORCH_INTERNAL_ASSERT(
            !loop->isTrivial(), "The double buffer loop must be materialized");

        auto loop_id = loop_indexing.loopDomains()[loop_idx];

        auto concrete_loop_id =
            GpuLower::current()->caMap()->getConcreteMappedID(
                loop_id, IdMappingMode::EXACT);

        auto stage_depth =
            GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                loop->iter_domain());
        index_parameters.initial_concrete_id_index[concrete_loop_id] =
            SimplifyingIrBuilder::addExpr(
                index_parameters.initial_concrete_id_index[concrete_loop_id],
                SimplifyingIrBuilder::create<Int>(stage_depth - 1));
      }
    }
  }

  return index_parameters;
}

// Initial index parameters for shared and local case
IndexingParameters getNonGlobalInitialIndexParameters(
    const LoopIndexing& loop_indexing,
    const TensorView* consumer_tv,
    bool index_producer = false,
    const TensorView* producer_tv = nullptr,
    std::unordered_map<IterDomain*, IterDomain*> p2c_map = {}) {
  IndexingParameters index_parameters;
  const auto& loops = loop_indexing.loops();
  const auto& loop_domains = loop_indexing.loopDomains();

  // TODO:
  //  The non-global path should become shorter as we
  // pull more info into id graph.
  std::unordered_map<IterDomain*, IterDomain*> alloc_id_map;

  if (index_producer) {
    alloc_id_map = mapAllProducerDomainsToConsumer(producer_tv, consumer_tv);
  }

  auto alloc_tv = index_producer ? producer_tv : consumer_tv;
  auto alloc_info = lower_utils::getAllocInformation(
      alloc_tv, loops, alloc_id_map, index_producer);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::unordered_set<kir::ForLoop*> zero_loops;

  kir::ForLoop* double_buffer_loop = nullptr;

  if (index_producer) {
    double_buffer_loop =
        GpuLower::current()->doubleBufferInfo().getDoubleBufferLoop(
            consumer_tv, loops, true);
  }

  std::tie(loop_to_ind_map, zero_loops) = indexMapFromTV(
      alloc_tv,
      loops,
      alloc_info.init_for_loop,
      !index_producer,
      double_buffer_loop);

  ensureStaticIndexing(alloc_tv, alloc_info.init_for_loop, loops, alloc_id_map);

  TORCH_INTERNAL_ASSERT(
      loops.size() <= loop_domains.size(),
      "Loop domain didn't replay all loops");

  for (auto loop_idx : c10::irange(loops.size())) {
    auto loop = loops[loop_idx];
    auto loop_domain = loop_domains[loop_idx];

    auto concrete_loop_domain =
        GpuLower::current()->caMap()->getConcreteMappedID(
            loop_domain, IdMappingMode::EXACT);

    index_parameters.initial_concrete_id_index[concrete_loop_domain] =
        loop_to_ind_map.at(loop);

    if (zero_loops.count(loop)) {
      index_parameters.zero_domains.insert(concrete_loop_domain);
    }
  }

  // Derive preferred path from loop indexing result.
  const TensorView* target_tv = index_producer ? producer_tv : consumer_tv;
  index_parameters.preferred_concrete_ids = buildLoopIndexingPreferredPath(
      target_tv, loop_indexing, index_producer, p2c_map);

  // Derive the halo extents from the loop indexing result.
  index_parameters.concrete_id_to_halo_extent =
      GpuLower::current()->haloInfo()->buildConcreteHaloExtentMap(
          loop_indexing);

  return index_parameters;
}

// Return true if it is sufficient to predicate the end of the loop
// iteration. An aligned vectorized loop is one example where it is
// guaranteed to be valid by the validation checks. More generally,
// the divisible split set is used to find such loops. The divisible
// split set contains splits used in view transformations as well as
// those whose output domains are vectorized. View transformations
// guarantee that any split involved is divisible, whereas
// vectorization only guarantees that the overall root extent is
// divisible by the split factor. Thus, if a loop IterDomain is
// an output of a split included in the divisible view splits, we can
// just predicate the end of the loop iteration. If a loop IterDomain
// is an output of a divisible split due to vectorization, it is only
// valid when the loop IterDomain is mapped with the vectorized inner
// output IterDomain. If it is mapped with an outer IterDomain, since
// the split input IterDomain may be an output IterDomain of a
// non-divisible split, we still need to predicate each loop iteration
// value.
bool predicateAtEnd(kir::ForLoop* loop) {
  auto loop_id = loop->iter_domain();
  auto split = dynamic_cast<Split*>(loop_id->definition());
  if (split == nullptr) {
    return false;
  }

  bool is_divisible = GpuLower::current()->divisibleSplitSet().count(split) > 0;

  if (!is_divisible) {
    return false;
  }

  // Find the other output of the split
  auto other_out_id =
      split->inner() == loop_id ? split->outer() : split->inner();

  // If the other output is mapped with a vectorized IterDomain,
  // this IterDomain needs to be predicated at each iteration point.
  const auto& other_id_exact_set = GpuLower::current()
                                       ->caMap()
                                       ->getIdSets(IdMappingMode::EXACT)
                                       .getDisjointSetOf(other_out_id);

  if (std::any_of(
          other_id_exact_set.begin(), other_id_exact_set.end(), [](auto id) {
            return id->getParallelType() == ParallelType::Vectorize;
          })) {
    return false;
  }

  // Now it is either loop_id is mapped with a vectorized IterDomain
  // or it's an output of view transformations.
  return true;
}

//! Initial index parameters for predicate, adjusts loop to indexing
//!  may according to the information annotated on the loop nest.
//!
//! TODO:
//!  This function is mostly copy pasted from previous implementation
//! at this step, further clean up is possible since:
//!  1. Much of the loop-to-ind adjustment will be issued from idgraph
//!  2. Much of the initial index logic could be shared across all
//! the 3 variants.
IndexingParameters getPredicateInitialIndexParameters(
    const LoopIndexing& loop_indexing,
    TensorView* consumer_tv,
    kir::ForLoop* unswitch_or_vec_loop,
    IterDomain* double_buffer_axis,
    bool is_start_predicate) {
  IndexingParameters index_parameters;
  const auto& loops = loop_indexing.loops();
  const auto& loop_domains = loop_indexing.loopDomains();

  // This shouldn't be needed.
  TORCH_INTERNAL_ASSERT(
      loops.size() <= loop_domains.size(),
      "Loop domain didn't replay all loops");

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  // Fill initial index with each forloop's index.
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  bool unswitch_pred = unswitch_or_vec_loop != nullptr &&
      (unswitch_or_vec_loop->iter_domain()->getParallelType() ==
           ParallelType::Unswitch ||
       unswitch_or_vec_loop->iter_domain()->getParallelType() ==
           ParallelType::Unroll);

  // Vectorized predicates are different from unswitch. Unswitch predicates
  // all loops within the unswitch (the outer most unswitch) are generated
  // with loop->extent-1 as the index. With vectorized predicates, only the
  // vectorized loop should be like this.

  bool within_unswitch = false;

  for (const auto loop_i : c10::irange(loops.size())) {
    auto loop = loops[loop_i];
    auto loop_id = loop->iter_domain();
    auto loop_pt = loop_id->getParallelType();
    auto ref_id = loop_domains.at(loop_i);

    if (!within_unswitch && unswitch_pred) {
      within_unswitch = loop == unswitch_or_vec_loop;
    }

    bool predicate_at_end =
        within_unswitch || loop == unswitch_or_vec_loop || predicateAtEnd(loop);

    if (predicate_at_end) {
      // Rely on the reference to check broadcasting. The for loop could be
      // broadcasted on a constant value from an unroll split. Since reference
      // may convert this to an iter domain, that for loop could be valid to
      // generate predication from.

      // Note that loop->stop() is not used below. Instead,
      // loop->iter_domain()->extent() is used, which is uniform
      // across the mapped domains irrespective of halo. Predicates are
      // compared with each to pick the most restrictive ones. The
      // comparison is done by only using the offset, which is the
      // term added to the index. So, the index term must be the
      // same among all predicates, otherwise the comparison would
      // be invalid. The effect by halo is added to the offset
      // term. See getUnswitchStopOffset.

      if (ref_id->isBroadcast()) {
        // Ignore indexing into broadcasted dimensions.
        continue;
      } else if (loop_id->isThread()) {
        // When parallelized, if the loop stop is the same as the
        // extent of the associated IterDomain, i.e., no extra
        // iterations for halo, predicating with the threading index
        // is sufficient for both the start and stop
        // predicates. That isn't the case if the loop has halo, and
        // in the case either the minimum and maximum values of the
        // iteration domain needs to be used.
        //
        // Note: Better performance was obtained if using
        // threadIdx in unswitch predicates was avoided. More
        // specifically, in the Hdiff stencil example, instead of
        // predicating with threadIdx.x for both the start and stop
        // predicates, using zero and (blockDim.x - 1) for the start
        // and stop predicates, respectively, resulted in less
        // register pressure. The alternative codegen can be done by
        // adding this to the first if condition:
        // loop_id->isBlockDim(). This would not be a concern if the
        // else part could be omitted, so canOmitElseClause should
        // be used as well.
        if (loop->stop() == loop_id->extent()) {
          loop_to_ind_map[loop] = loop->start();
        } else if (is_start_predicate) {
          loop_to_ind_map[loop] = GpuLower::current()->kernel()->zeroVal();
        } else {
          // Note that the parallel dimension is used rather than
          // loop-stop(). See the above comment.
          loop_to_ind_map[loop] =
              GpuLower::current()->parallelDimensionMap().get(loop_pt);
        }
      } else if (is_start_predicate) {
        loop_to_ind_map[loop] = GpuLower::current()->kernel()->zeroVal();
      } else {
        // Similar to the above, loop_id()->extent() is
        // used here instead of loop->stop(). See the above comment.
        loop_to_ind_map[loop] = SimplifyingIrBuilder::subExpr(
            loop_id->extent(), GpuLower::current()->kernel()->oneVal());
      }
    }
  }

  // Modify trivial loops to use the loop start value.
  //  FIXME: eventually should be all lifted in idgraph.
  for (const auto loop : loops) {
    auto& idx = loop_to_ind_map.at(loop);
    // If the loop is trivial, the loop index can only be the loop
    // start value.
    if (idx == loop->index() && loop->isTrivial()) {
      idx = loop->start();
    }
  }

  // Increment double buffer loop index
  if (double_buffer_axis != nullptr) {
    auto db_loop = GpuLower::current()->doubleBufferInfo().getDoubleBufferLoop(
        double_buffer_axis, loops, true);
    if (db_loop != nullptr) {
      auto loop_to_ind_map_it = loop_to_ind_map.find(db_loop);
      TORCH_INTERNAL_ASSERT(loop_to_ind_map_it != loop_to_ind_map.end());
      auto cur_index = loop_to_ind_map_it->second;
      // if cur_index is not the same as the index of db_loop, it must
      // be true that that index has been modified to support
      // unswitch. In that case, it is not necessary to move ahead the
      // index for double buffering.
      auto stage_depth =
          GpuLower::current()->doubleBufferInfo().getStageDepthFor(
              db_loop->iter_domain());
      if (cur_index == db_loop->index()) {
        loop_to_ind_map[db_loop] = SimplifyingIrBuilder::addExpr(
            cur_index, SimplifyingIrBuilder::create<Int>(stage_depth - 1));
      }
    }
  }

  // Convert loop-to-ind map to concrete-to-ind map
  for (int loop_idx : c10::irange(loops.size())) {
    auto loop = loops.at(loop_idx);
    auto concrete_loop_domain =
        GpuLower::current()->caMap()->getConcreteMappedID(
            loop_domains.at(loop_idx), IdMappingMode::EXACT);
    index_parameters.initial_concrete_id_index[concrete_loop_domain] =
        loop_to_ind_map.at(loop);
  }

  // Note that, unlike non-predicate indexing, magic-zero insertion is
  // not done at this point but is done individually for each indexed
  // domain. See Index::getReferenceRootPredicates.

  // Derive the halo extents from the loop indexing result.
  index_parameters.concrete_id_to_halo_extent =
      GpuLower::current()->haloInfo()->buildConcreteHaloExtentMap(
          loop_indexing);

  return index_parameters;
}

} // namespace

LoopIndexing LoopIndexingAnalysis::fromLoopAndConsumer(
    const std::vector<kir::ForLoop*>& loops,
    const TensorView* consumer_tv) {
  LoopIndexingAnalysis analysis(loops, consumer_tv);
  return analysis.getLoopIndexing(loops);
}

VectorOfUniqueEntries<IterDomain*> LoopIndexingAnalysis::
    getReplayableConcreteIDs(
        const std::vector<IterDomain*>& consumer_leaf_ids,
        const TensorView* consumer_tv) {
  LoopIndexingAnalysis analysis(consumer_leaf_ids, consumer_tv);
  return analysis.replayed_concrete_ids_;
}

LoopIndexingAnalysis::LoopIndexingAnalysis(
    const std::vector<kir::ForLoop*>& loops,
    const TensorView* consumer_tv)
    : consumer_tv_(consumer_tv) {
  // Validate consistency in given loop nest
  validateLoopStructure(loops);

  // Populate initial loop iter domains.
  std::transform(
      loops.begin(),
      loops.end(),
      std::back_inserter(initial_loop_domain_ids_),
      [](kir::ForLoop* fl) { return fl->iter_domain(); });

  run();
}

LoopIndexingAnalysis::LoopIndexingAnalysis(
    const std::vector<IterDomain*>& consumer_leaf_ids,
    const TensorView* consumer_tv)
    : consumer_tv_(consumer_tv) {
  // Populate initial loop iter domains.
  std::transform(
      consumer_leaf_ids.begin(),
      consumer_leaf_ids.end(),
      std::back_inserter(initial_loop_domain_ids_),
      [&](IterDomain* consumer_leaf_id) {
        // Make sure consumer_leaf_id is indeed a consumer leaf ID
        TORCH_INTERNAL_ASSERT(
            std::find(
                consumer_tv->domain()->domain().begin(),
                consumer_tv->domain()->domain().end(),
                consumer_leaf_id) != consumer_tv->domain()->domain().end(),
            "Not a consumer leaf ID: ",
            consumer_leaf_id->toString(),
            ", consumer: ",
            consumer_tv->toString());
        return GpuLower::current()->caMap()->getConcreteMappedID(
            consumer_leaf_id, IdMappingMode::LOOP);
      });

  run();
}

void LoopIndexingAnalysis::run() {
  // Collect consumer id's for view rfactor traversal.
  all_consumer_id_vals_ = DependencyCheck::getAllValsBetween(
      {consumer_tv_->getRootDomain().begin(),
       consumer_tv_->getRootDomain().end()},
      {consumer_tv_->domain()->domain().begin(),
       consumer_tv_->domain()->domain().end()});

  // Resolve definition of each exact concrete id's involved in the whole loop
  // nest transform history
  traverseFromDomainVals();

  // Construct concrete to consumer map. The replayed exprs are guaranteed to
  // consume each concrete id once so this map is well defined.
  for (auto expr : replayed_exprs_) {
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      auto concrete_input_id =
          GpuLower::current()->caMap()->getConcreteMappedID(
              input_id, IdMappingMode::EXACT);
      concrete_id_to_consumer_[concrete_input_id] = expr;
    }
  }

  // Reconstruct the iterdomain view of the original loopnest after resolving
  // the exact definition of each index.
  constructLoopDomains();

  //! Collect the set of indexing expressions that can be
  //!  resolved out of line.
  collectOutOfLineExprs();
}

void LoopIndexingAnalysis::validateLoopStructure(
    const std::vector<kir::ForLoop*>& loops) {
  // Throw an error when two loops are mapped with each other, which
  // violates an assumption that unique mappings between concrete
  // IterDomains and the IterDomains of the loop structure must be
  // established. It should be a reasonable assumption, but fusions
  // like below won't work:
  // tv0 = [I0]
  // tv1 = broadcast(tv0, {true, false});
  // tv2 = broadcast(tv0, {false, true});
  // tv3 = tv1 + tv2
  // Notice that the two axes of each of tv1, tv2 and tv3 are mapped
  // with each other. We believe it is unlikely this limitation
  // becomes a real concern in practice.
  // Map concrete id to the original loop iter domain.
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_loop;
  for (auto it_i = loops.begin(); it_i != loops.end(); ++it_i) {
    // Largely duplicating original logic
    auto loop_id = (*it_i)->iter_domain();
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        loop_id, IdMappingMode::EXACT);

    TORCH_INTERNAL_ASSERT(
        !concrete_to_loop.count(concrete_loop_id),
        "Unsupported loop structure. Two loops are mapped together.",
        loop_id->toString(),
        " and ",
        concrete_to_loop.at(concrete_loop_id)->toString());

    concrete_to_loop[concrete_loop_id] = loop_id;
  }
}

void LoopIndexingAnalysis::traverseFromDomainVals() {
  // Order is really important here, start with outer most for loops in a
  // depth first manner. The outer most loops are topologically closer to the
  // outputs, so their broadcast dimensions are "more" resolved than those
  // towards the inner most loops.
  std::deque<IterDomain*> to_visit(
      initial_loop_domain_ids_.begin(), initial_loop_domain_ids_.end());
  std::unordered_set<Expr*> visited_exprs;
  std::unordered_set<IterDomain*> visited_ids;

  while (!to_visit.empty()) {
    auto out_id = to_visit.front();
    to_visit.pop_front();

    if (!visited_ids.emplace(out_id).second) {
      continue;
    }
    auto expr = out_id->definition();

    if (auto rfactor_id =
            getRfactorIDToTraverse(out_id, all_consumer_id_vals_)) {
      to_visit.emplace_front(rfactor_id);
    }

    // ID's will be copied for the reference as we replay transformations. If
    // there was no transformations on an iteration domain, a copy of the
    // iteration domain for the reference is made here.
    if (expr == nullptr) {
      if (std::find(
              initial_loop_domain_ids_.begin(),
              initial_loop_domain_ids_.end(),
              out_id) != initial_loop_domain_ids_.end()) {
        concretizeAndVisitId(out_id);
      }
      continue;
    }

    if (!visited_exprs.emplace(expr).second) {
      continue;
    }

    visitExpr(expr);

    auto inp_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
    // Make sure to put at the begining of the deque to maintain correct
    // ordering.
    to_visit.insert(to_visit.begin(), inp_ids.begin(), inp_ids.end());
  }
}

IterDomain* LoopIndexingAnalysis::concretizeAndVisitId(IterDomain* id) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::EXACT);
  if (replayed_concrete_ids_.pushBack(concrete_id)) {
    concrete_to_original_id_[concrete_id] = id;
  }
  return concrete_id;
}

namespace {
// Alias used for std::transform
IterDomain* exactConcreteId(IterDomain* id) {
  return GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::EXACT);
}
} // namespace

void LoopIndexingAnalysis::visitExpr(Expr* expr) {
  if (auto swizzle2d = dynamic_cast<Swizzle2D*>(expr)) {
    // Swizzle outputs are already forwarded through
    //  by exact CA map, so currently they are just
    //  ignored in the replay pass except
    //  that we want to note this node visited.
    concretizeAndVisitId(swizzle2d->outX());
    concretizeAndVisitId(swizzle2d->outY());
    return;
  }

  // Current implementation just tries to
  //  follow the exact behavior of reference replay
  //  except that no expr was actually "replayed".

  // Record all inputs, and stop if current expr
  //  duplicates id consumption or production.
  if (visitIdsAndCheckDuplication(expr->inputs(), consumed_concrete_)) {
    return;
  }
  if (visitIdsAndCheckDuplication(expr->outputs(), produced_concrete_)) {
    return;
  }

  // Record the expr if no duplication on input or output found
  replayed_exprs_.push_back(expr);

  // Record the consumed and produced concrete ids by the newly
  //  recorded expression.
  auto consumed_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
  std::transform(
      consumed_ids.begin(),
      consumed_ids.end(),
      std::inserter(consumed_concrete_, consumed_concrete_.end()),
      exactConcreteId);

  auto produced_ids = ir_utils::filterByType<IterDomain>(expr->outputs());
  std::transform(
      produced_ids.begin(),
      produced_ids.end(),
      std::inserter(produced_concrete_, produced_concrete_.end()),
      exactConcreteId);
}

bool LoopIndexingAnalysis::visitIdsAndCheckDuplication(
    const std::vector<Val*>& vals,
    const std::unordered_set<IterDomain*>& existing_ids) {
  bool duplication = false;
  for (auto id : ir_utils::filterByType<IterDomain>(vals)) {
    duplication = duplication || existing_ids.count(concretizeAndVisitId(id));
  }
  return duplication;
}

void LoopIndexingAnalysis::constructLoopDomains() {
  for (auto loop_id : initial_loop_domain_ids_) {
    // Find the replayed_concrete_id mapping to the loop id.
    auto ref_id_it = std::find_if(
        replayed_concrete_ids_.vector().begin(),
        replayed_concrete_ids_.vector().end(),
        [&](IterDomain* concrete_id) {
          return
              // Make sure the replayed_concrete_id is a leaf ID
              !concrete_id_to_consumer_.count(concrete_id) &&
              // Use permissive map so the selected ID indeed represents the
              // loop.
              // This mapping look up is part of a staged indexing scheme.
              //  When we find a replayed exact id that exactly map to the loop
              //  id, this means that we can resolve indexing involved in this
              //  loop "locally", i.e. only with and with only the iterdomains
              //  on the given consumer tv.
              //  When we cannot find an exact mapping, the permissive mapping
              //  would help defering the indexing resolution for this loop nest
              //   level to other iterdomain expressions from tv's that are
              //   further concretized and usually they are further down the
              //   consumer chain of the given consumer tv.
              GpuLower::current()->caMap()->areMapped(
                  concrete_id, loop_id, IdMappingMode::PERMISSIVE);
        });

    TORCH_INTERNAL_ASSERT(
        ref_id_it != replayed_concrete_ids_.vector().end(),
        "Could not find required iter domain in reference replay: ",
        loop_id->toString());

    auto ref_id = *ref_id_it;
    loop_domains_.pushBack(concrete_to_original_id_.at(ref_id));
  }

  // Construct the root domain as the inputs of the replayed domain
  auto loops_replayed_domain_vals =
      ir_utils::filterByType<Val>(loop_domains_.vector());
  auto root_domain_vals = IterVisitor::getInputsTo(
      {loops_replayed_domain_vals.begin(), loops_replayed_domain_vals.end()});

  // Fill loop roots:
  auto root_domain_ids = ir_utils::filterByType<IterDomain>(root_domain_vals);
  loop_root_domains_ =
      std::vector<IterDomain*>(root_domain_ids.begin(), root_domain_ids.end());

  // The domain may have dangling iteration domains, i.e. the inner output of
  // a split but not the outer. Find which replayed vals are dependant on the
  // root domains.
  auto all_replayed_vals =
      ir_utils::filterByType<Val>(replayed_concrete_ids_.vector());
  auto all_ids_from_root = DependencyCheck::getAllValsBetween(
      {root_domain_vals.begin(), root_domain_vals.end()},
      {all_replayed_vals.begin(), all_replayed_vals.end()});

  // Fill all dangling outputs as otherwise backwards visitor in index compute
  // will complain for not having all outputs of the traversal.
  for (auto id : ir_utils::filterByType<IterDomain>(all_ids_from_root)) {
    if (id->uses().empty()) {
      loop_domains_.pushBack(GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT));
    }
  }
}

IndexFromIdGraph getTensorIndexFromIdGraph(
    const std::vector<kir::ForLoop*>& loops,
    const TensorView* consumer_tv,
    const TensorView* producer_tv,
    bool is_global,
    std::unordered_map<IterDomain*, IterDomain*> c2p_map) {
  bool index_producer = producer_tv != nullptr;
  auto target_tv = index_producer ? producer_tv : consumer_tv;

  auto loop_indexing =
      LoopIndexingAnalysis::fromLoopAndConsumer(loops, consumer_tv);

  IndexingParameters index_parameters;

  std::unordered_map<IterDomain*, IterDomain*> p2c_map;

  // The p2c map is only needed when indexing producer
  //  as producer has replayed ids.
  if (index_producer) {
    p2c_map = invertOneToOneMap(c2p_map);
  }

  if (is_global) {
    index_parameters = getLinearIndexParameters(loop_indexing, index_producer);
  } else {
    index_parameters = getNonGlobalInitialIndexParameters(
        loop_indexing, consumer_tv, index_producer, producer_tv, p2c_map);
  }

  IndexCompute indexing(
      index_parameters.initial_concrete_id_index,
      index_parameters.zero_domains,
      index_parameters.preferred_concrete_ids,
      index_parameters.concrete_id_to_halo_extent);

  // Run first backward traversal to generate
  //  loop nest based indexing math.
  indexing.run(loop_indexing);

  // Populate indexing through exact map from initial indexing
  auto consumer_root = index_producer ? consumer_tv->getRootDomain()
                                      : consumer_tv->getMaybeRFactorDomain();

  // First collect all iterdomains in consumer transform history.
  auto all_consumer_vals = DependencyCheck::getAllValsBetween(
      {consumer_root.begin(), consumer_root.end()},
      {consumer_tv->domain()->domain().begin(),
       consumer_tv->domain()->domain().end()});

  // Want update map to be based on almost exact, but indexing is on exact, make
  // a map from one space to the other.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      almost_exact_2_target_ids;

  for (IterDomain* consumer_id :
       ir_utils::filterByType<IterDomain>(all_consumer_vals)) {
    auto target_id = consumer_id;

    // use mapped producer id when indexing producer
    if (index_producer) {
      auto target_id_it = c2p_map.find(consumer_id);
      if (target_id_it == c2p_map.end()) {
        // consumer id not found in c2p map
        // skip binding for this id.
        continue;
      }
      target_id = target_id_it->second;
    }

    auto almost_exact_concrete_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            consumer_id, IdMappingMode::ALMOSTEXACT);

    auto almost_exact_2_target_ids_it =
        almost_exact_2_target_ids.find(almost_exact_concrete_id);
    if (almost_exact_2_target_ids_it == almost_exact_2_target_ids.end()) {
      almost_exact_2_target_ids_it =
          almost_exact_2_target_ids
              .emplace(
                  almost_exact_concrete_id,
                  VectorOfUniqueEntries<IterDomain*>())
              .first;
    }
    auto& mapped_dims = almost_exact_2_target_ids_it->second;
    mapped_dims.pushBack(target_id);
  }

  // Map the concrete id indexing back to the producer or consumer tv
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      index_update_map;
  for (auto entry : indexing.indexMap()) {
    auto ref_exact_id = entry.first;
    auto almost_exact_concrete_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            ref_exact_id, IdMappingMode::ALMOSTEXACT);

    if (almost_exact_2_target_ids.find(almost_exact_concrete_id) ==
        almost_exact_2_target_ids.end()) {
      continue;
    }

    auto consumer_ids = almost_exact_2_target_ids.at(almost_exact_concrete_id);

    for (auto consumer_id : consumer_ids) {
      auto index_update_map_it = index_update_map.find(ref_exact_id);
      if (index_update_map_it == index_update_map.end()) {
        index_update_map_it =
            index_update_map
                .emplace(std::make_pair(
                    ref_exact_id, VectorOfUniqueEntries<IterDomain*>()))
                .first;
      }
      auto& mapped_dims = index_update_map_it->second;
      mapped_dims.pushBack(consumer_id);
    }
  }

  // No contig indexing was done in reference indexing
  ContigIDs contig_finder(
      target_tv->domain()->domain(),
      target_tv->getMaybeRFactorDomain(),
      target_tv->domain()->contiguity(),
      {},
      indexing.indexMap(),
      GpuLower::current()->divisibleSplitSet(),
      GpuLower::current()->caMap(),
      GpuLower::current()->haloInfo(),
      GpuLower::current()->concretizedBroadcastDomains(),
      p2c_map);

  auto target_indexing = indexing.updateIndexCompute(
      target_tv->domain(), index_update_map, contig_finder);

  // Fill validation info.
  // TODO: cleanup seems possible.
  if (index_producer) {
    fillProducerVectorizedContigRootDomains(
        producer_tv, consumer_tv, c2p_map, contig_finder);
  } else {
    fillConsumerVectorizedContigRootDomains(consumer_tv, contig_finder);
  }

  return IndexFromIdGraph(
      target_indexing,
      indexing,
      index_parameters.initial_concrete_id_index,
      loop_indexing.loopDomains());
}

IndexFromIdGraph getPredicateIndexingFromIdGraph(
    const std::vector<kir::ForLoop*>& loops,
    TensorView* consumer_tv,
    kir::ForLoop* unswitch_or_vec_loop,
    IterDomain* double_buffer_axis,
    bool is_start_predicate) {
  // Run replay pass on the loop nest to generate the deterministic
  //  traversal info from loop structure.
  auto loop_indexing =
      LoopIndexingAnalysis::fromLoopAndConsumer(loops, consumer_tv);

  // Bind initial index variables to the loop nodes and adjust
  //  according to loop and unswitch info.
  auto index_parameters = getPredicateInitialIndexParameters(
      loop_indexing,
      consumer_tv,
      unswitch_or_vec_loop,
      double_buffer_axis,
      is_start_predicate);

  // Run first backward traversal to generate
  //  loop nest based indexing math.
  IndexCompute indexing(
      index_parameters.initial_concrete_id_index,
      index_parameters.zero_domains,
      index_parameters.preferred_concrete_ids,
      index_parameters.concrete_id_to_halo_extent);

  indexing.run(loop_indexing);

  // First collect all iterdomains in consumer transform history.
  auto all_consumer_vals = DependencyCheck::getAllValsBetween(
      {consumer_tv->getMaybeRFactorDomain().begin(),
       consumer_tv->getMaybeRFactorDomain().end()},
      {consumer_tv->domain()->domain().begin(),
       consumer_tv->domain()->domain().end()});

  // Want update map to be based on almost exact, but indexing is on exact, make
  // a map from one space to the other.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      almost_exact_2_consumer_ids;

  for (IterDomain* consumer_id :
       ir_utils::filterByType<IterDomain>(all_consumer_vals)) {
    auto almost_exact_concrete_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            consumer_id, IdMappingMode::ALMOSTEXACT);

    auto almost_exact_2_consumer_ids_it =
        almost_exact_2_consumer_ids.find(almost_exact_concrete_id);
    if (almost_exact_2_consumer_ids_it == almost_exact_2_consumer_ids.end()) {
      almost_exact_2_consumer_ids_it =
          almost_exact_2_consumer_ids
              .emplace(std::make_pair(
                  almost_exact_concrete_id,
                  VectorOfUniqueEntries<IterDomain*>()))
              .first;
    }
    auto& mapped_dims = almost_exact_2_consumer_ids_it->second;
    mapped_dims.pushBack(consumer_id);
  }

  // Map the concrete id indexing back to the consumer tv
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      index_update_map;
  for (auto entry : indexing.indexMap()) {
    auto ref_exact_id = entry.first;
    auto almost_exact_concrete_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            ref_exact_id, IdMappingMode::ALMOSTEXACT);

    if (almost_exact_2_consumer_ids.find(almost_exact_concrete_id) ==
        almost_exact_2_consumer_ids.end()) {
      continue;
    }
    auto consumer_ids =
        almost_exact_2_consumer_ids.at(almost_exact_concrete_id);

    for (auto consumer_id : consumer_ids) {
      auto index_update_map_it = index_update_map.find(ref_exact_id);
      if (index_update_map_it == index_update_map.end()) {
        index_update_map_it =
            index_update_map
                .emplace(std::make_pair(
                    ref_exact_id, VectorOfUniqueEntries<IterDomain*>()))
                .first;
      }
      auto& mapped_dims = index_update_map_it->second;
      mapped_dims.pushBack(consumer_id);
    }
  }

  // No contiguity info is used in the predicate indexing pass, the predicate
  // generation logic that uses the index math generated here will take
  // contiguity into account. Send an empty ContigID class so nothing is marked
  // as contiguous.
  auto contig_finder = ContigIDs::getNonContigIDs();

  // Run second backward traversal to map back to the consumer_tv
  auto target_indexing = indexing.updateIndexCompute(
      consumer_tv->domain(), index_update_map, contig_finder);

  return IndexFromIdGraph(
      target_indexing,
      indexing,
      index_parameters.initial_concrete_id_index,
      loop_indexing.loopDomains());
}

namespace {

class LoopIndexingTraversal {
  enum class TraversalOrder { ForwardTopological, BackwardTopological };

 public:
  static std::vector<Expr*> forwardTopologicalOrder(
      const std::vector<Expr*>& exprs) {
    LoopIndexingTraversal traversal(exprs, TraversalOrder::ForwardTopological);
    return traversal.getExprList();
  }

  static std::vector<Expr*> backwardTopologicalOrder(
      const std::vector<Expr*>& exprs) {
    LoopIndexingTraversal traversal(exprs, TraversalOrder::BackwardTopological);
    return traversal.getExprList();
  }

 private:
  explicit LoopIndexingTraversal(
      const std::vector<Expr*>& exprs,
      TraversalOrder traversal_order);

  // Returns the vals following the expression in either
  //  forward or backward order.
  const std::vector<Val*>& nextValsInTraversalOrder(Expr* expr);

  // Returns the vals that the expression follows in either
  //  forward or backward order.
  const std::vector<Val*>& prevValsInTraversalOrder(Expr* expr);

  // Returns the sorted list according to the given traversal order.
  std::vector<Expr*> getExprList();

 private:
  // Reference to original un-sorted expression list.
  const std::vector<Expr*>& exprs_;

  // The traversal order in this pass.
  const TraversalOrder traversal_order_ = TraversalOrder::ForwardTopological;

  // Internal record of concrete id's and it's corresponding
  //  iterdomain expression that defines the exact index.
  std::unordered_map<IterDomain*, Expr*> concrete_id_to_dependency_;
};

LoopIndexingTraversal::LoopIndexingTraversal(
    const std::vector<Expr*>& exprs,
    TraversalOrder traversal_order)
    : exprs_(exprs), traversal_order_(traversal_order) {
  // Populate concrete id dependencies:
  for (auto expr : exprs_) {
    auto next_ids =
        ir_utils::filterByType<IterDomain>(nextValsInTraversalOrder(expr));
    for (auto id : next_ids) {
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT);
      TORCH_INTERNAL_ASSERT(
          concrete_id_to_dependency_.insert(std::make_pair(concrete_id, expr))
              .second,
          "Repeated dependency, invalid iterdomain traversal.");
    }
  }
}

const std::vector<Val*>& LoopIndexingTraversal::nextValsInTraversalOrder(
    Expr* expr) {
  switch (traversal_order_) {
    case TraversalOrder::ForwardTopological:
      return expr->outputs();
      break;
    case TraversalOrder::BackwardTopological:
      return expr->inputs();
      break;

    default:
      TORCH_INTERNAL_ASSERT(false, "unimplemented traversal order");
  }
  return expr->inputs();
}

const std::vector<Val*>& LoopIndexingTraversal::prevValsInTraversalOrder(
    Expr* expr) {
  switch (traversal_order_) {
    case TraversalOrder::ForwardTopological:
      return expr->inputs();
      break;
    case TraversalOrder::BackwardTopological:
      return expr->outputs();
      break;

    default:
      TORCH_INTERNAL_ASSERT(false, "unimplemented traversal order");
  }
  return expr->inputs();
}

std::vector<Expr*> LoopIndexingTraversal::getExprList() {
  std::deque<Expr*> to_visit(exprs_.begin(), exprs_.end());

  // pre-allocate result space.
  std::vector<Expr*> result;
  result.reserve(exprs_.size());

  // Keeps track of visited and inserted expressions.
  // An expr is visited if it has been placed in result list.
  // An expr is inserted if the traversal has put the expr on
  //  the top of the stack once. Repeated insertion of the same
  //  expression would never be observed if the underlying
  //  dependency of the expressions is cycle free.
  std::unordered_set<Expr*> visited, inserted;

  while (!to_visit.empty()) {
    auto top = to_visit.front();
    if (visited.count(top)) {
      to_visit.pop_front();
      continue;
    }

    bool ready = true;

    for (auto prev_id :
         ir_utils::filterByType<IterDomain>(prevValsInTraversalOrder(top))) {
      auto prev_expr_it = concrete_id_to_dependency_.find(
          GpuLower::current()->caMap()->getConcreteMappedID(
              prev_id, IdMappingMode::EXACT));
      if (prev_expr_it != concrete_id_to_dependency_.end()) {
        auto prev_expr = prev_expr_it->second;
        if (!visited.count(prev_expr)) {
          ready = false;
          to_visit.push_front(prev_expr);
          TORCH_INTERNAL_ASSERT(
              inserted.insert(prev_expr).second,
              "Circular dependency in loop index expressions.");
          break;
        }
      }
    }

    if (ready) {
      visited.insert(top);
      result.emplace_back(top);
      to_visit.pop_front();
    }
  }

  return result;
}

} // namespace

void LoopIndexingAnalysis::collectOutOfLineExprs() {
  // Keep track of all the id's that can be resolved without
  //  iterdomains on the left of ca axes.
  std::unordered_set<IterDomain*> out_of_line_ids;

  // Start the set with all the leaf ids.
  std::transform(
      consumer_tv_->domain()->domain().begin() +
          consumer_tv_->getComputeAtPosition(),
      consumer_tv_->domain()->domain().end(),
      std::inserter(out_of_line_ids, out_of_line_ids.end()),
      exactConcreteId);

  // Get the original selected list of index expressions
  //  in reverse topological order.
  auto backward_expr_list =
      LoopIndexingTraversal::backwardTopologicalOrder(replayed_exprs_);

  for (auto expr : backward_expr_list) {
    auto id_outputs = ir_utils::filterByType<IterDomain>(expr->outputs());
    if (
        // Check that all of the outputs are out of line
        std::all_of(
            id_outputs.begin(),
            id_outputs.end(),
            [&out_of_line_ids](IterDomain* id) {
              return out_of_line_ids.count(
                  GpuLower::current()->caMap()->getConcreteMappedID(
                      id, IdMappingMode::EXACT));
            })) {
      // Record out of line expression
      out_of_line_exprs_.push_back(expr);

      // Add all of the expression inputs as out of line id's.
      auto id_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
      std::transform(
          id_inputs.begin(),
          id_inputs.end(),
          std::inserter(out_of_line_ids, out_of_line_ids.end()),
          exactConcreteId);
    }
  }
}

std::vector<Expr*> LoopIndexing::getForwardExprList() const {
  return LoopIndexingTraversal::forwardTopologicalOrder(index_exprs_);
}

std::vector<Expr*> LoopIndexing::getBackwardExprList() const {
  return LoopIndexingTraversal::backwardTopologicalOrder(index_exprs_);
}

std::unordered_set<IterDomain*> LoopIndexing::getAllExactConcreteIdSet() const {
  std::unordered_set<IterDomain*> all_id_set;
  for (auto expr : index_exprs_) {
    auto out_ids = ir_utils::filterByType<IterDomain>(expr->outputs());
    std::transform(
        out_ids.begin(),
        out_ids.end(),
        std::inserter(all_id_set, all_id_set.end()),
        exactConcreteId);

    auto in_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
    std::transform(
        in_ids.begin(),
        in_ids.end(),
        std::inserter(all_id_set, all_id_set.end()),
        exactConcreteId);
  }
  return all_id_set;
}

namespace {

//! Returns true if id is mapped together with any id in
//!  the vector ids by permissive compute at map.
bool isPermissivelyMappedWithAny(IterDomain* id, const std::vector<Val*>& ids) {
  return std::any_of(ids.begin(), ids.end(), [&](Val* val) {
    return val->isA<IterDomain>() &&
        GpuLower::current()->caMap()->areMapped(
            id, val->as<IterDomain>(), IdMappingMode::PERMISSIVE);
  });
}

class LoopIndexingPreferredPathCompute : public IterVisitor {
 public:
  static std::unordered_set<IterDomain*> compute(
      const TensorView* original_tv,
      const LoopIndexing& loop_indexing,
      bool use_replay_map,
      const std::unordered_map<IterDomain*, IterDomain*>& p2c_map) {
    LoopIndexingPreferredPathCompute compute;

    auto all_concrete_ids = loop_indexing.getAllExactConcreteIdSet();

    // Annotate all ids
    auto all_original_ids = DependencyCheck::getAllValsBetween(
        {original_tv->getMaybeRFactorDomain().begin(),
         original_tv->getMaybeRFactorDomain().end()},
        {original_tv->domain()->domain().begin(),
         original_tv->domain()->domain().end()});

    for (auto original_id :
         ir_utils::filterByType<IterDomain>(all_original_ids)) {
      auto mapped_id = original_id;
      if (use_replay_map) {
        auto c_id_it = p2c_map.find(original_id);
        if (c_id_it == p2c_map.end()) {
          continue;
        }
        mapped_id = c_id_it->second;
      }
      auto concrete_original_id =
          GpuLower::current()->caMap()->getConcreteMappedID(
              mapped_id, IdMappingMode::EXACT);
      if (all_concrete_ids.count(concrete_original_id)) {
        if (original_id->isBroadcast() || original_id->isReduction() ||
            original_id->isStride()) {
          continue;
        }
        compute.preferred_path_.insert(concrete_original_id);
      }
    }

    for (auto expr : loop_indexing.getForwardExprList()) {
      compute.handle(expr);
    }

    return compute.preferred_path_;
  }

 private:
  void handle(Expr* e) override {
    // If an input ID is marked, propagate the marking to outputs of the
    // expression
    auto all_iter_inputs = ir_utils::filterByType<IterDomain>(e->inputs());
    if (std::any_of(
            all_iter_inputs.begin(),
            all_iter_inputs.end(),
            [&](IterDomain* inp_id) {
              return this->preferred_path_.find(
                         GpuLower::current()->caMap()->getConcreteMappedID(
                             inp_id, IdMappingMode::EXACT)) !=
                  this->preferred_path_.end();
            })) {
      auto all_iter_outputs = ir_utils::filterByType<IterDomain>(e->outputs());

      std::transform(
          all_iter_outputs.begin(),
          all_iter_outputs.end(),
          std::inserter(preferred_path_, preferred_path_.end()),
          exactConcreteId);
    }
  }

  std::unordered_set<IterDomain*> preferred_path_;
};

} // namespace

// External interface for preferred path propagation.
std::unordered_set<IterDomain*> buildLoopIndexingPreferredPath(
    const TensorView* original_tv,
    const LoopIndexing& loop_indexing,
    bool use_replay_map,
    std::unordered_map<IterDomain*, IterDomain*> p2c_map) {
  return LoopIndexingPreferredPathCompute::compute(
      original_tv, loop_indexing, use_replay_map, p2c_map);
}

// Get an rfactor IterDomain that is mapped with an IterDomain. If
// multiple such IDs exist, select one whose input IDs are mapped with
// the consumer IDs. This is to ensure the path from the leaf
// IterDomains to the root matches with the consumer tensor.
IterDomain* getRfactorIDToTraverse(
    IterDomain* id,
    const std::vector<Val*>& consumer_all_ids) {
  const auto& rfactor_ids =
      GpuLower::current()->caMap()->getViewRfactorDomainsOfIdGroup(
          id, IdMappingMode::PERMISSIVE);

  if (rfactor_ids.empty()) {
    return nullptr;
  }

  for (auto rfactor_id : rfactor_ids) {
    auto def = rfactor_id->definition();
    if (def == nullptr) {
      continue;
    }

    auto rfactor_id_inputs = ir_utils::filterByType<IterDomain>(def->inputs());
    if (std::all_of(
            rfactor_id_inputs.begin(),
            rfactor_id_inputs.end(),
            [&](IterDomain* rfactor_id_input) {
              return isPermissivelyMappedWithAny(
                  rfactor_id_input, consumer_all_ids);
            })) {
      return rfactor_id;
    }
  }

  // No mapped ID found, which means the consumer is a post-view
  // tensor. In that case, it shouldn't matter which view path to
  // traverse, so just return the first one.
  return rfactor_ids.at(0);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
