#include <predicate_compute.h>

#include <arith.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <transform_iter.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool isTensorIndexOp(Expr* expr) {
  const auto& outputs = expr->outputs();
  return outputs.size() >= 1 && outputs[0]->isA<kir::TensorIndex>();
}

bool isOutputLocal(const Expr* expr) {
  return std::all_of(
      expr->outputs().begin(), expr->outputs().end(), [](const Val* output) {
        return !output->isA<TensorView>() ||
            output->as<TensorView>()->getMemoryType() == MemoryType::Local;
      });
}

} // namespace

bool ParallelizedDomainPredicate::PredicateInfo::addDomain(IterDomain* id) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::EXACT);
  if (std::find(ids_.begin(), ids_.end(), concrete_id) == ids_.end()) {
    ids_.push_back(concrete_id);
    return true;
  } else {
    return false;
  }
}

Bool* ParallelizedDomainPredicate::PredicateInfo::getPredicate() const {
  Bool* pred = nullptr;

  auto index = SimplifyingIrBuilder::create<NamedScalar>(
      stringifyThread(pt_), DataType::Int);

  for (const auto& pred_id : ids()) {
    // Just sanity check that pred_id is concrete
    TORCH_INTERNAL_ASSERT(
        pred_id ==
        GpuLower::current()->caMap()->getConcreteMappedID(
            pred_id, IdMappingMode::EXACT));
    auto new_pred = SimplifyingIrBuilder::ltExpr(index, pred_id->extent());
    pred = SimplifyingIrBuilder::andExpr(pred, new_pred)->as<Bool>();
  }

  return pred;
}

namespace {

std::unordered_set<Val*> getNonUnswitchedRootDomains(
    const std::vector<kir::ForLoop*>& loops,
    size_t unswitched_loop_index) {
  std::vector<Val*> non_unswited_leaf_domains;
  std::transform(
      loops.begin(),
      loops.begin() + unswitched_loop_index,
      std::back_inserter(non_unswited_leaf_domains),
      [&](kir::ForLoop* loop) { return loop->iter_domain(); });

  auto non_unswitched_inputs =
      IterVisitor::getInputsTo(non_unswited_leaf_domains);

  auto non_unswitched_root_doms =
      ir_utils::filterByType<IterDomain>(non_unswitched_inputs);

  std::unordered_set<Val*> non_unswitched_concrete_root_domains;

  std::transform(
      non_unswitched_root_doms.begin(),
      non_unswitched_root_doms.end(),
      std::inserter(
          non_unswitched_concrete_root_domains,
          non_unswitched_concrete_root_domains.end()),
      [&](auto root_dom) {
        return GpuLower::current()->caMap()->getConcreteMappedID(
            root_dom, IdMappingMode::EXACT);
      });

  return non_unswitched_concrete_root_domains;
}

bool isFullyUnswitched(
    IterDomain* loop_id,
    const std::unordered_set<Val*>& non_unswitched_root_domains) {
  auto root_vals = IterVisitor::getInputsTo({loop_id});

  auto root_domains = ir_utils::filterByType<IterDomain>(root_vals);

  return std::none_of(
      root_domains.begin(), root_domains.end(), [&](auto root_dom) {
        auto concrete_root_dom =
            GpuLower::current()->caMap()->getConcreteMappedID(
                root_dom, IdMappingMode::EXACT);
        return non_unswitched_root_domains.count(concrete_root_dom) > 0;
      });
}

} // namespace

std::unordered_map<
    ParallelType,
    ParallelizedDomainPredicate::PredicateInfo,
    TypeHash>
ParallelizedDomainPredicate::getPredicateMap(
    const Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::ForLoop* unswitched_loop) {
  const auto gpu_lower = GpuLower::current();
  auto output_tvs = ir_utils::getTvs(expr->outputs());

  if (output_tvs.empty()) {
    return {};
  }

  // Initialize a map with empty predicate info
  std::unordered_map<ParallelType, PredicateInfo, TypeHash> map;
  for (auto pt : kParallelTypeThreads) {
    map.insert({pt, PredicateInfo(pt)});
  }

  // For each loop, check if it's parallelized by an non-exact
  // threading dimension. If yes and it's used in the given expr, the
  // domain needs to be protected by a predicate on the thread/block
  // index.

  bool within_unswitch = false;
  std::unordered_set<Val*> non_unswitched_root_domains;

  for (const auto i : c10::irange(loops.size())) {
    auto loop = loops[i];

    // Parallel dimensions need not be predicated if fully unswitched.
    if (loop == unswitched_loop) {
      within_unswitch = true;
      non_unswitched_root_domains = getNonUnswitchedRootDomains(loops, i);
    }

    auto loop_id = loop->iter_domain();
    auto loop_ptype = loop_id->getParallelType();

    // Not necessary to add a predicate if the paralle type is exact
    if (!isParallelTypeThread(loop_ptype) ||
        gpu_lower->parallelDimensionMap().isExact(loop_ptype)) {
      continue;
    }

    // Parallel dimensions need not be predicated if fully unswitched.
    if (within_unswitch &&
        isFullyUnswitched(loop_id, non_unswitched_root_domains)) {
      continue;
    }

    for (auto tv : output_tvs) {
      // Check if the loop domain is used by the output tensor
      auto it = std::find_if(
          tv->domain()->domain().begin(),
          tv->domain()->domain().end(),
          [&](auto tv_id) {
            return gpu_lower->caMap()->areMapped(
                loop_id, tv_id, IdMappingMode::EXACT);
          });
      if (it == tv->domain()->domain().end()) {
        continue;
      }

      IterDomain* tv_id = *it;

      // If the corresponding domain is a broadcast, it's not really used.
      if (tv_id->isBroadcast()) {
        continue;
      }

      // If it's a root domain, it should be covered by the root
      // predicates, so no extra predicate is required.
      if (std::find(
              tv->domain()->getRootDomain().begin(),
              tv->domain()->getRootDomain().end(),
              tv_id) != tv->domain()->getRootDomain().end()) {
        continue;
      }

      // tv_id needs to be predicated. Adds it to the PredicateInfo map.
      auto& info = map.at(loop_ptype);
      info.addDomain(tv_id);
    }
  }

  return map;
}

Bool* ParallelizedDomainPredicate::getPredicate(
    const Expr* expr,
    const std::vector<kir::ForLoop*>& loops) {
  auto pred_map = getPredicateMap(expr, loops);

  Val* pred = GpuLower::current()->kernel()->trueVal();

  for (auto pt : kParallelTypeThreads) {
    auto pred_info_it = pred_map.find(pt);
    if (pred_info_it != pred_map.end()) {
      const auto& pred_info = pred_info_it->second;
      auto tid_pred = pred_info.getPredicate();
      pred = SimplifyingIrBuilder::andExpr(pred, tid_pred);
    }
  }

  TORCH_INTERNAL_ASSERT(pred != nullptr);
  return pred->as<Bool>();
}

UnswitchPredicateKey::UnswitchPredicateKey()
    : predicated_concrete_id_(nullptr) {
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }
}

// For a predicated concrete domain, id, find which thread parallel
// types are used. For each used parallel type, find the concrete
// domain that the paralllel type is associated with. The parallelized
// concrete domains are used to uniquely collect all necessary
// unswitch predicates.
UnswitchPredicateKey::UnswitchPredicateKey(
    IterDomain* predicated_consumer_id,
    TensorView* consumer_tv,
    IterDomain* predicated_concrete_id)
    : predicated_concrete_id_(predicated_concrete_id) {
  // Initialize the parallelized domain map
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }

  std::vector<Val*> all_parallelized_consumer_leaf_ids;
  std::copy_if(
      consumer_tv->domain()->domain().begin(),
      consumer_tv->domain()->domain().end(),
      std::back_inserter(all_parallelized_consumer_leaf_ids),
      [](IterDomain* x) { return isParallelTypeThread(x->getParallelType()); });

  // If the consumer domais are not parallelized at all, no need to
  // differentiate keys based on how the predicated id is parallelized
  if (all_parallelized_consumer_leaf_ids.empty()) {
    return;
  }

  // All domains that are parallelized descendants of predicated_consumer_id
  auto all_parallelized_consumer_ids = DependencyCheck::getAllValsBetween(
      {predicated_consumer_id}, all_parallelized_consumer_leaf_ids);
  // Just pick leaf domains
  std::vector<IterDomain*> parallelized_consumer_leaf_ids;
  std::copy_if(
      consumer_tv->domain()->domain().begin(),
      consumer_tv->domain()->domain().end(),
      std::back_inserter(parallelized_consumer_leaf_ids),
      [&](IterDomain* x) {
        return std::find(
                   all_parallelized_consumer_ids.begin(),
                   all_parallelized_consumer_ids.end(),
                   x) != all_parallelized_consumer_ids.end();
      });

  if (parallelized_consumer_leaf_ids.empty()) {
    // None of the parallelized leaf domains are derived from
    // predicated_consumer_id
    return;
  }

  // Find the corresponding concrete id for each parallel type
  for (auto consumer_leaf : parallelized_consumer_leaf_ids) {
    auto pt = consumer_leaf->getParallelType();
    auto concrete_leaf = GpuLower::current()->caMap()->getConcreteMappedID(
        consumer_leaf, IdMappingMode::EXACT);
    parallel_concrete_ids_.at(pt) = concrete_leaf;
  }
}

std::string UnswitchPredicateKey::toString() const {
  std::stringstream ss;
  ss << "Predicated domain: ";
  if (predicatedId() != nullptr) {
    ss << predicatedId();
  } else {
    ss << "null";
  }
  for (auto pt : kParallelTypeThreads) {
    auto pid = parallelId(pt);
    ss << ", " << pt << ": ";
    if (pid) {
      ss << pid;
    } else {
      ss << "null";
    }
  }
  return ss.str();
}

std::size_t UnswitchPredicateKeyHash::operator()(
    const UnswitchPredicateKey& key) const {
  auto h = std::hash<const IterDomain*>{}(key.predicatedId());
  for (auto pt : kParallelTypeThreads) {
    h = h ^ std::hash<const IterDomain*>{}(key.parallelId(pt));
  }
  return h;
};

Bool* PredicateCompute::getInlinePredicate(
    const Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    Bool* thread_pred,
    PredicateType pred_type) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getInlinePredicate");

  const auto gpu_lower = GpuLower::current();

  // If outputs are registers, no need to predicate for threads
  if (isOutputLocal(expr)) {
    thread_pred = gpu_lower->kernel()->trueVal();
  }

  if (loops.empty()) {
    TORCH_INTERNAL_ASSERT(thread_pred != nullptr);
    return thread_pred;
  }

  auto out_tv = ir_utils::getTvOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Missing TensorView output");

  // Predicates for non-exact parallel dimensions must be used even
  // when PredicateElimination::canOmitPredicate is true.
  auto parallel_dom_pred =
      ParallelizedDomainPredicate::getPredicate(expr, loops);
  TORCH_INTERNAL_ASSERT(parallel_dom_pred != nullptr);

  if (gpu_lower->predicateElimination().canOmitPredicate(expr)) {
    return SimplifyingIrBuilder::andExpr(thread_pred, parallel_dom_pred)
        ->as<Bool>();
  }

  auto pred_info_vec = Index::getReferenceRootPredicates(
      out_tv, loops, nullptr, pred_type == PredicateType::Padding);

  std::vector<Bool*> preds;

  // When pred_type is ReductionWrite, filter out predicates for
  // reduction axes. For blockReduce, this is necessary when reduction
  // axes start at non-zero offsets and parallelized with TID since
  // blockReduce returns a valid output only at offset-zero
  // threads. Similarly, for gridReduce, the last block to store the
  // output may be predicated out with the read predicate, so the
  // write predicate needs to ignore the reduction axes.
  bool non_zero_start_found = false;
  for (const auto& pred_info : pred_info_vec) {
    if (pred_type == PredicateType::ReductionWrite) {
      const auto& consumer_ids = pred_info.rootIds();
      bool pred_for_reduction_axis = false;
      for (auto consumer_id : consumer_ids) {
        if (consumer_id->isReduction()) {
          if (!consumer_id->start()->isZeroInt()) {
            non_zero_start_found = true;
          }
          pred_for_reduction_axis = true;
          break;
        }
      }
      // Don't add the predicate if it corresponds to a reduction axis
      if (pred_for_reduction_axis) {
        continue;
      }
    }
    preds.push_back(pred_info.startPredicate());
    preds.push_back(pred_info.stopPredicate());
  }

  // When generating a predicate for blockReduce writes and not for
  // gridReduce, if all reduction axes start with zero, we can just
  // use the same predicate for reads. nullptr is returned then.
  if (pred_type == PredicateType::ReductionWrite && !non_zero_start_found &&
      !out_tv->domain()->hasGridReduction()) {
    return nullptr;
  }

  preds.push_back(parallel_dom_pred);

  if (thread_pred != nullptr) {
    preds.push_back(thread_pred);
  }

  if (preds.empty()) {
    return GpuLower::current()->kernel()->trueVal();
  }

  Val* cond = preds[0];
  for (const auto i : c10::irange(1, preds.size())) {
    cond = SimplifyingIrBuilder::andExpr(cond, preds[i]);
  }

  return cond->as<Bool>();
}

Bool* UnswitchPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::get");

  UnswitchPredicate up(outer_loops, unrolled_loop);

  Val* unswitch_pred = GpuLower::current()->kernel()->trueVal();
  for (auto pred : up.predicates_) {
    unswitch_pred = SimplifyingIrBuilder::andExpr(unswitch_pred, pred);
  }

  return unswitch_pred->as<Bool>();
}

void UnswitchPredicate::predicateOn(Expr* tv_expr) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  const auto gpu_lower = GpuLower::current();

  // FIXME:
  //   Needed to keep the predicate of cp.async initialization to get the
  //   inverted predicate,
  // see [Predicate Inversion for CpAsync]. In a follow up both this part and
  // the [Predicate Inversion for CpAsync] should be cleaned up together.
  if (gpu_lower->predicateElimination().canOmitPredicate(tv_expr) &&
      !ir_utils::isCpAsyncInit(tv_expr)) {
    addParallelizedDomainPredicates(tv_expr);
    return;
  }

  auto out_tv = ir_utils::getTvOutput(tv_expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Missing TensorView output");

  auto ref_pred_info = Index::getReferenceRootPredicates(
      out_tv, for_loops_, unrolled_loop_, false);

  // If RootPredicateInfo has a static predicate that is more
  // restrictive than the current one, replace the current with the
  // new one. If it has a dynamic predicate, add it to the dynamic
  // predicate list. Since the final static predicate can't be
  // determined until all expressions are analyzed, predicates are
  // temporarily placed in the predicated_keys map and the final
  // predicates are generated in the finalize function.

  for (const auto& pred_info : ref_pred_info) {
    TORCH_INTERNAL_ASSERT(pred_info.startPredicate() != nullptr);
    TORCH_INTERNAL_ASSERT(pred_info.stopPredicate() != nullptr);

    const auto& root_ids = pred_info.rootIds();

    bool add_pred = false;

    // Used to find a matching existing MergedPredicates
    UnswitchPredicateKey first_key;
    bool first_key_set = false;

    for (auto root_id : root_ids) {
      auto concrete_root_id = gpu_lower->caMap()->getConcreteMappedID(
          root_id, IdMappingMode::EXACT);

      if (root_id->isBroadcast()) {
        continue;
      }

      UnswitchPredicateKey key(root_id, out_tv, concrete_root_id);
      auto inserted = predicated_keys_.insert(key).second;
      add_pred = add_pred || inserted;

      if (!first_key_set) {
        first_key = key;
        first_key_set = true;
      }
    }

    if (!first_key_set) {
      // No predicate generated
      continue;
    }

    // The start and stop offsets may need to be merged to avoid
    // redundant predicates. When these offsets are zero, nothing is
    // done. When non-zero, find the corresponding MergedPredicates
    // and merge both the start and stop offsets. Note that the
    // offsets are non-zero, the predicates must be generated at a
    // root domain, so root_ids.size() must be one. That unique root
    // domain is used as a key to find the corresponding
    // MergedPredicate.

    // Initialize with an invalid iterator to signal no corresponding
    // MergedPredicates is found yet.
    auto merged_pred_it = pending_predicates_.end();

    if (add_pred) {
      // This is a new predicate for the root domain. Initialize a new
      // MergedPredicates and add it to the pending list.
      UnswitchPredicate::MergedPredicates merged_pred;

      // To look up this MergedPredicates for other predicates
      // generated for the same predicate key
      if (root_ids.size() == 1) {
        merged_pred.predicate_key = first_key;
      }

      pending_predicates_.push_back(merged_pred);

      merged_pred_it =
          pending_predicates_.begin() + pending_predicates_.size() - 1;
    } else if (root_ids.size() == 1) {
      // If not new, try to find a corresponding MergedPredicates.
      merged_pred_it = std::find_if(
          pending_predicates_.begin(),
          pending_predicates_.end(),
          [&first_key](const auto& merged_predicates) {
            return merged_predicates.predicate_key == first_key;
          });
      // Note: It is possible that no matching merged predicate info
      // is found. Since add_pred is false here, the root domain is
      // already predicated. It must mean that the root domain
      // is included in a contiguous merged domain, which means there
      // must be no halo-extended domain involved.
    }

    // If a corresponding MergedPredicates is found, merge both the
    // start and stop offsets.
    if (merged_pred_it != pending_predicates_.end()) {
      mergeUnswitchPredicateOffsets(
          pred_info.startPredicate(),
          pred_info.startOffset(),
          merged_pred_it->start,
          true);

      mergeUnswitchPredicateOffsets(
          pred_info.stopPredicate(),
          pred_info.stopOffset(),
          merged_pred_it->stop,
          false);
    }
  }

  addParallelizedDomainPredicates(tv_expr);
}

void UnswitchPredicate::addParallelizedDomainPredicates(Expr* tv_expr) {
  auto pred_map = ParallelizedDomainPredicate::getPredicateMap(
      tv_expr, for_loops_, unrolled_loop_);
  for (auto pt : kParallelTypeThreads) {
    auto pred_info_it = pred_map.find(pt);
    if (pred_info_it == pred_map.end()) {
      continue;
    }
    const auto& new_info = pred_info_it->second;
    auto& predicated =
        parallelized_dom_predicates_
            .insert({pt, ParallelizedDomainPredicate::PredicateInfo{pt}})
            .first->second;
    for (auto id : new_info.ids()) {
      if (predicated.addDomain(id)) {
        predicates_.push_back(new_info.getPredicate());
      }
    }
  }
}

void UnswitchPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openLoop");

  for_loops_.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTvOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }

  for_loops_.pop_back();
}

void UnswitchPredicate::openIte(kir::IfThenElse* ite) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openIte");

  // only expand the ite thenBody
  for (auto expr : ite->thenBody().exprs()) {
    if (ir_utils::isTvOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }
}

void UnswitchPredicate::finalize() {
  for (const auto& merged_pred : pending_predicates_) {
    const auto& start_info = merged_pred.start;
    if (start_info.static_pred) {
      predicates_.push_back(start_info.static_pred);
    }
    for (auto dynamic_pred : start_info.dynamic_preds) {
      predicates_.push_back(dynamic_pred);
    }
    const auto& stop_info = merged_pred.stop;
    if (stop_info.static_pred) {
      predicates_.push_back(stop_info.static_pred);
    }
    for (auto dynamic_pred : stop_info.dynamic_preds) {
      predicates_.push_back(dynamic_pred);
    }
  }
}

void UnswitchPredicate::mergeUnswitchPredicateOffsets(
    Bool* predicate,
    Val* offset,
    MergedPredicates::Info& merged_predicate_info,
    bool is_start) {
  auto is_more_restrictive = [&is_start](int64_t new_val, int64_t current_val) {
    if (is_start) {
      return new_val < current_val;
    } else {
      return new_val > current_val;
    }
  };

  auto offset_int = dynamic_cast<Int*>(offset);
  // If it's a static predicate, replace the current one if it's
  // more restrictive. If it's dynamic, just adds it to the dynamic
  // predicate list.
  if (offset_int && offset_int->isConst()) {
    auto offset_const = offset_int->value().value();
    auto& static_pred = merged_predicate_info.static_pred;
    auto& static_offset = merged_predicate_info.static_offset;
    if (static_pred == nullptr ||
        is_more_restrictive(offset_const, static_offset)) {
      static_pred = predicate;
      static_offset = offset_const;
    }
  } else {
    merged_predicate_info.dynamic_preds.push_back(predicate);
  }
}

UnswitchPredicate::UnswitchPredicate(
    std::vector<kir::ForLoop*> outer_loops,
    kir::ForLoop* unrolled_loop)
    : for_loops_(std::move(outer_loops)), unrolled_loop_(unrolled_loop) {
  openLoop(unrolled_loop);
  finalize();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
