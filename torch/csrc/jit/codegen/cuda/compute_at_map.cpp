#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

//! Class to figure out how many non-broadcast axes and how many broadcast axes
//! were used to produce an iter domain. This is important for figuring out what
//! the correct broadcasted extent is of an iteration domain.
//!
//! When GpuLower is available, trivial reductions are not counted as
//! concrete domains so that they should not be used to generate
//! for-loops.
class InputDomainCounter : public IterVisitor {
 public:
  // Returns <number of {non-braodcast non-reduction iteration domains,
  // broadcast and trivial reduction domains}, number of broadcast domains> used
  // to generate the iteration domains in provided target domain.
  static std::unordered_map<IterDomain*, std::pair<int, int>> produceCounts(
      const std::vector<TensorView*>& tvs,
      const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
          rfactor_dep_map,
      const TrivialReductionInfo* trivial_reduction_info) {
    InputDomainCounter counter(rfactor_dep_map);

    for (auto tv : tvs) {
      auto& domain = tv->domain()->domain();
      counter.traverse(domain);
    }

    // Accumulate for count map
    std::unordered_map<IterDomain*, std::pair<int, int>> count_map;
    for (const auto& entry : counter.domain_set_) {
      auto id = entry.first;
      auto input_id_set = entry.second;
      int concrete_counts = 0;
      int broadcast_counts = 0;
      for (auto input_id : input_id_set) {
        if (input_id->isBroadcast() ||
            trivial_reduction_info->isDerived(input_id)) {
          broadcast_counts++;
        } else {
          concrete_counts++;
        }
      }
      count_map[id] = {concrete_counts, broadcast_counts};
    }

    // Make sure all domains are mapped.
    for (auto tv : tvs) {
      auto& domain = tv->domain()->domain();
      for (auto id : domain) {
        TORCH_INTERNAL_ASSERT(
            count_map.find(id) != count_map.end(),
            "Missing count of ",
            id->toString());
      }
    }

    return count_map;
  }

 private:
  InputDomainCounter(
      const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
          rfactor_dep_map)
      : rfactor_dep_map_(rfactor_dep_map) {}

 private:
  void traverse(std::vector<IterDomain*> domain) {
    if (domain.empty()) {
      return;
    }
    traverseFrom(
        domain[0]->fusion(), std::vector<Val*>(domain.begin(), domain.end()));

    // Inputs may be root domains which wouldn't have any entries if no exprs
    // were traversed, so manually insert their count
    for (auto id : domain) {
      if (domain_set_.find(id) == domain_set_.end()) {
        TORCH_INTERNAL_ASSERT(
            id->definition() == nullptr,
            "Expected id: ",
            id->toString(),
            " to not have transformations in its history.");
        getEntry(id);
      }
    }
  }

  std::unordered_set<IterDomain*>& getEntry(IterDomain* id) {
    auto domain_set_it = domain_set_.find(id);

    if (domain_set_it == domain_set_.end()) {
      domain_set_it =
          domain_set_
              .emplace(std::make_pair(id, std::unordered_set<IterDomain*>()))
              .first;
      domain_set_it->second.emplace(id);
    }

    // If id is a consumer of an rfactor ID, propagates the domains
    // accumulated in the rfactor ID.
    auto rf_dep_it = rfactor_dep_map_.find(id);
    if (rf_dep_it != rfactor_dep_map_.end()) {
      auto& ds = domain_set_it->second;
      for (IterDomain* rf_dep : rf_dep_it->second) {
        auto rf_dep_domain_set_it = domain_set_.find(rf_dep);
        TORCH_INTERNAL_ASSERT(
            rf_dep_domain_set_it != domain_set_.end(),
            "Domains for an rfactor domain should have already been computed but not found: ",
            rf_dep->toString());
        ds.insert(
            rf_dep_domain_set_it->second.begin(),
            rf_dep_domain_set_it->second.end());
      }
    }

    return domain_set_it->second;
  }

  void handle(Expr* expr) override {
    // If we end up moving swizzle to an Expr it would be identity here, instead
    // of outputs being a function of all inputs
    switch (expr->getExprType().value()) {
      case (ExprType::Split):
      case (ExprType::Merge):
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Invalid expr type found in transform traversal.");
    }

    // Gather all input domains
    std::unordered_set<IterDomain*> resulting_set;
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      const auto& input_entry = getEntry(input_id);
      resulting_set.insert(input_entry.begin(), input_entry.end());
    }

    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      TORCH_INTERNAL_ASSERT(
          domain_set_.emplace(std::make_pair(output_id, resulting_set)).second);
      if (rfactor_dep_map_.find(output_id) != rfactor_dep_map_.end()) {
        // If output_id has rfactor dependency, just call getEntry
        // which will propagate the dependent domains to output_id
        getEntry(output_id);
      }
    }
  }

 private:
  const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
      rfactor_dep_map_;
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>> domain_set_;
};

// Only used once, consider removing.
template <class T>
std::deque<T*> deduplicateDeque(const std::deque<T*>& deque) {
  std::unordered_set<T*> used;
  std::deque<T*> deduped;
  for (auto entry : deque) {
    if (used.find(entry) == used.end()) {
      deduped.push_back(entry);
      used.emplace(entry);
    }
  }
  return deduped;
}

void assertLowered(bool lowered) {
  TORCH_INTERNAL_ASSERT(
      lowered,
      "Tried to accessed lowered values of compute at map,",
      " however a valid lowering was not set when compute at map was created.");
}

} // namespace

ComputeAtMap::ComputeAtMap(
    Fusion* fusion,
    MappingMode mapping_mode,
    const TrivialReductionInfo* _trivial_reduction_info)
    : mapping_mode_(mapping_mode) {
  if (_trivial_reduction_info == nullptr) {
    TrivialReductionInfo trivial_reduction_info;
    trivial_reduction_info.build(fusion);
    build(fusion, &trivial_reduction_info);
  } else {
    build(fusion, _trivial_reduction_info);
  }
}

void ComputeAtMap::mapIds(IterDomain* id0, IterDomain* id1) {
  auto set_it_0 = disjoint_iter_set_maps_.find(id0);
  auto set_it_1 = disjoint_iter_set_maps_.find(id1);
  if (set_it_0 == disjoint_iter_set_maps_.end() &&
      set_it_1 == disjoint_iter_set_maps_.end()) {
    // Neither iter domain has been mapped, so make a new disjoint set
    auto new_set = std::make_shared<std::deque<IterDomain*>>();
    new_set.get()->push_back(id0);
    new_set.get()->push_back(id1);
    disjoint_iter_set_maps_.emplace(std::make_pair(id0, new_set));
    disjoint_iter_set_maps_.emplace(std::make_pair(id1, new_set));
    disjoint_iter_sets_.push_back(new_set);

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      if (id0->isParallelized() && id1->isParallelized()) {
        // Both are parallelized, make sure they're the same, set entry for
        // parallel map
        TORCH_INTERNAL_ASSERT(
            id0->getParallelType() == id1->getParallelType(),
            "Parallel type of ",
            id0,
            " should match ",
            id1);
        parallel_type_map_[new_set] = id0->getParallelType();
      } else if (id0->isParallelized() || id1->isParallelized()) {
        // Only one is parallelized, set entry for parallel map
        parallel_type_map_[new_set] = id0->isParallelized()
            ? id0->getParallelType()
            : id1->getParallelType();
      }
    }

  } else if (
      set_it_0 != disjoint_iter_set_maps_.end() &&
      set_it_1 != disjoint_iter_set_maps_.end()) {
    // Both iter domains have been mapped, so join their sets together
    auto set0_ptr = set_it_0->second;
    auto set1_ptr = set_it_1->second;

    // If the sets are already the same, do nothing
    if (set0_ptr == set1_ptr) {
      return;
    }

    // Place everything in set1 into set0 and remap all ID's in set1 to set0
    auto& set1 = *set1_ptr;
    for (auto id : set1) {
      set0_ptr->push_back(id);
      disjoint_iter_set_maps_[id] = set0_ptr;
    }

    // set1 no longer needed as its IDs are copied into set0
    disjoint_iter_sets_.erase(std::find(
        disjoint_iter_sets_.begin(), disjoint_iter_sets_.end(), set1_ptr));

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_type_0_it = parallel_type_map_.find(set0_ptr);
      auto parallel_type_1_it = parallel_type_map_.find(set1_ptr);
      if (parallel_type_0_it != parallel_type_map_.end() &&
          parallel_type_1_it != parallel_type_map_.end()) {
        // If both sets had a parallel type associated with them, make sure they
        // are the same
        TORCH_INTERNAL_ASSERT(
            parallel_type_0_it->second == parallel_type_1_it->second);
      } else if (parallel_type_1_it != parallel_type_map_.end()) {
        // Set 1 has a parallel type, set 0 does not, set parallel entry
        parallel_type_map_[set0_ptr] = parallel_type_1_it->second;
      }
      // Else set 0 already has the right parallel type set in the map, if at
      // all

      // Remove set1 from the parallel type map as it shouldn't exist anymore
      parallel_type_map_.erase(set1_ptr);
    }

  } else {
    auto existing_set = set_it_0 != disjoint_iter_set_maps_.end()
        ? set_it_0->second
        : set_it_1->second;
    auto missing_id = set_it_0 != disjoint_iter_set_maps_.end() ? id1 : id0;
    existing_set->push_back(missing_id);
    disjoint_iter_set_maps_[missing_id] = existing_set;

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_type_it = parallel_type_map_.find(existing_set);
      if (parallel_type_it != parallel_type_map_.end() &&
          missing_id->isParallelized()) {
        // existing_set has a parallel type already and missing_id has a
        // parallel type, make sure they match. No need to update map
        TORCH_INTERNAL_ASSERT(
            parallel_type_it->second == missing_id->getParallelType());
      } else if (
          parallel_type_it == parallel_type_map_.end() &&
          id1->isParallelized()) {
        // Set parallel type of existing_set as the newly added missing_id is
        // parallel
        parallel_type_map_[existing_set] = missing_id->getParallelType();
      }
    }
  }
}

bool ComputeAtMap::pullConcreteCountResetIds(
    const torch::jit::fuser::cuda::TrivialReductionInfo* trivial_reduction_info,
    TensorView* tv) {
  bool is_view_like_rfactor = false;
  // If consumer is not used in any other expression we don't have to
  // worry about resolving its view like rfactor in any of the maps.
  if (tv->uses().size()) {
    // Check if this consumer has an rfactor domain without any
    // reductions, this would make it a view-like operation
    if (tv->hasRFactor() &&
        std::none_of(
            // TODO: This should be rfactor domain, but it is breaking some
            // tests likely because ParallelMap only maps leaf domains. Will
            // fix in a follow up.
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            [&trivial_reduction_info](IterDomain* id) {
              return id->isRFactorProduct() &&
                  // If one of the rfactor domains is a reduction we don't
                  // have to perform special handling.
                  (id->isReduction() && !trivial_reduction_info->isDerived(id));
            })) {
      // Definitely rfactor like iteration domain add rfactor iteration
      // domains to the rfactor_concrete_count_reset_domains_ set

      for (auto id : tv->domain()->domain()) {
        if (id->isRFactorProduct()) {
          // Add id as a domain to reset concrete count on for map propagation
          rfactor_concrete_count_reset_domains_.emplace(id);
        }
      }

      is_view_like_rfactor = true;
    }
  }
  return is_view_like_rfactor;
}

void ComputeAtMap::markRFactorDependency(
    IterDomain* producer_id,
    IterDomain* consumer_id) {
  if (rfactor_concrete_count_reset_domains_.find(producer_id) !=
      rfactor_concrete_count_reset_domains_.end()) {
    rf_dep_map_[consumer_id].insert(producer_id);
  } else {
    auto it = rf_dep_map_.find(producer_id);
    if (it != rf_dep_map_.end()) {
      rf_dep_map_[consumer_id].insert(producer_id);
    }
  }
}

void ComputeAtMap::build(
    Fusion* fusion,
    const TrivialReductionInfo* trivial_reduction_info) {
  TORCH_INTERNAL_ASSERT(
      trivial_reduction_info != nullptr,
      "Trivial reduction info needs to be constructed and passed in to compute at map build.");

  auto fusion_inputs = fusion->inputs();
  // Pull view like rfactor id's out of inputs for later processing. This isn't
  // needed today because rfactor domains are removed from inputs, however, we
  // should likely support rfactor directly on inputs instead of doing that so
  // leaving this in.
  for (auto tv_inp : ir_utils::filterByType<TensorView>(fusion_inputs)) {
    pullConcreteCountResetIds(trivial_reduction_info, tv_inp);
  }

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* first_output_tv = nullptr;

    // Probably could get away with just checking if it's a view op and if any
    // consumer is used, but to support other view-like operations and limit
    // impact of trivial view like operations, I want to leave this detection
    // generic. This bool just helps prevent us doing extra logic later when
    // we're not processing a view op.
    bool is_view_like_rfactor = false;

    for (auto c_tv : tv_outputs) {
      // Pull view like rfactor id's out of c_tv for later processing
      is_view_like_rfactor =
          pullConcreteCountResetIds(trivial_reduction_info, c_tv);
    }

    for (auto c_tv : tv_outputs) {
      if (first_output_tv == nullptr) {
        first_output_tv = c_tv;
      } else {
        // Map multi outputs of an expression to each other. c is current
        // output, and f as first output. Keep consistent with the later section
        // of producer and consumers. Which here producer is now "first output",
        // and consumer is still consumer. One exception is how the
        // domains left of CA positions are handled in the Parallel
        // map. Those domains are not mapped in producer and consumer
        // mappings as they do not share loops, but are mapped in the
        // case of mapping multiple outputs since they do share the
        // same loops.

        TORCH_INTERNAL_ASSERT(
            c_tv->getRootDomain().size() ==
                first_output_tv->getRootDomain().size(),
            "Multiple outputs with mismatched dimensions is not supported. ",
            "Only supported case is welford op where all outputs tvs have idential domains.");
        // p->f, c->c
        std::unordered_map<IterDomain*, IterDomain*> c2f_root_map;
        for (const auto i :
             c10::irange(first_output_tv->getRootDomain().size())) {
          c2f_root_map.insert(std::make_pair(
              c_tv->getRootDomain()[i], first_output_tv->getRootDomain()[i]));
        }

        // Multi output mapping
        auto replay_FasC = BestEffortReplay(
            first_output_tv->domain()->domain(),
            c_tv->domain()->domain(),
            c2f_root_map);

        auto c2f_map = replay_FasC.getReplay();

        // Map the entire replay map between the multiple
        // consumers even for the Parallel map as they share the same
        // loop.
        for (auto entry : c2f_map) {
          auto c_id = entry.first;
          auto f_id = entry.second;
          // Map the id's together
          mapIds(f_id, c_id);
        }
      }

      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        // If outside computeAt axis, we don't want to directly map
        // consumer/producer as their thread mappings could change as long as
        // it's across shared/global memory.
        auto pairwise_map = PairwiseRootDomainMap(p_tv, c_tv);
        auto c2p_root_map =
            pairwise_map.mapConsumerToProducer(c_tv->domain(), p_tv->domain());

        // For index map do not map any broadcast dimensions to non-broadcast
        // dimensions
        if (mapping_mode_ == MappingMode::INDEX) {
          // Prevent any broadcasted axes being mapped to non-broadcasted axes.
          for (auto it = c2p_root_map.begin(); it != c2p_root_map.end();) {
            auto c_id = it->first;
            auto p_id = it->second;
            if (p_id->isBroadcast() != c_id->isBroadcast()) {
              it = c2p_root_map.erase(it);
            } else {
              ++it;
            }
          }
        }

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We want to replay producer as consumer instead
        // of the other way around since consumer may have some broadcasted axes
        // producer doesn't have merged into loops producer may use. If we did
        // consumer as producer we wouldn't have this information in the
        // mapping. If we're using this map for indexing, we do not want to
        // propagate broadcast mismatches. If we're using it to identify loop
        // nests, we do want to propagate mismatches.
        auto replay_PasC = mapping_mode_ == MappingMode::LOOP ||
                mapping_mode_ == MappingMode::PARALLEL
            ? BestEffortReplay::replayPasC(p_tv, c_tv, -1, pairwise_map)
            : BestEffortReplay(
                  p_tv->domain()->domain(),
                  c_tv->domain()->domain(),
                  c2p_root_map);

        auto c2p_map = replay_PasC.getReplay();

        // If we're creating parallel map, only map the leaf
        // axes. Also, the producer axis must be left of the CA
        // point.
        // Otherwise, map the entire replay map.
        if (mapping_mode_ == MappingMode::PARALLEL) {
          // Mark axes left of compute at point for parallel type tracking
          std::unordered_set<IterDomain*> producer_axes_to_map(
              p_tv->domain()->domain().begin(),
              p_tv->domain()->domain().begin() + p_tv->getComputeAtPosition());

          for (auto c_id : c_tv->domain()->domain()) {
            auto it = c2p_map.find(c_id);
            if (it == c2p_map.end()) {
              continue;
            }
            auto p_id = it->second;
            if (producer_axes_to_map.find(p_id) == producer_axes_to_map.end()) {
              continue;
            }
            mapIds(p_id, c_id);
            markRFactorDependency(p_id, c_id);
          }
        } else {
          for (auto entry : c2p_map) {
            auto c_id = entry.first;
            auto p_id = entry.second;
            // Map the id's together
            mapIds(p_id, c_id);
            markRFactorDependency(p_id, c_id);
          }

          // Make sure we always get root mapping for the loop map. Because of
          // forwarding we could otherwise miss some root mappings.
          if (mapping_mode_ == MappingMode::LOOP) {
            for (auto entry : c2p_root_map) {
              auto c_id = entry.first;
              auto p_id = entry.second;
              // Map the id's together
              mapIds(p_id, c_id);
              markRFactorDependency(p_id, c_id);
            }
          }
        }
      }
    }
  }

  // deduplicate iter domain entries in each set
  for (const auto& iter_set : disjoint_iter_sets_) {
    *iter_set = deduplicateDeque(*iter_set);
  }

  // Compute the concrete id counts to resolve concrete id's in the map, in
  // index map mode this is unnecessary as it doesn't need to resolve concrete
  // ID's the ID's in a disjoint set are all the same extent there.
  concrete_id_count_map_ = InputDomainCounter::produceCounts(
      ir_utils::allTvs(fusion), rf_dep_map_, trivial_reduction_info);

  for (const auto& set : disjoint_iter_sets_) {
    int max_concrete_count = -1;
    // I really don't know if we need to take broadcast ops into concrete id,
    // this does make sure it's not just the largest but is likely to have the
    // most transformations in its history. When we deal with view better in
    // reference replay, we may find out we need to do a better job of ensuring
    // history so we don't have to worry here.
    int max_broadcast_count = -1;
    IterDomain* concrete_id = nullptr;

    // Indicate if the previous ID was an rfactor domain
    for (auto id : *set) {
      // If the previous ID is an rfactor, reset the concrete ID with
      // this ID no matter how many IDs the previous concrete ID has.
      auto counts = getConcreteIdCountOf(id);
      int concrete_count = counts.first;
      int broadcast_count = counts.second;

      if (concrete_count > max_concrete_count) {
        max_concrete_count = concrete_count;
        max_broadcast_count = broadcast_count;
        concrete_id = id;
      } else if (concrete_count == max_concrete_count) {
        if (broadcast_count > max_broadcast_count) {
          max_concrete_count = concrete_count;
          max_broadcast_count = broadcast_count;
          concrete_id = id;
        }
      }
    }

    TORCH_INTERNAL_ASSERT(
        concrete_id != nullptr, "Could not concretize an IterDomain set.");

    for (auto id : *set) {
      concrete_id_map_[id] = concrete_id;
      if (mapping_mode_ == MappingMode::PARALLEL) {
        auto parallel_map_it = parallel_type_map_.find(set);
        // Parallelize all IterDomains to simplify lowering and codegen
        if (parallel_map_it != parallel_type_map_.end()) {
          // Don't propogate vectorize like other parallel types
          if (parallel_map_it->second != ParallelType::Vectorize) {
            id->parallelize(parallel_map_it->second);
          }
        }
      }
    }
  }
}

bool ComputeAtMap::areMapped(IterDomain* id0, IterDomain* id1) const {
  if (id0 == id1) {
    return true;
  }
  auto set0_it = disjoint_iter_set_maps_.find(id0);
  auto set1_it = disjoint_iter_set_maps_.find(id1);
  if (set0_it == disjoint_iter_set_maps_.end() ||
      set1_it == disjoint_iter_set_maps_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

IterDomain* ComputeAtMap::getConcreteMappedID(IterDomain* id) const {
  auto it = concrete_id_map_.find(id);
  if (it != concrete_id_map_.end()) {
    return it->second;
  }
  return id;
}

std::string ComputeAtMap::toString() const {
  std::stringstream ss;

  // We may not have cleaned up non active sets as this is intended for debug,
  // so first grab unique entries and iterate over them.
  std::unordered_set<std::shared_ptr<std::deque<IterDomain*>>> disjoint_sets;

  for (const auto& entry : disjoint_iter_set_maps_) {
    disjoint_sets.emplace(entry.second);
  }

  for (const auto& disjoint_set : disjoint_sets) {
    ss << "  disjoint_set{ ";
    TORCH_INTERNAL_ASSERT(disjoint_set->size() > 0);
    auto concrete_id = concrete_id_map_.at(disjoint_set->front());
    for (auto it = disjoint_set->begin(); it != disjoint_set->end(); it++) {
      if (it != disjoint_set->begin()) {
        ss << ", ";
      }
      ss << (*it);
      if (*it == concrete_id) {
        ss << "*";
      }
    }
    ss << " }";
    if (mapping_mode_ == MappingMode::PARALLEL) {
      if (parallel_type_map_.find(disjoint_set) != parallel_type_map_.end()) {
        ss << "  -> " << parallel_type_map_.at(disjoint_set);
      } else {
        ss << "  -> " << ParallelType::Serial;
      }
    }
    ss << "\n";
  }
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
