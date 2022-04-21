#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

// Is the provided IterDomain an Leaf of provided TensorView and within its
// computeAtPosition
bool idIsAComputeAtLeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->domain()->domain().begin();
  auto end = tv->domain()->domain().begin() + tv->getComputeAtPosition();
  return std::find(begin, end, id) != end;
}

// Is the provided IterDomain an Leaf of provided TensorView
bool idIsALeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->domain()->domain().begin();
  auto end = tv->domain()->domain().end();
  return std::find(begin, end, id) != end;
}

} // namespace

IterDomainGraph::IterDomainGraph(Fusion* fusion) {
  build(fusion);
}

void IterDomainGraph::build(Fusion* fusion) {
  // Initialize a node for every iteration domain
  for (auto tv : ir_utils::allTvs(fusion)) {
    const auto& root_domain = tv->getRootDomain();
    const auto& domain = tv->domain()->domain();

    // Grab all values in the history of the tensor view's domain
    auto all_vals = DependencyCheck::getAllValsBetween(
        {root_domain.begin(), root_domain.end()},
        {domain.begin(), domain.end()});

    // Filter so we only have iteration domains (ignore Ints used in split)
    auto all_ids = ir_utils::filterByType<IterDomain>(all_vals);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      bool is_view_rfactor_id = false;
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration domain
        // is marked as an rfactor product and is in the rfactor domain, it's a
        // view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->getMaybeRFactorDomain();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          is_view_rfactor_id = true;
        }
      }
      bool is_leaf_id =
          std::find(domain.begin(), domain.end(), id) != domain.end();
      initializeId(id, is_view_rfactor_id, is_leaf_id);
    }
  }

  // All ID's are initialized, start connecting them on the permissive, exact,
  // and loop dimensions.

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* first_output_tv = nullptr;

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

        // Multi output mapping, outputs are required to have the same domain
        // and same transformations, so they can be mapped in permissive/exact,
        // and when within compute at position of domain()->domain() in the
        // parallel map.
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
          permissive_nodes_.mapEntries(f_id, c_id);
          exact_nodes_.mapEntries(f_id, c_id);
          if (idIsALeafDomain(f_id, first_output_tv)) {
            loop_nodes_.mapEntries(f_id, c_id);
          }
          sibling_sets_.mapEntries(f_id, c_id);
        }
      }

      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        // If outside computeAt axis, we don't want to directly map
        // consumer/producer as their thread mappings could change as long as
        // it's across shared/global memory.
        auto pairwise_map = PairwiseRootDomainMap(p_tv, c_tv);
        const auto& permissive_c2p_root_map =
            pairwise_map.mapConsumerToProducer(c_tv->domain(), p_tv->domain());

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We want to replay producer as consumer instead
        // of the other way around since consumer may have some broadcasted axes
        // producer doesn't have merged into loops producer may use. If we did
        // consumer as producer we wouldn't have this information in the
        // mapping. If we're using this map for indexing, we do not want to
        // propagate broadcast mismatches. If we're using it to identify loop
        // nests, we do want to propagate mismatches.
        auto permissive_replay_PasC =
            BestEffortReplay::replayPasC(p_tv, c_tv, -1, pairwise_map);

        const auto& permissive_c2p_map = permissive_replay_PasC.getReplay();

        // For exact mapings do not map any broadcast dimensions to
        // non-broadcast dimensions. Prevent any broadcasted axes being mapped
        // to non-broadcasted axes.
        auto exact_c2p_root_map = permissive_c2p_root_map;
        for (auto it = exact_c2p_root_map.begin();
             it != exact_c2p_root_map.end();) {
          auto c_id = it->first;
          auto p_id = it->second;
          if (p_id->isBroadcast() != c_id->isBroadcast()) {
            it = exact_c2p_root_map.erase(it);
          } else {
            ++it;
          }
        }

        // Same as permissive above but for exact
        auto exact_replay_PasC = BestEffortReplay(
            p_tv->domain()->domain(),
            c_tv->domain()->domain(),
            exact_c2p_root_map);

        const auto& exact_c2p_map = exact_replay_PasC.getReplay();

        for (auto entry : exact_c2p_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          exact_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);
        }

        for (auto entry : permissive_c2p_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          if (idIsAComputeAtLeafDomain(p_id, p_tv)) {
            loop_nodes_.mapEntries(c_id, p_id);
          }
          permissive_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);
        }

        // Make sure we always get root mapping for the permissive map. Because
        // of forwarding we could otherwise miss some root mappings.
        for (auto entry : permissive_c2p_root_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          // Map the id's together
          permissive_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);
        }
      }
    }
  }
}

void IterDomainGraph::initializeId(
    IterDomain* id,
    bool is_view_rfactor_id,
    bool is_leaf_id) {
  permissive_nodes_.initializeSet(id);
  exact_nodes_.initializeSet(id);
  if (is_leaf_id) {
    loop_nodes_.initializeSet(id);
  }
  consumers_[id] = {};
  producers_[id] = {};
  sibling_sets_.initializeSet(id);

  all_ids_.pushBack(id);

  if (is_view_rfactor_id) {
    view_rfactor_ids_.emplace(id);
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion) : id_graph_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  trivial_reduction_info_.build(fusion);
  buildConcreteIds();
}

void ComputeAtMap::validateAndPropagatePType() {
  for (const auto& loop_disjoint_set : id_graph_.loopNodes().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();
      TORCH_INTERNAL_ASSERT(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());
      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

bool ComputeAtMap::areMapped(
    IterDomain* id0,
    IterDomain* id1,
    IdMappingMode mode) const {
  return disjointSetOf(id0, mode)->has(id1);
}

IterDomain* ComputeAtMap::computeConcreteId(
    IterDomain* id,
    IdMappingMode mode) {
  const auto& disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size(),
      "Empty disjoint set found for ",
      id->toString());

  if (disjoint_set_shared_ptr->vector().size() == 1) {
    return disjoint_set_shared_ptr->vector().front();
  }

  VectorOfUniqueEntries<IterDomain*> maybe_concrete_ids;
  for (auto id : disjoint_set_shared_ptr->vector()) {
    bool id_output = true;
    for (auto consumer_id : id_graph_.consumers().at(id).vector()) {
      if (disjoint_set_shared_ptr->has(consumer_id)) {
        id_output = false;
        break;
      }
    }
    if (id_output) {
      maybe_concrete_ids.pushBack(id);
    }
  }

  TORCH_INTERNAL_ASSERT(
      maybe_concrete_ids.vector().size(),
      "No potential concrete_id's found for ",
      id->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  IterDomain* concrete_id = nullptr;
  int max_iter_root_count = 0;
  int max_bcast_root_count = 0;

  for (auto maybe_concrete_id : maybe_concrete_ids.vector()) {
    std::unordered_set<IterDomain*> root_ids;
    std::deque<IterDomain*> to_visit;

    to_visit.push_back(maybe_concrete_id);
    while (to_visit.size()) {
      auto current_id = to_visit.front();
      to_visit.pop_front();
      if (isViewRfactor(current_id)) {
        root_ids.emplace(current_id);
        continue;
      }

      // push back producer IterDomains or add root if they don't exist
      auto producer_vals = ir_utils::producerValsOf(current_id);
      auto producer_ids = ir_utils::filterByType<IterDomain>(producer_vals);

      if (producer_ids.empty()) {
        root_ids.emplace(current_id);
      } else {
        to_visit.insert(
            to_visit.end(), producer_ids.begin(), producer_ids.end());
      }
    }

    int bcast_root_count = std::count_if(
        root_ids.begin(), root_ids.end(), [&](IterDomain* root_id) {
          return root_id->isBroadcast()
              // TODO: This shouldn't have a negative impact, but (emperically)
              // might not be necessary
              || trivial_reduction_info_.isDerived(root_id);
        });
    int iter_root_count = (int)root_ids.size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_id;
    }
  } // end maybe_concrete_id
  TORCH_INTERNAL_ASSERT(
      concrete_id != nullptr,
      "Something went wrong, could not find a concrete id.");

  return concrete_id;
}

void ComputeAtMap::buildConcreteIds() {
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.permissiveNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::EXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.loopNodes().disjointSets()) {
    TORCH_INTERNAL_ASSERT(
        disjoint_set_shared_ptr->vector().size(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

IterDomain* ComputeAtMap::getConcreteMappedID(
    IterDomain* id,
    IdMappingMode mode) const {
  auto disjoint_set_shared_ptr = disjointSetOf(id, mode);

  TORCH_INTERNAL_ASSERT(
      disjoint_set_shared_ptr->vector().size() > 0,
      "Empty disjoint set found for ",
      id->toString());

  auto cache_it = concrete_id_cache_.find(disjoint_set_shared_ptr);

  TORCH_INTERNAL_ASSERT(
      cache_it != concrete_id_cache_.end(),
      "Could not find concrete id for: ",
      id->toString(),
      " with mode ",
      mode);

  return cache_it->second;
}

std::string ComputeAtMap::toString() const {
  std::stringstream ss;
  ss << "Compute at map { \n";
  ss << "Permissive map:\n" << id_graph_.permissiveNodes().toString() << "\n";
  ss << "Exact map:\n" << id_graph_.exactNodes().toString() << "\n";
  ss << "Loop map:\n" << id_graph_.loopNodes().toString() << "\n";
  ss << "Consumer maps:\n";
  for (auto entry : id_graph_.consumers()) {
    ss << "  " << entry.first->toString() << " :: " << entry.second.toString()
       << "\n";
  }

  ss << "Producer maps:\n";
  for (auto entry : id_graph_.producers()) {
    ss << "  " << entry.first->toString() << " :: " << entry.second.toString()
       << "\n";
  }

  ss << "Sibling map:\n" << id_graph_.siblings().toString() << "\n";

  ss << "} compute at map" << std::endl;
  return ss.str();
}

bool ComputeAtMap::isViewRfactor(IterDomain* ref_id) const {
  return id_graph_.viewRfactorIds().find(ref_id) !=
      id_graph_.viewRfactorIds().end();
}

std::vector<IterDomain*> ComputeAtMap::getViewRfactorDomainsOfIdGroup(
    IterDomain* ref_id,
    IdMappingMode mode) const {
  auto disjoint_set = disjointSetOf(ref_id, mode);
  std::vector<IterDomain*> rfactor_ids;
  for (auto disjoint_id : disjoint_set->vector()) {
    if (id_graph_.viewRfactorIds().find(disjoint_id) !=
        id_graph_.viewRfactorIds().end()) {
      rfactor_ids.push_back(disjoint_id);
    }
  }
  return rfactor_ids;
}

const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& ComputeAtMap::
    disjointSetOf(IterDomain* id, IdMappingMode mode) const {
  switch (mode) {
    case IdMappingMode::PERMISSIVE:
      return id_graph_.permissiveNodes().disjointSetMap().at(id);
    case IdMappingMode::EXACT:
      return id_graph_.exactNodes().disjointSetMap().at(id);
    case IdMappingMode::LOOP:
      return id_graph_.loopNodes().disjointSetMap().at(id);
  }
  TORCH_INTERNAL_ASSERT(false, "Error with mapping mode provided.");
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
