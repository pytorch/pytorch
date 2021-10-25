#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
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
  // Returns number of {non-braodcast non-reduction iteration domains, broadcast
  // and trivial reduction domains} used to generate the iteration domains in
  // provided target domain.
  static std::unordered_map<IterDomain*, std::pair<int, int>> produceCounts(
      const std::vector<IterDomain*>& domain,
      GpuLower* gpu_lower) {
    if (domain.empty()) {
      return std::unordered_map<IterDomain*, std::pair<int, int>>();
    }

    InputDomainCounter counter(domain);

    std::unordered_map<IterDomain*, std::pair<int, int>> count_map;
    for (const auto& entry : counter.domain_set_) {
      auto id = entry.first;
      auto input_id_set = entry.second;
      int concrete_counts = 0;
      int broadcast_counts = 0;
      for (auto input_id : input_id_set) {
        if (input_id->isBroadcast() ||
            (gpu_lower &&
             gpu_lower->trivialReductionInfo().isDerived(input_id))) {
          broadcast_counts++;
        } else {
          concrete_counts++;
        }
      }
      count_map[id] = {concrete_counts, broadcast_counts};
    }

    // Inputs may be root domains which wouldn't have any entries if no exprs
    // were traversed, so manually insert their count
    for (auto id : domain) {
      if (count_map.find(id) == count_map.end()) {
        count_map[id] =
            (id->isBroadcast() ||
             (gpu_lower && gpu_lower->trivialReductionInfo().isDerived(id)))
            ? std::make_pair(0, 1)
            : std::make_pair(1, 0);
      }
    }
    return count_map;
  }

 private:
  InputDomainCounter(const std::vector<IterDomain*>& domain_) {
    traverseFrom(
        domain_[0]->fusion(),
        std::vector<Val*>(domain_.begin(), domain_.end()));
  }

 private:
  std::unordered_set<IterDomain*>& getEntry(IterDomain* id) {
    auto domain_set_it = domain_set_.find(id);
    if (domain_set_it == domain_set_.end()) {
      domain_set_it =
          domain_set_
              .emplace(std::make_pair(id, std::unordered_set<IterDomain*>()))
              .first;
      domain_set_it->second.emplace(id);
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

    // Gather all non-broadcast input domains
    std::unordered_set<IterDomain*> resulting_set;
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      auto input_entry = getEntry(input_id);
      resulting_set.insert(input_entry.begin(), input_entry.end());
    }
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      domain_set_.emplace(std::make_pair(output_id, resulting_set));
    }
  }

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
        TORCH_INTERNAL_ASSERT(id0->getParallelType() == id1->getParallelType());
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

void ComputeAtMap::build(Fusion* fusion, GpuLower* gpu_lower) {
  // Consumers can only show up once in an expression, keep track of all of them
  std::vector<TensorView*> consumer_tvs;

  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* first_output_tv = nullptr;
    for (auto c_tv : tv_outputs) {
      consumer_tvs.push_back(c_tv);

      if (first_output_tv == nullptr) {
        first_output_tv = c_tv;
      } else {
        // Map multi outputs of an expression to eachother. c is current output,
        // and f as first output. Keep consistent with the later section of
        // producer and consumers. Which here producer is now "first output",
        // and consumer is still consumer.

        TORCH_INTERNAL_ASSERT(
            c_tv->getRootDomain().size() ==
                first_output_tv->getRootDomain().size(),
            "Multiple outputs with mismatched dimensions is not supported. ",
            "Only supported case is welford op where all outputs tvs have idential domains.");
        // p->f, c->c
        std::unordered_map<IterDomain*, IterDomain*> c2f_root_map;
        for (size_t i = 0; i < first_output_tv->getRootDomain().size(); i++) {
          c2f_root_map.insert(std::make_pair(
              c_tv->getRootDomain()[i], first_output_tv->getRootDomain()[i]));
        }

        // Multi output mapping
        auto replay_FasC = BestEffortReplay(
            first_output_tv->domain()->domain(),
            c_tv->domain()->domain(),
            c2f_root_map);

        auto c2f_map = replay_FasC.getReplay();

        // If we're creating parallel map, only map the leaf
        // axes. Also, the producer axis must be left of the CA
        // point.
        // Otherwise, map the entire replay map.
        if (mapping_mode_ == MappingMode::PARALLEL) {
          // Mark axes left of compute at point for parallel type tracking
          std::unordered_set<IterDomain*> producer_axes_to_map(
              first_output_tv->domain()->domain().begin(),
              first_output_tv->domain()->domain().begin() +
                  first_output_tv->getComputeAtPosition());

          for (auto c_id : c_tv->domain()->domain()) {
            auto it = c2f_map.find(c_id);
            if (it == c2f_map.end()) {
              continue;
            }
            auto f_id = it->second;
            if (producer_axes_to_map.find(f_id) == producer_axes_to_map.end()) {
              continue;
            }
            mapIds(f_id, c_id);
          }
        } else {
          for (auto entry : c2f_map) {
            auto c_id = entry.first;
            auto f_id = entry.second;
            // Map the id's together
            mapIds(f_id, c_id);
          }
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
          }
        } else {
          for (auto entry : c2p_map) {
            auto c_id = entry.first;
            auto p_id = entry.second;
            // Map the id's together
            mapIds(p_id, c_id);
          }
        }
      }
    }
  }

  // deduplicate iter domain entries in each set
  for (const auto& iter_set : disjoint_iter_sets_) {
    *iter_set = deduplicateDeque(*iter_set);
  }

  // For each IterDomain set we will track how many concrete root domains were
  // used to generate the IterDomain. Used to populate conrete_id_map. Concrete
  // ID has maximum of concrete ids, ties are decided based on n_broadcast_ids.
  // Refer to AdvancedLowering5 for why we need to split ties with broadcast
  // dims.
  std::unordered_map<IterDomain*, int> n_concrete_ids_;
  std::unordered_map<IterDomain*, int> n_broadcast_ids_;

  for (auto c_tv : consumer_tvs) {
    auto counts =
        InputDomainCounter::produceCounts(c_tv->domain()->domain(), gpu_lower);
    std::transform(
        counts.begin(),
        counts.end(),
        std::inserter(n_concrete_ids_, n_concrete_ids_.end()),
        [](auto counts_entry) {
          return std::make_pair(counts_entry.first, counts_entry.second.first);
        });
    std::transform(
        counts.begin(),
        counts.end(),
        std::inserter(n_broadcast_ids_, n_broadcast_ids_.end()),
        [](auto counts_entry) {
          return std::make_pair(counts_entry.first, counts_entry.second.second);
        });
  }

  for (auto inp_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    auto counts = InputDomainCounter::produceCounts(
        inp_tv->domain()->domain(), gpu_lower);
    std::transform(
        counts.begin(),
        counts.end(),
        std::inserter(n_concrete_ids_, n_concrete_ids_.end()),
        [](auto counts_entry) {
          return std::make_pair(counts_entry.first, counts_entry.second.first);
        });
    std::transform(
        counts.begin(),
        counts.end(),
        std::inserter(n_broadcast_ids_, n_broadcast_ids_.end()),
        [](auto counts_entry) {
          return std::make_pair(counts_entry.first, counts_entry.second.second);
        });
  }

  // Populate concrete id map
  for (const auto& set : disjoint_iter_sets_) {
    int max_concrete_count = -1;
    int max_broadcast_count = -1;
    IterDomain* concrete_id = nullptr;
    for (auto id : *set) {
      int concrete_count = n_concrete_ids_.at(id);
      if (concrete_count >= max_concrete_count) {
        int broadcast_count = n_broadcast_ids_.at(id);
        if (concrete_count > max_concrete_count ||
            broadcast_count > max_broadcast_count) {
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

  if (gpu_lower != nullptr) {
    convertToKir(fusion, gpu_lower);
  }
}

void ComputeAtMap::convertToKir(Fusion* fusion, GpuLower* gpu_lower) {
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  TORCH_INTERNAL_ASSERT(gpu_lower != nullptr);

  has_lowered_kir_ = true;

  std::unordered_map<
      std::shared_ptr<std::deque<IterDomain*>>,
      std::shared_ptr<std::deque<kir::IterDomain*>>>
      disjoint_set_2_kir;

  for (const auto& disjoint_iter_set : disjoint_iter_set_maps_) {
    auto fusion_set = disjoint_iter_set.second;
    auto kir_set_it = disjoint_set_2_kir.find(fusion_set);
    std::shared_ptr<std::deque<kir::IterDomain*>> kir_set;
    if (kir_set_it == disjoint_set_2_kir.end()) {
      kir_set = std::make_shared<std::deque<kir::IterDomain*>>();
      std::transform(
          fusion_set->begin(),
          fusion_set->end(),
          std::inserter(*kir_set, kir_set->begin()),
          [&gpu_lower](IterDomain* id) {
            return gpu_lower->lowerValue(id)->as<kir::IterDomain>();
          });
      disjoint_set_2_kir.emplace(std::make_pair(fusion_set, kir_set));
    } else {
      kir_set = kir_set_it->second;
    }
    kir_disjoint_iter_set_maps_.emplace(std::make_pair(
        gpu_lower->lowerValue(disjoint_iter_set.first)->as<kir::IterDomain>(),
        kir_set));
  }

  for (auto entry : concrete_id_map_) {
    kir_concrete_id_map_.emplace(std::make_pair(
        gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>(),
        gpu_lower->lowerValue(entry.second)->as<kir::IterDomain>()));
  }

  for (const auto& entry : disjoint_iter_set_maps_) {
    kir_2_fusion_[gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>()] =
        entry.first;
  }

  // Make sure we have all IterDomains that could be used to generate a ForLoop
  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    for (auto out : tv_outputs) {
      for (auto entry : out->domain()->domain()) {
        kir_2_fusion_[gpu_lower->lowerValue(entry)->as<kir::IterDomain>()] =
            entry;
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

bool ComputeAtMap::areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const {
  assertLowered(has_lowered_kir_);
  if (id0 == id1) {
    return true;
  }
  auto set0_it = kir_disjoint_iter_set_maps_.find(id0);
  auto set1_it = kir_disjoint_iter_set_maps_.find(id1);
  if (set0_it == kir_disjoint_iter_set_maps_.end() ||
      set1_it == kir_disjoint_iter_set_maps_.end()) {
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

kir::IterDomain* ComputeAtMap::getConcreteMappedID(kir::IterDomain* id) const {
  assertLowered(has_lowered_kir_);
  auto it = kir_concrete_id_map_.find(id);
  if (it != kir_concrete_id_map_.end()) {
    return it->second;
  }
  return id;
}

IterDomain* ComputeAtMap::toFusion(kir::IterDomain* kir) const {
  assertLowered(has_lowered_kir_);
  auto kir_2_fusion_it = kir_2_fusion_.find(kir);
  TORCH_INTERNAL_ASSERT(
      kir_2_fusion_it != kir_2_fusion_.end(),
      "Kernel ir is not guarneteed to be reversible into fusion ir, could not find fusion entry. ",
      kir::toString(kir, false));
  return kir_2_fusion_it->second;
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
