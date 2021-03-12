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

//! Class to figure out how many non-broadcast axes were used to produce an iter
//! domain. This is important for figuring out what the correct broadcasted
//! extent is of an iteration domain.
//!
//! When GpuLower is available, trivial reductions are not counted as
//! concrete domains so that they should not be used to generate
//! for-loops.
class ConcreteInputCounter : public IterVisitor {
 public:
  // Returns number of non-braodcast non-reduction iteration domains used to
  // generate the iteration domains in provided target domain.
  static std::unordered_map<IterDomain*, int> produceCounts(
      const std::vector<IterDomain*>& domain,
      GpuLower* gpu_lower) {
    std::unordered_map<IterDomain*, int> count_map;
    if (domain.empty()) {
      return count_map;
    }
    ConcreteInputCounter counter(domain, gpu_lower);
    std::transform(
        counter.concrete_domain_set_.begin(),
        counter.concrete_domain_set_.end(),
        std::inserter(count_map, count_map.begin()),
        [](const std::pair<IterDomain*, std::unordered_set<IterDomain*>>&
               entry) {
          return std::make_pair(entry.first, entry.second.size());
        });
    // Inputs may be root domains which wouldn't have any entries if no exprs
    // were traversed, so manually insert their count
    for (auto id : domain) {
      if (count_map.find(id) == count_map.end()) {
        count_map[id] =
            (id->isBroadcast() ||
             (gpu_lower && gpu_lower->trivialReductionInfo().isDerived(id)))
            ? 0
            : 1;
      }
    }
    return count_map;
  }

 private:
  ConcreteInputCounter(
      const std::vector<IterDomain*>& domain_,
      GpuLower* gpu_lower)
      : gpu_lower_(gpu_lower) {
    traverseFrom(
        domain_[0]->fusion(),
        std::vector<Val*>(domain_.begin(), domain_.end()));
  }

  std::unordered_set<IterDomain*>& getEntry(IterDomain* id) {
    auto concrete_set_it = concrete_domain_set_.find(id);
    if (concrete_set_it == concrete_domain_set_.end()) {
      concrete_set_it =
          concrete_domain_set_
              .emplace(std::make_pair(id, std::unordered_set<IterDomain*>()))
              .first;
      if (!id->isBroadcast() &&
          (gpu_lower_ && !gpu_lower_->trivialReductionInfo().isDerived(id))) {
        concrete_set_it->second.emplace(id);
      }
    }

    return concrete_set_it->second;
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
      concrete_domain_set_.emplace(std::make_pair(output_id, resulting_set));
    }
  }

  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      concrete_domain_set_;
  GpuLower* gpu_lower_ = nullptr;
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
    for (auto c_tv : tv_outputs) {
      consumer_tvs.push_back(c_tv);

      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        // If outside computeAt axis, we don't want to directly map
        // consumer/producer as their thread mappings could change as long as
        // it's across shared/global memory.

        // Mark axes outside compute at point for parallel type tracking
        std::unordered_set<IterDomain*> right_of_ca_point;
        if (mapping_mode_ == MappingMode::PARALLEL &&
            p_tv->getComputeAtPosition() < p_tv->nDims()) {
          right_of_ca_point.insert(
              p_tv->domain()->domain().begin() + p_tv->getComputeAtPosition(),
              p_tv->domain()->domain().end());
        }

        auto c2p_root_map =
            PairwiseRootDomainMap(p_tv, c_tv)
                .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We want to replay producer as consumer instead
        // of the other way around since consumer may have some broadcasted axes
        // producer doesn't have merged into loops producer may use. If we did
        // consumer as producer we wouldn't have this information in the
        // mapping. If we're using this map for indexing, we do not want to
        // propagate broadcast mismatches. If we're using it to identify loop
        // nests, we do want to propagate mismatches.
        BestEffortReplay replay_PasC(
            p_tv->domain()->domain(),
            c_tv->domain()->domain(),
            c2p_root_map,
            mapping_mode_ == MappingMode::LOOP ||
                mapping_mode_ == MappingMode::PARALLEL);

        auto c2p_map = replay_PasC.getReplay();

        // Find this computeAt position in consumer. This could be removed if we
        // changed computeAt of TensorViews to always have a this computeAt
        // position even for terminating outputs
        std::unordered_set<IterDomain*> within_producer_compute_at;
        for (unsigned int p_i = 0; p_i < p_tv->getComputeAtPosition(); p_i++) {
          within_producer_compute_at.insert(p_tv->axis((int)p_i));
        }

        // Map the entire replay map
        for (auto entry : c2p_map) {
          auto c_id = entry.first;
          auto p_id = entry.second;
          // If outside CA point and we're creating parallel map, do not map the
          // axis
          if (mapping_mode_ == MappingMode::PARALLEL &&
              right_of_ca_point.find(p_id) != right_of_ca_point.end()) {
            continue;
          }
          // Map the id's together
          mapIds(p_id, c_id);
        }
      }
    }
  }

  // deduplicate iter domain entries in each set
  for (const auto& iter_set : disjoint_iter_sets_) {
    *iter_set = deduplicateDeque(*iter_set);
  }

  // For each IterDomain set we will track how many concrete root domains were
  // used to generate the IterDomain. Used to populate conrete_id_map
  std::unordered_map<IterDomain*, int> n_concrete_ids_;

  for (auto c_tv : consumer_tvs) {
    auto counts = ConcreteInputCounter::produceCounts(
        c_tv->domain()->domain(), gpu_lower);
    n_concrete_ids_.insert(counts.begin(), counts.end());
  }

  for (auto inp_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    auto counts = ConcreteInputCounter::produceCounts(
        inp_tv->domain()->domain(), gpu_lower);
    n_concrete_ids_.insert(counts.begin(), counts.end());
  }

  // Populate concrete id map
  for (const auto& set : disjoint_iter_sets_) {
    int max_pos = -1;
    IterDomain* concrete_id = nullptr;
    for (auto id : *set) {
      // Uncertain if the following is needed, Maybe it makes sense to not
      // create loop nests based on rfactor axes if we can avoid it
      // if(id->isRFactorProduct() && id->definition() == nullptr){
      //   continue;
      // }
      int pos = n_concrete_ids_.at(id);
      if (pos > max_pos) {
        max_pos = pos;
        concrete_id = id;
      }
    }
    // Uncertain if the following is needed, Maybe it makes sense to not
    // create loop nests based on rfactor axes if we can avoid it
    // if(concrete_id == nullptr){
    //   // Same thing as above, but consider non-input rfactor iter domains
    //   for (auto id : *set) {
    //     int pos = n_concrete_ids_.at(id);
    //     if (pos > max_pos) {
    //       max_pos = pos;
    //       concrete_id = id;
    //     }
    //   }
    // }
    TORCH_INTERNAL_ASSERT(
        concrete_id != nullptr, "Could not concretize an IterDomain set.");

    // If parallel mode, parallelize the the concrete id
    // TODO: Would be good to simply keep a parallelization map and make lookups
    // to it through lowering.
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_map_it = parallel_type_map_.find(set);
      if (parallel_map_it != parallel_type_map_.end()) {
        concrete_id->parallelize(parallel_map_it->second);
      }
    }

    for (auto id : *set) {
      concrete_id_map_[id] = concrete_id;
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
      "Kernel ir is not guarneteed to be reversible into fusion ir, could not find fusion entry.");
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
