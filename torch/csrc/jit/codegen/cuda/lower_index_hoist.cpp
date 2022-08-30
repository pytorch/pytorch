#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/lower_index_hoist.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Return leaf domains of a given domain.
std::unordered_set<IterDomain*> getUsedLeafIds(
    IterDomain* id,
    TensorDomain* td) {
  const auto all_vals_between = DependencyCheck::getAllValsBetween(
      {id}, {td->domain().begin(), td->domain().end()});

  std::unordered_set<IterDomain*> used_leaf_ids;

  for (const auto leaf : td->domain()) {
    if (std::find(all_vals_between.begin(), all_vals_between.end(), leaf) !=
        all_vals_between.end()) {
      used_leaf_ids.insert(leaf);
    }
  }

  TORCH_INTERNAL_ASSERT(
      !used_leaf_ids.empty(),
      "No used id found: ",
      id->toString(),
      ", ",
      td->toString());

  return used_leaf_ids;
}

} // namespace

CommonIndexKey::CommonIndexKey(
    IterDomain* consumer_indexed_id,
    TensorDomain* consumer_td,
    TensorDomain* ref_td,
    const std::unordered_map<IterDomain*, Val*>& ref_index_map,
    const std::vector<kir::ForLoop*>& loops) {
  auto gpu_lower = GpuLower::current();

  concrete_indexed_id_ = gpu_lower->caMap()->getConcreteMappedID(
      consumer_indexed_id, IdMappingMode::EXACT);

  const auto consumer_leaf_ids =
      getUsedLeafIds(consumer_indexed_id, consumer_td);

  // Convert to Parallel concrete IDs to find matching loops.
  std::unordered_set<IterDomain*> concrete_leaf_ids;
  for (auto& id : consumer_leaf_ids) {
    concrete_leaf_ids.insert(
        gpu_lower->caMap()->getConcreteMappedID(id, IdMappingMode::LOOP));
  }

  // Find used loops and their index vals
  for (const auto i : c10::irange(loops.size())) {
    auto loop = loops.at(i);
    auto loop_id = gpu_lower->caMap()->getConcreteMappedID(
        loop->iter_domain(), IdMappingMode::LOOP);
    auto it = concrete_leaf_ids.find(loop_id);
    if (it != concrete_leaf_ids.end()) {
      // This leaf reference id is used for indexing the consumer id
      used_loops_.push_back(loop);
      auto index_it = ref_index_map.find(ref_td->axis(i));
      TORCH_INTERNAL_ASSERT(
          index_it != ref_index_map.end(),
          "Index not found for leaf ID, ",
          ref_td->axis(i)->toString());
      loop_index_vals_.push_back(index_it->second);
    }
  }

  TORCH_INTERNAL_ASSERT(
      !used_loops_.empty(),
      "No loop used for indexing found. ",
      consumer_indexed_id->toString());

  TORCH_INTERNAL_ASSERT(
      consumer_leaf_ids.size() == used_loops_.size(),
      "consumer_leaf_ids.size() = ",
      consumer_leaf_ids.size(),
      ", used_loops_.size() == ",
      used_loops_.size(),
      ", loops.size() == ",
      loops.size());
}

CommonIndexKey::CommonIndexKey(
    IterDomain* consumer_indexed_id,
    TensorDomain* consumer_td,
    const std::vector<IterDomain*>& loop_domains,
    const std::unordered_map<IterDomain*, Val*>& loop_index_map,
    const std::vector<kir::ForLoop*>& loops) {
  auto gpu_lower = GpuLower::current();

  concrete_indexed_id_ = gpu_lower->caMap()->getConcreteMappedID(
      consumer_indexed_id, IdMappingMode::EXACT);

  const auto consumer_leaf_ids =
      getUsedLeafIds(consumer_indexed_id, consumer_td);

  // Convert to Parallel concrete IDs to find matching loops.
  std::unordered_set<IterDomain*> concrete_leaf_ids;
  for (auto& id : consumer_leaf_ids) {
    concrete_leaf_ids.insert(
        gpu_lower->caMap()->getConcreteMappedID(id, IdMappingMode::LOOP));
  }

  // Find used loops and their index vals
  for (const auto i : c10::irange(loops.size())) {
    auto loop = loops.at(i);
    auto loop_id = gpu_lower->caMap()->getConcreteMappedID(
        loop->iter_domain(), IdMappingMode::LOOP);
    auto it = concrete_leaf_ids.find(loop_id);
    if (it != concrete_leaf_ids.end()) {
      // This leaf reference id is used for indexing the consumer id
      used_loops_.push_back(loop);
      auto loop_concrete_id = gpu_lower->caMap()->getConcreteMappedID(
          loop_domains.at(i), IdMappingMode::EXACT);
      auto index_it = loop_index_map.find(loop_concrete_id);
      TORCH_INTERNAL_ASSERT(
          index_it != loop_index_map.end(),
          "Index not found for leaf ID, ",
          loop_domains.at(i)->toString(),
          ", concrete ID: ",
          loop_concrete_id->toString());
      loop_index_vals_.push_back(index_it->second);
    }
  }

  TORCH_INTERNAL_ASSERT(
      !used_loops_.empty(),
      "No loop used for indexing found. ",
      consumer_indexed_id->toString());

  TORCH_INTERNAL_ASSERT(
      consumer_leaf_ids.size() == used_loops_.size(),
      "consumer_leaf_ids.size() = ",
      consumer_leaf_ids.size(),
      ", used_loops_.size() == ",
      used_loops_.size(),
      ", loops.size() == ",
      loops.size());
}

bool CommonIndexKey::operator==(const CommonIndexKey& other) const {
  auto gpu_lower = GpuLower::current();

  if (concrete_indexed_id_ != other.concrete_indexed_id_) {
    return false;
  }

  if (used_loops_.size() != other.used_loops_.size()) {
    return false;
  }

  // Check if both CommonIndexKeys use the same loops. If not, it's
  // still valid to share the same hoisted index as long as: 1) each
  // loop pair is mapped with the CA index map, and 2) they are not
  // instantiated as actual loops.
  for (const auto i : c10::irange(used_loops_.size())) {
    auto lhs_loop = used_loops_.at(i);
    auto rhs_loop = other.used_loops_.at(i);
    if (lhs_loop == rhs_loop) {
      continue;
    }
    if (gpu_lower->caMap()->areMapped(
            lhs_loop->iter_domain(),
            rhs_loop->iter_domain(),
            IdMappingMode::EXACT) &&
        lhs_loop->isTrivial() && rhs_loop->isTrivial()) {
      continue;
    }
    return false;
  }

  for (const auto i : c10::irange(loop_index_vals_.size())) {
    auto lhs_index = loop_index_vals_.at(i);
    auto rhs_index = other.loop_index_vals_.at(i);
    if (lhs_index == rhs_index) {
      continue;
    }
    // Initial index variables can have some additions such as magic
    // zero and "1" when used in producer indexing for double buffered
    // tensors. Thus, the initial variables themselves may be
    // different, and its components need to be examined. An easy way
    // is to flatten them to strings as follows.
    auto lhs_str = loop_index_vals_.at(i)->toInlineString();
    auto rhs_str = other.loop_index_vals_.at(i)->toInlineString();
    if (lhs_str == rhs_str) {
      continue;
    }

    return false;
  }

  return true;
}

std::string CommonIndexKey::toString() const {
  TORCH_INTERNAL_ASSERT(concrete_indexed_id_ != nullptr);
  std::stringstream ss;
  ss << "CommonIndexKey: " << concrete_indexed_id_->toString();
  ss << ", { ";
  for (auto loop : used_loops_) {
    ss << loop->iter_domain()->toString() << " ";
  }
  ss << "}";
  ss << ", { ";
  for (auto val : loop_index_vals_) {
    ss << val->toString() << " ";
  }
  ss << "}";
  return ss.str();
}

std::pair<Val*, bool> CommonIndexMap::insert(
    IterDomain* indexed_consumer_id,
    TensorDomain* consumer_td,
    TensorDomain* ref_td,
    const std::unordered_map<IterDomain*, Val*>& ref_index_map,
    const std::vector<kir::ForLoop*>& loops,
    Val* index) {
  if (index->definition() == nullptr) {
    // Only defined val is eligible to hoist
    return {index, false};
  }

  const CommonIndexKey key(
      indexed_consumer_id, consumer_td, ref_td, ref_index_map, loops);

  return tryInsertNewIndex(key, index);
}

std::pair<Val*, bool> CommonIndexMap::insert(
    IterDomain* indexed_consumer_id,
    TensorDomain* consumer_td,
    const std::vector<IterDomain*>& loop_domains,
    const std::unordered_map<IterDomain*, Val*>& loop_index_map,
    const std::vector<kir::ForLoop*>& loops,
    Val* index) {
  if (index->definition() == nullptr) {
    // Only defined val is eligible to hoist
    return {index, false};
  }

  const CommonIndexKey key(
      indexed_consumer_id, consumer_td, loop_domains, loop_index_map, loops);

  return tryInsertNewIndex(key, index);
}

std::pair<Val*, bool> CommonIndexMap::tryInsertNewIndex(
    CommonIndexKey key,
    Val* index) {
  Val* hoisted_index = nullptr;
  bool new_index_inserted = false;

  // Hoisting is not possible if any of used loops is grouped.
  if (std::any_of(
          key.usedLoops().begin(), key.usedLoops().end(), [](const auto loop) {
            return loop->iter_domain()->getParallelType() ==
                ParallelType::Group;
          })) {
    return {index, false};
  }

  // If already mapped, return the previously mapped index
  auto it = common_index_map_.find(key);
  if (it != common_index_map_.end()) {
    hoisted_index = it->second;
    new_index_inserted = false;
    ++use_counts_.at(key);
  } else {
    common_index_map_.emplace(key, index);
    hoisted_index = index;
    new_index_inserted = true;
    use_counts_[key] = 1;
  }
  return {hoisted_index, new_index_inserted};
}

namespace {

//! Insertion point of allocation
struct CommonIndexInsertionInfo {
  Expr* ref = nullptr;
  kir::Scope* scope = nullptr;
};

// Inserts allocations of hoisted indices
class CommonIndexInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      const CommonIndexMap& common_indices) {
    CommonIndexInserter inserter(exprs, common_indices);
    return inserter.exprs_;
  }

 private:
  CommonIndexInserter(
      const std::vector<Expr*>& exprs,
      const CommonIndexMap& common_index_map)
      : common_index_map_(common_index_map) {
    // Create a map to keys from loops where they should be inserted
    for (const auto& kv : common_index_map.commonIndexMap()) {
      const auto& key = kv.first;
      // Only consider indices used multiple times
      if (!usedMultipleTimes(key)) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(!key.usedLoops().empty());
      auto insertion_loop = key.usedLoops().back();
      innermost_used_loop_map_[insertion_loop].push_back(key);
    }

    traverseAndInsert(exprs);
  }

  CommonIndexInsertionInfo findInsertionPoint(
      const CommonIndexKey& key,
      kir::ForLoop* current_loop) const {
    CommonIndexInsertionInfo info;

    // Allocation must be inside any used non-trivial loop. Since the
    // loop index value is constant if a loop is trivial, allocation
    // does not need to be inside trivial loops.
    for (const auto loop : key.usedLoops()) {
      if (!loop->isTrivial()) {
        info.ref = loop->body()[0];
        info.scope = &(loop->body());
      }
    }

    // If no non-trivial used loop is found, insert at the top-level
    // scope just before the outer-most loop.
    if (info.ref == nullptr) {
      info.ref = scope_exprs_.empty() ? current_loop : scope_exprs_.at(0);
      info.scope = nullptr;
    }

    return info;
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    auto innermost_loop_map_it = innermost_used_loop_map_.find(loop);
    if (innermost_loop_map_it == innermost_used_loop_map_.end()) {
      kir::ExprMutator::handle(loop);
      return;
    }

    for (const auto& key : innermost_loop_map_it->second) {
      auto common_index = common_index_map_.commonIndexMap().at(key);

      // Insert only when the index is used multiple times and is not
      // yet inserted.
      if (inserted_indices_.find(common_index) != inserted_indices_.end()) {
        continue;
      }

      // Make the type of the hoisted index be the index type of the
      // kernel, which can be either int64_t or int. Not very clean,
      // but this seems to be the quickest way to use the index type
      // as we don't have a scalar IR node for the index type.
      common_index->resolveIndexDtype();

      auto alloc = IrBuilder::create<kir::Allocate>(
          common_index,
          MemoryType::Local,
          GpuLower::current()->kernel()->oneVal());
      const auto common_index_def = common_index->definition();
      TORCH_INTERNAL_ASSERT(
          common_index_def != nullptr,
          "Hosted index must have a definition. ",
          common_index->toString());

      const auto insertion_info = findInsertionPoint(key, loop);
      registerInsertBefore(insertion_info.ref, alloc, insertion_info.scope);
      registerInsertBefore(
          insertion_info.ref, common_index_def, insertion_info.scope);

      // Track inserted index
      inserted_indices_.emplace(common_index);
    }

    kir::ExprMutator::handle(loop);
  }

  bool usedMultipleTimes(const CommonIndexKey& key) {
    auto it = common_index_map_.useCounts().find(key);
    TORCH_INTERNAL_ASSERT(
        it != common_index_map_.useCounts().end(),
        "Key not found in the use-count map: ",
        key.toString());
    return it->second > 1;
  }

 private:
  const CommonIndexMap& common_index_map_;
  //! Map to CommonIndexKeys from their innermost used loops
  std::unordered_map<kir::ForLoop*, std::vector<CommonIndexKey>>
      innermost_used_loop_map_;
  //! Keep track of inserted indices
  std::unordered_set<Val*> inserted_indices_;
};

} // namespace

std::vector<Expr*> allocateCommonIndices(const std::vector<Expr*>& exprs) {
  return CommonIndexInserter::run(exprs, GpuLower::current()->commonIndexMap());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
