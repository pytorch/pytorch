#include <torch/nativert/executor/memory/DisjointStorageGroups.h>

#include <list>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>

namespace {

using namespace torch::nativert;

// A StorageGroup represents a collection of allocations that share backing
// storage
class StorageGroup {
 public:
  // every storage group must contain at least one allocation spec.
  explicit StorageGroup(const AllocationSpec* spec)
      : max_spec_size_(spec->size),
        lifetime_(spec->lifetime),
        spec_group_({spec}) {}

  void add_spec(const AllocationSpec* spec) {
    spec_group_.push_back(spec);
    max_spec_size_ = std::max(max_spec_size_, spec->size);
    TORCH_DCHECK_LT(lifetime_.end, spec->lifetime.end);
    lifetime_.end = spec->lifetime.end;
    is_free_ = false;
  }

  const std::vector<const AllocationSpec*>& spec_group() const {
    return spec_group_;
  }

  size_t max_spec_size() const {
    return max_spec_size_;
  }

  size_t num_specs() const {
    return spec_group_.size();
  }

  const AllocationLifetime& lifetime() const {
    return lifetime_;
  }

  bool is_free() const {
    return is_free_;
  }

  void set_free(bool is_free) {
    is_free_ = is_free;
  }

 private:
  // whether or not this storage group is free
  // to add new specs
  bool is_free_{false};
  // represents the amount of memory that will be
  // allocated for all specs in this group...
  size_t max_spec_size_;
  // the lifetime of this storage group
  AllocationLifetime lifetime_;
  // all the specs in this group
  std::vector<const AllocationSpec*> spec_group_;
};

} // namespace

namespace torch::nativert {

LayoutPlan DisjointStorageGroupsPlanner(
    const std::vector<AllocationSpec>& allocation_specs) {
  struct CompareAllocationSpecsBySize {
    bool operator()(const AllocationSpec* a, const AllocationSpec* b)
        const /* noexcept */
    {
      return a->size > b->size;
    }
  };

  std::vector<
      std::multiset<const AllocationSpec*, CompareAllocationSpecsBySize>>
      allocation_indices;
  std::vector<std::vector<const AllocationSpec*>> deallocation_indices;

  for (const auto& spec : allocation_specs) {
    size_t alloc_index = spec.lifetime.start;
    size_t dealloc_index = spec.lifetime.end;

    TORCH_DCHECK_LT(alloc_index, dealloc_index);

    if (alloc_index >= allocation_indices.size()) {
      allocation_indices.resize(alloc_index + 1);
    }

    if (dealloc_index >= deallocation_indices.size()) {
      deallocation_indices.resize(dealloc_index + 1);
    }

    allocation_indices[alloc_index].insert(&spec);
    deallocation_indices[dealloc_index].emplace_back(&spec);
  }

  // don't want to invalidate pointers
  // so let's make this a list
  std::list<StorageGroup> storage_groups;
  // maps each AllocationSpec to its assigned storage group.
  c10::FastMap<const AllocationSpec*, StorageGroup*> spec_to_storage_group;
  // stores the set of storage groups that
  // are available for re-use.
  std::vector<StorageGroup*> free_storage_groups;

  auto createStorageGroup = [&](const AllocationSpec* spec) {
    auto& group = storage_groups.emplace_back(spec);
    spec_to_storage_group.emplace(spec, &group);
  };

  auto assignToAvailableStorageGroup = [&](const AllocationSpec* spec) {
    DCHECK(!free_storage_groups.empty());
    auto* storage_group = free_storage_groups.back();
    TORCH_DCHECK_NOTNULL(storage_group);
    TORCH_DCHECK_EQ(storage_group->is_free(), true);
    storage_group->add_spec(spec);
    spec_to_storage_group.emplace(spec, storage_group);
    free_storage_groups.pop_back();
  };

  for (const auto i : c10::irange(allocation_indices.size())) {
    for (auto* spec : allocation_indices[i]) {
      TORCH_DCHECK_NOTNULL(spec);
      if (free_storage_groups.empty()) {
        createStorageGroup(spec);
      } else {
        assignToAvailableStorageGroup(spec);
      }
    }

    if (i < deallocation_indices.size()) {
      for (auto* spec : deallocation_indices[i]) {
        TORCH_DCHECK_NOTNULL(spec);
        auto* storage_group = spec_to_storage_group.at(spec);
        if (!storage_group->is_free() &&
            storage_group->lifetime().end == spec->lifetime.end) {
          storage_group->set_free(true);
          free_storage_groups.push_back(storage_group);
        }
      }
    }
  }

  LayoutPlan plan;

  c10::FastMap<const StorageGroup*, size_t> storage_group_to_offset;
  size_t offset = 0;
  for (const auto& storage_group : storage_groups) {
    storage_group_to_offset.emplace(&storage_group, offset);
    offset += storage_group.max_spec_size();
  }

  plan.total_size = offset;
  plan.allocations.reserve(allocation_specs.size());

  for (const auto& spec : allocation_specs) {
    // specs in storage groups lifetime's shouldn't be overlapping
    // so we can just set their offset to the offset of the group
    plan.allocations.emplace_back(Allocation{
        spec.size,
        storage_group_to_offset.at(spec_to_storage_group.at(&spec))});
  }

  return plan;
}

} // namespace torch::nativert
