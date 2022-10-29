#include <torch/csrc/profiler/data_flow.h>

#include <c10/util/overloaded.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/collection.h>

namespace torch {
namespace profiler {
namespace impl {

void calculateUniqueTensorIDs(
    std::vector<std::shared_ptr<Result>>& sorted_results) {
  // This task is equivilent to https://leetcode.com/problems/number-of-islands/
  // We first cluster events with a greedy index assignment, and then merge
  // groups that overlap.

  using storage_id_t = strong::type<
      size_t,
      struct _StorageID,
      strong::regular,
      strong::hashable,
      strong::arithmetic,
      strong::ordered>;

  struct TensorStoragePair {
    TensorImplAddress impl_;
    storage_id_t storage_id_;

    // Used to assign the result.
    std::reference_wrapper<c10::optional<TensorID>> id_ref_;
  };
  std::vector<TensorStoragePair> tensors;

  // Step 1) Flatten and convert storage data pointers. (Handle address reuse.)
  // --------------------------------------------------------------------------
  {
    storage_id_t current_id{0};
    ska::flat_hash_map<StorageImplData, storage_id_t> live_storage;
    auto lookup = [&current_id, &live_storage](const StorageImplData data) {
      auto inserted = live_storage.insert({data, current_id});
      current_id += storage_id_t(inserted.second);
      return inserted.first->second;
    };

    ska::flat_hash_set<storage_id_t> tensor_set;
    auto insert_tensor = [&lookup, &tensors, &tensor_set](TensorMetadata& m) {
      if (m.impl() && m.data_) {
        const auto id = lookup(m.data_);
        tensor_set.insert(id);
        tensors.emplace_back(TensorStoragePair{m.impl(), id, m.id_});
      }
    };

    for (auto& result : sorted_results) {
      result->visit(c10::overloaded(
          [&](ExtraFields<EventType::TorchOp>& torch_op) {
            for (auto& m : torch_op.inputs_.tensor_metadata_) {
              if (m.has_value()) {
                insert_tensor(*m);
              }
            }
          },
          [&](ExtraFields<EventType::Allocation>& alloc_op) {
            // We won't know which allocations are for Tensor storage yet.
            // We'll filter after we see all of the op inputs.
            tensors.emplace_back(TensorStoragePair{
                TensorImplAddress(nullptr),
                lookup(StorageImplData(alloc_op.ptr_)),
                alloc_op.id_});

            // Handle deallocation
            if (alloc_op.alloc_size_ < 0) {
              live_storage.erase(StorageImplData(alloc_op.ptr_));
            }
          },
          [&](ExtraFields<EventType::PyCall>& py_call) {
            // torch.nn.Module
            if (py_call.module_.has_value()) {
              for (auto& p : py_call.module_->parameters_) {
                insert_tensor(p.metadata_);
                if (p.grad_metadata_.has_value()) {
                  insert_tensor(*p.grad_metadata_);
                }
              }
            }

            // torch.optim.Optimizer
            if (py_call.optimizer_.has_value()) {
              for (auto& p : py_call.optimizer_->parameters_) {
                insert_tensor(p.metadata_);
                if (p.grad_metadata_.has_value()) {
                  insert_tensor(*p.grad_metadata_);
                }
                for (auto& state_i : p.state_) {
                  insert_tensor(state_i.second);
                }
              }
            }
          },
          [](const auto&) {}));
    }

    // Handle any allocation events which we cannot prove are for
    // `StorageImpl`s.
    tensors.erase(
        std::remove_if(
            tensors.begin(),
            tensors.end(),
            [&tensor_set](const auto& i) {
              return tensor_set.find(i.storage_id_) == tensor_set.end();
            }),
        tensors.end());
  }

  // Step 2) Handle the case that the storage of a TensorImpl changed.
  // --------------------------------------------------------------------------
  using storage_id_pair_t = std::pair<storage_id_t, storage_id_t>;
  ska::flat_hash_set<storage_id_pair_t, HashCombine> same_group_set;
  {
    ska::flat_hash_map<TensorImplAddress, storage_id_t> impl_map;
    for (const auto& t : tensors) {
      // Storage allocations / frees don't have an associated TensorImpl, so
      // we don't want all storages to merge through nullptr.
      if (!t.impl_) {
        continue;
      }

      const auto it = impl_map.insert({t.impl_, t.storage_id_}).first;

      // The pair needs to be sorted for the coalesce step to work properly.
      it->second < t.storage_id_
          ? same_group_set.insert({it->second, t.storage_id_})
          : same_group_set.insert({t.storage_id_, it->second});
    }
  }

  // Step 3) Coalesce groups and assign final IDs.
  // --------------------------------------------------------------------------
  ska::flat_hash_map<storage_id_t, size_t> id_map;
  {
    std::vector<storage_id_pair_t> unique_pairs;
    for (const auto& i : same_group_set) {
      unique_pairs.push_back(i);
    }
    std::sort(unique_pairs.begin(), unique_pairs.end());

    size_t current_id{0};
    for (const auto& i : unique_pairs) {
      auto inserted = id_map.insert({i.first, current_id});
      current_id += inserted.second;
      id_map.insert({i.second, inserted.first->second});
    }
  }

  // Step 4) Write back to metadata
  // --------------------------------------------------------------------------
  for (const auto& t : tensors) {
    t.id_ref_.get() = TensorID(id_map.at(t.storage_id_));
  }
}

} // namespace impl
} // namespace profiler
} // namespace torch
