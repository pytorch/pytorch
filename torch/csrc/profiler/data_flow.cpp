#include <torch/csrc/profiler/data_flow.h>

#include <c10/util/overloaded.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/collection.h>

namespace torch {
namespace profiler {
namespace impl {

namespace {
static constexpr TensorImplAddress NoTensorImpl{nullptr};

struct RawTensorInfo {
  TensorImplAddress impl_;
  StorageImplData storage_;
  c10::Device device_;
  bool is_free_;

  // Used to assign back to the original structs.
  std::reference_wrapper<c10::optional<AllocationID>> allocation_id_ref_;
  std::reference_wrapper<c10::optional<TensorID>> id_ref_;
};
} // namespace

void calculateUniqueTensorIDs(
    std::vector<std::shared_ptr<Result>>& sorted_results) {
  // This task is equivilent to https://leetcode.com/problems/number-of-islands/
  // We first cluster events with a greedy index assignment, and then merge
  // groups that overlap.
  std::vector<RawTensorInfo> tensors;

  // Flatten results to a uniform representation.
  // --------------------------------------------------------------------------
  {
    auto insert_tensor = [&tensors](TensorMetadata& m) {
      tensors.emplace_back(RawTensorInfo{
          m.impl(),
          m.data_,
          m.device(),
          /*is_free_=*/false,
          m.allocation_id_,
          m.id_});
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
            tensors.emplace_back(RawTensorInfo{
                NoTensorImpl,
                StorageImplData(alloc_op.ptr_),
                alloc_op.device(),
                /*is_free_=*/alloc_op.alloc_size_ < 0,
                alloc_op.allocation_id_,
                alloc_op.id_});
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
  }

  // Assign IDs to solve ABA for Storage.
  // --------------------------------------------------------------------------
  {
    size_t counter{1};
    using key_t = std::pair<StorageImplData, c10::Device>;
    ska::flat_hash_map<key_t, size_t, HashCombine> versions;
    for (auto& t : tensors) {
      auto inserted = versions.insert({{t.storage_, t.device_}, counter});
      counter += inserted.second;
      t.allocation_id_ref_.get().emplace(AllocationID(inserted.first->second));
      if (t.is_free_) {
        versions.erase(inserted.first);
      }
    }
  }

  // Handle any allocation events which we cannot prove are for Tensor storage.
  // --------------------------------------------------------------------------
  {
    ska::flat_hash_set<AllocationID> tensor_set;
    for (const auto& t : tensors) {
      if (t.impl_ != NoTensorImpl) {
        tensor_set.insert(*t.allocation_id_ref_.get());
      }
    }
    tensors.erase(
        std::remove_if(
            tensors.begin(),
            tensors.end(),
            [&tensor_set](const auto& i) {
              auto it = tensor_set.find(*i.allocation_id_ref_.get());
              return it == tensor_set.end();
            }),
        tensors.end());
  }

  // Handle the case that the storage of a TensorImpl changed.
  // --------------------------------------------------------------------------
  using storage_id_pair_t = std::pair<AllocationID, AllocationID>;
  ska::flat_hash_set<storage_id_pair_t, HashCombine> same_group_set;
  {
    ska::flat_hash_map<TensorImplAddress, AllocationID> impl_map;
    for (const auto& t : tensors) {
      // Storage allocations / frees don't have an associated TensorImpl, so
      // we don't want all storages to merge through nullptr.
      if (!t.impl_) {
        continue;
      }

      const auto allocation_id = *t.allocation_id_ref_.get();
      const auto it = impl_map.insert({t.impl_, allocation_id}).first;

      // The pair needs to be sorted for the coalesce step to work properly.
      it->second < allocation_id
          ? same_group_set.insert({it->second, allocation_id})
          : same_group_set.insert({allocation_id, it->second});
    }
  }

  // Coalesce groups and assign final IDs.
  // --------------------------------------------------------------------------
  ska::flat_hash_map<AllocationID, size_t> id_map;
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

  // Write back to Tensor IDs.
  // --------------------------------------------------------------------------
  for (const auto& t : tensors) {
    const auto id = id_map.at(*t.allocation_id_ref_.get());
    t.id_ref_.get().emplace(TensorID(id));
  }
}

} // namespace impl
} // namespace profiler
} // namespace torch
