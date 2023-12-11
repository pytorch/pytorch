#include <torch/csrc/profiler/data_flow.h>

#include <c10/util/overloaded.h>
#include <torch/csrc/profiler/collection.h>

namespace torch {
namespace profiler {
namespace impl {

namespace {
static constexpr TensorImplAddress NoTensorImpl{nullptr};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct RawTensorInfo {
  TensorImplAddress impl_;
  StorageImplData storage_;
  c10::Device device_;
  bool is_free_;

  // Used to assign back to the original structs.
  std::reference_wrapper<c10::optional<AllocationID>> allocation_id_ref_;
  std::reference_wrapper<c10::optional<TensorID>> id_ref_;
};

struct RawTensors {
  std::vector<RawTensorInfo>& get() {
    return tensors_;
  }

  void operator()(TensorMetadata& t) {
    tensors_.emplace_back(RawTensorInfo{
        t.impl(), t.data_, t.device_, false, t.allocation_id_, t.id_});
  }

  void operator()(c10::optional<TensorMetadata>& t) {
    if (t.has_value()) {
      (*this)(*t);
    }
  }

  void operator()(ExtraFields<EventType::Allocation>& a) {
    const StorageImplData ptr{a.ptr_};
    const auto is_free = a.alloc_size_ < 0;
    tensors_.emplace_back(RawTensorInfo{
        NoTensorImpl, ptr, a.device(), is_free, a.allocation_id_, a.id_});
  }

  void operator()(std::vector<TensorMetadata>& t) {
    for (auto& ti : t) {
      (*this)(ti);
    }
  }

  template <typename T>
  void operator()(T&) {}

  std::vector<RawTensorInfo> tensors_;
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
    RawTensors raw_tensors;

    // The python tracer caches values, so it's only safe to use the first case.
    ska::flat_hash_set<PyModuleSelf> seen_modules;
    ska::flat_hash_set<PyOptimizerSelf> seen_optimizers;
    for (auto& result : sorted_results) {
      result->visit(c10::overloaded(
          [&](ExtraFields<EventType::TorchOp>& torch_op) {
            for (auto& i : torch_op.inputs_) {
              std::visit(raw_tensors, i);
            }
          },
          [&](ExtraFields<EventType::PyCall>& py_call) {
            // torch.nn.Module
            if (py_call.module_.has_value() &&
                seen_modules.insert(py_call.module_->self_).second) {
              for (auto& p : py_call.module_->parameters_) {
                raw_tensors(p.metadata_);
                raw_tensors(p.grad_metadata_);
              }
            }

            // torch.optim.Optimizer
            if (py_call.optimizer_.has_value() &&
                seen_optimizers.insert(py_call.optimizer_->self_).second) {
              for (auto& p : py_call.optimizer_->parameters_) {
                raw_tensors(p.metadata_);
                raw_tensors(p.grad_metadata_);
                for (auto& state_i : p.state_) {
                  raw_tensors(state_i.second);
                }
              }
            }
          },
          [&](auto& i) { raw_tensors(i); }));
    }
    tensors = std::move(raw_tensors.tensors_);
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
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
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

      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
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
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto id = id_map.at(*t.allocation_id_ref_.get());
    t.id_ref_.get().emplace(TensorID(id));
  }
}

} // namespace impl
} // namespace profiler
} // namespace torch
