#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/util/SmallVector.h>
#include <c10/util/hash.h>
#include <ATen/core/Tensor.h>

namespace torch::distributed {
  // not using existing Placement in torch/csrc/distributed/Placement.h
  struct PlacementHashData {
    enum class PlacementType : uint8_t { Replicate, Shard, Partial, StridedShard };
    PlacementType placement_type;
    // should ask some checks for these limitations
    int64_t dim = 0;           // Shard/StridedShard
    int64_t split_factor = 0;  // StridedShard only
    std::string reduce_op;     // Partial only

    bool operator==(const PlacementHashData& rhs) const {
      return placement_type == rhs.placement_type && dim == rhs.dim &&
          split_factor == rhs.split_factor && reduce_op == rhs.reduce_op;
  }

    size_t hash() const {
      size_t h = c10::get_hash(static_cast<uint8_t>(placement_type), dim, split_factor);
      if (!reduce_op.empty()) {
        h = c10::hash_combine(h, std::hash<std::string>()(reduce_op));
      }
      return h;
    }
  };

  struct ShardOrderEntry {
    // corresponds to ShardOrderEntry in torch/distirbuted/tensor/_dtensor_spec.py
    int64_t tensor_dim;
    c10::SmallVector<int64_t, 2> mesh_dims;

    bool operator==(const ShardOrderEntry& rhs) const {
      return tensor_dim == rhs.tensor_dim && mesh_dims == rhs.mesh_dims;
    }

    size_t hash() const {
      size_t h = std::hash<int64_t>()(tensor_dim);
      for (auto d : mesh_dims) {
        h = c10::hash_combine(h, std::hash<int64_t>()(d));
      }
      return h;
    }
  };

  struct DTensorSpec : public c10::BackendMeta {
    at::Tensor local_tensor;

    PyObject* mesh_ptr = nullptr;
    c10::SmallVector<PlacementHashData, 4> placements;
    c10::SmallVector<ShardOrderEntry, 4> shard_order;

    size_t spec_hash = 0;

    bool spec_eq(
    const DTensorSpec& rhs,
    const at::Tensor& self_outer,
    const at::Tensor& rhs_outer) const
    {
      return mesh_ptr == rhs.mesh_ptr && placements == rhs.placements &&
        shard_order == rhs.shard_order &&
        self_outer.sizes() == rhs_outer.sizes() &&
        self_outer.strides() == rhs_outer.strides() &&
        self_outer.dtype() == rhs_outer.dtype();
      }

    static size_t compute_hash(
    PyObject* mesh,
    c10::ArrayRef<PlacementHashData> placements,
    c10::ArrayRef<ShardOrderEntry> shard_order,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    c10::ScalarType dtype)
    {
      size_t h = std::hash<const void*>()(mesh);
      for (const auto& p : placements) {
        h = c10::hash_combine(h, p.hash());
      }
      for (const auto& s : shard_order) {
        h = c10::hash_combine(h, s.hash());
      }
      for (auto s : sizes) {
        h = c10::hash_combine(h, std::hash<int64_t>()(s));
      }
      for (auto s : strides) {
        h = c10::hash_combine(h, std::hash<int64_t>()(s));
      }
      h = c10::hash_combine(
          h, std::hash<int16_t>()(static_cast<int16_t>(dtype)));
      return h;
    }
  };
} // namespace torch::distributed
