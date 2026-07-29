#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace at {

class Tensor;

namespace native {

// Numeric values cross the dispatcher and must match Python callers.
enum class PhiloxDistributionKind : int64_t {
  Normal = 0,
};

using philox_distribution_shards_fn = void (*)(
    Tensor& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    PhiloxDistributionKind distribution,
    ArrayRef<Scalar> params,
    std::optional<Generator> generator);

DECLARE_DISPATCH(
    philox_distribution_shards_fn,
    philox_distribution_shards_stub)

namespace detail {

struct ValidatedPhiloxShardMetadata {
  int64_t global_numel;
  std::vector<int64_t> chunk_numels;
};

TORCH_API ValidatedPhiloxShardMetadata validate_philox_shard_metadata(
    const TensorBase& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count);

} // namespace detail
} // namespace native
} // namespace at
