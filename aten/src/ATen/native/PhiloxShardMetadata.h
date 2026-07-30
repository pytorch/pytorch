#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <vector>

namespace at::native::detail {

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
    int64_t chunk_count,
    const char* op_name);

} // namespace at::native::detail
