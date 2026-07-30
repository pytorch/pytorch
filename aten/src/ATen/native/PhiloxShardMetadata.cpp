#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/MemoryOverlap.h>
#include <ATen/native/PhiloxShardMetadata.h>
#include <c10/util/irange.h>
#include <c10/util/overflows.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace at::native::detail {
namespace {

bool rectangles_overlap(
    IntArrayRef offsets,
    IntArrayRef sizes,
    size_t first,
    size_t second,
    size_t ndim) {
  for (const auto dim : c10::irange(ndim)) {
    const auto first_offset = offsets[first * ndim + dim];
    const auto second_offset = offsets[second * ndim + dim];
    const auto first_size = sizes[first * ndim + dim];
    const auto second_size = sizes[second * ndim + dim];
    if (first_offset >= second_offset + second_size ||
        second_offset >= first_offset + first_size) {
      return false;
    }
  }
  return true;
}

} // namespace

ValidatedPhiloxShardMetadata validate_philox_shard_metadata(
    const TensorBase& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count,
    const char* op_name) {
  TORCH_CHECK(
      self.layout() == kStrided,
      op_name,
      ": self must be a strided tensor, got ",
      self.layout());
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) == global_shape.size(),
      op_name,
      ": global_shape and self must have the same number of dimensions");
  TORCH_CHECK(
      chunk_count >= 0, op_name, ": chunk_count must be non-negative");

  const size_t ndim = global_shape.size();
  if (ndim == 0) {
    TORCH_CHECK(
        chunk_count <= 1,
        op_name,
        ": a scalar tensor can have at most one shard");
  } else {
    TORCH_CHECK(
        static_cast<uint64_t>(chunk_count) <=
            std::numeric_limits<size_t>::max() / ndim,
        op_name,
        ": shard metadata is too large");
  }
  const size_t expected_metadata_size =
      static_cast<size_t>(chunk_count) * ndim;
  TORCH_CHECK(
      global_offsets.size() == expected_metadata_size &&
          local_offsets.size() == expected_metadata_size &&
          local_sizes.size() == expected_metadata_size,
      op_name,
      ": flattened shard metadata arrays must contain chunk_count * ndim values");

  for (const auto size : global_shape) {
    TORCH_CHECK(
        size >= 0, op_name, ": global_shape must contain non-negative sizes");
  }
  int64_t global_numel = 0;
  if (std::none_of(global_shape.begin(), global_shape.end(), [](int64_t size) {
        return size == 0;
      })) {
    global_numel = 1;
    for (const auto size : global_shape) {
      TORCH_CHECK(
          !c10::mul_overflows(global_numel, size, &global_numel),
          op_name,
          ": global tensor has more than int64 elements");
    }
  }

  std::vector<int64_t> chunk_numels(static_cast<size_t>(chunk_count));
  int64_t mapped_numel = 0;
  for (const auto chunk : c10::irange(static_cast<size_t>(chunk_count))) {
    bool empty_chunk = false;
    for (const auto dim : c10::irange(ndim)) {
      const size_t index = chunk * ndim + dim;
      const int64_t global_offset = global_offsets[index];
      const int64_t local_offset = local_offsets[index];
      const int64_t local_size = local_sizes[index];
      const int64_t global_size = global_shape[dim];
      const int64_t tensor_size = self.size(static_cast<int64_t>(dim));
      TORCH_CHECK(
          global_offset >= 0 && local_size >= 0 &&
              local_size <= global_size &&
              global_offset <= global_size - local_size,
          op_name,
          ": global shard ",
          chunk,
          " dimension ",
          dim,
          " is outside global_shape");
      TORCH_CHECK(
          local_offset >= 0 && local_size <= tensor_size &&
              local_offset <= tensor_size - local_size,
          op_name,
          ": local shard ",
          chunk,
          " dimension ",
          dim,
          " is outside self shape");
      empty_chunk |= local_size == 0;
    }

    int64_t chunk_numel = empty_chunk ? 0 : 1;
    if (!empty_chunk) {
      for (const auto dim : c10::irange(ndim)) {
        const int64_t size = local_sizes[chunk * ndim + dim];
        TORCH_CHECK(
            !c10::mul_overflows(chunk_numel, size, &chunk_numel),
            op_name,
            ": shard ",
            chunk,
            " has more than int64 elements");
      }
    }
    TORCH_CHECK(
        chunk_numel <= self.numel() - mapped_numel,
        op_name,
        ": local shard metadata describes more than self.numel() elements");
    mapped_numel += chunk_numel;
    chunk_numels[chunk] = chunk_numel;
  }

  for (size_t first = 0; first < static_cast<size_t>(chunk_count); ++first) {
    if (chunk_numels[first] == 0) {
      continue;
    }
    for (size_t second = first + 1;
         second < static_cast<size_t>(chunk_count);
         ++second) {
      if (chunk_numels[second] == 0) {
        continue;
      }
      TORCH_CHECK(
          !rectangles_overlap(
              global_offsets, local_sizes, first, second, ndim),
          op_name,
          ": global shards ",
          first,
          " and ",
          second,
          " must not overlap");
      TORCH_CHECK(
          !rectangles_overlap(local_offsets, local_sizes, first, second, ndim),
          op_name,
          ": local shards ",
          first,
          " and ",
          second,
          " must not overlap");
    }
  }
  at::assert_no_internal_overlap(self);

  return {global_numel, std::move(chunk_numels)};
}

} // namespace at::native::detail
