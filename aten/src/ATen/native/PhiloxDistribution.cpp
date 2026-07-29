#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/PhiloxDistribution.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/irange.h>

#include <limits>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_distribution_shards_native.h>
#endif

namespace at::native {

DEFINE_DISPATCH(philox_distribution_shards_stub);
REGISTER_NO_CPU_DISPATCH(philox_distribution_shards_stub)

namespace {

void validate_normal_std(double stddev);

template <typename scalar_t>
void validate_uniform_bounds(const Tensor& self, double low, double high);

} // anonymous namespace

Tensor& _philox_distribution_shards_symint(
    Tensor& self,
    c10::SymIntArrayRef global_shape,
    c10::SymIntArrayRef global_offsets,
    c10::SymIntArrayRef local_offsets,
    c10::SymIntArrayRef local_sizes,
    int64_t chunk_count,
    int64_t distribution,
    ArrayRef<Scalar> params,
    std::optional<Generator> generator) {
  const auto global_shape_int = C10_AS_INTARRAYREF_SLOW_ALLOC(global_shape);
  const auto global_offsets_int = C10_AS_INTARRAYREF_SLOW_ALLOC(global_offsets);
  const auto local_offsets_int = C10_AS_INTARRAYREF_SLOW_ALLOC(local_offsets);
  const auto local_sizes_int = C10_AS_INTARRAYREF_SLOW_ALLOC(local_sizes);
  const auto distribution_kind =
      static_cast<PhiloxDistributionKind>(distribution);
  TORCH_CHECK(
      distribution_kind == PhiloxDistributionKind::Normal ||
          distribution_kind == PhiloxDistributionKind::Uniform,
      "_philox_distribution_shards_: unsupported distribution kind ",
      distribution);
  switch (distribution_kind) {
    case PhiloxDistributionKind::Normal:
      TORCH_CHECK(
          params.size() == 2,
          "_philox_distribution_shards_: distribution kind ",
          distribution,
          " expects 2 parameters, got ",
          params.size());
      TORCH_CHECK(
          !params[0].isComplex() && !params[1].isComplex(),
          "_philox_distribution_shards_: parameters must be real");
      validate_normal_std(params[1].toDouble());
      break;
    case PhiloxDistributionKind::Uniform: {
      TORCH_CHECK(
          params.size() == 2,
          "_philox_distribution_shards_: distribution kind ",
          distribution,
          " expects 2 parameters, got ",
          params.size());
      TORCH_CHECK(
          !params[0].isComplex() && !params[1].isComplex(),
          "_philox_distribution_shards_: parameters must be real");
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          self.scalar_type(),
          "_philox_distribution_shards_",
          [&] {
            validate_uniform_bounds<scalar_t>(
                self, params[0].toDouble(), params[1].toDouble());
          });
      break;
    }
  }
  philox_distribution_shards_stub(
      self.device().type(),
      self,
      global_shape_int,
      global_offsets_int,
      local_offsets_int,
      local_sizes_int,
      chunk_count,
      distribution_kind,
      params,
      generator);
  return self;
}

namespace {

void validate_normal_std(double stddev) {
  TORCH_CHECK(
      stddev >= 0.0,
      "normal expects std >= 0.0, but found std ",
      stddev);
}

template <typename scalar_t>
void validate_uniform_bounds(const Tensor& self, double low, double high) {
  const auto min =
      static_cast<double>(std::numeric_limits<scalar_t>::lowest());
  const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
  TORCH_CHECK(
      low >= min && low <= max, "from is out of bounds for ", self.dtype());
  TORCH_CHECK(
      high >= min && high <= max, "to is out of bounds for ", self.dtype());
  TORCH_CHECK(
      low <= high,
      "uniform_ expects to return a [from, to) range, but found from=",
      low,
      " > to=",
      high);
  TORCH_CHECK(
      high - low <= max,
      "uniform_ expects to-from <= std::numeric_limits<",
      toString(self.scalar_type()),
      ">::max(), but found to=",
      high,
      " and from=",
      low,
      " which result in to-from to exceed the limit");
}

bool philox_shard_rectangles_overlap(
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

} // anonymous namespace

namespace detail {

ValidatedPhiloxShardMetadata validate_philox_shard_metadata(
    const TensorBase& self,
    IntArrayRef global_shape,
    IntArrayRef global_offsets,
    IntArrayRef local_offsets,
    IntArrayRef local_sizes,
    int64_t chunk_count) {
  constexpr const char* op_name = "_philox_distribution_shards_";
  TORCH_CHECK(
      self.layout() == kStrided,
      op_name,
      ": self must be a strided tensor, got ",
      self.layout());
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) == global_shape.size(),
      op_name,
      ": global_shape and self must have the same number of dimensions");
  TORCH_CHECK(chunk_count >= 0, op_name, ": chunk_count must be non-negative");

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
  const size_t expected_metadata_size = static_cast<size_t>(chunk_count) * ndim;
  TORCH_CHECK(
      global_offsets.size() == expected_metadata_size &&
          local_offsets.size() == expected_metadata_size &&
          local_sizes.size() == expected_metadata_size,
      op_name,
      ": flattened shard metadata arrays must contain chunk_count * ndim values");

  bool has_zero_global_dim = false;
  for (const auto dim : c10::irange(ndim)) {
    TORCH_CHECK(
        global_shape[dim] >= 0, op_name, ": global_shape must be non-negative");
    has_zero_global_dim |= global_shape[dim] == 0;
  }
  int64_t global_numel = has_zero_global_dim ? 0 : 1;
  if (!has_zero_global_dim) {
    for (const auto size : global_shape) {
      TORCH_CHECK(
          size <= std::numeric_limits<int32_t>::max() / global_numel,
          op_name,
          ": global_shape has more than INT_MAX elements");
      global_numel *= size;
    }
    // Dense CUDA distributions change launch policy when TensorIterator splits here.
    const auto max_32bit_offset = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(
        global_numel - 1 <=
            (max_32bit_offset - 1) /
                static_cast<int64_t>(self.element_size()),
        op_name,
        ": the logical global tensor requires 64-bit indexing, which is not supported");
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
          global_offset >= 0 && local_size >= 0 && local_size <= global_size &&
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
        TORCH_INTERNAL_ASSERT(
            size <= std::numeric_limits<int32_t>::max() / chunk_numel);
        chunk_numel *= size;
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
    for (size_t second = first + 1; second < static_cast<size_t>(chunk_count);
         ++second) {
      if (chunk_numels[second] == 0) {
        continue;
      }
      TORCH_CHECK(
          !philox_shard_rectangles_overlap(
              global_offsets, local_sizes, first, second, ndim),
          op_name,
          ": global shards ",
          first,
          " and ",
          second,
          " must not overlap");
      TORCH_CHECK(
          !philox_shard_rectangles_overlap(
              local_offsets, local_sizes, first, second, ndim),
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

} // namespace detail

} // namespace at::native
