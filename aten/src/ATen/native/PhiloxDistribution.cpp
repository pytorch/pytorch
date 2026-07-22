#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/PhiloxDistribution.h>
#include <c10/core/SymIntArrayRef.h>

#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_distribution_shards_native.h>
#endif

namespace at::native {

DEFINE_DISPATCH(philox_distribution_shards_stub);
REGISTER_NO_CPU_DISPATCH(philox_distribution_shards_stub)

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
  TORCH_CHECK(low >= min && low <= max, "from is out of bounds for ", self.dtype());
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
  TORCH_CHECK(
      params.size() == 2,
      "_philox_distribution_shards_: distribution kind ",
      distribution,
      " expects 2 parameters, got ",
      params.size());
  TORCH_CHECK(
      !params[0].isComplex() && !params[1].isComplex(),
      "_philox_distribution_shards_: parameters must be real");

  const double param0 = params[0].toDouble();
  const double param1 = params[1].toDouble();
  switch (distribution_kind) {
    case PhiloxDistributionKind::Normal:
      validate_normal_std(param1);
      break;
    case PhiloxDistributionKind::Uniform:
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          self.scalar_type(),
          "_philox_distribution_shards_",
          [&] { validate_uniform_bounds<scalar_t>(self, param0, param1); });
      break;
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
      param0,
      param1,
      generator);
  return self;
}

} // namespace at::native
