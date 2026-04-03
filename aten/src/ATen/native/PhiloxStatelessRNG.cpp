#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cpu/StatelessPhilox4x32.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/TransformationHelper.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <cmath>

namespace at::native {

using at::cpu::philox_4x32;

namespace {

// Elements produced per Philox 4x32 call: 4 for float/half/bfloat16, 2 for double.
template <typename scalar_t>
constexpr int elems_per_call = std::is_same_v<scalar_t, double> ? 2 : 4;

// Derive a new (seed, offset) key from 4 random uint32 values.
inline void philox_derive_key(
    std::array<uint32_t, 4> r,
    uint64_t* out_seed,
    uint64_t* out_offset) {
  *out_seed = static_cast<uint64_t>(r[0]) | (static_cast<uint64_t>(r[1]) << 32);
  *out_offset = static_cast<uint64_t>(r[2]) | (static_cast<uint64_t>(r[3]) << 32);
}

// Box-Muller: convert 4 uniform uint32 values into 4 standard normal floats.
inline std::array<float, 4> box_muller_float(std::array<uint32_t, 4> r) {
  constexpr float M = 2.3283064365386963e-10f; // 1/2^32
  constexpr float TWO_PI = 6.2831853071795864f;
  float u1 = std::fma(static_cast<float>(r[0]), M, M * 0.5f);
  float u2 = std::fma(static_cast<float>(r[1]), M, M * 0.5f);
  float u3 = std::fma(static_cast<float>(r[2]), M, M * 0.5f);
  float u4 = std::fma(static_cast<float>(r[3]), M, M * 0.5f);

  float radius1 = std::sqrt(-2.0f * std::log(u1));
  float radius2 = std::sqrt(-2.0f * std::log(u3));
  float s1 = std::sin(TWO_PI * u2);
  float c1 = std::cos(TWO_PI * u2);
  float s2 = std::sin(TWO_PI * u4);
  float c2 = std::cos(TWO_PI * u4);
  return {radius1 * c1, radius1 * s1, radius2 * c2, radius2 * s2};
}

// Box-Muller: convert 4 uint32 values into 2 standard normal doubles.
inline std::array<double, 2> box_muller_double(std::array<uint32_t, 4> r) {
  constexpr double M = 2.3283064365386963e-10; // 1/2^32
  constexpr double TWO_PI = 6.2831853071795864;
  double u1 = std::fma(static_cast<double>(r[0]), M,
                  static_cast<double>(r[1]) * M * M + M * M * 0.5);
  double u2 = std::fma(static_cast<double>(r[2]), M,
                  static_cast<double>(r[3]) * M * M + M * M * 0.5);

  double radius = std::sqrt(-2.0 * std::log(u1));
  return {radius * std::cos(TWO_PI * u2), radius * std::sin(TWO_PI * u2)};
}

// Distribution dispatch: iterates over chunks of elements per key,
// calling sample_func to produce raw samples from a Philox call and
// param_func to apply per-element parameter scaling.
template <typename scalar_t, typename sample_func_t, typename param_func_t>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self, const Tensor& key,
    const sample_func_t& sample_func, const param_func_t& param_func) {
  TORCH_CHECK(self.is_floating_point(),
      op_name, ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      op_name, ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      op_name, ": self and key must be on the same device, got ",
      self.device(), " and ", key.device());
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      op_name, ": key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());
  if (key.dim() > 1) {
    TORCH_CHECK(key.dim() == self.dim() + 1,
        op_name, ": batched key must have ndim == output ndim + 1, "
        "got key shape ", key.sizes(), " with output shape ", self.sizes());
    auto key_batch = key.sizes().slice(0, self.dim());
    TORCH_CHECK(is_expandable_to(key_batch, self.sizes()),
        op_name, ": key batch shape ", key_batch,
        " is not broadcastable with output shape ", self.sizes());
  }

  if (self.numel() == 0) {
    return;
  }

  auto output = self.contiguous();
  scalar_t* out_ptr = output.mutable_data_ptr<scalar_t>();

  constexpr int epc = elems_per_call<scalar_t>;

  if (key.dim() == 1) {
    auto key_contig = key.contiguous();
    uint64_t seed = key_contig.const_data_ptr<uint64_t>()[0];
    uint64_t offset = key_contig.const_data_ptr<uint64_t>()[1];
    int64_t num_elems = self.numel();
    int64_t num_full_chunks = num_elems / epc;

    for (int64_t chunk = 0; chunk < num_full_chunks; chunk++) {
      auto sample = sample_func(seed, offset + static_cast<uint64_t>(chunk));
      for (int j = 0; j < epc; j++) {
        out_ptr[chunk * epc + j] = param_func(sample[j]);
      }
    }
    // Scalar tail for remaining elements.
    int64_t tail_start = num_full_chunks * epc;
    if (tail_start < num_elems) {
      auto sample = sample_func(seed, offset + static_cast<uint64_t>(num_full_chunks));
      for (int j = 0; j < num_elems - tail_start; j++) {
        out_ptr[tail_start + j] = param_func(sample[j]);
      }
    }
  } else {
    // Batched key path: determine elems_per_key from trailing size-1 key dims.
    int64_t elems_per_key = 1;
    int64_t key_dims = self.dim();
    for (int64_t i = self.dim() - 1; i >= 0; i--) {
      if (key.size(i) != 1) break;
      elems_per_key *= self.size(i);
      key_dims--;
    }
    int64_t num_keys = self.numel() / elems_per_key;

    // Expand and flatten key to (num_keys, 2).
    std::vector<int64_t> expand_sizes;
    expand_sizes.reserve(key.dim());
    for (int64_t i = 0; i < key_dims; i++) {
      expand_sizes.push_back(self.size(i));
    }
    for (int64_t i = key_dims; i < self.dim(); i++) {
      expand_sizes.push_back(1);
    }
    expand_sizes.push_back(2);
    auto key_flat = key.expand(expand_sizes).reshape({num_keys, 2}).contiguous();
    const uint64_t* keys_ptr = key_flat.const_data_ptr<uint64_t>();

    int64_t chunks_per_key = (elems_per_key + epc - 1) / epc;

    for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
      uint64_t seed = keys_ptr[key_idx * 2];
      uint64_t offset = keys_ptr[key_idx * 2 + 1];

      for (int64_t chunk = 0; chunk < chunks_per_key; chunk++) {
        auto sample = sample_func(seed, offset + static_cast<uint64_t>(chunk));
        int64_t base = key_idx * elems_per_key + chunk * epc;
        int64_t remaining = std::min(static_cast<int64_t>(epc), elems_per_key - chunk * epc);
        for (int64_t j = 0; j < remaining; j++) {
          out_ptr[base + j] = param_func(sample[j]);
        }
      }
    }
  }

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

} // anonymous namespace

Tensor _philox_key_split_cpu(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  auto output_sizes = key.sizes().vec();
  output_sizes.insert(output_sizes.begin(), num_splits);
  Tensor output = at::empty(output_sizes, key.options());

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t key_idx = 0; key_idx < num_keys; key_idx++) {
    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];

    for (int64_t split_idx = 0; split_idx < num_splits; split_idx++) {
      auto r = philox_4x32(seed, offset + static_cast<uint64_t>(split_idx));
      int64_t out = (split_idx * num_keys + key_idx) * 2;
      philox_derive_key(r, &out_ptr[out], &out_ptr[out + 1]);
    }
  }

  return output;
}

Tensor _philox_key_fold_in_cpu(const Tensor& key, int64_t data) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  auto key_contig = key.contiguous();
  int64_t num_keys = key.numel() / 2;

  Tensor output = at::empty_like(key_contig);

  if (num_keys == 0) {
    return output;
  }

  const uint64_t* input = key_contig.const_data_ptr<uint64_t>();
  uint64_t* out_ptr = output.data_ptr<uint64_t>();

  for (int64_t idx = 0; idx < num_keys; idx++) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];

    auto r = philox_4x32(seed, offset + static_cast<uint64_t>(data));
    philox_derive_key(r, &out_ptr[idx * 2], &out_ptr[idx * 2 + 1]);
  }

  return output;
}

Tensor& _philox_uniform_cpu_(Tensor& self, const Tensor& key, double low, double high) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_", [&] {
    auto sample_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [](uint64_t seed, uint64_t offset) -> std::array<uint64_t, 2> {
          auto r = philox_4x32(seed, offset);
          return {
            (static_cast<uint64_t>(r[0]) << 32) | r[1],
            (static_cast<uint64_t>(r[2]) << 32) | r[3]
          };
        };
      } else {
        return [](uint64_t seed, uint64_t offset) {
          return philox_4x32(seed, offset);
        };
      }
    }();

    auto lo = static_cast<scalar_t>(low);
    auto hi = static_cast<scalar_t>(high);
    auto param_func = [lo, hi](auto rand) {
      return static_cast<scalar_t>(
          at::transformation::uniform_real(rand, lo, hi));
    };

    philox_distribution_kernel<scalar_t>(
        "_philox_uniform_", self, key, sample_func, param_func);
  });
  return self;
}

Tensor& _philox_normal_cpu_(Tensor& self, const Tensor& key, double mean, double stddev) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_normal_", [&] {
    using compute_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    auto sample_func = []() {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return [](uint64_t seed, uint64_t offset) {
          return box_muller_double(philox_4x32(seed, offset));
        };
      } else {
        return [](uint64_t seed, uint64_t offset) {
          return box_muller_float(philox_4x32(seed, offset));
        };
      }
    }();

    auto mu = static_cast<compute_t>(mean);
    auto sigma = static_cast<compute_t>(stddev);
    auto param_func = [mu, sigma](compute_t rand) {
      return static_cast<scalar_t>(rand * sigma + mu);
    };

    philox_distribution_kernel<scalar_t>(
        "_philox_normal_", self, key, sample_func, param_func);
  });
  return self;
}

} // namespace at::native
