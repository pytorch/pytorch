#include <ATen/ATen.h>
#include <iostream>

#include <benchmark/benchmark.h>

static void quantize_per_channel_4d_contiguous(benchmark::State& state) {
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  at::Tensor a = at::rand({batches, channels, height, width});
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel_cpu(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

static void quantize_per_channel_4d_channels_last(benchmark::State& state) {
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  at::Tensor a = at::rand(
      {batches, channels, height, width},
      at::TensorOptions().memory_format(at::MemoryFormat::ChannelsLast));
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel_cpu(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

static void quantize_per_channel_2d(benchmark::State& state) {
  const size_t channels = static_cast<size_t>(state.range(0));
  const size_t nelem = static_cast<size_t>(state.range(1));

  at::Tensor a = at::rand({channels, nelem});
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel_cpu(
        a, scales, zero_points, 0, at::ScalarType::QUInt8);
  }
}

static void GenerateSizes4d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C", "H", "W"});

  for (size_t n = 16; n < 256; n *= 2) {
    for (size_t c = 4; c < 256; c *= 2) {
      for (size_t hw = 4; hw < 256; hw *= 2) {
        b->Args({n, c, hw, hw});
      }
    }
  }
}

static void GenerateSizes2d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"C", "N"});

  for (size_t c = 4; c < 512; c *= 2) {
    for (size_t n = 4; n < 512; n *= 2) {
      b->Args({c, n});
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
BENCHMARK(quantize_per_channel_2d)->Apply(GenerateSizes2d);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
BENCHMARK(quantize_per_channel_4d_contiguous)->Apply(GenerateSizes4d);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
BENCHMARK(quantize_per_channel_4d_channels_last)->Apply(GenerateSizes4d);
BENCHMARK_MAIN();
