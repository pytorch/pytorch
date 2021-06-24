#include <ATen/ATen.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <benchmark/benchmark.h>
#include <c10/core/InferenceMode.h>
#include <sstream>

struct ConvParams {
  std::vector<int64_t> input;
  std::vector<int64_t> weight;
  std::vector<int64_t> bias;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups;
};

struct xs {
  explicit xs(const std::vector<int64_t>& v_) : v(v_) {}
  const std::vector<int64_t>& v;
};

std::ostream& operator<<(std::ostream& os, const xs& x) {
  bool first = true;
  for (auto const& xx : x.v) {
    if (!first) {
      os << "x";
    }
    first = false;
    os << xx;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ConvParams& params) {
  os << "I" << xs(params.input) << "_W" << xs(params.weight) << "_B"
     << xs(params.bias) << "_S" << xs(params.stride) << "_P"
     << xs(params.padding) << "_D" << xs(params.dilation) << "_G"
     << params.groups;
  return os;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<ConvParams> MobileNetV3Params = {
    {{1, 3, 224, 224}, {16, 3, 3, 3}, {16}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 16, 112, 112}, {16, 16, 1, 1}, {16}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 16, 112, 112}, {16, 1, 3, 3}, {16}, {2, 2}, {1, 1}, {1, 1}, 16},
    {{1, 16, 56, 56}, {16, 16, 1, 1}, {16}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 16, 56, 56}, {72, 16, 1, 1}, {72}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 72, 56, 56}, {72, 1, 3, 3}, {72}, {2, 2}, {1, 1}, {1, 1}, 72},
    {{1, 72, 28, 28}, {24, 72, 1, 1}, {24}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 24, 28, 28}, {88, 24, 1, 1}, {88}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 88, 28, 28}, {88, 1, 3, 3}, {88}, {1, 1}, {1, 1}, {1, 1}, 88},
    {{1, 88, 28, 28}, {24, 88, 1, 1}, {24}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 24, 28, 28}, {96, 24, 1, 1}, {96}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 96, 28, 28}, {96, 1, 5, 5}, {96}, {2, 2}, {2, 2}, {1, 1}, 96},
    {{1, 96, 14, 14}, {40, 96, 1, 1}, {40}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 40, 14, 14}, {240, 40, 1, 1}, {240}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 240, 14, 14}, {240, 1, 5, 5}, {240}, {1, 1}, {2, 2}, {1, 1}, 240},
    {{1, 240, 14, 14}, {40, 240, 1, 1}, {40}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 40, 14, 14}, {240, 40, 1, 1}, {240}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 240, 14, 14}, {240, 1, 5, 5}, {240}, {1, 1}, {2, 2}, {1, 1}, 240},
    {{1, 240, 14, 14}, {40, 240, 1, 1}, {40}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 40, 14, 14}, {120, 40, 1, 1}, {120}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 120, 14, 14}, {120, 1, 5, 5}, {120}, {1, 1}, {2, 2}, {1, 1}, 120},
    {{1, 120, 14, 14}, {48, 120, 1, 1}, {48}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 48, 14, 14}, {144, 48, 1, 1}, {144}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 144, 14, 14}, {144, 1, 5, 5}, {144}, {1, 1}, {2, 2}, {1, 1}, 144},
    {{1, 144, 14, 14}, {48, 144, 1, 1}, {48}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 48, 14, 14}, {288, 48, 1, 1}, {288}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 288, 14, 14}, {288, 1, 5, 5}, {288}, {2, 2}, {2, 2}, {1, 1}, 288},
    {{1, 288, 7, 7}, {96, 288, 1, 1}, {96}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 96, 7, 7}, {576, 96, 1, 1}, {576}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 576, 7, 7}, {576, 1, 5, 5}, {576}, {1, 1}, {2, 2}, {1, 1}, 576},
    {{1, 576, 7, 7}, {96, 576, 1, 1}, {96}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 96, 7, 7}, {576, 96, 1, 1}, {576}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 576, 7, 7}, {576, 1, 5, 5}, {576}, {1, 1}, {2, 2}, {1, 1}, 576},
    {{1, 576, 7, 7}, {96, 576, 1, 1}, {96}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 96, 7, 7}, {576, 96, 1, 1}, {576}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 576, 1, 1}, {1280, 576, 1, 1}, {1280}, {1, 1}, {0, 0}, {1, 1}, 1},
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<ConvParams> ResNet18Params = {
    {{1, 3, 224, 224}, {64, 3, 7, 7}, {}, {2, 2}, {3, 3}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {128, 64, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {128, 64, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {256, 128, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {256, 128, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {512, 256, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 512, 7, 7}, {512, 512, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {512, 256, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 512, 7, 7}, {512, 512, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 512, 7, 7}, {512, 512, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<ConvParams> ResNet50Params = {
    {{1, 3, 224, 224}, {64, 3, 7, 7}, {}, {2, 2}, {3, 3}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {256, 64, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 64, 56, 56}, {256, 64, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 56, 56}, {64, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {256, 64, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 56, 56}, {64, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 64, 56, 56}, {64, 64, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 64, 56, 56}, {256, 64, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 56, 56}, {128, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 128, 56, 56}, {128, 128, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {512, 128, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 56, 56}, {512, 256, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 512, 28, 28}, {128, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {512, 128, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 28, 28}, {128, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {512, 128, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 28, 28}, {128, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 128, 28, 28}, {128, 128, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 128, 28, 28}, {512, 128, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 28, 28}, {256, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 28, 28}, {256, 256, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 28, 28}, {1024, 512, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {256, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {256, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {256, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {256, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {256, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 256, 14, 14}, {256, 256, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 256, 14, 14}, {1024, 256, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {512, 1024, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 14, 14}, {512, 512, 3, 3}, {}, {2, 2}, {1, 1}, {1, 1}, 1},
    {{1, 512, 7, 7}, {2048, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 1024, 14, 14}, {2048, 1024, 1, 1}, {}, {2, 2}, {0, 0}, {1, 1}, 1},
    {{1, 2048, 7, 7}, {512, 2048, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 7, 7}, {512, 512, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 512, 7, 7}, {2048, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 2048, 7, 7}, {512, 2048, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
    {{1, 512, 7, 7}, {512, 512, 3, 3}, {}, {1, 1}, {1, 1}, {1, 1}, 1},
    {{1, 512, 7, 7}, {2048, 512, 1, 1}, {}, {1, 1}, {0, 0}, {1, 1}, 1},
};

struct EnableMklDnn {
  explicit EnableMklDnn(bool enable)
      : prev_(at::globalContext().userEnabledMkldnn()) {
    at::globalContext().setUserEnabledMkldnn(enable);
  }

  ~EnableMklDnn() {
    at::globalContext().setUserEnabledMkldnn(prev_);
  }

  bool prev_;
};

template <bool WithMklDnn>
static void BM_conv2d_native(
    benchmark::State& state,
    const ConvParams& params) {
  EnableMklDnn mkl(WithMklDnn);
  auto input = at::randn(params.input);
  auto weight = at::randn(params.weight);
  auto bias = params.bias.size() > 0 ? at::randn(params.bias) : at::Tensor{};
  auto output = at::conv2d(
      input,
      weight,
      bias,
      params.stride,
      params.padding,
      params.dilation,
      params.groups);
  for (auto _ : state) {
    output = at::conv2d(
        input,
        weight,
        bias,
        params.stride,
        params.padding,
        params.dilation,
        params.groups);
  }
  state.counters["GFLOPS/s"] = benchmark::Counter(
      2.0f * output.numel() * weight.numel() / weight.size(0) *
          state.iterations(),
      benchmark::Counter::kIsRate);
  state.counters["GB/s"] = benchmark::Counter(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      state.iterations() * (input.nbytes() + weight.nbytes() + output.nbytes()),
      benchmark::Counter::kIsRate);
}

enum MklDnnReorder {
  None,
  WeightOnly,
  WeightAndInput,
};

template <MklDnnReorder Reorder>
static void BM_conv2d_mkldnn(
    benchmark::State& state,
    const ConvParams& params) {
  auto input = at::randn(params.input);
  auto weight = at::randn(params.weight);
  auto bias = params.bias.size() > 0 ? at::randn(params.bias) : at::Tensor{};

  if (Reorder == WeightAndInput) {
    auto it_input = at::native::itensor_from_mkldnn(input.to_mkldnn());
    auto r = ideep::tensor(
        params.input, ideep::data_type::f32, ideep::format_tag::aBcd16b);
    it_input.reorder_to(r);
    input = at::native::new_with_itensor_mkldnn(
        std::move(r), at::kFloat, at::Device(at::kCPU));
  }

  if (Reorder == WeightOnly || Reorder == WeightAndInput) {
    weight = at::mkldnn_reorder_conv2d_weight(
        weight.to_mkldnn(),
        params.padding,
        params.stride,
        params.dilation,
        params.groups);

    bias = params.bias.size() > 0 ? bias.to_mkldnn() : bias;
  }

  auto output = at::mkldnn_convolution(
      input,
      weight,
      bias,
      params.padding,
      params.stride,
      params.dilation,
      params.groups);
  for (auto _ : state) {
    output = at::mkldnn_convolution(
        input,
        weight,
        bias,
        params.padding,
        params.stride,
        params.dilation,
        params.groups);
  }
  state.counters["GFLOPS/s"] = benchmark::Counter(
      2.0f * output.numel() * weight.numel() / weight.size(0) *
          state.iterations(),
      benchmark::Counter::kIsRate);
  state.counters["GB/s"] = benchmark::Counter(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      state.iterations() * (input.nbytes() + weight.nbytes() + output.nbytes()),
      benchmark::Counter::kIsRate);
}

std::string name(
    const char* base,
    const char* suffix,
    const ConvParams& params) {
  std::ostringstream os;
  os << base << "_" << suffix << "_" << params;
  return os.str();
}

void registerOne(const char* base, const ConvParams& params) {
  benchmark::RegisterBenchmark(
      name(base, "native", params).data(), BM_conv2d_native<true>, params);
  benchmark::RegisterBenchmark(
      name(base, "native_nomkl", params).data(),
      BM_conv2d_native<false>,
      params);
  benchmark::RegisterBenchmark(
      name(base, "mkldnn_none", params).data(), BM_conv2d_mkldnn<None>, params);
  benchmark::RegisterBenchmark(
      name(base, "mkldnn_weight", params).data(),
      BM_conv2d_mkldnn<WeightOnly>,
      params);
  benchmark::RegisterBenchmark(
      name(base, "mkldnn_input", params).data(),
      BM_conv2d_mkldnn<WeightAndInput>,
      params);
}

int main(int argc, char** argv) {
  c10::InferenceMode guard;

#define BENCH(x)                         \
  for (auto const& params : x##Params) { \
    registerOne(#x, params);             \
  }
  BENCH(MobileNetV3);
  BENCH(ResNet18);
  BENCH(ResNet50);
#undef BENCH

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
