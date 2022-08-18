#include <vector>

#include <benchmark/benchmark.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#ifdef FBCODE_CAFFE2

#include <caffe2/core/context.h>
#include <caffe2/utils/math/transpose.h>
#include <torch/fb/sparsenn/sparsenn_operators.h>

#endif // FBCODE_CAFFE2

using torch::jit::Module;
using torch::jit::StaticModule;
using torch::jit::StaticModuleOptions;

#define BENCHMARK_WITH_INPUTS(X) BENCHMARK(X) \
  ->Iterations(100) \
  ->Repetitions(3) \
  ->Threads(1) \
  ->Ranges({{1<<0, 1<<8}, {10<<0, 10<<8}, {10<<0, 10<<8}}) \
  ;

constexpr StaticModuleOptions kDisableCopyVariants = {
  .enable_out_variant = true,
  .optimize_memory = true,
  .manage_output_tensors = false,
  .use_copy_variants = false,
  .use_maybe_copy_variants = true,
  .enable_tensorexpr_fusion = false,
};

static StaticModule makeStaticModuleFromIR(const std::string& irsrc,
    StaticModuleOptions opts = kDisableCopyVariants) {
  auto graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, torch::jit::Value*> vmap;
  torch::jit::parseIR(irsrc, graph.get(), vmap);
  return StaticModule{std::move(graph), opts, {}};
}

static StaticModule makeStaticModuleFromJIT(const std::string& jitsrc,
    StaticModuleOptions opts = kDisableCopyVariants) {
  Module jit_module("m");
  jit_module.define(jitsrc);
  return StaticModule{jit_module, /*frozen*/ false, opts, {}};
}

static void BM_ir_permute_copy(benchmark::State& state) {
  const auto irsrc = R"IR(
  graph(%input: Tensor, %output: Tensor):
    %indices: int[] = prim::Constant[value=[0, 2, 1]]()
    %mf: MemoryFormat = prim::Constant[value=0]()
    %result: Tensor = aten::permute_copy(%input, %indices, %output)
    %result_contig: Tensor = aten::contiguous(%result, %mf)
    return (%result_contig)
)IR";
  auto smod = makeStaticModuleFromIR(irsrc);
  const auto B = state.range(0);
  const auto M = state.range(1);
  const auto N = state.range(2);
  auto tensor = torch::randn({B, M, N});
  auto output = torch::zeros({B, N, M});
  std::vector<c10::IValue> args{tensor, output};
  smod(args, /*kwargs*/ {});
  for (auto _ : state) {
    smod(args, /*kwargs*/ {});
  }
  auto result = smod(args, /*kwargs*/ {}).toTensor();
  LOG(INFO) << "BM_ir_permute_copy final result is contiguous: " << result.is_contiguous();
}

static void BM_jit_permute(benchmark::State& state) {
  const auto jitsrc = R"JIT(
  def forward(self, input):
      return torch.permute(input, [0, 2, 1]).clone()
)JIT";
  auto smod = makeStaticModuleFromJIT(jitsrc);
  const auto B = state.range(0);
  const auto M = state.range(1);
  const auto N = state.range(2);
  auto tensor = torch::randn({B, M, N});
  std::vector<c10::IValue> args{tensor};
  smod(args, /*kwargs*/ {});
  for (auto _ : state) {
    smod(args, /*kwargs*/ {});
  }
  auto result = smod(args, /*kwargs*/ {}).toTensor();
  LOG(INFO) << "BM_jit_permute final result is contiguous: " << result.is_contiguous();
}

#ifdef FBCODE_CAFFE2

static void BM_fb_permute_out(benchmark::State& state) {
  const auto irsrc = R"IR(
  graph(%input: Tensor, %output: Tensor):
    %indices: int[] = prim::Constant[value=[0, 2, 1]]()
    %result: Tensor = fb::permute_out(%output, %input, %indices)
    return (%result)
)IR";
  auto smod = makeStaticModuleFromIR(irsrc);
  const auto B = state.range(0);
  const auto M = state.range(1);
  const auto N = state.range(2);
  auto tensor = torch::randn({B, M, N});
  auto output = torch::zeros({B, N, M});
  std::vector<c10::IValue> args{tensor, output};
  smod(args, /*kwargs*/ {});
  for (auto _ : state) {
    smod(args, /*kwargs*/ {});
  }
}

static void BM_caffe2_transpose(benchmark::State& state) {
  const auto B = state.range(0);
  const auto M = state.range(1);
  const auto N = state.range(2);
  const auto tensor = torch::randn({B, M, N});
  auto out = torch::zeros({B, N, M});
  const std::vector<int> axes {0, 2, 1};
  caffe2::CPUContext c;
  for (auto _ : state) {
    caffe2::math::Transpose(/*ndim*/ 3,
                             tensor.sizes().data(),
                             axes.data(),
                             tensor.data_ptr<float>(),
                             out.data_ptr<float>(),
                             &c);
  }
}

static void BM_at_permute_out(benchmark::State& state) {
  const auto B = state.range(0);
  const auto M = state.range(1);
  const auto N = state.range(2);
  const auto tensor = torch::randn({B, M, N});
  auto out = torch::zeros({B, N, M});
  const std::vector<long> axes {0, 2, 1};
  for (auto _ : state) {
    at::permute_out(out, tensor, axes);
  }
}

#endif // FBCODE_CAFFE2


BENCHMARK_WITH_INPUTS(BM_ir_permute_copy);
BENCHMARK_WITH_INPUTS(BM_jit_permute);

#ifdef FBCODE_CAFFE2

BENCHMARK_WITH_INPUTS(BM_fb_permute_out);
BENCHMARK_WITH_INPUTS(BM_at_permute_out);
BENCHMARK_WITH_INPUTS(BM_caffe2_transpose);

#endif

int main(int argc, char** argv) {
  c10::ParseCommandLineFlags(&argc, &argv);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
