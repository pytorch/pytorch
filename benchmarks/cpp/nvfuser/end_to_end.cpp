
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

using namespace torch::jit::fuser::cuda;

static void LstmCellBenchmark(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tvs[16];
  for (size_t i = 0; i < 16; i++) {
    tvs[i] = TensorViewBuilder().ndims(2).dtype(DataType::Float).build();
    fusion.addInput(tvs[i]);
  }

  const auto ingate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[0], tvs[1]), tvs[2]), tvs[3]));

  const auto forgetgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[4], tvs[5]), tvs[6]), tvs[7]));

  const auto cellgate = unaryOp(
      UnaryOpType::Tanh, add(add(add(tvs[8], tvs[9]), tvs[10]), tvs[11]));

  const auto outgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[12], tvs[13]), tvs[14]), tvs[15]));

  const auto cx = TensorViewBuilder()
                      .ndims(2)
                      .dtype(DataType::Float)
                      .contiguity(std::vector<bool>(2, true))
                      .build();

  const auto cy = add(mul(forgetgate, cx), mul(ingate, cellgate));

  const auto hy = mul(outgate, unaryOp(UnaryOpType::Tanh, cy));

  fusion.addInput(cx);
  fusion.addOutput(cy);
  fusion.addOutput(hy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const at::Tensor large_tensor0 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor1 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor2 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor3 =
      at::randn({batch_size, hidden_features * 4}, options);

  const auto chunked0 = large_tensor0.chunk(4, 1);
  const auto chunked1 = large_tensor1.chunk(4, 1);
  const auto chunked2 = large_tensor2.chunk(4, 1);
  const auto chunked3 = large_tensor3.chunk(4, 1);

  std::vector<c10::IValue> inputs;
  inputs.insert(inputs.end(), chunked0.begin(), chunked0.end());
  inputs.insert(inputs.end(), chunked1.begin(), chunked1.end());
  inputs.insert(inputs.end(), chunked2.begin(), chunked2.end());
  inputs.insert(inputs.end(), chunked3.begin(), chunked3.end());

  const auto at_cx = at::randn({batch_size, hidden_features}, options);
  inputs.push_back(at_cx);

  std::vector<at::Tensor> outputs;

  scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.compileFusion(&fusion);

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK_CAPTURE(LstmCellBenchmark, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(LstmCellBenchmark, Medium, 1024, 128)
    ->Unit(benchmark::kMicrosecond);
