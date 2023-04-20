#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

static auto getLayerBackwardNormRuntime(
    std::unique_ptr<Fusion> fusion_ptr,
    std::unique_ptr<FusionExecutorCache>& fec,
    std::vector<at::IValue>& aten_inputs,
    std::vector<int64_t>& shape,
    std::vector<int64_t>& norm_shape) {
  Fusion& fusion = *fusion_ptr.get();

  const size_t kM = shape.size();
  const size_t kN = norm_shape.size();
  const size_t kOuterNumDims = kM - kN;

  std::vector<int64_t> outer_shape;
  for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
    outer_shape.push_back(shape[idx]);
  }
  for (size_t idx = kOuterNumDims; idx < kM; ++idx) {
    outer_shape.push_back(1);
  }

  auto grad_out = makeSymbolicTensor(shape.size());
  auto input = makeSymbolicTensor(shape.size());
  auto mean = makeConcreteTensor(outer_shape);
  auto rstd = makeConcreteTensor(outer_shape);
  auto weight = makeSymbolicTensor(norm_shape.size());
  auto bias = makeSymbolicTensor(norm_shape.size());
  fusion.addInput(grad_out);
  fusion.addInput(input);
  fusion.addInput(mean);
  fusion.addInput(rstd);
  fusion.addInput(weight);
  fusion.addInput(bias);

  auto grads = layer_norm_backward(
      grad_out,
      input,
      norm_shape,
      mean,
      rstd,
      weight,
      bias,
      {true, true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);
  fusion.addOutput(grads.grad_bias);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_grad_out = at::randn(shape, options);
  at::Tensor aten_input = at::randn(shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  at::Tensor aten_bias = at::randn(norm_shape, options);
  auto at_weight = c10::optional<at::Tensor>(aten_weight);
  auto at_bias = c10::optional<at::Tensor>(aten_bias);

  const float kEps = 1e-5;
  auto aten_results =
      at::native_layer_norm(aten_input, norm_shape, at_weight, at_bias, kEps);
  auto aten_output = std::get<0>(aten_results);
  auto aten_mean = std::get<1>(aten_results);
  auto aten_rstd = std::get<2>(aten_results);

  fec = std::make_unique<FusionExecutorCache>(std::move(fusion_ptr));
  aten_inputs = {
      aten_grad_out, aten_input, aten_mean, aten_rstd, aten_weight, aten_bias};
  auto cg_outputs = fec->runFusionWithInputs(aten_inputs);

  return fec->getMostRecentKernelRuntime();
}

static void LayerNormBackward_HeuristicLookup(
    benchmark::State& benchmark_state) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // PreAllocate
  std::unique_ptr<FusionExecutorCache> fec;
  std::vector<at::IValue> aten_inputs;

  std::vector<int64_t> shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  auto runtime = getLayerBackwardNormRuntime(
      std::move(fusion_ptr), fec, aten_inputs, shape, norm_shape);

  KernelArgumentHolder args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);

  TORCH_INTERNAL_ASSERT(
      runtime->getMaybeHeuristicsFor(args).has_value());

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    runtime->getMaybeHeuristicsFor(args);
  }
}

static auto getLayerForwardNormRuntime(
    std::unique_ptr<Fusion> fusion_ptr,
    std::unique_ptr<FusionExecutorCache>& fec,
    std::vector<at::IValue>& aten_inputs,
    std::vector<int64_t>& shape,
    std::vector<int64_t>& norm_shape) {
  Fusion& fusion = *fusion_ptr.get();

  const float kEps = 1e-5;
  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  auto input = makeSymbolicTensor(shape.size());
  fusion.addInput(input);

  auto result = layer_norm(input, norm_shape, nullptr, nullptr, eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(shape, options);

  fec = std::make_unique<FusionExecutorCache>(std::move(fusion_ptr));
  aten_inputs = {aten_input};
  auto cg_outputs = fec->runFusionWithInputs(aten_inputs);

  return fec->getMostRecentKernelRuntime();
}

static void LayerNormForward_HeuristicLookup(
    benchmark::State& benchmark_state) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // PreAllocate
  std::unique_ptr<FusionExecutorCache> fec;
  std::vector<at::IValue> aten_inputs;

  std::vector<int64_t> shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  auto runtime = getLayerForwardNormRuntime(
      std::move(fusion_ptr), fec, aten_inputs, shape, norm_shape);

  KernelArgumentHolder args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);

  TORCH_INTERNAL_ASSERT(
      runtime->getMaybeHeuristicsFor(args).has_value());

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    runtime->getMaybeHeuristicsFor(args);
  }
}

BENCHMARK(LayerNormBackward_HeuristicLookup)->Unit(benchmark::kMicrosecond);
BENCHMARK(LayerNormForward_HeuristicLookup)->Unit(benchmark::kMicrosecond);
