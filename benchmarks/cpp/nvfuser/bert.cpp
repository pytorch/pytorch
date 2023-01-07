#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

// Return reduction tensor view and output of reduction
static void setupDivMaxSoftmaxDropoutForward(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, false, false, true})
                        .shape({-1, 1, 1, -1})
                        .build();
  TensorView* tv1 = makeContigTensor(4, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // TODO: should be input
  auto d16 = IrBuilder::create<Double>(1.0);

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv1 = castOp(DataType::Float, tv1);
  }

  auto tv2 = div(tv1, d16);
  auto tv3 = add(tv2, tv0);

  auto tv10 = softmax(tv3, 3);
  auto dropout_tvs = dropout(tv10, IrBuilder::create<Double>(0.9));
  auto tv12 = dropout_tvs.mask;
  auto tv14 = dropout_tvs.output;

  if (is_fp16) {
    tv14 = castOp(DataType::Half, tv14);
    tv10 = castOp(DataType::Half, tv10);
    tv3 = castOp(DataType::Half, tv3);
  }

  fusion->addOutput(tv14);
  fusion->addOutput(tv12);
  fusion->addOutput(tv10);
  fusion->addOutput(tv3);
}

static void setupDivMaxSoftmaxDropoutBackward(Fusion* fusion, DataType dtype) {
  TensorView* tv0 = makeContigTensor(4, dtype);
  // Strangely tv1 isn't used anywhere, need to come back to that...
  TensorView* tv1 = makeContigTensor(4, dtype);
  TensorView* tv2 = makeContigTensor(4, dtype);
  TensorView* tv3 = makeContigTensor(4, DataType::Bool);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);

  bool is_fp16 = dtype == DataType::Half;
  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv1 = castOp(DataType::Float, tv1);
    tv2 = castOp(DataType::Float, tv2);
  }

  // TODO: should be inputs
  auto d32 = IrBuilder::create<Double>(1.0);
  // fusion->addInput(d32);
  auto d33 = IrBuilder::create<Double>(2.0);
  // fusion->addInput(d33);

  auto tv4 = mul(tv2, tv3);
  auto tv5 = mul(tv4, d33);
  auto tv6 = mul(tv5, tv0);
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, false, false, true});
  auto tv9 = mul(tv0, tv8);
  auto tv10 = sub(tv6, tv9);
  auto tv11 = div(tv10, d32);

  if (is_fp16) {
    tv10 = castOp(DataType::Half, tv10);
    tv11 = castOp(DataType::Half, tv11);
  }

  fusion->addOutput(tv11);
  fusion->addOutput(tv10);
}

static void MagicScheduler_DivMaxSoftDropFwd(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto w = benchmark_state.range(0);
  auto x = benchmark_state.range(1);
  auto y = benchmark_state.range(2);
  auto z = benchmark_state.range(3);

  setupDivMaxSoftmaxDropoutForward(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, 1, 1, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> at_inputs = {t0, t1};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getPersistentHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  schedulePersistentKernel(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion({t0, t1}, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  for (auto tensor : std::vector<at::Tensor>({t0, t1})) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

static void MagicScheduler_DivMaxSoftDropBwd(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto w = benchmark_state.range(0);
  auto x = benchmark_state.range(1);
  auto y = benchmark_state.range(2);
  auto z = benchmark_state.range(3);

  setupDivMaxSoftmaxDropoutBackward(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);
  at::Tensor t2 = at::randn({w, x, y, z}, options);
  at::Tensor t3 = at::randn({w, x, y, z}, options).round().to(at::kBool);

  std::vector<c10::IValue> at_inputs = {t0, t1, t2, t3};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getPersistentHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  schedulePersistentKernel(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion({t0, t1, t2, t3}, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  // Some reason t1 isn't used, ignore it.
  for (auto tensor : std::vector<at::Tensor>({t0, t2, t3})) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

static void setupBiasDropoutAddLayernormFwd(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = makeContigTensor(1, dtype);
  TensorView* tv1 = makeContigTensor(1, dtype);
  TensorView* tv2 = makeContigTensor(3, dtype);
  TensorView* tv3 = makeContigTensor(3, dtype);
  TensorView* tv4 = makeContigTensor(1, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv1 = castOp(DataType::Float, tv1);
    tv2 = castOp(DataType::Float, tv2);
    tv3 = castOp(DataType::Float, tv3);
    tv4 = castOp(DataType::Float, tv4);
  }

  auto tv5 = broadcast(tv4, {true, true, false});
  auto tv6 = add(tv3, tv5);
  auto dropout_outs = dropout(tv6, IrBuilder::create<Double>(0.9));

  auto tv8 = dropout_outs.output;
  auto tv10 = dropout_outs.mask;

  auto tv11 = add(tv10, tv2);

  auto layer_norm_outs =
      layer_norm(tv11, 1, tv0, tv1, IrBuilder::create<Double>(1e-5));
  auto tv14 = layer_norm_outs.output;
  auto tv21 = layer_norm_outs.mean;
  auto tv26 = layer_norm_outs.invstd;

  if (is_fp16) {
    tv11 = castOp(DataType::Half, tv11);
    tv14 = castOp(DataType::Half, tv14);
    tv21 = castOp(DataType::Half, tv21);
    tv26 = castOp(DataType::Half, tv26);
  }

  fusion->addOutput(tv8);
  fusion->addOutput(tv11);
  fusion->addOutput(tv14);
  fusion->addOutput(tv21);
  fusion->addOutput(tv26);
}

static void MagicScheduler_BiasDropoutAddLayernormFwd(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = benchmark_state.range(0);
  auto y = benchmark_state.range(1);
  auto z = benchmark_state.range(2);

  setupBiasDropoutAddLayernormFwd(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({z}, options);
  at::Tensor t1 = at::randn({z}, options);
  at::Tensor t2 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  at::Tensor t4 = at::randn({z}, options);

  std::vector<c10::IValue> at_inputs = {t0, t1, t2, t3, t4};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getPersistentHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  schedulePersistentKernel(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion(at_inputs, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  for (auto inp : at_inputs) {
    auto tensor = inp.toTensor();
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

static void setupBiasDropoutAddLayernormBwd1(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv1 = makeContigTensor(3, dtype);
  TensorView* tv2 = makeContigTensor(3, dtype);
  TensorView* tv3 = TensorViewBuilder()
                        .ndims(3)
                        .dtype(dtype)
                        .contiguity({true, true, true})
                        .shape({-1, -1, 1})
                        .build();
  TensorView* tv4 = TensorViewBuilder()
                        .ndims(3)
                        .dtype(dtype)
                        .contiguity({true, true, true})
                        .shape({-1, -1, 1})
                        .build();

  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  if (is_fp16) {
    tv1 = castOp(DataType::Float, tv1);
    tv2 = castOp(DataType::Float, tv2);
    tv3 = castOp(DataType::Float, tv3);
    tv4 = castOp(DataType::Float, tv4);
  }

  auto tv7 = sub(tv2, tv3);
  auto tv8 = mul(tv7, tv4);
  auto tv24 = sum(tv1, {0, 1});
  auto tv22 = mul(tv1, tv8);
  auto tv23 = sum(tv22, {0, 1});

  if (is_fp16) {
    tv24 = castOp(DataType::Half, tv24);
    tv23 = castOp(DataType::Half, tv23);
    tv8 = castOp(DataType::Half, tv8);
  }

  fusion->addOutput(tv24);
  fusion->addOutput(tv23);
  fusion->addOutput(tv8);
}

static void MagicScheduler_BiasDropoutAddLayernormBwd1(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = benchmark_state.range(0);
  auto y = benchmark_state.range(1);
  auto z = benchmark_state.range(2);

  setupBiasDropoutAddLayernormBwd1(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({x, y, z}, options);
  at::Tensor t2 = at::randn({x, y, 1}, options);
  at::Tensor t3 = at::randn({x, y, 1}, options);

  std::vector<c10::IValue> at_inputs = {t0, t1, t2, t3};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getReductionHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  scheduleReduction(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    clearL2Cache();
    cg_outputs = fe.runFusion(at_inputs, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  for (auto inp : at_inputs) {
    auto tensor = inp.toTensor();
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

static void setupBiasDropoutAddLayernormBwd2(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv4 = TensorViewBuilder()
                        .ndims(3)
                        .dtype(dtype)
                        .contiguity({true, true, true})
                        .shape({-1, -1, 1})
                        .build();
  TensorView* tv5 = makeContigTensor(1, dtype);
  TensorView* tv1 = makeContigTensor(3, dtype);
  TensorView* tv8 = makeContigTensor(3, dtype);

  fusion->addInput(tv4);
  fusion->addInput(tv5);
  fusion->addInput(tv1);
  fusion->addInput(tv8);

  if (is_fp16) {
    tv4 = castOp(DataType::Float, tv4);
    tv5 = castOp(DataType::Float, tv5);
    tv1 = castOp(DataType::Float, tv1);
    tv8 = castOp(DataType::Float, tv8);
  }
  auto d36 = mul(IrBuilder::create<Double>(1.0), tv1->axis(2)->extent());
  auto d47 = unaryOp(UnaryOpType::Reciprocal, d36);

  auto tv9 = broadcast(tv5, {true, true, false});
  auto tv10 = mul(tv1, tv9);
  auto tv14 = mul(tv10, tv8);
  auto tv15 = sum(tv14, {2});
  auto tv16 = broadcast(tv15, {false, false, true});
  auto tv17 = mul(tv8, tv16);
  auto tv12 = sum(tv10, {2});
  auto tv13 = broadcast(tv12, {false, false, true});
  auto tv11 = mul(d36, tv10);
  auto tv18 = sub(tv11, tv13);
  auto tv20 = mul(d47, tv4);
  auto tv19 = sub(tv18, tv17);
  auto tv21 = mul(tv20, tv19);

  if (is_fp16) {
    tv21 = castOp(DataType::Half, tv21);
  }

  fusion->addOutput(tv21);
}

static void MagicScheduler_BiasDropoutAddLayernormBwd2(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = benchmark_state.range(0);
  auto y = benchmark_state.range(1);
  auto z = benchmark_state.range(2);

  setupBiasDropoutAddLayernormBwd2(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t4 = at::randn({x, y, 1}, options);
  at::Tensor t5 = at::randn({z}, options);
  at::Tensor t1 = at::randn({x, y, z}, options);
  at::Tensor t8 = at::randn({x, y, z}, options);

  std::vector<c10::IValue> at_inputs = {t4, t5, t1, t8};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getPersistentHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  schedulePersistentKernel(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion(at_inputs, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  for (auto inp : at_inputs) {
    auto tensor = inp.toTensor();
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

static void setupBiasDropoutAddLayernormBwd3(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = makeContigTensor(3, dtype);
  TensorView* tv21 = makeContigTensor(3, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv21);

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv21 = castOp(DataType::Float, tv21);
  }

  // Uncertain this is the right value, but going for it anyways
  auto d34 = div(IrBuilder::create<Double>(1.0), tv0->axis(2)->extent());

  auto tv25 = mul(tv21, tv0);
  auto tv26 = mul(tv25, d34);
  auto tv27 = sum(tv26, {0, 1});

  if (is_fp16) {
    tv26 = castOp(DataType::Half, tv27);
    tv27 = castOp(DataType::Half, tv27);
  }

  fusion->addOutput(tv26);
  fusion->addOutput(tv27);
}

static void MagicScheduler_BiasDropoutAddLayernormBwd3(
    benchmark::State& benchmark_state,
    DataType dtype) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = benchmark_state.range(0);
  auto y = benchmark_state.range(1);
  auto z = benchmark_state.range(2);

  setupBiasDropoutAddLayernormBwd3(&fusion, dtype);

  auto tvs = ir_utils::allTvs(&fusion);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t21 = at::randn({x, y, z}, options);

  std::vector<c10::IValue> at_inputs = {t0, t21};
  std::vector<at::Tensor> cg_outputs;

  auto norm_params = getReductionHeuristics(&fusion, at_inputs);
  TORCH_CHECK(norm_params != nullptr, "Norm scheduler can't be used!");
  scheduleReduction(&fusion, *norm_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    cg_outputs = fe.runFusion(at_inputs, norm_params->lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  int64_t bytes = 0;
  for (auto inp : at_inputs) {
    auto tensor = inp.toTensor();
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  for (auto tensor : cg_outputs) {
    bytes += tensor.numel() *
        (int64_t)dataTypeSize(aten_to_data_type(tensor.scalar_type()));
  }

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

//------------------------------------------------------------------------------

static void DivMaxSoftDropFwd_fp32(benchmark::State& benchmark_state) {
  MagicScheduler_DivMaxSoftDropFwd(benchmark_state, DataType::Float);
}

static void DivMaxSoftDropBwd_fp32(benchmark::State& benchmark_state) {
  MagicScheduler_DivMaxSoftDropBwd(benchmark_state, DataType::Float);
}

static void DivMaxSoftDropFwd_fp16(benchmark::State& benchmark_state) {
  MagicScheduler_DivMaxSoftDropFwd(benchmark_state, DataType::Half);
}

static void DivMaxSoftDropBwd_fp16(benchmark::State& benchmark_state) {
  MagicScheduler_DivMaxSoftDropBwd(benchmark_state, DataType::Half);
}

static void BiasDropoutAddLayernormFwd_fp32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormFwd(benchmark_state, DataType::Float);
}

static void BiasDropoutAddLayernormFwd_tf32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormFwd(benchmark_state, DataType::Float);
}

static void BiasDropoutAddLayernormBwd1_fp32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormBwd1(benchmark_state, DataType::Float);
}

// Use full ampere wave here
static void BiasDropoutAddLayernormBwd1_tf32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormBwd1(benchmark_state, DataType::Float);
}

static void BiasDropoutAddLayernormBwd2_fp32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormBwd2(benchmark_state, DataType::Float);
}

static void BiasDropoutAddLayernormBwd3_fp32(
    benchmark::State& benchmark_state) {
  MagicScheduler_BiasDropoutAddLayernormBwd3(benchmark_state, DataType::Float);
}

//------------------------------------------------------------------------------

BENCHMARK(DivMaxSoftDropFwd_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(DivMaxSoftDropBwd_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(DivMaxSoftDropFwd_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(DivMaxSoftDropBwd_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(BiasDropoutAddLayernormFwd_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32, 1024}, {128, 128}, {1024, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Use full ampere wave here
BENCHMARK(BiasDropoutAddLayernormFwd_tf32)
    // ->RangeMultiplier(2)
    ->Ranges({{32, 1024}, {128, 128}, {864, 864}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(BiasDropoutAddLayernormBwd1_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32, 1024}, {128, 128}, {1024, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Use full ampere wave here
BENCHMARK(BiasDropoutAddLayernormBwd1_tf32)
    // ->RangeMultiplier(2)
    ->Ranges({{32, 1024}, {128, 128}, {864, 864}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(BiasDropoutAddLayernormBwd2_fp32)
    ->Ranges({{32, 1024}, {128, 128}, {1024, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(BiasDropoutAddLayernormBwd3_fp32)
    ->Ranges({{32, 1024}, {128, 128}, {1024, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
