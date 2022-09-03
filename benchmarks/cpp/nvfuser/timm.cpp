#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>

#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

static void setup_vit_base_patch16_224_bcast7(Fusion* fusion, void* null) {
  FusionGuard fg(fusion);

  auto t2 = makeContigTensor(3, DataType::Float);
  auto t3 = TensorViewBuilder()
                .shape({-1, -1, 1})
                .dtype(DataType::Float)
                .contiguity({true, true, false})
                .build();
  auto t4 = TensorViewBuilder()
                .shape({-1, -1, 1})
                .dtype(DataType::Float)
                .contiguity({true, true, false})
                .build();
  auto t7 = makeContigTensor(3, DataType::Half);

  fusion->addInput(t2);
  fusion->addInput(t3);
  fusion->addInput(t4);
  fusion->addInput(t7);

  auto t8 = castOp(DataType::Float, t7);
  auto t9 = set(t8);
  auto t10 = sub(t2, t3);
  auto t11 = mul(t10, t4);
  auto t25 = mul(t9, t11);
  auto t26 = sum(t25, {0, 1});
  auto t36 = set(t26);
  auto t27 = sum(t9, {0, 1});
  auto t37 = set(t27);
  auto t39 = castOp(DataType::Half, t11);

  fusion->addOutput(t36);
  fusion->addOutput(t37);
  fusion->addOutput(t39);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_bcast7(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t2 = at::randn(input_shape, fp32_options);
  auto t3 = at::randn({input_shape[0], input_shape[1], 1}, fp32_options);
  auto t4 = at::randn({input_shape[0], input_shape[1], 1}, fp32_options);
  auto t7 = at::randn(input_shape, fp16_options);

  std::vector<c10::IValue> aten_inputs({t2, t3, t4, t7});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // full tensor - float + halfx2 - t2, t7, t39
  // Inner most dimension only - floatx2 - t36, t37
  // Outer two dimensions only - floatx2 - t3, t4
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      //           t2 + t7    t3 + t4                 t36 + t37
      t2.numel() * (4 + 2) + t3.numel() * 4 * 2 + input_shape[2] * (4 * 2) +
      // T39
      t2.numel() * 2);
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast7,
    setup_vit_base_patch16_224_bcast7,
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast7,
    nullptr);

// pwise case, broadcasting both sides
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast7)
    ->Args({64, 197, 768})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void setup_vit_base_patch16_224_bcast5(Fusion* fusion, void* null) {
  FusionGuard fg(fusion);

  auto t2 = makeContigTensor(3, DataType::Float);
  auto t5 = makeContigTensor(1, DataType::Float);
  auto t3 = makeContigTensor(3, DataType::Half);
  auto t0 = makeContigTensor(1, DataType::Float);
  auto t1 = makeContigTensor(1, DataType::Float);

  fusion->addInput(t2);
  fusion->addInput(t5);
  fusion->addInput(t3);
  fusion->addInput(t0);
  fusion->addInput(t1);

  std::vector<bool> bcast_pattern0({true, true, false});
  std::vector<bool> bcast_pattern1({false, false, true});

  auto t4 = castOp(DataType::Float, t3);
  auto t6 = set(t5);
  auto t7 = broadcast(t6, bcast_pattern0);
  auto t8 = add(t4, t7);
  auto t9 = rand_like(t8);
  auto d34 =
      sub(IrBuilder::create<Double>(1.0), IrBuilder::create<Double>(0.0));
  auto t10 = lt(t9, d34);
  auto t11 = castOp(DataType::Float, t10);
  auto t12 = mul(t8, t11);
  auto b36 = eq(d34, IrBuilder::create<Double>(0.0));
  auto d37 = castOp(DataType::Double, b36);
  auto d38 = add(d37, d34);
  auto d40 = div(IrBuilder::create<Double>(1.0), d38);
  auto t13 = mul(t12, d40);
  auto t14 = set(t13);
  auto t15 = add(t2, t14);
  auto t16 = set(t15);
  auto t36 = sum(t16, {2});
  auto d151 = castOp(DataType::Double, t2->axis(2)->extent());
  auto d152 = mul(IrBuilder::create<Double>(1.0), d151);
  auto t19 = div(t36, d152);
  auto t22 = broadcast(t19, bcast_pattern1);
  auto t23 = sub(t16, t22);
  auto t37 = mul(t23, t23);
  auto t20 = sum(t37, {2});
  auto t24 = broadcast(t20, bcast_pattern1);
  auto d95 = castOp(DataType::Double, t2->axis(2)->extent());
  auto d105 = reciprocal(d95);
  auto t25 = mul(t24, d105);
  auto t26 = add(t25, IrBuilder::create<Double>(1e-6));
  auto t27 = rsqrt(t26);
  auto t28 = mul(t23, t27);
  auto t17 = set(t1);
  auto t29 = broadcast(t17, bcast_pattern0);
  auto t30 = mul(t28, t29);
  auto t18 = set(t0);
  auto t31 = broadcast(t18, bcast_pattern0);
  auto t32 = add(t30, t31);
  auto t33 = set(t32);
  auto t34 = castOp(DataType::Half, t33);

  fusion->addOutput(t16); // full 3d float
  fusion->addOutput(t10); // full 3d bool
  fusion->addOutput(t22); // bcast last dim float
  fusion->addOutput(t27); // bcast last dim float
  fusion->addOutput(t18); // passthrough t0 float
  fusion->addOutput(t17); // passthrough t1 float
  fusion->addOutput(t34); // full 3d half
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_bcast5(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t2 = at::randn(input_shape, fp32_options);
  auto t5 = at::randn({input_shape[2]}, fp32_options);
  auto t3 = at::randn(input_shape, fp16_options);
  auto t0 = at::randn({input_shape[2]}, fp32_options);
  auto t1 = at::randn({input_shape[2]}, fp32_options);

  std::vector<c10::IValue> aten_inputs({t2, t5, t3, t0, t1});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // Full tensor - floatx2, halfx2, bool - t2, t16, t3, t34, t16
  // Inner most dim only - floatx5 - t5, t0, t1, t7, t17
  // Outer two dims only - floatx2 - t22, t27

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      t2.numel() * (2 * 4 + 2 * 2 + 1) + t5.numel() * 5 * 4 +
      input_shape[0] * input_shape[1] * 2 * 4);
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast5_NCHW,
    setup_vit_base_patch16_224_bcast5,
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast5,
    nullptr);

// Broadcast on both sides
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_vit_base_patch16_224_bcast5_NCHW)
    ->Args({64, 197, 768})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void setup_vit_base_patch16_224_bcast_outer2(
    Fusion* fusion,
    void* null) {
  FusionGuard fg(fusion);

  auto t0 = makeContigTensor(3, DataType::Half);
  auto t2 = makeContigTensor(1, DataType::Float);

  fusion->addInput(t0);
  fusion->addInput(t2);

  auto t1 = castOp(DataType::Float, t0);
  auto t3 = set(t2);
  auto t4 = broadcast(t3, {true, true, false});
  auto t5 = add(t1, t4);
  auto t6 = castOp(DataType::Half, t5);
  auto t7 = castOp(DataType::Half, t3);

  fusion->addOutput(t6);
  fusion->addOutput(t7);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_outer2(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, fp16_options);
  auto t2 = at::randn({input_shape[2]}, fp32_options);

  std::vector<c10::IValue> aten_inputs({t0, t2});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // full tensor - halfx2 - t0, t6
  // inner dimension only - halfx2 - t2, t7
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * t0.numel() * (2 + 2) +
      input_shape[2] * (2 + 4));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_outer2,
    setup_vit_base_patch16_224_bcast_outer2,
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_outer2,
    nullptr);

NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_outer2)
    ->Args({64, 197, 2304})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void setup_vit_base_patch16_224_norm_inner3(Fusion* fusion, void* null) {
  FusionGuard fg(fusion);

  auto t0 = makeContigTensor(4, DataType::Half);
  fusion->addInput(t0);
  auto d13 = IrBuilder::create<Double>();
  fusion->addInput(d13);

  auto t1 = castOp(DataType::Float, t0);
  auto t2 = set(t1);
  auto t3 = mul(t2, d13);
  auto t4 = set(t3);
  auto t5 = max(t4, {3});
  auto t6 = broadcast(t5, {false, false, false, true});
  auto t7 = sub(t4, t6);
  auto t8 = exp(t7);
  auto t9 = sum(t8, {3});
  auto t10 = broadcast(t9, {false, false, false, true});
  auto t11 = reciprocal(t10);
  auto t12 = mul(t8, t11);
  auto t13 = rand_like(t12);
  auto d79 = sub(IrBuilder::create<Double>(1), IrBuilder::create<Double>(0));
  auto t14 = lt(t13, d79);
  auto t15 = castOp(DataType::Float, t14);
  auto b81 = eq(d79, IrBuilder::create<Double>(0));
  auto d82 = castOp(DataType::Double, b81);
  auto d83 = add(d82, d79);
  auto d85 = div(IrBuilder::create<Double>(1), d83);
  auto t16 = mul(t12, t15);
  auto t17 = mul(t16, d85);
  auto t18 = set(t17);
  auto t19 = castOp(DataType::Half, t18);

  fusion->addOutput(t19);
  fusion->addOutput(t14);
  fusion->addOutput(t12);
  fusion->addOutput(t4);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_norm_inner3(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, fp16_options);

  std::vector<c10::IValue> aten_inputs({t0, 0.125});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // Full tensors - floatx2, half x2, bool - t12, t4, t0, t19, t14
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * t0.numel() * 13);
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_norm_inner3,
    setup_vit_base_patch16_224_norm_inner3,
    NvFuserScheduler_TIMM_vit_base_patch16_224_norm_inner3,
    nullptr);

// Norm inner dim
NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_norm_inner3)
    ->Args({64, 12, 197})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void setup_vit_base_patch16_224_bcast_outer6(
    Fusion* fusion,
    void* null) {
  FusionGuard fg(fusion);

  auto t0 = makeContigTensor(3, DataType::Half);
  auto t2 = makeContigTensor(1, DataType::Float);

  fusion->addInput(t0);
  fusion->addInput(t2);

  auto t1 = castOp(DataType::Float, t0);
  auto t3 = set(t2);
  auto t4 = broadcast(t3, {true, true, false});
  auto t5 = add(t1, t4);
  auto t6 = set(t5);
  auto t7 = mul(t6, IrBuilder::create<Double>(0.707106));
  auto t8 = erf(t7);
  auto t9 = add(IrBuilder::create<Double>(1), t8);
  auto t10 = mul(IrBuilder::create<Double>(0.5), t9);
  auto t11 = mul(t6, t10);
  auto t12 = rand_like(t11);
  auto d66 = sub(IrBuilder::create<Double>(1), IrBuilder::create<Double>(0));
  auto t13 = lt(t12, d66);
  auto t14 = castOp(DataType::Float, t13);
  auto t15 = mul(t11, t14);
  auto b68 = eq(d66, IrBuilder::create<Double>(0));
  auto d69 = castOp(DataType::Double, b68);
  auto d70 = add(d69, d66);
  auto d72 = div(IrBuilder::create<Double>(1), d70);
  auto t16 = mul(t15, d72);
  auto t17 = set(t16);
  auto t18 = castOp(DataType::Half, t17);
  auto t19 = castOp(DataType::Half, t3);

  fusion->addOutput(t18);
  fusion->addOutput(t13);
  fusion->addOutput(t6);
  fusion->addOutput(t19);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_outer6(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, fp16_options);
  auto t2 = at::randn({input_shape[2]}, fp32_options);

  std::vector<c10::IValue> aten_inputs({t0, t2});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);
  // full tensors - float, halfx2, bool - t6, t0, t18, t13
  // inner dimension only - float, half - t2, t19
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * t0.numel() * (2 + 2 + 1 + 4) +
      input_shape[2] * (4 + 2));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_outer6,
    setup_vit_base_patch16_224_bcast_outer6,
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_outer6,
    nullptr);

NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_outer6)
    // First size is original, the rest are variations to check perf
    // reliability.
    ->Args({64, 197, 3 * 1024})
    ->Args({64, 197, 2 * 1024})
    ->Args({64, 197, 1024})
    ->Args({64, 197, 512})
    ->Args({3, 1024, 64 * 197})
    ->Args({2, 1024, 64 * 197})
    ->Args({1, 1024, 64 * 197})
    ->Args({2, 256, 64 * 197})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Reverse the broadcast dimensions to check for consistency in scheduling.
static void setup_vit_base_patch16_224_bcast_inner6(
    Fusion* fusion,
    void* null) {
  FusionGuard fg(fusion);

  auto t0 = makeContigTensor(3, DataType::Half);
  auto t2 = makeContigTensor(2, DataType::Float);

  fusion->addInput(t0);
  fusion->addInput(t2);

  auto t1 = castOp(DataType::Float, t0);
  auto t3 = set(t2);
  auto t4 = broadcast(t3, {false, false, true});
  auto t5 = add(t1, t4);
  auto t6 = set(t5);
  auto t7 = mul(t6, IrBuilder::create<Double>(0.707106));
  auto t8 = erf(t7);
  auto t9 = add(IrBuilder::create<Double>(1), t8);
  auto t10 = mul(IrBuilder::create<Double>(0.5), t9);
  auto t11 = mul(t6, t10);
  auto t12 = rand_like(t11);
  auto d66 = sub(IrBuilder::create<Double>(1), IrBuilder::create<Double>(0));
  auto t13 = lt(t12, d66);
  auto t14 = castOp(DataType::Float, t13);
  auto t15 = mul(t11, t14);
  auto b68 = eq(d66, IrBuilder::create<Double>(0));
  auto d69 = castOp(DataType::Double, b68);
  auto d70 = add(d69, d66);
  auto d72 = div(IrBuilder::create<Double>(1), d70);
  auto t16 = mul(t15, d72);
  auto t17 = set(t16);
  auto t18 = castOp(DataType::Half, t17);
  auto t19 = castOp(DataType::Half, t3);

  fusion->addOutput(t18);
  fusion->addOutput(t13);
  fusion->addOutput(t6);
  fusion->addOutput(t19);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_inner6(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, fp16_options);
  auto t2 = at::randn({input_shape[0], input_shape[1]}, fp32_options);

  std::vector<c10::IValue> aten_inputs({t0, t2});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // full tensors - float, halfx2, bool - t6, t0, t18, t13
  // outer two dimensions only - float, half - t2, t19
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * t0.numel() * (2 + 2 + 1 + 4) +
      input_shape[0] * input_shape[1] * (4 + 2));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_inner6,
    setup_vit_base_patch16_224_bcast_inner6,
    NvFuserScheduler_TIMM_vit_base_patch16_224_bcast_inner6,
    nullptr);

NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_bcast_inner6)
    ->Args({64, 197, 3 * 1024})
    ->Args({64, 197, 2 * 1024})
    ->Args({64, 197, 1024})
    ->Args({64, 197, 512})
    ->Args({3, 1024, 64 * 197})
    ->Args({2, 1024, 64 * 197})
    ->Args({1, 1024, 64 * 197})
    ->Args({2, 256, 64 * 197})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void setup_vit_base_patch16_224_LN_BWD(Fusion* fusion, void* null) {
  FusionGuard fg(fusion);

  auto t0 = makeContigTensor(3, DataType::Bool);
  fusion->addInput(t0);

  auto t1 = makeContigTensor(3, DataType::Half);
  fusion->addInput(t1);

  auto t2 = castOp(DataType::Float, t1);

  auto t3 = makeContigTensor(3, DataType::Half);
  fusion->addInput(t3);

  auto t4 = castOp(DataType::Float, t3);

  auto d35 = t3->axis(2)->extent();

  auto t5 = TensorViewBuilder()
                .shape({-1, -1, 1})
                .dtype(DataType::Float)
                .contiguity({true, true, false})
                .build();
  fusion->addInput(t5);

  auto t6 = TensorViewBuilder()
                .shape({-1, -1, 1})
                .dtype(DataType::Float)
                .contiguity({true, true, false})
                .build();
  fusion->addInput(t6);

  auto t7 = makeContigTensor(1, DataType::Half);
  fusion->addInput(t7);

  auto t8 = castOp(DataType::Float, t7);

  auto t9 = makeContigTensor(1, DataType::Half);
  fusion->addInput(t9);

  auto t11 = sub(t4, t5);
  auto t12 = mul(t11, t6);

  auto t13 = broadcast(t8, {true, true, false});
  auto t14 = mul(t2, t13);
  auto t15 = mul(d35, t14);
  auto t16 = sum(t14, {2});
  auto t17 = broadcast(t16, {false, false, true});
  auto t18 = mul(t14, t12);
  auto t19 = sum(t18, {2});
  auto t20 = broadcast(t19, {false, false, true});

  auto t40 = castOp(DataType::Half, t12);
  auto t41 = castOp(DataType::Float, t40);
  auto t42 = castOp(DataType::Half, t20);
  auto t43 = castOp(DataType::Float, t42);
  auto t21 = mul(t42, t43);

  auto t38 = castOp(DataType::Half, t15);
  auto t39 = castOp(DataType::Float, t38);
  auto t44 = castOp(DataType::Half, t17);
  auto t45 = castOp(DataType::Float, t44);
  auto t22 = sub(t39, t45);

  auto t23 = sub(t22, t21);

  auto d87 = reciprocal(d35);
  auto t24 = mul(d87, t6);

  auto t25 = mul(t24, t23);
  auto t26 = mul(t2, t41);
  auto t27 = sum(t26, {0, 1});
  auto t28 = sum(t2, {0, 1});

  auto t29 = castOp(DataType::Float, t0);
  auto t30 = mul(t25, t29);

  auto d33 = IrBuilder::create<Double>();
  fusion->addInput(d33);
  auto t31 = mul(t30, d33);
  auto t32 = sum(t31, {0, 1});
  auto t33 = castOp(DataType::Half, t32);
  auto t34 = castOp(DataType::Half, t31);
  auto t35 = castOp(DataType::Half, t25);
  auto t36 = castOp(DataType::Half, t27);
  auto t37 = castOp(DataType::Half, t28);

  fusion->addOutput(t33);
  fusion->addOutput(t34);
  fusion->addOutput(t35);
  fusion->addOutput(t36);
  fusion->addOutput(t37);
}

static void NvFuserScheduler_TIMM_vit_base_patch16_224_LN_BWD(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);
  // auto bool_options = at::TensorOptions().dtype(at::kBool).device(at::kCUDA,
  // 0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, fp16_options).to(at::kBool);
  auto t1 = at::randn(input_shape, fp16_options);
  auto t3 = at::randn(input_shape, fp16_options);
  auto t5 = at::randn({input_shape[0], input_shape[1], 1}, fp32_options);
  auto t6 = at::randn({input_shape[0], input_shape[1], 1}, fp32_options);
  auto t7 = at::randn({input_shape[2]}, fp16_options);
  auto t9 = at::randn({input_shape[2]}, fp16_options);

  std::vector<c10::IValue> aten_inputs({t0, t1, t3, t5, t6, t7, t9, 1.0});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // Full tensors - bool, halfx4 - t0, t1, t3, t34, t35
  // Outer two dimensions - floatx2 - t5, t6
  // Inner dimension - halfx5 - t7, t9, t33, t36, t37
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * ((t0.numel() * (4 * 2 + 1))) +
      (t5.numel() * 4 * 2) + (t7.numel() * 5 * 2));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_LN_BWD,
    setup_vit_base_patch16_224_LN_BWD,
    NvFuserScheduler_TIMM_vit_base_patch16_224_LN_BWD,
    nullptr);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_NCHW_vit_base_patch16_224_LN_BWD)
    ->Args({128, 197, 768})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void nhwc_seresnet152d_transpose65(Fusion* fusion, void* null) {
  FusionGuard fg(fusion);

  auto t2 = makeContigTensor(4, DataType::Half);
  auto t5 = makeContigTensor(4, DataType::Half);
  auto t7 = makeContigTensor(4, DataType::Half);
  auto t9 = makeContigTensor(4, DataType::Half);
  auto t4 = makeConcreteTensor({}, DataType::Half);

  fusion->addInput(t2);
  fusion->addInput(t5);
  fusion->addInput(t7);
  fusion->addInput(t9);
  fusion->addInput(t4);

  auto d86 = IrBuilder::create<Double>(0);

  auto t3 = castOp(DataType::Float, t2);
  auto t6 = castOp(DataType::Float, t5);
  auto t8 = castOp(DataType::Float, t7);
  auto t10 = castOp(DataType::Float, t9);
  auto t11 = add(t8, t10);
  auto t12 = set(t11);
  auto t13 = set(t6);
  auto t14 = lt(t13, d86);
  auto t15 = broadcast(t4, {true, true, true, true});
  auto t16 = where(t14, t15, t12);
  auto t17 = set(t16);
  auto t29 = castOp(DataType::Half, t17);
  auto t18 = mul(t17, t3);
  auto t19 = permute(t18, {0, 2, 3, 1});
  auto t30 = castOp(DataType::Half, t19);

  fusion->addOutput(t29);
  fusion->addOutput(t30);
}

static void NvFuserScheduler_nhwc_seresnet152d_transpose65(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    void* null) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(2),
      benchmark_state.range(1)};

  at::manual_seed(0);
  auto fp16_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t2 = at::randn(input_shape, fp16_options);
  auto t5 = at::randn(input_shape, fp16_options);
  auto t7 = at::randn(input_shape, fp16_options);
  auto t9 = at::randn(input_shape, fp16_options);
  // Need zero dim tensor don't know how to do that, so just going to reduce a
  // 1D tensor
  auto t4 = at::randn({2}, fp16_options).sum();

  std::vector<c10::IValue> aten_inputs({t2, t5, t7, t9, t4});
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // Full tensors - halfx6 - t2, t5, t7, t9, t29, t30
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * t2.numel() * 6 * 2);
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_nhwc_seresnet152d_transpose65,
    nhwc_seresnet152d_transpose65,
    NvFuserScheduler_nhwc_seresnet152d_transpose65,
    nullptr);

// Norm inner dim Half version of vit_base_patch16_224_norm_inner3
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_nhwc_seresnet152d_transpose65)
    ->Args({128, 12, 197})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
