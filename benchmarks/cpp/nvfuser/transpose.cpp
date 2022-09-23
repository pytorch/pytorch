#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

#define TRANSPOSE_CONFIG {true, false, false, false}

using namespace torch::jit::fuser::cuda;

struct TransposeConfig {
    bool input1_transpose_axes = false;
    bool input2_transpose_axes = false;
    bool intermediate_transpose_axes = false;
    bool output_transpose_axes = false;
};

std::vector<at::Tensor> generateInputs(
    DataType dtype,
    int num_dims,
    std::pair<int, int> axes,
    int perm_size,
    int innerdim_size,
    bool input1_transpose_axes,
    bool input2_transpose_axes,
    bool non_vectorize_offset = false,
    int iter_size = 32) {
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  std::vector<int64_t> transpose_shape(num_dims, iter_size);
  transpose_shape[axes.second] = innerdim_size;
  transpose_shape[axes.first] = perm_size;

  std::vector<int64_t> non_transpose_shape(num_dims, iter_size);
  non_transpose_shape[axes.first] = innerdim_size;
  non_transpose_shape[axes.second] = perm_size;

  // TensorType: Concrete, Contig, Symbolic
  // Vectorization | Unroll - Add 1 to sizes
  // Shift axis by 1 to disable vectorize loads
  if (non_vectorize_offset) {
    for (auto idx : c10::irange(transpose_shape.size())) {
      transpose_shape[idx] += 1;
    }
    for (auto idx : c10::irange(non_transpose_shape.size())) {
      non_transpose_shape[idx] += 1;
    }
  }

  auto optionalTransposeSize =
      [&transpose_shape, &non_transpose_shape](bool transpose_tensor) {
        return (transpose_tensor) ? transpose_shape : non_transpose_shape;
      };

  at::Tensor aten_input1 =
      at::randn(optionalTransposeSize(input1_transpose_axes), options);
  at::Tensor aten_input2 =
      at::randn(optionalTransposeSize(input2_transpose_axes), options);
  return {aten_input1, aten_input2};
}

//------------------------------------------------------------------------------

static void setupTranspose(
    Fusion* fusion,
    DataType dtype,
    int num_dims,
    std::pair<int, int> axes,
    TransposeConfig tc) {
  FusionGuard fg(fusion);

  auto optionalTranspose = [axes](TensorView* tv, bool is_transpose) {
    return (is_transpose) ? transpose(tv, axes.first, axes.second) : tv;
  };

  auto input1 = makeContigTensor(num_dims, dtype);
  auto input2 = makeContigTensor(num_dims, dtype);
  fusion->addInput(input1);
  fusion->addInput(input2);

  auto ot_input1 = optionalTranspose(input1, tc.input1_transpose_axes);
  auto ot_input2 = optionalTranspose(input2, tc.input2_transpose_axes);
  auto intermediate = add(ot_input1, ot_input2);
  auto ot_intermediate =
      optionalTranspose(intermediate, tc.intermediate_transpose_axes);
  auto output = relu(ot_intermediate);
  auto ot_output = optionalTranspose(output, tc.output_transpose_axes);
  fusion->addOutput(ot_output);
}

static void NvFuserScheduler_Transpose(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int num_dims,
    std::pair<int, int> axes,
    TransposeConfig tc) {
  auto aten_inputs = generateInputs(
      dtype,
      num_dims,
      axes,
      benchmark_state.range(0),
      benchmark_state.range(1),
      tc.input1_transpose_axes,
      tc.input2_transpose_axes);
  auto at_input1 = aten_inputs[0];
  auto at_input2 = aten_inputs[1];

  std::vector<c10::IValue> fuser_inputs = {at_input1, at_input2};
  runBenchmarkIterations(benchmark_state, fusion_executor_cache, fuser_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      ((at_input1.numel() * 3) * int64_t(dataTypeSize(dtype))));
}

//------------------------------------------------------------------------------

#define NVFUSER_TRANSPOSE_SQUARE_RUN(             \
    TITLE, DTYPE, NUM_DIMS, AXIS1, AXIS2, CONFIG) \
  NVFUSER_BENCHMARK_DEFINE(                       \
      TITLE,                                      \
      setupTranspose,                             \
      NvFuserScheduler_Transpose,                 \
      DTYPE,                                      \
      NUM_DIMS,                                   \
      {AXIS1, AXIS2},                             \
      CONFIG);                                    \
                                                  \
  NVFUSER_BENCHMARK_RUN(TITLE)                    \
      ->RangeMultiplier(8)                        \
      ->Args({9, 2408})                           \
      ->Args({16, 512})                           \
      ->Args({18, 96})                            \
      ->Args({24, 96})                            \
      ->Args({24, 256})                           \
      ->Args({24, 512})                           \
      ->Args({32, 27})                            \
      ->Args({32, 96})                            \
      ->Args({32, 288})                           \
      ->Args({32, 864})                           \
      ->Args({40, 120})                           \
      ->Args({48, 128})                           \
      ->Args({48, 256})                           \
      ->Args({49, 512})                           \
      ->Args({49, 1024})                          \
      ->Args({49, 2048})                          \
      ->Args({49, 4608})                          \
      ->Args({64, 64})                            \
      ->Args({64, 96})                            \
      ->Args({64, 128})                           \
      ->Args({64, 147})                           \
      ->Args({64, 192})                           \
      ->Args({64, 256})                           \
      ->Args({64, 288})                           \
      ->Args({64, 512})                           \
      ->Args({80, 64})                            \
      ->Args({81, 1728})                          \
      ->Args({83, 1728})                          \
      ->Args({96, 864})                           \
      ->Args({100, 1280})                         \
      ->Args({100, 4032})                         \
      ->Args({120, 40})                           \
      ->Args({128, 128})                          \
      ->Args({128, 512})                          \
      ->Args({128, 1152})                         \
      ->Args({192, 128})                          \
      ->Args({192, 256})                          \
      ->Args({192, 720})                          \
      ->Args({192, 768})                          \
      ->Args({192, 1120})                         \
      ->Args({192, 1728})                         \
      ->Args({196, 256})                          \
      ->Args({196, 512})                          \
      ->Args({196, 1024})                         \
      ->Args({196, 2304})                         \
      ->Args({256, 256})                          \
      ->Args({256, 1024})                         \
      ->Args({256, 2304})                         \
      ->Args({284, 512})                          \
      ->Args({320, 1280})                         \
      ->Args({320, 1728})                         \
      ->Args({324, 2592})                         \
      ->Args({361, 768})                          \
      ->Args({361, 1120})                         \
      ->Args({384, 2})                            \
      ->Args({384, 32})                           \
      ->Args({384, 128})                          \
      ->Args({384, 256})                          \
      ->Args({384, 512})                          \
      ->Args({384, 1280})                         \
      ->Args({384, 2592})                         \
      ->Args({384, 4032})                         \
      ->Args({448, 1280})                         \
      ->Args({480, 16})                           \
      ->Args({480, 256})                          \
      ->Args({512, 2})                            \
      ->Args({512, 16})                           \
      ->Args({512, 128})                          \
      ->Args({512, 256})                          \
      ->Args({512, 1024})                         \
      ->Args({512, 2048})                         \
      ->Args({512, 3072})                         \
      ->Args({512, 4608})                         \
      ->Args({784, 40})                           \
      ->Args({784, 120})                          \
      ->Args({784, 128})                          \
      ->Args({784, 1152})                         \
      ->Args({1001, 2408})                        \
      ->Args({1024, 16})                          \
      ->Args({1024, 256})                         \
      ->Args({1024, 512})                         \
      ->Args({1024, 1024})                        \
      ->Args({1024, 3072})                        \
      ->Args({1369, 192})                         \
      ->Args({1369, 256})                         \
      ->Args({1369, 288})                         \
      ->Args({2048, 512})                         \
      ->Args({2048, 1024})                        \
      ->Args({2250, 27})                          \
      ->Args({3072, 512})                         \
      ->Args({3072, 1024})                        \
      ->Args({3136, 64})                          \
      ->Args({5329, 720})                         \
      ->Args({5625, 64})                          \
      ->Args({12544, 147})                        \
      ->Args({22201, 288})                        \
      ->Unit(benchmark::kMicrosecond)

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp32_Inner_2D_01_Axis,
    DataType::Float,
    2 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp32_Inner_3D_02_Axis,
    DataType::Float,
    3 /* num_dims */,
    0 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp32_Inner_3D_12_Axis,
    DataType::Float,
    3 /* num_dims */,
    1 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp32_Outer_3D_01_Axis,
    DataType::Float,
    3 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

//------------------------------------------------------------------------------

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp16_Inner_2D_01_Axis,
    DataType::Half,
    2 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp16_Inner_3D_02_Axis,
    DataType::Half,
    3 /* num_dims */,
    0 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp16_Inner_3D_12_Axis,
    DataType::Half,
    3 /* num_dims */,
    1 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_SQUARE_RUN(
    NF_Transpose_Random_fp16_Outer_3D_01_Axis,
    DataType::Half,
    3 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

//------------------------------------------------------------------------------


#define NVFUSER_TRANSPOSE_RUN(TITLE, DTYPE, NUM_DIMS, AXIS1, AXIS2, CONFIG) \
  NVFUSER_BENCHMARK_DEFINE(                                                 \
      TITLE,                                                                \
      setupTranspose,                                                       \
      NvFuserScheduler_Transpose,                                           \
      DTYPE,                                                                \
      NUM_DIMS,                                                             \
      {AXIS1, AXIS2},                                                       \
      CONFIG);                                                              \
                                                                            \
  NVFUSER_BENCHMARK_RUN(TITLE)                                              \
      ->RangeMultiplier(8)                                                  \
      ->Ranges({{2, 256 * 256}, {160, 320}})                                \
      ->Unit(benchmark::kMicrosecond)                                       \

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp32_Inner_2D_01_Axis,
    DataType::Float,
    2 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp32_Inner_3D_02_Axis,
    DataType::Float,
    3 /* num_dims */,
    0 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp32_Inner_3D_12_Axis,
    DataType::Float,
    3 /* num_dims */,
    1 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp32_Outer_3D_01_Axis,
    DataType::Float,
    3 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

//------------------------------------------------------------------------------

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp16_Inner_2D_01_Axis,
    DataType::Half,
    2 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp16_Inner_3D_02_Axis,
    DataType::Half,
    3 /* num_dims */,
    0 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp16_Inner_3D_12_Axis,
    DataType::Half,
    3 /* num_dims */,
    1 /* axis1 */,
    2 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

NVFUSER_TRANSPOSE_RUN(
    NF_Transpose_fp16_Outer_3D_01_Axis,
    DataType::Half,
    3 /* num_dims */,
    0 /* axis1 */,
    1 /* axis2 */,
    TransposeConfig(TRANSPOSE_CONFIG));

//------------------------------------------------------------------------------

static void Baseline_Transpose(
    benchmark::State& benchmark_state,
    DataType dtype,
    int num_dims,
    std::pair<int, int> axes,
    TransposeConfig tc) {
  auto aten_inputs = generateInputs(
      dtype,
      num_dims,
      axes,
      benchmark_state.range(0),
      benchmark_state.range(1),
      tc.input1_transpose_axes,
      tc.input2_transpose_axes);
  auto at_input1 = aten_inputs[0];
  auto at_input2 = aten_inputs[1];

  auto optionalTransposeAten = [&axes](at::Tensor x, bool is_transpose) {
    return (is_transpose) ? at::transpose(x, axes.first, axes.second) : x;
  };

  for (auto _ : benchmark_state) {
    clearL2Cache();
    CudaKernelTimer timer;

    auto at_ot_input1 =
        optionalTransposeAten(at_input1, tc.input1_transpose_axes);
    auto at_ot_input2 =
        optionalTransposeAten(at_input2, tc.input2_transpose_axes);
    auto at_intermediate = add(at_ot_input1, at_ot_input2);
    auto at_ot_intermediate =
        optionalTransposeAten(at_intermediate, tc.intermediate_transpose_axes);
    auto at_output = relu(at_ot_intermediate);
    auto at_ot_output =
        optionalTransposeAten(at_output, tc.output_transpose_axes);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (at_input1.numel() * 3 * int64_t(dataTypeSize(dtype))));
}

//------------------------------------------------------------------------------

static void Baseline_Transpose_fp32_Inner_2D_01_Axis(
    benchmark::State& benchmark_state) {
  Baseline_Transpose(
      benchmark_state,
      DataType::Float,
      2 /* num_dims */,
      {0, 1} /* axes */,
      TRANSPOSE_CONFIG);
}

static void Baseline_Transpose_fp16_Inner_2D_01_Axis(
    benchmark::State& benchmark_state) {
  Baseline_Transpose(
      benchmark_state,
      DataType::Half,
      2 /* num_dims */,
      {0, 1} /* axes */,
      TRANSPOSE_CONFIG);
}

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Transpose_fp32_Inner_2D_01_Axis)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Transpose_fp16_Inner_2D_01_Axis)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
