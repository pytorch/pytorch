#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/inlining.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/transpose.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionTranspose1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = transpose(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  at::Tensor aten_output = t0.t();

  testValidate(
      &fusion, outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTranspose2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = transpose(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->split(0, 32);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  at::Tensor aten_output = t0.t();

  testValidate(
      &fusion, outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTransposeWithSwizzle_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0);
  fusion.addOutput(tv1);

  // tv0: [I0, I1]
  // tv1: [I1, I0]

  const int BS = 32;

  // CTA tiling by BS*BS
  tv1->split(1, BS);
  tv1->split(0, BS);
  tv1->reorder({{1, 2}});
  // tv1: [I1/BS, I0/BS, BS(I1), BS(I0)]

  // Create a smem buffer to cache each tile
  auto tv0_cache = tv0->cacheAfter();
  tv0_cache->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv1, 2);
  // tv0: [I0, I1]
  // tv0_cache: [I1/BS, I0/BS, BS(I1), BS(I0)]
  // tv1: [I1/BS, I0/BS, BS(I1), BS(I0)]

  // Assign each thread block to a tile
  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);

  // Thread mapping for each tile. For both of the input and output
  // tiles, map TIDx to the fastest-changing dimension to facilitate
  // coalesced gmem accesses.
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(3)->parallelize(ParallelType::TIDx);
  // Note that the fastest-changing axis is next to the inner-most
  // axis since computeAt reorders the axes as the output tensor.
  tv0_cache->axis(2)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(3)->parallelize(ParallelType::TIDy);

  // Swizzles the smem cache to avoid bank conflicts
  tv0_cache->swizzle(SwizzleType::Transpose, {3, 2});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 100;
  const int by = 200;
  at::Tensor t0 = at::randn({bx, by}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0.t();

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTransposeWithSwizzle1DThreadBlock_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0);
  fusion.addOutput(tv1);

  // tv0: [I0, I1]
  // tv1: [I1, I0]

  const int BS = 32;
  const int BDIM = 256;

  // CTA tiling by BS*BS
  tv1->split(1, BS);
  tv1->split(0, BS);
  tv1->reorder({{1, 2}});
  // tv1: [I1/BS, I0/BS, BS(I1), BS(I0)]

  // Create a smem buffer to cache each tile
  auto tv0_cache = tv0->cacheAfter();
  tv0_cache->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv1, 2);
  // tv0: [I0, I1]
  // tv0_cache: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]
  // tv1: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Tranform the tile axes for 1D thread mapping
  tv1->merge(-2, -1);
  tv1->split(-1, BDIM);
  // tv1: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Transform the cache similarly but apply swizzle to the 2D tile axes.
  tv0_cache->reorder({{-2, -1}});
  tv0_cache->swizzle(SwizzleType::Transpose, {2, 3});
  tv0_cache->merge(-2, -1);
  tv0_cache->split(-1, BDIM);
  // tv0: [I1/BS, I0/BS, BS*BS/BDIM, BDIM]

  // Assign each thread block to a tile
  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);

  // Thread mapping for each tile.
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 100;
  const int by = 200;
  at::Tensor t0 = at::randn({bx, by}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0.t();

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSimple_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->tanspose->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSinTransposeCos_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 2);
  auto tv2 = sin(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = cos(tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.transpose(0, 2).sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

/*
 * t0->transpose--.
 *                 \
 * t1->transpose---add-->sin->t5
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleInput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 0, 2);
  auto tv3 = transpose(tv1, 0, 2);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sin(tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 1024, 1024}, options);
  at::Tensor input1 = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = (input0.transpose(0, 2) + input1.transpose(0, 2)).sin();

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// t0->sin->transpose->t5
//  `->cos->transpose->t6
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv2 = sin(tv0);
  auto tv3 = cos(tv0);
  auto tv5 = transpose(tv2, 0, 2);
  auto tv6 = transpose(tv3, 0, 2);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref1 = input.sin().transpose(0, 2);
  auto tv_ref2 = input.cos().transpose(0, 2);

  testValidate(
      &fusion, outputs, {input}, {tv_ref1, tv_ref2}, __LINE__, __FILE__);
}

/*
 * t0->transpose->sin->t3
 *   \_.-->cos->t5
 *   /
 * t1
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleInputOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 0, 2);
  auto tv3 = sin(tv2);
  fusion.addOutput(tv3);
  auto tv4 = add(tv0, tv1);
  auto tv5 = cos(tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 1024, 1024}, options);
  at::Tensor input1 = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref1 = input0.transpose(0, 2).sin();
  auto tv_ref2 = (input0 + input1).cos();

  testValidate(
      &fusion,
      outputs,
      {input0, input1},
      {tv_ref1, tv_ref2},
      __LINE__,
      __FILE__);
}

/*
 *             .------>sin------>z
 * x->transpose->transpose->add->y
 *  \_______________________/
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMatchingSkipConnection_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 2);
  auto tv2 = transpose(tv1, 0, 2);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);
  auto tv4 = sin(tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref1 = input.transpose(0, 2).transpose(0, 2) + input;
  auto tv_ref2 = input.transpose(0, 2).sin();

  testValidate(
      &fusion, outputs, {input}, {tv_ref1, tv_ref2}, __LINE__, __FILE__);
}

// x->transpose--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleTransposeBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 1, 2);
  auto tv3 = broadcast(tv1, {false, false, true});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1024, 256, 1024}, options);
  at::Tensor input1 = at::randn({1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});
  // auto lparams = schedulePointwise(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = input0.transpose(1, 2) + input1.unsqueeze(2);

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// x->broadcast--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleTransposeNoReference_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true, false});
  auto tv3 = broadcast(tv1, {false, false, true});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 1024}, options);

  EXPECT_THAT(
      [&]() {
        scheduleTranspose(&fusion, {input0, input1});
      },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("reference tensor")));
}

// x->broadcast--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleBroadcastOnly_CUDA) {
  for (bool contig0 : {true, false}) {
    for (bool contig1 : {true, false}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto tv0 = contig0 ? makeContigConcreteTensor({-1, 1, -1})
                         : makeConcreteTensor({-1, 1, -1});
      auto tv1 = contig1 ? makeContigConcreteTensor({-1, -1, 1})
                         : makeConcreteTensor({-1, -1, 1});
      fusion.addInput(tv0);
      fusion.addInput(tv1);
      auto tv2 = add(tv0, tv1);
      fusion.addOutput(tv2);

      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor input0 = at::randn({1024, 1, 256}, options);
      at::Tensor input1 = at::randn({1024, 1024, 1}, options);

      auto lparams = scheduleTranspose(&fusion, {input0, input1});

      FusionExecutor fe;
      fe.compileFusion(&fusion, {input0, input1}, lparams);
      auto outputs = fe.runFusion({input0, input1}, lparams);

      auto tv_ref = input0 + input1;

      testValidate(
          &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
    }
  }
}

// mermaid graph:
// ```mermaid
// %%{
//   init: {
//     'theme': 'base',
//     'themeVariables': { 'fontSize': '30px', 'fontFamily': 'times'}}
// }%%
// graph TD
//   T0("T0(M, N, K)")
//   T1("T1(N, M, K)")
//   T2("T2(M, K, N)")
//   T0 --> A("transpose(1, 2)") --> T3("T3(M, K, N)")
//   T1 ---> sigmoid --> T5("T5(N, M, K)")
//   T5 --> B("transpose(0, 2)") --> T7("T7(K, M, N)")
//   T2 ----> C("add")
//   T3 --> C --> T6("T6(M, K, N)")
//   T6 --> D("transpose(0, 1)") --> T11("T11(K, M, N)")
//   T11 --> E("add") -->T12("T12(K, M, N)")
//   T7 --> E
//   T1 ---> F("transpose(0, 1)") --> T4("T4(M, N, K)")
//   T0 --> G("add") --> T8("T8(M, N, K)") --> relu ---> T9("T9(M, N, K)")
//   T4 --> G
//   T6 ---> sin ---> T10("T10(M, K, N)")
//   style T0 fill:lightgreen
//   style T1 fill:lightgreen
//   style T2 fill:lightgreen
//   style T12 fill:lightblue
//   style T9 fill:lightblue
//   style T10 fill:lightblue
// ```
TEST_F(NVFuserTest, FusionScheduleTransposeComplexDAG1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  auto tv2 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = transpose(tv0, 1, 2);
  auto tv4 = transpose(tv1, 0, 1);
  auto tv5 = sigmoid(tv1);
  auto tv6 = add(tv2, tv3);
  auto tv7 = transpose(tv5, 0, 2);
  auto tv8 = add(tv4, tv0);
  auto tv9 = relu(tv8);
  fusion.addOutput(tv9);
  auto tv10 = sin(tv6);
  fusion.addOutput(tv10);
  auto tv11 = transpose(tv6, 0, 1);
  auto tv12 = add(tv7, tv11);
  fusion.addOutput(tv12);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({512, 1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 512, 256}, options);
  at::Tensor input2 = at::randn({512, 256, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1, input2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2}, lparams);
  auto outputs = fe.runFusion({input0, input1, input2}, lparams);

  auto t3 = input0.transpose(1, 2);
  auto t4 = input1.transpose(0, 1);
  auto t5 = input1.sigmoid();
  auto t6 = input2 + t3;
  auto t7 = t5.transpose(0, 2);
  auto t8 = t4 + input0;
  auto t9 = t8.relu();
  auto t10 = t6.sin();
  auto t11 = t6.transpose(0, 1);
  auto t12 = t7 + t11;

  testValidate(
      &fusion,
      outputs,
      {input0, input1, input2},
      {t9, t10, t12},
      __LINE__,
      __FILE__);
}

// mermaid graph:
// ```mermaid
// %%{
//   init: {
//     'theme': 'base',
//     'themeVariables': { 'fontSize': '30px', 'fontFamily': 'times'}}
// }%%
// graph TD
//   T0("T0(M, N, K)")
//   T1("T1(N, M, K)")
//   T2("T2(M, K, N)")
//   T0 --> A("transpose(1, 2)") --> T3("T3(M, K, N)")
//   T1 ---> sigmoid --> T5("T5(N, M, K)")
//   T5 --> B("transpose(0, 2)") --> T7("T7(K, M, N)")
//   T2 ----> C("add")
//   T3 --> C --> T6("T6(M, K, N)")
//   T6 --> D("transpose(0, 1)") --> T11("T11(K, M, N)")
//   T11 --> E("add") -->T12("T12(K, M, N)")
//   T7 --> E
//   T1 ---> F("transpose(0, 1)") --> T4("T4(M, N, K)")
//   T0 --> G("add") --> T8("T8(M, N, K)") --> relu ---> T9("T9(M, N, K)")
//   T4 --> G
//   T6 ---> sin ---> T10("T10(M, K, N)")
//   style T0 fill:lightgreen
//   style T1 fill:lightgreen
//   style T2 fill:lightgreen
//   style T12 fill:lightblue
//   style T9 fill:lightblue
//   style T10 fill:lightblue
// ```
TEST_F(NVFuserTest, FusionManualScheduleTransposeComplexDAG1_CUDA) {
  // achieved: 833.526 GB/s on RTX 3090 (theoretical bandwidth: 936 GB/s)
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  auto tv2 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = transpose(tv0, 1, 2);
  auto tv4 = transpose(tv1, 0, 1);
  auto tv5 = sigmoid(tv1);
  auto tv6 = add(tv2, tv3);
  auto tv7 = transpose(tv5, 0, 2);
  auto tv8 = add(tv4, tv0);
  auto tv9 = relu(tv8);
  fusion.addOutput(tv9);
  auto tv10 = sin(tv6);
  fusion.addOutput(tv10);
  auto tv11 = transpose(tv6, 0, 1);
  auto tv12 = add(tv7, tv11);
  fusion.addOutput(tv12);

  // group 1: tv0, tv1, *tv9, innermost dim K
  // group 2: tv2, *tv10, tv12, innermost dim N

  // cache inputs and outputs
  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();
  auto tv2_cache = tv2->cacheAfter();
  auto tv9_cache = tv9->cacheBefore();
  auto tv10_cache = tv10->cacheBefore();
  auto tv12_cache = tv12->cacheBefore();

  // Step 1: Make 32x32 tiles, schedule outer dimensions
  {
    // Pick an arbitrary tensor as a reference tensor for this step. There is no
    // requirement on which group this reference tensor should belong to. Here
    // we pick tv9, which belongs to group 1.

    // Make 32x32 tile:
    // [M, N, K]
    tv9->split(1, 32);
    tv9->reorder({{2, -1}});
    tv9->split(2, 32);
    tv9->reorder({{3, -1}});
    // [M, N/32, K/32, 32(N), 32(K)]

    // merge outer dims, parallelize on BIDx, and unswitch
    tv9->merge(0);
    tv9->merge(0);
    tv9->split(0, 1);
    // [M * N/32 * K/32, 1, 32(N), 32(K)]
    tv9->axis(0)->parallelize(ParallelType::BIDx);
    tv9->axis(1)->parallelize(ParallelType::Unswitch);
    // [BIDx, Unswitch, 32(N), 32(K)]

    // propagate to the entire DAG
    MaxRootDomainInfoSpanningTree entire_dag(tv9);
    TransformPropagator tp(tv9);
    entire_dag.traverse(&tp);
    scheduler_utils::parallelizeAllLike(tv9);
  }

  constexpr int threads_per_block = 128;

  // Step 2, schedule group 2
  {
    // group 2: tv2, *tv10, tv12, innermost dim N

    tv2_cache->setMemoryType(MemoryType::Shared);
    tv10_cache->setMemoryType(MemoryType::Shared);
    tv12_cache->setMemoryType(MemoryType::Shared);

    // pick tv10 as reference tensor for group 2
    // [BIDx, Unswitch, 32(N), 32(K)]
    tv10->reorder({{-1, -2}});
    // [BIDx, Unswitch, 32(K), 32(N)]
    tv10->merge(2);
    tv10->split(2, 4);
    tv10->split(2, threads_per_block);
    tv10->axis(-1)->parallelize(ParallelType::Vectorize);
    tv10->axis(-2)->parallelize(ParallelType::TIDx);
    tv10->axis(-3)->parallelize(ParallelType::Unroll);
    // [BIDx, Unswitch, Unroll, TIDx, Vectorize]

    // Propagate to group 2 and its cache. Note that group 2 and its cache are
    // not connected, so we need to borrow other tensors of the DAG to be able
    // to propagate. The transformations on borrowed tensors will be overwritten
    // in the next step. We can not borrow the reference tensor of group 1.
    auto all_tvs_except_ref1 = ir_utils::allTvsExcept(&fusion, {tv9});
    auto all_tvs_except_ref1_set = std::unordered_set<TensorView*>(
        all_tvs_except_ref1.begin(), all_tvs_except_ref1.end());
    SetSelector selector(all_tvs_except_ref1_set);
    MaxRootDomainInfoSpanningTree tree(tv10, &selector);
    TransformPropagator tp(tv10);
    tree.traverse(&tp);
    scheduler_utils::parallelizeAllLike(
        tv10, {tv2_cache, tv10, tv12}, {ParallelType::TIDx});
    scheduler_utils::parallelizeAllLike(
        tv10,
        {tv2_cache, tv10, tv12},
        {ParallelType::Vectorize, ParallelType::Unroll});
  }

  // Step 3, schedule group 1
  {
    // group 1: tv0, tv1, *tv9, innermost dim K
    // [BIDx, Unswitch, 32(N), 32(K)]
    tv9->merge(2);
    tv9->split(2, 4);
    tv9->split(2, threads_per_block);
    tv9->axis(-1)->parallelize(ParallelType::Vectorize);
    tv9->axis(-2)->parallelize(ParallelType::TIDx);
    tv9->axis(-3)->parallelize(ParallelType::Unroll);
    // [BIDx, Unswitch, Unroll, TIDx, Vectorize]

    // Propagate to the entire DAG except for group 2 and its cached inputs
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(&fusion, {tv2, tv2_cache, tv10, tv12});
    auto all_tvs_except2_set = std::unordered_set<TensorView*>(
        all_tvs_except2.begin(), all_tvs_except2.end());
    SetSelector selector(all_tvs_except2_set);
    MaxRootDomainInfoSpanningTree tree(tv9, &selector);
    TransformPropagator tp(tv9);
    tree.traverse(&tp);
    scheduler_utils::parallelizeAllLike(
        tv9, all_tvs_except2, {ParallelType::TIDx});
    scheduler_utils::parallelizeAllLike(
        tv9,
        {tv0_cache, tv1_cache, tv9},
        {ParallelType::Vectorize, ParallelType::Unroll});
  }

  // inline
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({512, 1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 512, 256}, options);
  at::Tensor input2 = at::randn({512, 256, 1024}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2});
  auto outputs = fe.runFusion({input0, input1, input2});

  auto t3 = input0.transpose(1, 2);
  auto t4 = input1.transpose(0, 1);
  auto t5 = input1.sigmoid();
  auto t6 = input2 + t3;
  auto t7 = t5.transpose(0, 2);
  auto t8 = t4 + input0;
  auto t9 = t8.relu();
  auto t10 = t6.sin();
  auto t11 = t6.transpose(0, 1);
  auto t12 = t7 + t11;

  testValidate(
      &fusion,
      outputs,
      {input0, input1, input2},
      {t9, t10, t12},
      __LINE__,
      __FILE__);
}

// x->view->y
TEST_F(NVFuserTest, FusionViewNoTranspose_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = flatten(tv0, 1, 2);
  fusion.addOutput(tv1);

  TORCH_CHECK(!hasAtLeastTwoValidGroups(&fusion));
}

TEST_F(NVFuserTest, FusionTransposeSelfMapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  EXPECT_THAT(
      [&]() { IterDomainGraph(fusion_ptr.get()); },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Unsupported domain mapping detected")));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({5, 5}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.transpose(0, 1) + t0;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

#if 0
// silent wrong result
TEST_F(NVFuserTest, FusionTransposeViewSelfMapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = view(tv0, {2, 3}, {3, 2});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.transpose(0, 1) + t0.view({3, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}
#endif

// t0------------.
// t2->broadcast->sub->mul->relu->t6
// t1------------------'
TEST_F(NVFuserTest, FusionScheduleTransposeMissingDim_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigConcreteTensor({1, -1, 1});
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = broadcast(tv2, {true, false, true});
  auto tv4 = sub(tv0, tv3);
  auto tv5 = mul(tv4, tv1);
  auto tv6 = relu(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({512, 1024, 512}, options);
  at::Tensor input1 = at::randn({1, 1024, 1}, options);
  at::Tensor input2 = at::randn({1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1, input2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2}, lparams);
  auto outputs = fe.runFusion({input0, input1, input2}, lparams);

  auto t3 = input2.unsqueeze(0).unsqueeze(-1);
  auto t4 = input0 - t3;
  auto t5 = t4 * input1;
  auto t6 = at::relu(t5);

  testValidate(
      &fusion, outputs, {input0, input1, input2}, {t6}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmall_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1024, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({64 * 1024 * 1024, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 0, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 64 * 1024 * 1024, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(0, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(8);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 4, 7);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1024 * 1024, 2, 2, 2, 2, 2, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(4, 7).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTranspose2DSmallInnerSize_CUDA) {
  std::array<std::vector<int64_t>, 2> shapes{
      std::vector<int64_t>{1024 * 1024 * 128, 2},
      std::vector<int64_t>{2, 1024 * 1024 * 128}};
  for (const auto& shape : shapes) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(2);
    fusion.addInput(tv0);
    auto tv1 = sin(tv0);
    auto tv2 = transpose(tv1, 0, 1);
    auto tv3 = cos(tv2);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::randn(shape, options);

    auto lparams = scheduleTranspose(&fusion, {input});

    FusionExecutor fe;
    fe.compileFusion(&fusion, {input}, lparams);
    auto outputs = fe.runFusion({input}, lparams);

    auto tv_ref = input.sin().transpose(0, 1).cos();

    testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{32, 0};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{0, 32};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32}, DataType::Bool);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{8, 0};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 8);
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  // T1 [TIDx(32), 8, V(4)]

  tv2->setMemoryType(MemoryType::Shared);
  tv2->merge(0);
  tv2->split(0, 4);
  tv2->split(0, 32);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  // T2 [8, TIDx(32), 4]

  tv3->merge(0);
  tv3->split(0, 2);
  tv3->split(0, 32);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  // T3 [16, TIDx(32), 2]

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect1{0, 8};
    std::pair<int, int> expect2{8, 4};
    std::pair<int, int> expect3{2, 0};
    TORCH_CHECK(
        info.second == expect1 || info.second == expect2 ||
        info.second == expect3);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{32, 0};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{32, 0};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 8, 8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{0, 2};
    TORCH_CHECK(info.second == expect);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 8, 8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  // no bank confliction
  TORCH_CHECK(bank_conflict_info.empty());
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
