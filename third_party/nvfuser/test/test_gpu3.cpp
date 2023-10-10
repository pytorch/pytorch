#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <arith.h>
#include <codegen.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_launch_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <grouped_reduction.h>
#include <inlining.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_graphviz.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_expr_evaluator.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>
#include <lower_magic_zero.h>
#include <mutator.h>
#include <ops/all_ops.h>
#include <parser.h>
#include <register_interface.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

TEST_F(NVFuserTest, FusionNonDivisibleSplit1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  // [I]
  tv1->split(0, 5);
  // [ceilDiv(I, 5), 5]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 5), 2, 3]

  auto tv2 = sum(tv0, {0});
  fusion.addOutput(tv2);

  // tv2 shouldn't need to have another predicate
  tv2->split(0, 4);
  tv2->split(1, 2);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv1 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref, ref}, __LINE__, __FILE__);
}

// Repro of issue #1074
TEST_F(NVFuserTest, FusionNonDivisibleSplit2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->reorder({{1, 2}, {2, 1}});
  tv0->computeAt(tv2, 2);

  tv2->split(-1, 3);

  // To make the sanitizer catch the invalid accesses. Not necessary
  // to expose the bug.
  tv1->setMemoryType(MemoryType::Shared);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({13, 17}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Similar to FusionNonDivisibleSplit1 but with unswitch
TEST_F(NVFuserTest, FusionNonDivisibleSplit3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::Unswitch);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Non-divisible split through merge
TEST_F(NVFuserTest, FusionNonDivisibleSplit4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0, 1});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->merge(1, 2);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Nested splits
TEST_F(NVFuserTest, FusionNonDivisibleSplit5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  // [I]
  tv2->split(0, 8);
  // [I/8, 8]
  tv2->split(1, 2);
  // [I/8, 4, 2]
  tv2->split(1, 3); // non-divisible split of outer output
  // [I/8, 2, 3, 2]

  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Vectorized non-divisible split. Must be validated at run time
TEST_F(NVFuserTest, FusionNonDivisibleSplitVectorize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->split(0, 8, false);
  tv1->split(1, 4);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    TORCH_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);

  auto t0_non_divisible = at::randn({8}, options);
  // Since ceilDiv(8, 8) is not divisible by 4, the vectorization is
  // illegal. The run-time validation of vectorization should throw an error.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0_non_divisible}));
}

// If a split is validated at run time, it's not necessary to predicate.
TEST_F(NVFuserTest, FusionNonDivisibleSplitVectorize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv3->split(0, 8, false);
  tv3->split(1, 4);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, {tv1, tv2});

  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    TORCH_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({1024}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1284Repro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {10, 20};
  std::vector<int64_t> input_shape_1 = {15};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  fusion.addInput(in_0);
  fusion.addInput(in_1);

  TensorView* out_0 = add(in_0, IrBuilder::create<Double>(0.f));
  TensorView* out_1 = add(in_1, IrBuilder::create<Double>(2.f));

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  std::vector<IValue> aten_inputs = {at_in_0, at_in_1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t1 = at_in_1 + 2;

  auto runtime = fec.getMostRecentKernelRuntime();
  TORCH_INTERNAL_ASSERT(runtime->isSegmented());
  TORCH_INTERNAL_ASSERT(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion, outputs, {at_in_0, at_in_1}, {at_in_0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1284Repro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {4, 4};
  std::vector<int64_t> input_shape_1 = {3, 4, 4};
  std::vector<int64_t> input_shape_2 = {2, 8, 4, 4};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  TensorView* in_2 = makeSymbolicTensor(input_shape_2.size());

  fusion.addInput(in_0);
  fusion.addInput(in_1);
  fusion.addInput(in_2);

  TensorView* out_0 = add(in_0, in_1);
  TensorView* out_1 = add(in_0, in_2);

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  at::Tensor at_in_2 = at::randn(input_shape_2, options);

  std::vector<IValue> aten_inputs = {at_in_0, at_in_1, at_in_2};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t0 = at_in_0 + at_in_1;
  auto t1 = at_in_0 + at_in_2;

  auto runtime = fec.getMostRecentKernelRuntime();
  TORCH_INTERNAL_ASSERT(runtime->isSegmented());
  TORCH_INTERNAL_ASSERT(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion,
      outputs,
      {at_in_0, at_in_1, at_in_2},
      {t0, t1},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1305Repro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto t0 = makeContigTensor(1);
  auto t1 = makeContigTensor(2);

  fusion.addInput(t0);
  fusion.addInput(t1);

  auto t2 = broadcast(t0, {true, false});
  auto t3 = add(t1, t2);
  auto t4 = add(t3, t2);
  auto t5 = sum(t4, {1});
  auto t6 = broadcast(t5, {false, true});
  auto t7 = add(t3, t6);

  fusion.addOutput(t7);

  t3->computeAt(t7, -1, ComputeAtMode::MostInlined);

  TORCH_INTERNAL_ASSERT(t3->getComputeAtPosition() == 1);
}

TEST_F(NVFuserTest, FusionDoubleBuffering1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDoubleBuffering2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, -1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDoubleBuffering3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // tv2 is invalid to double-buffer as its producer, tv1, is
  // computed inside the double-buffering loop.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv2->doubleBuffer());

  // Moving tv2 inner makes tv1 large enough to double-buffer tv2
  tv2->computeAt(tv3, 2);

  tv2->doubleBuffer();

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unswitch
TEST_F(NVFuserTest, FusionDoubleBuffering4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  tv3->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering gmem to shared and unswitch
TEST_F(NVFuserTest, FusionDoubleBuffering5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 128);
  tv2->split(-1, 32);
  tv2->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv2);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unroll
TEST_F(NVFuserTest, FusionDoubleBuffering6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 16);
  tv3->split(-2, 4);
  tv3->split(-2, 2);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({199}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering and vectorize
TEST_F(NVFuserTest, FusionDoubleBuffering7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv2);

  tv2->split(-1, 128);
  tv2->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->computeAt(tv2, 2);

  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Multiple tensors to double-buffer
TEST_F(NVFuserTest, FusionDoubleBuffering8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 32);
  tv4->split(0, 4);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({100}, options);
  auto t1 = at::randn({100}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Nested double buffering from gmem to smem and smem to register
TEST_F(NVFuserTest, FusionDoubleBuffering9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto out = tv1;
  fusion.addOutput(out);

  auto tv2 = tv0->cacheAfter();
  auto tv3 = tv2->cacheAfter();

  out->split(0, 32);
  out->split(0, 4);
  TransformPropagatorWithCheck propagator(out);
  MaxRootDomainInfoSpanningTree(out).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Shared);

  tv2->computeAt(out, 1);
  tv3->computeAt(out, -1);

  out->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(out);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1001}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// FusionSmemBlockGemmCache + double buffering at both smem and local
TEST_F(NVFuserTest, FusionSmemBlockGemmCacheDoubleBuffer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(2); // (M, K)
  TensorView* tv1 = makeSymbolicTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  TensorView* tv6 = tv5->cacheBefore();

  // For smem double buffering
  auto tv0_cache_local = tv0->cacheAfter();
  auto tv1_cache_local = tv1->cacheAfter();

  // For register double buffering
  auto tv0_cache_smem = tv0->cacheAfter();
  auto tv1_cache_smem = tv1->cacheAfter();

  const int BSX = 32;
  const int TSX = 8;

  // [M, K, N]
  tv6->split(-1, BSX);
  tv6->split(-1, TSX);
  tv6->split(1, BSX);
  tv6->split(0, BSX);
  tv6->split(1, TSX);
  // [M/BSX, BSX/TSX, TSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 5}, {2, 4}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});
  // [M/BSX, N/BSX, K/BSX, BSX/TSX, BSX/TSX, TSX, TSX, BSX]

  auto tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv6_rf->computeAt(tv6, -1);
  tv0_cache_local->computeAt(tv6_rf, -1);
  tv1_cache_local->computeAt(tv6_rf, -1);

  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-3)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  tv0_cache_local->doubleBuffer();
  tv1_cache_local->doubleBuffer();

  tv0_cache_smem->doubleBuffer();
  tv1_cache_smem->doubleBuffer();

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
  // The smem cache write in this test case is redundant predicated,
  //   and also double buffered. Currently we are relying on WAR sync
  //   insertion to ensure ordering of double buffered tensor access.
  // The check below makes sure that the sync is inserted so that the
  //   test isn't running on a race condition.
  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count > 0);
}

TEST_F(NVFuserTest, FusionIntermediateTensorVectorize_CUDA) {
  std::vector<MemoryType> mem_types = {MemoryType::Shared, MemoryType::Local};

  for (auto mem_type : mem_types) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(1);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    tv1->setMemoryType(mem_type);

    tv3->split(-1, 4);
    TransformPropagatorWithCheck propagator(tv3);
    MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

    tv1->computeAt(tv3, -2);

    tv2->axis(-1)->parallelize(ParallelType::Vectorize);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::manual_seed(0);
    auto t0 = at::randn({15}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion);

    // This should throw an exception as the extent of t0 is not
    // divisible by the vector width
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(fe.runFusion({t0}));

    auto t1 = at::randn({16}, options);
    auto cg_outputs = fe.runFusion({t1});

    auto ref = t1;

    testValidate(&fusion, cg_outputs, {t1}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionBroadcastConcretization1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10, 1});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({10, 20});
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor({10, 10});
  fusion.addInput(tv2);

  // Not concretized
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv0, tv4);
  fusion.addOutput(tv5);

  // Concretized
  auto tv6 = sum(tv2, {1});
  auto tv7 = broadcast(tv6, {false, true});
  auto tv8 = add(tv1, tv7);
  fusion.addOutput(tv8);

  for (auto tv : {tv3, tv4, tv5, tv6, tv7, tv8}) {
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  GpuLower gpulw(&fusion);
  TORCH_CHECK(!gpulw.concretizedBroadcastDomains()->isConcretized(
      loweredTv(tv4, gpulw)->axis(1)));
  TORCH_CHECK(gpulw.concretizedBroadcastDomains()->isConcretized(
      loweredTv(tv7, gpulw)->axis(1)));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({10, 1}, options);
  auto t1 = at::randn({10, 20}, options);
  auto t2 = at::randn({10, 10}, options);
  std::vector<IValue> aten_inputs = {t0, t1, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t5 = t0 + t2.sum({1}).unsqueeze(-1);
  auto t8 = t1 + t2.sum({1}).unsqueeze(-1);

  testValidate(&fusion, outputs, aten_inputs, {t5, t8}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastConcretization2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0, 1});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = broadcast(tv2, {false, true});
  fusion.addOutput(tv3);

  // tv1 is thread-predicated with TIDx and TIDy
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  // tv2 broadcasts along TIDx
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  // tv3 broadcasts along TIDy
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);

  // Both tv2 and tv3 broadcast along predicated TID dimensions, but
  // since the broadcast domains are not concretized, there should be
  // no actual parallel broadcast

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({10, 11}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t3 = t0.sum().unsqueeze(-1).unsqueeze(-1);

  testValidate(&fusion, outputs, aten_inputs, {t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastConcretization3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape({10, 4, 8});
  std::vector<int64_t> output_shape({8, 4, 1});

  auto tv0 = makeConcreteTensor(input_shape);
  fusion.addInput(tv0);

  auto tv2 = sum(tv0, {0});
  auto tv3 = set(tv2);
  auto tv4 =
      view(tv3, {input_shape.begin() + 1, input_shape.end()}, output_shape);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  // The view op adds a broadcast domain in tv4, which is
  // parallelized. Howver, it is never materialized, so there should
  // be no parallel broadcast.

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(input_shape, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t5 = at::native::view(t0.sum(0), output_shape) + 1;

  testValidate(&fusion, outputs, aten_inputs, {t5}, __LINE__, __FILE__);
}

// Merging non-broadcast and broadcast domains
// TODO: Fix use case see issue https://github.com/csarofeen/pytorch/issues/1418
// validateParallelize does not pass. Even if it's skipped,
// generated code is invalid as blockBroadcast is not used.
#if 0
TEST_F(NVFuserTest, FusionBroadcastConcretization4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  fusion.addOutput(tv3);

  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->merge(0, 1);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  // TODO: When set to shared memory, this kernel should be correct, but fails
  // validation and when skipped produces incorrect code
  tv2->setMemoryType(MemoryType::Shared);

  tv3->merge(0, 1);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  fusion.printMath();
  fusion.printKernel();
}
#endif

TEST_F(NVFuserTest, FusionBroadcastConcretization5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = makeSymbolicTensor(1);
  fusion.addInput(tv3);

  // Assert tv2 and tv3 have the same shape
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Concretize a broadcast domain to multiple non-concrete domains
  // through a multi-output expression. It should be considered to be
  // non-uniquely concretized.
  auto tv5 = broadcast(tv0, {false, true});
  // Reduce only the non-broadcast domain.
  auto tvs = Welford(tv5, {0});
  auto tv9 = add(tvs.avg, tv1);
  auto tv10 = add(tvs.var_sum, tv2);
  fusion.addOutput(tv9);
  fusion.addOutput(tv10);

  // Same pattern as the above, but concretize the broadcast domain
  // with tv2 and tv3, which have the exactly same shape, so the
  // broadcast should be considered uniquely concretized.
  auto tv11 = broadcast(tv0, {false, true});
  // Reduce only the non-broadcast domain.
  auto tvs2 = Welford(tv11, {0});
  auto tv15 = add(tvs2.avg, tv2);
  auto tv16 = add(tvs2.var_sum, tv3);
  fusion.addOutput(tv15);
  fusion.addOutput(tv16);

  // Reduce only the broadcast domain. Since it's reduced, it should
  // not be considered to be concretized.
  auto tv17 = broadcast(tv0, {false, true});
  auto tvs3 = Welford(tv17, {1});
  fusion.addOutput(tvs3.avg);

  ConcretizedBroadcastDomains bcast_concretization_info(&fusion);

  TORCH_CHECK(
      bcast_concretization_info.maybeNonUniquelyConcretized(tv5->axis(1)),
      "Failed to detect non-unique concretization of ",
      tv5->toString());

  TORCH_CHECK(
      bcast_concretization_info.isUniquelyConcretized(tv11->axis(1)),
      "Failed to detect unique concretization of ",
      tv11->toString());

  TORCH_CHECK(
      !bcast_concretization_info.isConcretized(tv17->axis(1)),
      "Failed to detect non-concretization of ",
      tv17->toString());
}

TEST_F(NVFuserTest, FusionIssue1430_CUDA) {
  // Derived from an expression sorting issue when using loop map, now expr
  // sorting uses parallel map.
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int V = 2, W = 3, X = 4, Y = 5, Z = 6;

  // setup fusion
  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .dtype(DataType::Half)
                 .contiguity(std::vector<bool>(5, true))
                 .shape({V, W, X, Y, Z})
                 .build();

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = castOp(DataType::Float, tv1);

  auto tvs = Welford(tv2, {1, 2, 3, 4});
  auto tv3 = tvs.avg;
  auto tv4 = tvs.var_sum;
  auto tv5 = tvs.n;

  // avg
  auto tv6 = broadcast(tvs.avg, {false, true, true, true, true});

  // var
  auto tv7 = mul(tv4, IrBuilder::create<Double>(1. / (W * X * Y * Z)));
  auto tv8 = add(tv7, IrBuilder::create<Double>(1.e-6));
  auto tv9 = broadcast(tv8, {false, true, true, true, true});
  auto tv10 = rsqrt(tv9);

  auto tv11 = castOp(DataType::Float, tv1);
  auto tv12 = sub(tv11, tv6);
  auto tv13 = mul(tv12, tv10);

  auto tv14 = set(tv13);
  fusion.addOutput(tv14);

  tv3->axis(0)->parallelize(ParallelType::BIDy);
  tv3->axis(2)->parallelize(ParallelType::BIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(4)->parallelize(ParallelType::Vectorize);

  // tv3->reorder({{1, -2}});

  auto rfactor = ir_utils::rfactorHelper(tv3, {1, 4});

  scheduler_utils::parallelizeAllLike(rfactor);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv != tv1 || tv != tv3) {
      for (auto i : c10::irange(tv->nDims())) {
        if (isParallelTypeVectorize(tv->axis(i)->getParallelType())) {
          tv->axis(i)->parallelize(ParallelType::Serial);
        }
      }
    }
  }

  tv0->computeAt(tv14, 1);
  tv13->computeAt(tv14, -2);
  tv2->computeAt(tv14, -1, ComputeAtMode::MostInlined);
  tv11->computeAt(tv14, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({V, W, X, Y, Z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto cg_outputs = fe.runFusion({t0}, LaunchParams(X, V, -1, Y, -1, -1));

  auto t0_double = t0.to(at::kDouble);

  auto at_mu = at::mean(t0_double, {1, 2, 3, 4})
                   .unsqueeze(-1)
                   .unsqueeze(-1)
                   .unsqueeze(-1)
                   .unsqueeze(-1);
  auto at_var = at::var(t0_double, {1, 2, 3, 4}, false)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1);

  auto at_out = t0_double.sub(at_mu).div(at_var.add(1.e-6).sqrt());

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {at_out},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(X, V, -1, Y, -1, -1));
}

// Test code generation of allocated scalars
TEST_F(NVFuserTest, FusionCodegenAllocatedScalars_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Fusion is just a dummy container in this test, just used for
  // getting a Kernel container
  auto tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  GpuLower gpulw(&fusion);
  auto kernel = gpulw.kernel();

  // Set the kernel as the current fusion
  FusionGuard kg(kernel);

  // Create alocated scalars
  auto ks0 = add(kernel->zeroVal(), kernel->oneVal());
  auto ks0_alloc = IrBuilder::create<kir::Allocate>(
      ks0, MemoryType::Local, kernel->oneVal());

  auto ks1 = add(ks0, kernel->oneVal());
  auto ks1_alloc = IrBuilder::create<kir::Allocate>(
      ks1, MemoryType::Local, kernel->oneVal());

  auto tk0 = kernel->inputs()[0]->as<TensorView>();
  auto tki0 = IrBuilder::create<kir::TensorIndex>(tk0, std::vector<Val*>{ks0});
  auto tki1 = IrBuilder::create<kir::TensorIndex>(tk0, std::vector<Val*>{ks1});
  auto tk0_expr = IrBuilder::create<UnaryOp>(UnaryOpType::Set, tki0, tki1);

  // Insert the scalar expression and the allocation of the
  // output directly to the kernel
  auto proxy = kir::KernelInternalProxy(kernel);

  const auto indent = "  ";
  const auto ks0_name = "i" + std::to_string(ks0->name());
  const auto ks1_name = "i" + std::to_string(ks1->name());
  const auto tk0_name = "T" + std::to_string(tk0->name());

  auto& exprs = proxy.topLevelExprs();
  exprs.push_back(tk0_expr);

  // Invalid code gen
  const auto no_alloc_code = codegen::generateCudaKernel(kernel);

  // Without alloc, Int vals are just inlined, resulting in:
  // t0[(0 + 1)] = t0[((0 + 1) + 1)]
  std::stringstream no_alloc_ref;
  no_alloc_ref << "\n"
               << indent << tk0_name << "[(0 + 1)]\n"
               << indent << indent << " = " << tk0_name << "[((0 + 1) + 1)];\n";

  TORCH_CHECK(
      no_alloc_code.find(no_alloc_ref.str()) != std::string::npos,
      "Invalid code generation. Expected:",
      no_alloc_ref.str(),
      "Actual:\n",
      no_alloc_code);

  // Insert proper allocations and definitions
  exprs.insert(std::find(exprs.begin(), exprs.end(), tk0_expr), ks0_alloc);
  exprs.insert(
      std::find(exprs.begin(), exprs.end(), tk0_expr), ks0->definition());
  exprs.insert(std::find(exprs.begin(), exprs.end(), tk0_expr), ks1_alloc);
  exprs.insert(
      std::find(exprs.begin(), exprs.end(), tk0_expr), ks1->definition());

  const auto valid_code = codegen::generateCudaKernel(kernel);

  std::stringstream valid_ref;
  valid_ref << "\n"
            << indent << tk0_name << "[" << ks0_name << "]\n"
            << indent << indent << " = " << tk0_name << "[" << ks1_name
            << "];\n";

  TORCH_CHECK(
      valid_code.find(valid_ref.str()) != std::string::npos,
      "Invalid code generation. Expected:",
      valid_ref.str(),
      "Actual:\n",
      valid_code);
}

TEST_F(NVFuserTest, FusionIndexHoist1_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv1->split(-1, 4);
  tv2->split(-1, 4);
  tv3->merge(0, 1);
  tv3->split(0, 8);
  tv5->merge(0, 1);
  tv5->split(0, 8);
  tv4->computeAt(tv5, -1);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);

  // Use Int32 as the index type to verify Int32 is used as the type
  // of hoisted indices
  GpuLower gpulw(&fusion, DataType::Int32);
  auto kernel = gpulw.kernel();

  auto is_index_times_ns = [](Val* val, Val* index, std::string name) -> bool {
    auto def = dynamic_cast<BinaryOp*>(val->definition());
    if (def == nullptr) {
      return false;
    }
    return def->getBinaryOpType() == BinaryOpType::Mul &&
        def->rhs()->isA<NamedScalar>() &&
        def->rhs()->as<NamedScalar>()->name() == name && def->lhs() == index;
  };

  // Validate indices in the kernel are hoisted as
  // intended. Validation could be also done by just string comparison
  // as the parser test, but updating such tests would be tedious.
  for (auto top_level_loop :
       ir_utils::filterByType<kir::ForLoop>(kernel->topLevelExprs())) {
    auto innermost_loop = top_level_loop;
    while (auto first_expr_loop = dynamic_cast<kir::ForLoop*>(
               innermost_loop->body().exprs().at(0))) {
      innermost_loop = first_expr_loop;
    }
    const auto& exprs = innermost_loop->body().exprs();
    TORCH_CHECK(!exprs.empty(), "No expression found");
    TORCH_CHECK(
        exprs.at(0)->isA<kir::Allocate>(),
        "Invalid expression: ",
        exprs.at(0)->toString());
    auto hoisted_index = exprs.at(0)->as<kir::Allocate>()->buffer();
    TORCH_CHECK(
        hoisted_index->dtype() == DataType::Int32,
        "Invalid data type of hoisted indices. Should be Int32 but: ",
        hoisted_index->dtype());
    kir::Predicate* pred = nullptr;
    for (auto expr : exprs) {
      if (expr->isA<kir::IfThenElse>()) {
        pred = expr->as<kir::IfThenElse>()->predicate();
        auto arith_expr = expr->as<kir::IfThenElse>()->thenBody().exprs().at(0);
        auto out_ti = arith_expr->outputs()[0]->as<kir::TensorIndex>();
        if (out_ti->view()->name() == 1) {
          // Ref: T1[*, hoisted_index] = T0[*, hoisted_index * T0.stride];
          auto t1_index = out_ti->index(1);
          TORCH_CHECK(
              t1_index == hoisted_index,
              "Invalid index: ",
              t1_index->toInlineString());
          // Pred: hoisted_index < T0.size[1]
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 0);
          // hoisted_index * T0.stride[1]
          auto t0_index = in0->index(1);
          TORCH_CHECK(
              is_index_times_ns(t0_index, hoisted_index, "T0.stride[1]"),
              "Invalid index: ",
              t0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 2) {
          // Ref: T3[*, hoisted_index] = T2[*, hoisted_index];
          auto out_index = out_ti->index(1);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 1);
          auto in0_index = in0->index(1);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 3) {
          // Ref: T3[hoisted_index] = T2[hoisted_index];
          auto out_index = out_ti->index(0);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 2);
          auto in0_index = in0->index(0);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 4) {
          // Ref: T4[0] = T3[hoisted_index];
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 3);
          auto in0_index = in0->index(0);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 5) {
          // Ref: T5[hoisted_index] = T4[0]
          auto out_index = out_ti->index(0);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
        }
      }
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({15, 17}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Hoist indices for vectorized tensors
TEST_F(NVFuserTest, FusionIndexHoist2_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv5->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  tv4->split(-1, 3);

  tv0->computeAt(tv5, 1);
  tv1->computeAt(tv5, 1);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({16}, options);
  auto t1 = at::randn({16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTestGridComm_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  int X = 3, Y = 4, Z = 2;
  auto tv0 = makeConcreteTensor({X, Y, Z});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({X, Y, Z});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = add(tv2, tv1);
  auto tv4 = set(tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv4->setMemoryType(MemoryType::Global);

  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::Vectorize);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::BIDy);

  tv4->axis(0)->parallelize(ParallelType::BIDy);
  tv4->axis(1)->parallelize(ParallelType::BIDx);

  tv5->axis(0)->parallelize(ParallelType::BIDy);
  tv5->axis(1)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({X, Y, Z}, options);
  auto t1 = at::randn({X, Y, Z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue https://github.com/csarofeen/pytorch/issues/1497
TEST_F(NVFuserTest, FusionTestGridComm2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t W = 3, X = 4;

  auto tv0 = makeConcreteTensor({X});
  auto tv1 = makeConcreteTensor({W, X});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv3->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({X}, options);
  auto t1 = at::randn({W, X}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1 + 1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Vectorized reset test for double buffered registers
TEST_F(NVFuserTest, FusionDoubleBufferVector_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = sum(tv1, {0});
  auto tv2c = tv2->cacheBefore();

  fusion.addOutput(tv2);

  auto tv1cw = tv1->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter();

  tv1cw->split(-1, 32);
  tv1cr->split(-1, 32);
  tv1cr->split(-1, 4);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1cw->computeAt(tv1cr, 1);
  tv0->computeAt(tv1cw, -1);
  tv2c->split(-1, 32);
  tv2c->split(-1, 4);
  tv1cr->computeAt(tv2c, 2);

  tv1cw->setMemoryType(MemoryType::Shared);
  tv1cr->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({200}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto ref = (t0 + 1).sum({0});

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Request 48KB of data in shared mem,
//  should be large enough not to fit in
//  static allocations, but small enough
//  to fit in supported devices (sm70+).
TEST_F(NVFuserTest, FusionLargeSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  fusion.addOutput(tv2);

  tv2->split(0, 12288);
  tv2->split(1, 128);
  tv1->computeAt(tv2, 1);
  tv1->split(1, 128);
  tv0->computeAt(tv1, -1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({12288 * 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto ref = t0 + 1 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Request a smem allocation that is equal to the device limit
TEST_F(NVFuserTest, FusionTooLargeSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int device_limit = properties->sharedMemPerBlockOptin;

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  fusion.addOutput(tv2);

  // 4 byte per float
  tv2->split(0, device_limit / 4);
  tv2->split(1, 128);
  tv1->computeAt(tv2, 1);
  tv1->split(1, 128);
  tv0->computeAt(tv1, -1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({12288 * 4}, options);
  FusionExecutor fe;

  // First compile gets a compiled kernel
  fe.compileFusion(&fusion, {t0});

  // Should be throwing because the kernel
  //  requested absolute device limit
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}));
}

// Try to test alignment when multiple tensors are
//  in shared mem.
TEST_F(NVFuserTest, FusionSmemAlignment_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({3, 4, 7, 2, 5});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {4});
  auto tv2 = sum(tv1, {3});
  auto tv3 = sum(tv2, {2});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto tv0c = tv0->cacheAfter();
  auto tv1bc = tv1->cacheBefore();
  auto tv2bc = tv2->cacheBefore();
  auto tv3bc = tv3->cacheBefore();
  auto tv4bc = tv4->cacheBefore();

  tv0c->setMemoryType(MemoryType::Shared);
  tv1bc->setMemoryType(MemoryType::Shared);
  tv2bc->setMemoryType(MemoryType::Shared);
  tv3bc->setMemoryType(MemoryType::Shared);
  tv4bc->setMemoryType(MemoryType::Shared);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0->computeAt(tv4, 0);
  tv0->computeAt(tv2, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({3, 4, 7, 2, 5}, options);
  FusionExecutor fe;

  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto tref = t0.sum({1, 2, 3, 4});

  testValidate(&fusion, cg_outputs, {t0}, {tref}, __LINE__, __FILE__);
}

// Repro of #1521
TEST_F(NVFuserTest, FusionImmediateValueAsInput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto immediate_scalr = IrBuilder::create<Double>(0.1);
  // Adding an immediate scalar value as an input is not allowed
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.addInput(immediate_scalr));

  // Instead, use a symbolic value
  auto symbolic_scalar = IrBuilder::create<Double>();
  fusion.addInput(symbolic_scalar);

  auto tv1 = add(tv0, symbolic_scalar);
  fusion.addOutput(tv1);

  // Make sure the kernel is compiled.
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

// Repro of #1506
TEST_F(NVFuserTest, FusionVectorizeContigIndex_CUDA) {
  std::vector<int64_t> shape{14, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->merge(0);

  // Vectorize by 4 should be allowed
  tv2->split(0, 4);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv2, 1);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

// Make sure the same fusion as FusionVectorizeContigIndex fails if
// not contig.
TEST_F(NVFuserTest, FusionVectorizeContigIndexFail_CUDA) {
  std::vector<int64_t> shape{14, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->merge(0);

  tv2->split(0, 4);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv2, 1);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // This should fail at the launch time as 14 is not divisible by the
  // vector word size. The two domains are merged, but they are not
  // contiguous, so contig indexing is not involved in this case.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}));
}

TEST_F(NVFuserTest, FusionVectorizeInputToOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  const int n = 12;
  auto t0 = at::randn({n}, options);
  // Shift by one to make it non-aligned
  auto t0_misaligned = at::randn({n + 1}, options).index({Slice(1)});
  auto t1_misaligned = at::empty({n + 1}, options).index({Slice(1)});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  TORCH_CHECK(t0.equal(cg_outputs[0]));

  // Pass misaligned input. This must fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0_misaligned}));

  // Pass misaligned output. This must fail too.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}, {t1_misaligned}));
}

// Repro of issue #1530
TEST_F(NVFuserTest, FusionVectorizeContigIndexValidationFail_CUDA) {
  std::vector<int64_t> shape{1, 2, 1};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(shape.size());
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(1);
  tv1->merge(0);

  auto invalid_vec_size = shape[0] * shape[1] * shape[2];
  invalid_vec_size *= invalid_vec_size;

  tv1->split(0, invalid_vec_size);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}));
}

TEST_F(NVFuserTest, FusionContigIndexingWithBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({3, 4});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->merge(0);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Local);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4}, options);
  auto t1 = at::randn({3, 4}, options);

  auto t3 = t0.unsqueeze(0).add(t1);
  {
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0, t1});
    auto cg_outputs = fe.runFusion({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, {t3}, __LINE__, __FILE__);
  }

  // Make sure tv2 indexing also works when it's stored in global memory
  tv2->setMemoryType(MemoryType::Global);
  {
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0, t1});
    auto cg_outputs = fe.runFusion({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, {t3}, __LINE__, __FILE__);
  }
}

// Repro of #1534. Validation should detect invalid vectorization.
TEST_F(NVFuserTest, FusionVectorizeContigIndexValidationFail2_CUDA) {
  std::vector<int64_t> shape1{2, 3, 2};
  std::vector<int64_t> shape2{2, 2};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape2);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  tv4->merge(1, 2);
  tv4->merge(0, 1);
  tv4->split(0, 4);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  // Vectorization of tv2 should be detected as invalid.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0, t1}));
}

TEST_F(NVFuserTest, FusionVectorizeContigIndexWithBroadcast_CUDA) {
  std::vector<int64_t> shape1{2, 2, 2};
  std::vector<int64_t> shape2{1, 2, 2};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // [I0, I1, I2]
  auto tv0 = makeContigTensor(shape1.size());
  fusion.addInput(tv0);

  // [B3, I1, I2]
  auto tv1 = makeContigConcreteTensor(shape2);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  tv3->merge(1, 2);
  tv3->merge(0, 1);
  tv3->split(0, 4);

  // Don't modify tv1 so that it's replayed as tv2 with actual
  // transformations. It would create temporary IterDomains, and the
  // validation should still be able to detect vectorization by 4 is valid.
  // TransformPropagatorWithCheck propagator(tv3);
  // MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->merge(1, 2);
  tv2->merge(0, 1);
  tv2->split(0, 4);

  tv2->computeAt(tv3, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeContigIndexPointwiseSchedule_CUDA) {
  std::vector<int64_t> shape0{100, 14, 2, 14};
  std::vector<int64_t> shape1{100, 2, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(shape0.size());
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(shape1.size());
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true, false, false});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);

  auto lparams = schedulePointwise(&fusion, {t0, t1});

  GpuLower gpulw(&fusion);
  auto kernel = gpulw.kernel();

  // The innermost two dimensions are merged and contiguous, so
  // vectorization can be done against 2*14=28 rather than 14, so
  // vector word size should be 4. Broadcasting of tv1 should not
  // matter.
  for (const auto& vec_info : kernel->summary().vectorized_set_info) {
    TORCH_CHECK(
        vec_info.word_size == 4,
        "Invalid vector word size: ",
        vec_info.word_size);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, lparams);
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1.unsqueeze(-3);

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Repro of issue #1539.
TEST_F(NVFuserTest, FusionTrivialReductionForwarding1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false});
  auto tv2 = sum(tv1, {0});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(0, 4);

  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  // All tensors must be transformed to a 2D tensor with each axis
  // mapped with each other in the LOOP map.
  ComputeAtMap ca_map(&fusion);
  for (auto tv : ir_utils::allTvs(&fusion)) {
    TORCH_CHECK(
        tv->nDims() == 2, "Expected to be a 2D tensor but: ", tv->toString());
    for (const auto i : c10::irange(2)) {
      TORCH_CHECK(ca_map.areMapped(
          tv->axis(i), tv3->axis(i), IdMappingMode::PERMISSIVE));
    }
  }
}

TEST_F(NVFuserTest, FusionTrivialReductionForwarding2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false});
  auto tv2 = sum(tv1, {0});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  // Merging a trivial reduction with a non-reduction domain
  tv2->merge(0, 1);
  tv2->split(0, 4);

  tv3->split(0, 4);

  // tv2 and tv3 are different as tv3 lacks the trivial reduction, but
  // they are mapped with each other by BestEffortReplay as the merge
  // of trivial reduciton dim is forwarded.

  PairwiseRootDomainMap root_map(tv2, tv3);

  auto p2c = BestEffortReplay::replayCasP(tv3, tv2, 2, root_map).getReplay();
  for (const auto i : c10::irange(tv2->nDims())) {
    auto tv2_id = tv2->axis(i);
    auto it = p2c.find(tv2_id);
    TORCH_CHECK(
        it != p2c.end(),
        "Expected mapped consumer ID but not found: ",
        tv2_id->toString());
    auto tv3_mapped_id = it->second;
    TORCH_CHECK(
        tv3_mapped_id == tv3->axis(i),
        "Unexpected mapped consumer ID: ",
        tv3_mapped_id->toString());
  }

  auto c2p = BestEffortReplay::replayPasC(tv2, tv3, 2, root_map).getReplay();
  for (const auto i : c10::irange(tv3->nDims())) {
    auto tv3_id = tv3->axis(i);
    auto it = c2p.find(tv3_id);
    TORCH_CHECK(
        it != c2p.end(),
        "Expected mapped producer ID but not found: ",
        tv3_id->toString());
    auto tv2_mapped_id = it->second;
    TORCH_CHECK(
        tv2_mapped_id == tv2->axis(i),
        "Unexpected mapped consumer ID: ",
        tv2_mapped_id->toString());
  }
}

TEST_F(NVFuserTest, FusionTrivialReductionForwarding3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  // Similar pattern as FusionTrivialReductionForwarding2 but trivial
  // reduciton at non-root domain

  // Create a trivial reduction by splitting with a factor of 1
  tv1->split(1, 1, false);
  // Merging with a trivial reduction
  tv1->merge(0, 1);
  auto tv1_merge_out_id = tv1->axis(0);
  tv1->split(0, 5);

  tv2->split(0, 5);

  // The merge of tv1 is done with a non-root trivial
  // reduciton. BestEffortReplay should forward the merge.

  PairwiseRootDomainMap root_map(tv1, tv2);
  auto p2c = BestEffortReplay::replayCasP(tv2, tv1, 2, root_map).getReplay();

  // The two tensors should look like:
  // tv1: [I1*1//5, 5, I2//1]
  // tv2: [I1//5, 5]
  //
  // BestEffortRepaly should forward the merge of (I1 * 1) and create
  // mappings of:
  // I1*1//5 -> I1//5
  // 5 -> 5
  // I1*1 -> I1

  TORCH_CHECK(p2c.size() == 3, "Unexpected number of mappings");
  TORCH_CHECK(p2c.count(tv1->axis(0)) && p2c[tv1->axis(0)] == tv2->axis(0));
  TORCH_CHECK(p2c.count(tv1->axis(1)) && p2c[tv1->axis(1)] == tv2->axis(1));
  TORCH_CHECK(
      p2c.count(tv1_merge_out_id) &&
      p2c[tv1_merge_out_id] == tv2->getRootDomain()[0]);
}

TEST_F(NVFuserTest, FusionTrivialReductionForwarding4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  // tv4 has a trivial reduction axis
  auto tv4 = sum(tv2, {0});
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv3->merge(0, 1);
  tv3->split(0, 32);

  // This causes the trivial reduction of tv4 to be merged with
  // another axis of tv4, and then forward computeAt is done from tv4
  // to tv5. The split of the merged id of tv4 should be done on tv5
  // by forwarding the merge of the trivial reduction.
  tv0->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({111}, options);
  auto t1 = at::randn({123, 111}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto t2 = t0.unsqueeze(0);
  auto t3 = t1 + t2;
  auto t5 = sum(t2, {0}) + 1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {t3, t5}, __LINE__, __FILE__);
}

// See issue #1598
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Place tv2 on shared memory
  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->setMemoryType(MemoryType::Shared);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->split(0, 2);
  tv3->split(-1, 4);
  // swap tidx and tidy
  tv3->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDy);

  tv4->split(0, 2);
  tv4->split(-1, 4);
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 1);
  tv3->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 64}, options);
  auto t1 = at::randn({10, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue #1598
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->setMemoryType(MemoryType::Shared);

  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv4->split(0, 2);
  tv4->split(-1, 4);
  // Also do unroll for tv3 and tv4
  tv4->split(-2, 8, false);
  tv4->axis(-3)->parallelize(ParallelType::Unroll);
  // swap tidx and tidy
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 1);
  tv3->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 64}, options);
  auto t1 = at::randn({10, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue #1599
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Use unroll where a RAW-sync tensor is stored

  tv4->split(0, 2);
  tv4->split(0, 3);
  tv4->split(-1, 4);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 3);
  tv3->computeAt(tv4, -1);

  tv2->split(-1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({50, 64}, options);
  auto t1 = at::randn({50, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See #1618
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({16, 128});
  auto tv1 = makeConcreteTensor({16, 128});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = set(tv2);
  auto tv5 = set(tv3);
  auto tv6 = add(tv4, tv5);
  fusion.addOutput(tv6);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv2->computeAt(tv6, 0);
  tv3->computeAt(tv6, 1);
  tv4->computeAt(tv6, 1);
  tv5->computeAt(tv6, -1);
  tv2->split(1, 64);
  tv3->split(1, 64);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  // Check the block sync is inserted at the correct location.
  //  There is exactly one block sync needed in this test case
  //    and the sync needs to be after the 2 expressions
  //    that modify shared memory.
  class SyncInsertionPointChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(UnaryOp* uop) final {
      // Record number of unary ops that modifies shared memory.
      if (uop->out()->isA<kir::TensorIndex>() &&
          uop->out()->as<kir::TensorIndex>()->view()->getMemoryType() ==
              MemoryType::Shared &&
          // Filter out initialization expressions
          uop->in()->isA<kir::TensorIndex>()) {
        number_of_writes_++;
      }
    }
    void handle(kir::BlockSync* bsync) final {
      // Make sure both shared memory modifying expressions
      //  have been observed at the sync insertion point.
      TORCH_INTERNAL_ASSERT(
          number_of_writes_ == 2,
          "FusionRAWSyncInsertionPlace4 test fail:",
          "only 1 sync after the 2 shared mem writes is needed in this test,"
          "either a redundant sync has been inserted or the block sync is not inserted at the right place");
    }

   private:
    int number_of_writes_ = 0;
  } sync_insertion_checker;
  GpuLower gpulw(&fusion);
  sync_insertion_checker.handle(gpulw.kernel()->topLevelExprs());
}

// Test serial write and parallel read of shared mem: mapped case
TEST_F(NVFuserTest, FusionSerialSmemWriteParallelRead1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({128, 6});
  TensorView* tv1 = makeConcreteTensor({128, 6});
  TensorView* tv2 = makeConcreteTensor({128, 6});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = add(tv3, tv2);

  fusion.addOutput(tv4);

  //  Use shared memory
  tv3->setMemoryType(MemoryType::Shared);

  // Parallelize t4, in this case dim 0 on tv3 will
  //  not be parallelized but dim0 of t4 will be.
  // We will need to make sure a sync is inserted
  //  even if these dimensions are mapped.
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({128, 6}, options);
  at::Tensor t1 = at::randn({128, 6}, options);
  at::Tensor t2 = at::randn({128, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t2});
  auto cg_outputs = fe.runFusion({t0, t1, t2});

  auto ref = t0 + t1 + t2;

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, {ref}, __LINE__, __FILE__);
}

// Test serial write and parallel read of shared mem: un-mapped case
TEST_F(NVFuserTest, FusionSerialSmemWriteParallelRead2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({128, 6});
  TensorView* tv1 = makeConcreteTensor({128, 6});
  TensorView* tv2 = makeConcreteTensor({128, 6});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = add(tv3, tv2);

  fusion.addOutput(tv4);

  //  Use shared memory
  tv3->setMemoryType(MemoryType::Shared);

  // Split and parallelize t4,
  //  the parallelized dimension in t4 will not
  // map across to the shared mem tensor, t3. So
  // there will need to be a sync before use of t3.
  tv4->split(0, 2);
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({128, 6}, options);
  at::Tensor t1 = at::randn({128, 6}, options);
  at::Tensor t2 = at::randn({128, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t2});
  auto cg_outputs = fe.runFusion({t0, t1, t2});

  auto ref = t0 + t1 + t2;

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive
TEST_F(NVFuserTest, FusionSimpleCpAsync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int m = 33, n = 31;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv2, 1);
  tv0_shared->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 1, both block sync and async wait
//  are needed.
TEST_F(NVFuserTest, FusionDoubleBufferCpAsync1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 12);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 2, only async wait is needed
TEST_F(NVFuserTest, FusionDoubleBufferCpAsync2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test for double buffer in shared mem,
//  where we should not insert redundant syncs when
//  they are not needed.
TEST_F(NVFuserTest, FusionDoubleBufferNoSync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.kernel()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  TORCH_INTERNAL_ASSERT(!sync_inserted, "Un-expected block sync inserted");

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test predicate inversion for cp.async
TEST_F(NVFuserTest, FusionCpAsyncPredicate_CUDA) {
  // requires ampere+ GPU

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  auto tv0_reg = tv0_shared->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv1, 1);

  tv0_shared->split(-1, 32);
  tv0_shared->split(-1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);

  FusionExecutor fe;
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.sum({1});

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Test predicate removal on reg-to-reg expressions
TEST_F(NVFuserTest, FusionPredRemovalCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  TensorView* tv4 = set(tv3);

  fusion.addOutput(tv4);
  tv4->split(1, 4);
  tv0->computeAt(tv4, -2);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);

  class PredicateRemovalChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(UnaryOp* uop) final {
      assertOnLocalToLocal(uop);
    }

    // Utility to assert any local-to-local expr is only trivially predicated.
    void assertOnLocalToLocal(Expr* expr) {
      bool is_local = true;
      for (auto in : ir_utils::filterByType<kir::TensorIndex>(expr->inputs())) {
        if (in->view()->getMemoryType() != MemoryType::Local) {
          is_local = false;
        }
      }
      for (auto in :
           ir_utils::filterByType<kir::TensorIndex>(expr->outputs())) {
        if (in->view()->getMemoryType() != MemoryType::Local) {
          is_local = false;
        }
      }

      if (is_local) {
        if (auto ite = dynamic_cast<kir::IfThenElse*>(scope_exprs_.back())) {
          TORCH_INTERNAL_ASSERT(
              ite->predicate()->value()->isConst(),
              "redundant predicate on: ",
              expr);
        }
      }
    }

   private:
    bool within_ite_ = false;
  } pred_checker;

  GpuLower gpulw(&fusion);
  pred_checker.handle(gpulw.kernel()->topLevelExprs());
}

TEST_F(NVFuserTest, FusionPropagateParallelTypesToSiblings_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tvs = Welford(tv0, {0});
  auto tv_avg = tvs.avg;
  fusion.addOutput(tv_avg);

  tv_avg->split(0, 128);
  TransformPropagatorWithCheck propagator(tv_avg);
  MaxRootDomainInfoSpanningTree(tv_avg).traverse(&propagator);

  tv_avg->axis(0)->parallelize(ParallelType::BIDx);
  tv_avg->axis(1)->parallelize(ParallelType::TIDx);

  // Make sure the parallelization of tv_avg is propagated to the var
  // and count tensors.
  GpuLower gpulw(&fusion);
  for (const auto expr : gpulw.kernel()->exprs()) {
    auto wop = dynamic_cast<WelfordOp*>(expr);
    if (wop == nullptr) {
      continue;
    }
    auto ref = wop->outAvg()->as<TensorView>();
    for (auto sibling : ir_utils::filterByType<TensorView>(wop->outputs())) {
      if (ref == sibling) {
        continue;
      }
      TORCH_CHECK(
          ref->nDims() == sibling->nDims(),
          "Invalid sibling: ",
          sibling->toString());
      for (const auto i : c10::irange(ref->nDims())) {
        TORCH_CHECK(
            ref->axis(i)->getParallelType() ==
                sibling->axis(i)->getParallelType(),
            "Mismatched parallel types between siblings. ",
            ref->toString(),
            ", ",
            sibling->toString());
      }
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({9999}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  testValidate(fe.kernel(), outputs, {t0}, {t0.mean({0})}, __LINE__, __FILE__);
}

// Test ExactRootDomainMap
TEST_F(NVFuserTest, FusionExactRootDomainMap_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = transpose(tv2);
  auto tv4 = add(tv2, tv1);
  auto tv5 = add(tv2, tv3);
  auto tv6 = add(tv3, tv1);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  const auto exact_map = ExactRootDomainMap(&fusion);

  // In the exact mapping, the broadcast domain introduced at tv2 is
  // only mapped with the another one in tv3, which is just transposed
  // from tv2. Any other domain, including the second domain of tv4,
  // must not be mapped.

  auto tv2_bc = tv2->axis(1);
  auto tv3_bc = tv3->axis(0);

  TORCH_CHECK(
      exact_map.areMapped(tv2_bc, tv3_bc),
      "Invalid exact root domain map: ",
      exact_map.toString());

  // They must not be mapped with anything else.
  for (auto tv : ir_utils::allTvs(&fusion)) {
    for (auto root_id : tv->getRootDomain()) {
      if (root_id == tv2_bc || root_id == tv3_bc) {
        continue;
      }
      TORCH_CHECK(
          !exact_map.areMapped(root_id, tv2_bc),
          "Invalid exact root domain map: ",
          exact_map.toString());
      TORCH_CHECK(
          !exact_map.areMapped(root_id, tv3_bc),
          "Invalid exact root domain map: ",
          exact_map.toString());
    }
  }
}

class NVFuserMultithreadedTest : public ::testing::Test {
 protected:
  bool was_enabled = false;

  void SetUp() override {
    was_enabled = fuser::cuda::setEnabled(true);
  }

  void TearDown() override {
    fuser::cuda::setEnabled(was_enabled);
  }
};

TEST_F(NVFuserMultithreadedTest, SingleFunction_CUDA) {
  std::string ir = R"IR(
graph(%x.1 : Tensor,
      %y.1 : Tensor):
  %12 : NoneType = prim::Constant()
  %11 : bool = prim::Constant[value=0]()
  %9 : int = prim::Constant[value=1]()
  %3 : Tensor = aten::exp(%x.1)
  %5 : Tensor = aten::relu(%y.1)
  %6 : Tensor = aten::sin(%5)
  %8 : Tensor = aten::add(%3, %6, %9)
  %10 : int[] = prim::ListConstruct(%9)
  %13 : Tensor = aten::sum(%8, %10, %11, %12)
  return (%13)
)IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(ir, g.get());
  GraphFunction fn("nvfuser_test", g, nullptr);

  auto run_kernel = [&fn]() {
    auto x = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    auto y = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    std::vector<IValue> results;
    for (const auto& _ : c10::irange(10)) {
      auto stack = createStack({x.clone(), y.clone()});
      fn.run(stack);
      results.push_back(stack.back());
    }
    for (const auto& i : c10::irange(1, 10)) {
      auto t0 = results[0].toTensor();
      auto ti = results[i].toTensor();
      ASSERT_TRUE(at::allclose(t0, ti));
    }
  };

  constexpr size_t kNumThreads = 4;
  std::vector<std::thread> threads;
  for (size_t id = 0; id < kNumThreads; ++id) {
    threads.emplace_back(run_kernel);
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(NVFuserMultithreadedTest, MultipleFunctions_CUDA) {
  auto run_kernel = []() {
    const std::string ir = R"IR(
  graph(%x.1 : Tensor,
        %y.1 : Tensor):
    %12 : NoneType = prim::Constant()
    %11 : bool = prim::Constant[value=0]()
    %9 : int = prim::Constant[value=1]()
    %3 : Tensor = aten::exp(%x.1)
    %5 : Tensor = aten::relu(%y.1)
    %6 : Tensor = aten::sin(%5)
    %8 : Tensor = aten::add(%3, %6, %9)
    %10 : int[] = prim::ListConstruct(%9)
    %13 : Tensor = aten::sum(%8, %10, %11, %12)
    return (%13)
  )IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(ir, g.get());
    GraphFunction fn("nvfuser_test", g, nullptr);

    auto x = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    auto y = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    std::vector<IValue> results;
    constexpr size_t numRuns = 10;
    for (const auto& _ : c10::irange(numRuns)) {
      auto stack = createStack({x.clone(), y.clone()});
      fn.run(stack);
      results.push_back(stack.back());
    }
    for (const auto& i : c10::irange(1, numRuns)) {
      auto t0 = results[0].toTensor();
      auto ti = results[i].toTensor();
      ASSERT_TRUE(at::allclose(t0, ti));
    }
  };

  constexpr size_t kNumThreads = 4;
  std::vector<std::thread> threads;
  for (size_t id = 0; id < kNumThreads; ++id) {
    threads.emplace_back(run_kernel);
  }
  for (auto& t : threads) {
    t.join();
  }
}

// Repro of issue #1655
TEST_F(NVFuserTest, FusionIncompleteConcreteID_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv0, {true, true, false});
  auto tv4 = broadcast(tv1, {false, true, false});
  auto tv5 = broadcast(tv2, {true, false, false});

  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv3, tv5);

  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  tv6->merge(0);
  tv6->merge(0);

  TransformPropagatorWithCheck propagator(tv6);
  MaxRootDomainInfoSpanningTree(tv6).traverse(&propagator);

  tv0->computeAt(tv6, -1, ComputeAtMode::MostInlined);
  tv1->computeAt(tv6, -1, ComputeAtMode::MostInlined);
  tv2->computeAt(tv7, -1, ComputeAtMode::MostInlined);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST_F(NVFuserTest, FusionTestReEntrantGridWelford_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int X = 256, Y = 7, Z = 2048;

  // setup fusion
  auto tv0 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);

  auto tvs = Welford(tv1, {0, 1, 2});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);

  auto cached_input = tv0->cacheAfter();
  auto cached_avg = tv_avg->cacheBefore();
  auto cached_M2 = tv_M2->cacheBefore();

  auto reduction_tv = scheduler_utils::getReductionTvs(&fusion)[0];

  reduction_tv->merge(0);
  reduction_tv->merge(0);

  int TIDx = 16;
  int vec = 4;

  int TIDy = 16;
  int outer_tidy_fact = 16;

  reduction_tv->split(-1, TIDx * vec);
  reduction_tv->split(-1, vec);
  reduction_tv->axis(-2)->parallelize(ParallelType::TIDx);
  reduction_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  reduction_tv->axis(-3)->parallelize(ParallelType::BIDx);

  reduction_tv->split(0, TIDy);
  reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
  reduction_tv->split(0, outer_tidy_fact);
  reduction_tv->axis(0)->parallelize(ParallelType::BIDy);

  // T2_g[ rblockIdx.y, rS{16}, rthreadIdx.y, iblockIdx.x, ithreadIdx.x24,
  // iV25{4} ]
  reduction_tv->reorder({{3, 0}, {4, 1}, {0, 2}, {2, 3}, {1, 4}, {5, 5}});
  // T2_g[iblockIdx.x, ithreadIdx.x24, rblockIdx.y, rthreadIdx.y, rS{16},
  // iV25{4}]

  TransformPropagatorWithCheck propagator(reduction_tv);
  MaxRootDomainInfoSpanningTree(reduction_tv).traverse(&propagator);
  auto rfactor_tv = ir_utils::rfactorHelper(reduction_tv, {4});
  scheduler_utils::parallelizeAllLike(rfactor_tv);

  tv0->computeAt(tv_avg, 2);
  tv0->computeAt(cached_input, -2);

  cached_input->computeAt(rfactor_tv, 4, ComputeAtMode::BestEffort);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv == cached_input || tv == tv_avg || tv == tv_M2) {
      continue;
    }
    tv->axis(-1)->parallelize(ParallelType::Serial);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, {}, LaunchParams());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({X, Y, Y, Z}, options);

  auto cg_outputs = fe.runFusion({t0}, LaunchParams(-1, -1, -1, -1, -1, -1));

  // by default Welford outputs sum of square diff so need to divide to get var
  cg_outputs[1] = cg_outputs[1].div((float)(X * Y * Y));

  auto at_mu = at::mean(t0.to(at::kDouble), {0, 1, 2});
  auto at_var = at::var(t0.to(at::kDouble), {0, 1, 2}, false);

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {at_mu, at_var},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(-1, -1, -1, -1, -1, -1));
}

// Test sync insertion with redundant predicates
TEST_F(NVFuserTest, FusionRedundantPredSync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);

  fusion.addOutput(tv3);

  auto tv0c = tv0->cacheAfter();

  // Make a redundant write through smem
  tv0c->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv3, 0);
  tv1->computeAt(tv3, 0);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.kernel()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  TORCH_INTERNAL_ASSERT(sync_inserted, "Expected block sync not inserted");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test case for removing syncs on chain of redundant uses.
TEST_F(NVFuserTest, FusionRedundantPredSync2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);

  fusion.addOutput(tv3);

  auto tv0c = tv0->cacheAfter();

  // Make a redundant write through smem
  tv0c->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv3, 0);
  tv1->computeAt(tv3, 0);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  // Utility class to make sure one block sync
  //  is inserted by RAW pass.
  class SyncChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;
    int result() {
      return sync_seen_;
    }

   private:
    void handle(kir::BlockSync*) final {
      sync_seen_++;
    }

   private:
    int sync_seen_ = 0;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.kernel()->topLevelExprs());
  TORCH_INTERNAL_ASSERT(
      checker.result() < 2, "More syncs were inserted than expected");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test case for sync insertion after redundant predicated smem write
//  Check that syncs are removed only when all paths are redundant.
TEST_F(NVFuserTest, FusionRedundantPredSync3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = set(tv2);
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv2, tv1);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  auto tv0c = tv0->cacheAfter();

  // In this scheduling config,
  //  tv0c -> tv2 -> tv3 is a redundant path for tidy
  //  tv0c -> tv2 -> tv5 is not.
  //  So we need a RAW sync in tv0c->tv2 to make sure
  //  tv2 has the correct value to produce tv5.
  tv0c->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  tv5->axis(0)->parallelize(ParallelType::TIDy);
  tv5->axis(1)->parallelize(ParallelType::TIDx);

  // Utility class to make sure one block sync
  //  is inserted by RAW pass.
  class SyncChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;
    int result() {
      return sync_seen_;
    }

   private:
    void handle(kir::BlockSync* sync) final {
      if (!sync->isWarHazardSync()) {
        sync_seen_++;
      }
    }

   private:
    int sync_seen_ = 0;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.kernel()->topLevelExprs());

  // This is implicit checking. There are exactly 2 places
  //  where RAW hazards happen: one producing tv2 and the other
  //  producing tv3. This test case expect syncs in both of
  //  these places so we check that 2 RAW syncs are inserted.
  TORCH_INTERNAL_ASSERT(
      checker.result() == 2,
      "Exactly 2 RAW sync expected for the two shared memory transfers");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref, ref}, __LINE__, __FILE__);
}

// Unit test case for detecting thread redundant usage of shared tensors.
TEST_F(NVFuserTest, FusionRedundantUseCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);

  auto tv5 = set(tv4);

  auto tv6 = set(tv4);
  auto tv7 = set(tv6);

  fusion.addOutput(tv5);
  fusion.addOutput(tv7);

  tv2->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);

  tv7->axis(-1)->parallelize(ParallelType::TIDx);

  // Thread pred map cannot be built without an active lower
  //  object. So would need to lower the whole fusion for
  //  testing. However, lower also keeps an copy of the fusion
  //  so the original pointers cannot be used to querry the
  //  thread pred map. So have to traverse the new expr list
  //  to find the pointers;
  GpuLower gpulw(&fusion);

  TensorView *lowered_tv2 = nullptr, *lowered_tv4 = nullptr;
  auto used_vals = gpulw.kernel()->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->name() == 2) {
      lowered_tv2 = tv;
    }
    if (tv->name() == 4) {
      lowered_tv4 = tv;
    }
  }

  TORCH_INTERNAL_ASSERT(
      lowered_tv2 != nullptr && lowered_tv4 != nullptr,
      "tv2 or tv4 not lowered or mangled");

  auto tv2_info = gpulw.threadPredMap().getPredicateInfo(lowered_tv2);
  auto tv4_info = gpulw.threadPredMap().getPredicateInfo(lowered_tv4);

  // tv2 -> tv3 -> tv4 (shared) is the only use chain for tv2,
  //  and tv4 is redundantly written in tidx so tv2 is redundantly
  //  consumed in tidx.
  TORCH_INTERNAL_ASSERT(
      tv2_info.redundant_use_types.get(ParallelType::TIDx),
      "TV2 is redundantly used but not detected.");

  // tv4->tv5 (global) is a redundant use chain, but
  // tv4->tv6->tv7 is not, so tv4 should not be detected as
  // a redundant used tensor in tidx.
  TORCH_INTERNAL_ASSERT(
      !tv4_info.redundant_use_types.get(ParallelType::TIDx),
      "TV4 is not redundantly used but not detected.");
}

// Test a basic swizzle pattern
TEST_F(NVFuserTest, FusionSimpleSwizzle0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));

  fusion.addOutput(tv2);

  // Make a 2x8 Zshape tile
  tv1->split(-1, 16);
  tv1->split(-1, 8);
  // [O, 2, 8]

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv1->computeAt(tv2, 1);
  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test swizzle inlining
TEST_F(NVFuserTest, FusionSimpleSwizzle1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t3 = t0 + 3.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t3}, __LINE__, __FILE__);
}

// Test sync insertion and memory check in parallelized swizzles.
//  In this test, data is parallel written into smem in zcurve
//   pattern and then read out and output to global mem unswizzled.
TEST_F(NVFuserTest, FusionSimpleSwizzle2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));

  fusion.addOutput(tv2);

  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);

  // Validation should fail since TV1 is not in shared
  //  memory as required by sync info pass.
  ASSERT_ANY_THROW(GpuLower gpulw_throw(&fusion));

  tv1->setMemoryType(MemoryType::Shared);

  // Make sure that a sync is inserted:
  bool sync_found = false;
  GpuLower gpu_lw(&fusion);
  auto flattened_exps =
      ir_utils::flattenScopedExprs(gpu_lw.kernel()->topLevelExprs());

  for (auto expr : flattened_exps) {
    if (expr->isA<kir::BlockSync>()) {
      sync_found = true;
    }
    // Will require a sync thread before any shared memory read.
    for (auto inp_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (inp_tv->getMemoryType() == MemoryType::Shared) {
        TORCH_INTERNAL_ASSERT(
            sync_found, "Block sync required but not inserted");
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test BestEffortReplay behavior with swizzle op
TEST_F(NVFuserTest, FusionSwizzleMapping_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  // Check BestEffortReplay behavior with skip swizzles option on.
  PairwiseRootDomainMap root_map(tv1, tv2);

  // Check producer to consumer map,
  //  i.e. unswizzled tensor to swizzled tensor map
  //----------------------------------------------------------
  auto p2c = BestEffortReplay::replayCasP(tv2, tv1, -1, root_map).getReplay();
  auto swizzle_x_it0 = p2c.find(tv1->axis(-2));
  auto swizzle_y_it0 = p2c.find(tv1->axis(-1));
  // P2C map should exist and both the x and y map should
  //  map to the output of the swizzle op.
  TORCH_INTERNAL_ASSERT(
      swizzle_x_it0 != p2c.end() && swizzle_y_it0 != p2c.end());
  TORCH_INTERNAL_ASSERT(
      swizzle_x_it0->second == tv2->axis(-2) &&
      swizzle_y_it0->second == tv2->axis(-1));

  // Check consumer to producer map,
  //  i.e. swizzled tensor to unswizzled tensor map
  //----------------------------------------------------------
  auto c2p = BestEffortReplay::replayPasC(tv1, tv2, -1, root_map).getReplay();

  auto swizzle_op = tv2->axis(-1)->definition()->as<Swizzle2D>();

  // Find mapping for swizzle inputs
  auto swizzle_x_it1 = c2p.find(swizzle_op->inX());
  auto swizzle_y_it1 = c2p.find(swizzle_op->inY());

  // Find mapping for swizzle outputs
  auto swizzle_x_it2 = c2p.find(swizzle_op->outX());
  auto swizzle_y_it2 = c2p.find(swizzle_op->outY());

  // Input of swizzle ops will not be mapped to any
  //  by BestEffortReplay, as BestEffortReplay has to be
  //  one to one. IdGraph will further map them together.
  TORCH_INTERNAL_ASSERT(
      swizzle_x_it1 == c2p.end() && swizzle_y_it1 == c2p.end());

  // Mapping for swizzle outputs should be mapped and should
  //  also map to the corresponding axes on the unswizzled tensor.
  TORCH_INTERNAL_ASSERT(
      swizzle_x_it2 != c2p.end() && swizzle_y_it2 != c2p.end());
  TORCH_INTERNAL_ASSERT(
      swizzle_x_it2->second == tv1->axis(-2) &&
      swizzle_y_it2->second == tv1->axis(-1));

  // Check id graph behavior
  //----------------------------------------------------------
  ComputeAtMap ca_map(&fusion);
  // Corresponding inputs and outputs of swizzle ops are
  //  map through by exact and permissive map.
  TORCH_INTERNAL_ASSERT(
      ca_map.areMapped(tv1->axis(-2), swizzle_op->inX(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(
      ca_map.areMapped(tv1->axis(-1), swizzle_op->inY(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::EXACT));

  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->inX(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->inY(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::PERMISSIVE));
}

// Test a basic loop swizzle pattern
TEST_F(NVFuserTest, FusionLoopSwizzle0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  tv0->computeAt(tv2, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Outer block zshape pattern
TEST_F(NVFuserTest, FusionLoopSwizzle1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));

  fusion.addOutput(tv2);

  tv2->split(-2, 8);
  tv2->split(-1, 4);
  //[I0o, I0i, I1o, I1i]
  tv2->reorder({{1, 2}, {2, 1}});
  //[I0o, I1o, I0i, I1i]

  tv2->swizzle(Swizzle2DType::ZShape, 0, 1, SwizzleMode::Loop);
  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({45, 77}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test assertion in unsupported pattern: non-leaf loop swizzle.
TEST_F(NVFuserTest, FusionLoopSwizzleCheck0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  // Swizzle the inner tile.
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make swizzle output not a leaf domain.
  tv2->merge(-2);

  tv0->computeAt(tv2, -1);

  FusionExecutor fe;
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

// Test assertion in unsupported pattern: half-inlined loop swizzle.
TEST_F(NVFuserTest, FusionLoopSwizzleCheck1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  //[O, 4, 4]
  tv2->split(-1, 16);
  tv2->split(-1, 4);

  //[O, 4, 4]
  tv3->split(-1, 16);
  tv3->split(-1, 4);

  // Swizzle inner tile of tv2
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make tv2 swizzled and partially-inlined (unsupported).
  tv0->computeAt(tv3, -2);

  FusionExecutor fe;
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionUnsqueeze1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // [I, R]
  auto tv1 = sum(tv0, {1});
  // [I, B]
  auto tv2 = unsqueeze(tv1, -1);
  fusion.addOutput(tv2);

  TORCH_CHECK(
      tv2->nDims() == 2, "Unexpected unsqueeze result: ", tv2->toString());
  TORCH_CHECK(
      tv2->axis(1)->isBroadcast(),
      "Unexpected unsqueeze result: ",
      tv2->toString());

  // tv1 has only one non-reduction axis. An exception should be
  // thrown.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(unsqueeze(tv1, 2));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.sum(1).unsqueeze(-1);

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSqueeze1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // [I, B]
  auto tv1 = sum(tv0, {1}, true);
  // [I]
  auto tv2 = squeeze(tv1, {shape[0], 1});
  fusion.addOutput(tv2);

  TORCH_CHECK(
      tv2->nDims() == 2, "Unexpected squeeze result: ", tv2->toString());

  // [I, R]
  auto tv3 = sum(tv0, {1});
  // tv3 has only one non-reduction axis. The extent of the first axis
  // is not one, so squeeze should fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(squeeze(tv3, {shape[0], 1}));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.sum(1, true).squeeze(-1);

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionContigPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, true, false});
  fusion.addOutput(tv2);

  tv2->merge(-2, -1);
  tv2->merge(-2, -1);
  tv2->split(-1, 100);
  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(PredicatedChecker::isPredicated(tv1, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 4}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.unsqueeze(1);

  testValidate(fe.kernel(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Repro of https://github.com/csarofeen/pytorch/issues/1777
TEST_F(NVFuserTest, FusionDivScalarLhs_CUDA) {
  // tv1 = 2.0 / tv0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = div(IrBuilder::create<Double>(2.0), tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 3}, options);
  // There's no overload div(Scalar, Tensor) in ATen
  auto aten_output = at::div(
      at::native::wrapped_scalar_tensor(at::Scalar(2.0), options.device()), t0);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {aten_output}, __LINE__, __FILE__);
}

// Repro of an issue of the reduction scheduler with a broadcast
// domain concretized to multiple domains that are not proven to have
// the same extent
TEST_F(NVFuserTest, FusionRepro1713_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = broadcast(tv2, {false, true});

  auto tv4 = add(tv3, tv0);

  auto tv5 = add(tv3, tv1);
  auto tv6 = sum(tv5, {0});
  fusion->addOutput(tv4);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 204800}, options);
  // Original repro had the same shape as t0, but this should work
  // with a different extent at the second axis
  at::Tensor t1 = at::randn({1024, 123}, options);
  at::Tensor t2 = at::randn({1024}, options);
  std::vector<IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t3 = t2.unsqueeze(-1);
  auto t4 = t3 + t0;
  auto t5 = t3 + t1;
  auto t6 = sum(t5, {0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1, t2},
      {t4, t6},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionExpand_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto w = 2, x = 3, y = 4, z = 5;

  // Test
  // a simple expand
  // Expand that's propagated
  // expand_as
  // symbolic expand

  // x
  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = expand(tv1, {tv0->axis(0)->extent(), IrBuilder::create<Int>(y)});

  // x
  auto tv3 = makeSymbolicTensor(1);
  fusion->addInput(tv3);
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv4, tv2);
  // [x, e_y]

  // [x, y, z]
  auto tv6 = makeSymbolicTensor(3);
  fusion->addInput(tv6);

  // Disjoint set op will cause a segmentation for just this op.
  auto tmp_7 = set(tv6);
  fusion->addOutput(tmp_7);

  auto tv7 = broadcast(tv5, {false, false, true});

  auto tv8 = expand_as(tv7, tv6);
  // [x, e_y, e_z]

  auto w_symbolic = IrBuilder::create<Int>();
  fusion->addInput(w_symbolic);

  auto tv9 = broadcast(tv8, {true, false, false, false});
  //[1, x, e_y, e_z]

  auto tv10 = expand(
      tv9,
      {w_symbolic,
       tv9->axis(1)->extent(),
       tv9->axis(2)->expandedExtent(),
       tv9->axis(3)->expandedExtent()});

  fusion->addOutput(tv10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t3 = at::randn({x}, options);
  at::Tensor t6 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t6, w});
  auto cg_out = cg_outputs[1];

  TORCH_INTERNAL_ASSERT(cg_out.size(0) == w);
  TORCH_INTERNAL_ASSERT(cg_out.size(1) == x);
  TORCH_INTERNAL_ASSERT(cg_out.size(2) == y);
  TORCH_INTERNAL_ASSERT(cg_out.size(3) == z);
  TORCH_INTERNAL_ASSERT(cg_out.stride(0) == 0);
  TORCH_INTERNAL_ASSERT(cg_out.stride(1) == 1);
  TORCH_INTERNAL_ASSERT(cg_out.stride(2) == 0);
  TORCH_INTERNAL_ASSERT(cg_out.stride(3) == 0);

  auto t10 = t0.unsqueeze(-1)
                 .expand({x, y})
                 .add(t3.unsqueeze(-1))
                 .unsqueeze(-1)
                 .expand_as(t6)
                 .unsqueeze(0)
                 .expand({w, x, y, z});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t3, t6, w},
      {t6, t10},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionExpandIssue1751_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3, y = 4, z = 5;

  // y, z
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});

  // Two ways to propagate extents as is: use -1 or explicitly pass
  // the extent vals.

  auto tv2 = expand(
      tv1,
      {IrBuilder::create<Int>(x),
       IrBuilder::create<Int>(-1),
       IrBuilder::create<Int>(-1)});

  auto tv3 = expand(
      tv1,
      {IrBuilder::create<Int>(x),
       tv0->axis(0)->extent(),
       tv0->axis(1)->extent()});

  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    TORCH_INTERNAL_ASSERT(cg_out.size(0) == x);
    TORCH_INTERNAL_ASSERT(cg_out.size(1) == y);
    TORCH_INTERNAL_ASSERT(cg_out.size(2) == z);
  }

  auto t2 = t0.expand({x, y, z});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t2, t2}, __LINE__, __FILE__);
}

// TODO: Make sure the kernel uses the expanded concrete size instead
// of the symbolic size
TEST_F(NVFuserTest, FusionExpandToConcrete_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3, y = 4;

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false});

  auto tv2 =
      expand(tv1, {IrBuilder::create<Int>(x), IrBuilder::create<Int>(y)});

  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    TORCH_INTERNAL_ASSERT(cg_out.size(0) == x);
    TORCH_INTERNAL_ASSERT(cg_out.size(1) == y);
  }

  auto t2 = t0.expand({x, y});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReproNoncontigBroadcast_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 32, 16, 112, 112}, options).transpose(-1, -2);
  at::Tensor t1 = at::randn({32, 1, 112, 1}, options).transpose(-1, -2);

  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .contiguity({true, true, false, false, false}) // ttfff
                 .shape({-1, -1, -1, -1, -1})
                 .dtype(DataType::Half)
                 .build();
  auto tv1 = TensorViewBuilder()
                 .ndims(4)
                 .contiguity({true, false, false, true}) // tfft
                 .shape({-1, 1, 1, -1})
                 .dtype(DataType::Half)
                 .build();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);

  fusion->addOutput(tv2);

  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t2 = t0 + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

namespace {

// check that the resulting sibling are identical
void checkSiblingConsistency(TensorView* replay, TensorView* target) {
  auto replay_root = replay->getRootDomain();
  auto replay_dom = replay->domain()->domain();
  auto target_root = target->getRootDomain();
  auto target_dom = target->domain()->domain();
  std::unordered_map<IterDomain*, IterDomain*> target2replay_map;
  TORCH_CHECK(replay_root.size() == target_root.size());
  target2replay_map.reserve(replay_root.size());
  std::transform(
      target_root.begin(),
      target_root.end(),
      replay_root.begin(),
      std::inserter(target2replay_map, target2replay_map.begin()),
      [](auto a, auto b) { return std::make_pair(a, b); });
  BestEffortReplay replay_(replay_dom, target_dom, target2replay_map);
  auto r = replay_.getReplay();
  for (int64_t i = 0; i < (int64_t)replay_dom.size(); i++) {
    auto target_id = target_dom[i];
    auto replay_it = r.find(target_id);
    TORCH_CHECK(replay_it != r.end());
    TORCH_CHECK(
        replay_it->second == replay_dom[i],
        "IterDomain mismatch when checking ",
        replay,
        " and ",
        target,
        " at ",
        i,
        ", got ",
        replay_it->second,
        " and ",
        replay_dom[i]);
  }
};

} // namespace

TEST_F(NVFuserTest, FusionTransformPropagateSibling_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1760
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion.addOutput(tvs.var_sum);

  tvs.avg->split(1, 1);
  tvs.avg->split(1, 2);
  tvs.avg->split(1, 3);
  tvs.var_sum->split(1, 1);
  tvs.var_sum->split(1, 2);
  tvs.var_sum->split(1, 3);
  tvs.n->split(1, 1);
  tvs.n->split(1, 2);
  tvs.n->split(1, 3);

  auto var_sum_rf = ir_utils::rfactorHelper(tvs.var_sum, {1, 4});

  TransformPropagatorWithCheck propagator(var_sum_rf);
  MaxRootDomainInfoSpanningTree(var_sum_rf).traverse(&propagator);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  std::vector<TensorView*> siblings[] = {{tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
  for (auto tensors : siblings) {
    for (auto t1 : tensors) {
      for (auto t2 : tensors) {
        TORCH_CHECK(TransformReplay::fullSelfMatching(t1, t2));
      }
    }
  }
}

TEST_F(NVFuserTest, FusionTransformPropagateSelectorSibling_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion.addOutput(tvs.var_sum);

  tvs.avg->split(1, 1);
  tvs.avg->split(1, 2);
  tvs.avg->split(1, 3);
  tvs.var_sum->split(1, 1);
  tvs.var_sum->split(1, 2);
  tvs.var_sum->split(1, 3);
  tvs.n->split(1, 1);
  tvs.n->split(1, 2);
  tvs.n->split(1, 3);

  auto var_sum_rf = ir_utils::rfactorHelper(tvs.var_sum, {1, 4});

  struct DisableTv0 : public MaxInfoSpanningTree::Selector {
    TensorView* tv0;
    virtual bool allowC2P(TensorView* from, TensorView* to) override {
      return from != tv0 && to != tv0;
    };
    virtual bool allowP2C(TensorView* from, TensorView* to) override {
      return from != tv0 && to != tv0;
    };
    virtual bool allowSibling(TensorView* from, TensorView* to) override {
      return true;
    }
    DisableTv0(TensorView* tv0) : tv0(tv0) {}
  } selector1(tv0);

  struct DisableTv0AndSibling : public DisableTv0 {
    virtual bool allowSibling(TensorView* from, TensorView* to) override {
      return false;
    }
    using DisableTv0::DisableTv0;
  } selector2(tv0);

  TransformPropagatorWithCheck propagator(var_sum_rf);
  MaxRootDomainInfoSpanningTree good_path(var_sum_rf, &selector1);
  MaxRootDomainInfoSpanningTree bad_path(var_sum_rf, &selector2);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  auto check = [&]() {
    std::vector<TensorView*> siblings[] = {
        {tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
    for (auto tensors : siblings) {
      for (auto t1 : tensors) {
        for (auto t2 : tensors) {
          TORCH_CHECK(TransformReplay::fullSelfMatching(t1, t2));
        }
      }
    }
  };

  bad_path.traverse(&propagator);
  ASSERT_ANY_THROW(check());
  good_path.traverse(&propagator);
  check();
}

TEST_F(NVFuserTest, FusionTransformPropagatePosition_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeSymbolicTensor(6);
  fusion.addInput(tv0);

  auto tv2 = broadcast(tv0, {false, false, true, false, false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv0->merge(2);
  tv0->merge(0);
  TransformPropagatorWithCheck propagator(tv0);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);

  TORCH_CHECK(tv1->nDims() == 4);
}

TEST_F(NVFuserTest, FusionIgnoreZeroDimReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0});
  // tv1 is effectively a zero-dim tensor as it only has a reduction
  // axis.
  // Reducing it further is converted to just a set op.
  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto tv2_def = dynamic_cast<UnaryOp*>(tv2->definition());
  TORCH_CHECK(
      tv2_def != nullptr,
      "Expected UnaryOp but found ",
      tv2->definition()->toString());

  TORCH_CHECK(
      tv2_def->getUnaryOpType() == UnaryOpType::Set,
      "Expected UnaryOpType::Set but found ",
      tv2_def->getUnaryOpType());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({12345}, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = sum(t0, {0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Repro of issue #1770
TEST_F(NVFuserTest, FusionIssue1770Repro_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion->addInput(tv1);

  auto tv2 = ge(tv0, tv1);
  auto tv3 =
      where(tv2, IrBuilder::create<Double>(1), IrBuilder::create<Double>(2));
  fusion->addOutput(tv3);

  std::vector<int64_t> shape({999});
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = where(t0 >= t1, 1.0, 2.0);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionTransformPropagatorSelector_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);

  auto tv3 = sin(tv2);
  auto tv4 = cos(tv2);

  fusion->addOutput(tv3);
  fusion->addOutput(tv4);

  tv2->split(0, 10);

  struct Selector : public MaxInfoSpanningTree::Selector {
    TensorView* tv0;
    TensorView* tv3;
    virtual bool allowC2P(TensorView* from, TensorView* to) override {
      return to == tv0;
    }
    virtual bool allowP2C(TensorView* from, TensorView* to) override {
      return to == tv3;
    }
    virtual bool allowSibling(TensorView* from, TensorView* to) override {
      return false;
    }
    Selector(TensorView* tv0, TensorView* tv3) : tv0(tv0), tv3(tv3) {}
  } selector(tv0, tv3);

  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2, &selector).traverse(&propagator);

  TORCH_CHECK(tv0->nDims() == 2);
  TORCH_CHECK(tv1->nDims() == 1);
  TORCH_CHECK(tv2->nDims() == 2);
  TORCH_CHECK(tv3->nDims() == 2);
  TORCH_CHECK(tv4->nDims() == 1);
}

TEST_F(NVFuserTest, FusionTransformPropagatorPos_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({22, 105});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);
  fusion->addOutput(tv1);

  tv1->split(0, 2);
  tv1->split(-1, 3);
  tv1->split(-1, 5);

  TransformPropagatorWithCheck propagator(tv1, 2);
  MaxRootDomainInfoSpanningTree(tv1, 2).traverse(&propagator);

  auto expect = makeConcreteTensor({22, 105});
  expect->split(0, 2);
  TORCH_CHECK(TransformReplay::fullSelfMatching(expect, tv0));
}

TEST_F(NVFuserTest, FusionMaxRootDomainInfoSpanningTreePrintTwice_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = neg(tv1);

  fusion->addOutput(tv2);

  tv1->split(0, 10);

  struct Printer : public MaxInfoSpanningTree::Propagator {
    std::stringstream ss;
    virtual void propagateC2P(TensorView* from, TensorView* to) override {
      ss << "propagateC2P" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
    virtual void propagateP2C(TensorView* from, TensorView* to) override {
      ss << "propagateP2C" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
    virtual void propagateSibling(TensorView* from, TensorView* to) override {
      ss << "propagateSibling" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
  } printer1, printer2;
  printer1.ss << std::endl;
  printer2.ss << std::endl;

  MaxRootDomainInfoSpanningTree path(tv1);
  path.traverse(&printer1);
  path.traverse(&printer2);

  auto expect = R"ESCAPE(
propagateC2P
from: 1
to: 0
propagateP2C
from: 1
to: 2
)ESCAPE";
  TORCH_CHECK(printer1.ss.str() == expect);
  TORCH_CHECK(printer2.ss.str() == expect);
}

TEST_F(NVFuserTest, FusionTransformPropagatorNoOverwrite_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = broadcast(tv0, {true, false, true});
  auto tv2 = sin(tv1);
  fusion->addOutput(tv2);

  tv0->split(0, 2);
  tv2->split(1, 2);
  tv2->split(0, 4);

  MaxRootDomainInfoSpanningTree path1(tv2);
  TransformPropagatorWithCheck propagator1(tv2);
  path1.traverse(&propagator1);

  MaxRootDomainInfoSpanningTree path2(tv0);
  TransformPropagatorWithCheck propagator2(tv0);
  path2.traverse(&propagator2);

  TORCH_CHECK(tv1->axis(0)->isBroadcast());
  TORCH_CHECK(tv1->axis(1)->isBroadcast());
  TORCH_CHECK(!tv1->axis(2)->isBroadcast());
  TORCH_CHECK(!tv1->axis(3)->isBroadcast());
  TORCH_CHECK(tv1->axis(4)->isBroadcast());

  auto expect = makeSymbolicTensor(3);
  expect->split(1, 2);
  expect->split(0, 4);
  TORCH_CHECK(TransformReplay::fullSelfMatching(expect, tv1));
}

TEST_F(NVFuserTest, FusionIssue1785Repro_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(1);
  TensorView* tv1 = makeContigTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  // [B, I]
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  auto tv5 = set(tv4);

  // Register your outputs
  fusion.addOutput(tv5);

  tv5->split(0, 8);
  tv5->split(-1, 8);

  // [Serial, TIDy, TIDX, Serial]

  tv4->computeAt(tv5, -2);
  tv3->computeAt(tv4, -1);
  tv2->computeAt(tv3, 0);
  tv2->split(0, 8);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv1->computeAt(tv5, -2);

  tv5->axis(1)->parallelize(ParallelType::TIDy);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor in1 = at::randn({16}, options);
  at::Tensor in2 = at::randn({12, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {in1, in2});
  auto cg_outputs = fe.runFusion({in1, in2});

  auto tv_ref = in1 + in2;

  testValidate(&fusion, cg_outputs, {in1, in2}, {tv_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSkipReplay_CUDA) {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeContigTensor(1);
    TensorView* tv1 = makeContigTensor(2);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = broadcast(tv0, {false, true});
    auto tv3 = add(tv2, tv1);
    fusion.addOutput(tv3);

    tv3->split(1, 2, false);

    TransformPropagatorWithCheck propagator(tv3);
    MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeContigTensor(3);
    fusion.addInput(tv0);

    auto tv1 = sum(tv0, {0, 2});
    auto tv2 = sin(tv1);
    fusion.addOutput(tv2);

    tv0->split(1, 2, false);

    TransformPropagatorWithCheck propagator(tv0);
    MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);
  }
}

TEST_F(NVFuserTest, FusionInlineRepro1803_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tvs = Welford(tv1, {1});
  auto tvo = set(tvs.var_sum);
  fusion.addOutput(tvo);

  tvo->split(0, 16);
  tvo->axis(1)->parallelize(ParallelType::Unroll);

  tv0->computeAt(tvo, -1, ComputeAtMode::BestEffort);

  TORCH_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.avg->getComputeAtPosition());
  TORCH_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.n->getComputeAtPosition());
  TORCH_CHECK(tvs.var_sum->getComputeAtPosition() == 1);
}

// Unit test for the transform selection logic
TEST_F(NVFuserTest, FusionBoundedDirectionSelection1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->split(-1, 5);
  tv3->split(-1, 8);

  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      tv3, -1, {tv0, tv2});

  // Check that the splits are replayed on tv2
  TORCH_INTERNAL_ASSERT(
      tv2->nDims() == tv3->nDims(),
      "Propagator didn't propagate to tv2: ",
      tv2->toString());

  // Check that the splits are replayed on tv1 as well. Even though
  //  one of its consumers, tv2, is part of the boundary, another
  //  consumer is not a boundary, so tv1 should be transformed as well.
  TORCH_INTERNAL_ASSERT(
      tv1->nDims() == tv3->nDims(),
      "Propagator didn't propagate to tv1: ",
      tv1->toString());
}

TEST_F(NVFuserTest, FusionIssueRepro1844_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape = {2, 1, 768};
  std::vector<int64_t> sum_to_shape = {768};
  std::vector<int64_t> sum_to_axes = {0, 1};
  double kProb = 0.5;

  std::vector<Int*> sum_to_symb;
  std::transform(
      sum_to_shape.begin(),
      sum_to_shape.end(),
      std::back_inserter(sum_to_symb),
      [](int s) -> Int* { return IrBuilder::create<Int>(s); });

  TensorView* tv0 = makeContigConcreteTensor(shape);
  TensorView* tv1 = makeContigConcreteTensor(shape);
  TensorView* tv2 = makeContigConcreteTensor(shape, DataType::Bool);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  Double* prob = IrBuilder::create<Double>(kProb);
  auto grad_input = dropout_backward(tv1, tv2, prob);
  auto grad_gelu = gelu_backward(grad_input, tv0);
  auto grad_bias = sum_to(grad_gelu, sum_to_symb);

  fusion->addOutput(grad_gelu);
  fusion->addOutput(grad_bias);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  at::Tensor a = at::randn(shape, options);
  at::Tensor b = at::randn(shape, options);
  at::Tensor c = at::randn(shape, options);
  auto mask = at::gt(c, 0.0f);
  std::vector<IValue> aten_inputs = {a, b, mask};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto dinput = at::native_dropout_backward(b, mask, kProb);
  auto dgelu = at::gelu_backward(dinput, a, "none");
  auto dbias = dgelu.sum(sum_to_axes);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {dgelu, dbias},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionInsertMagicZero1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv2->split(-1, 2);
  tv2->reorder({{1, 2}, {2, 1}});
  tv2->merge(0);

  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 1);

  // The predicate of tv2 should be protected with magic zero
  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      PredicateMagicZeroChecker::isProtected(tv2, gpulw),
      "Failed to protect the predicates of ",
      tv2->toString());
}

TEST_F(NVFuserTest, FusionRepro1860_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);
  std::vector<bool> contiguity{true, false, false};

  std::vector<int64_t> shape{1, -1, -1};
  TensorView* tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  TensorView* tv1 = makeContigConcreteTensor(shape);
  fusion.addInput(tv1);
  TensorView* tv2 = makeContigConcreteTensor(shape);
  fusion.addInput(tv2);

  std::vector<IterDomain*> domain1(3, nullptr);
  for (const auto i : c10::irange(3)) {
    if (i == 0) {
      domain1[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
              .iter_type(IterType::Broadcast)
              .build();
    } else {
      domain1[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
              .expanded_extent(IrBuilder::create<Int>(1 + i))
              .iter_type(IterType::Broadcast)
              .build();
    }
  }

  TensorView* tv22 = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domain1, contiguity), DataType::Float);

  fusion.addInput(tv22);

  auto tv3 = add(tv0, tv1);
  auto tv4 = softmax(tv3, 0);
  auto tv5 = add(tv4, tv22);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({1, 2, 3}, options);
  at::Tensor input2 = at::randn({1, 2, 3}, options);
  at::Tensor input3 = at::randn({1, 2, 3}, options);
  at::Tensor input4 = at::randn({1, 1, 1}, options).expand({1, 2, 3});
  std::vector<IValue> aten_inputs = {input1, input2, input3, input4};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(aten_inputs);
}

TEST_F(NVFuserTest, FusionExpandReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 8});
  fusion->addInput(tv0);

  auto tv1 =
      expand(tv0, {IrBuilder::create<Int>(12), IrBuilder::create<Int>(8)});

  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.expand({12, 8}).sum({0});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Predicate elimination issue repro:
TEST_F(NVFuserTest, FusionExpandReduce2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 4});
  fusion->addInput(tv0);

  auto tv1 =
      expand(tv0, {IrBuilder::create<Int>(3), IrBuilder::create<Int>(4)});

  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  // tv2[r{3}, i{4}]
  tv2->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv2->split(0, NamedScalar::getParallelDim(ParallelType::BIDy), false);
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->split(-1, NamedScalar::getParallelDim(ParallelType::TIDx));
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::BIDx);
  // [rBIDy, rO, rTIDy, iBIDx, iTIDx]
  tv2->reorder({{-2, 0}, {-1, 1}, {2, 2}});
  // [iBIDx, iTIDx, rTIDy, rBIDy, rO]
  auto tv3 = tv2->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1, 4}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0}, LaunchParams(-1, 2, -1, 4, 2, 1));
  auto cg_outputs = fe.runFusion({t0}, LaunchParams(-1, 2, -1, 4, 2, 1));

  auto ref = t0.expand({3, 4}).sum({0});

  testValidate(
      fusion.get(),
      cg_outputs,
      {t0},
      {ref},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(-1, 2, -1, 4, 2, 1));
}

TEST_F(NVFuserTest, FusionExpandBadShapeTest_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);
  std::vector<bool> contiguity{false, false};

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  std::vector<IterDomain*> domains = {
      IterDomainBuilder(
          FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>())
          .build(),
      IterDomainBuilder(
          FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
          .expanded_extent(IrBuilder::create<Int>(10))
          .iter_type(IterType::Broadcast)
          .build()};

  // expand to 10
  TensorView* tv22 = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domains, contiguity), DataType::Float);

  fusion.addInput(tv22);

  auto tv3 = add(tv0, tv22);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // Incompatible shapes
  at::Tensor input1 = at::randn({2, 3}, options);
  // Passing expand size of 5, not 10. Should cause an error
  at::Tensor input4 = at::randn({2, 1}, options).expand({2, 5});

  std::vector<IValue> aten_inputs = {input1, input4};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  ASSERT_ANY_THROW(executor_cache.runFusionWithInputs(aten_inputs));
}

TEST_F(
    NVFuserTest,
    FusionPointwiseScheduleWithBroadcastAndTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv3 = sin(tv2);
  auto tv4 = add(tv3, tv1);
  auto tv5 = sum(tv4, {1});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100, 100, 10}, options);
  at::Tensor t1 = at::randn({10, 20}, options);

  auto aten_output = (t0.view({100, 1, 100, 1, 10, 1}).sin() + t1).squeeze(1);

  std::vector<IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineMost();

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().cos().transpose(1, 2).exp().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineAllAt(tv5, -1, true);

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().cos().transpose(1, 2).exp().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  // broadcasting
  auto tv2 = broadcast(tv1, {false, true, false, true, false, true});
  auto tv3 = relu(tv2);
  // trivial reduction
  auto tv4 = sum(tv3, {1, 3, 5});
  auto tv5 = cos(tv4);
  auto tv6 = transpose(tv5, 1, 2);
  auto tv7 = exp(tv6);
  auto tv8 = tan(tv7);
  fusion.addOutput(tv8);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  TORCH_CHECK(tv8->getComputeAtPosition() == 3);
  TORCH_CHECK(tv7->getComputeAtPosition() == 3);
  TORCH_CHECK(tv6->getComputeAtPosition() == 3);
  TORCH_CHECK(tv5->getComputeAtPosition() == 1);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 3);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().relu().cos().transpose(1, 2).exp().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = exp(tv1);
  auto tv3 = relu(tv2);
  auto tv4 = cos(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  tv3->merge(1);
  inlineMost();

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 1);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().exp().relu().cos().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  // broadcasting
  auto tv2 = broadcast(tv1, {false, true, false, true, false, true});
  auto tv3 = cos(tv2);
  auto tv4 = tan(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 3);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().view({2, 1, 3, 1, 4, 1}).cos().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningBroadcastTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  // broadcasting
  auto tv2 = broadcast(tv1, {false, true, false, true, false, true});
  auto tv3 = tan(tv2);
  // trivial reduction
  auto tv4 = sum(tv3, {1, 3, 5});
  auto tv5 = cos(tv4);
  auto tv6 = exp(tv5);
  fusion.addOutput(tv6);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  TORCH_CHECK(tv6->getComputeAtPosition() == 3);
  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 3);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().tan().cos().exp();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMatchedLeafPosWithoutReplayTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 1, 3, 1, 4, 1});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1, 3, 5});
  auto tv2 = sin(tv1);
  fusion.addOutput(tv1);

  for (auto tv : {tv0, tv1}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv0, tv1, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv1, tv0, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv1, tv2, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv2, tv1, 3) == 3);
}

TEST_F(NVFuserTest, FusionMatchedLeafPosWithoutReplayBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv2 = sin(tv1);
  fusion.addOutput(tv2);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv0, tv1, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv1, tv0, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv1, tv2, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv2, tv1, 3) == 3);
}

TEST_F(NVFuserTest, FusionIdGraphTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv2 = sum(tv1, {1, 3, 5});
  auto tv3 = sin(tv2);
  fusion.addOutput(tv3);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  ComputeAtMap ca_map(&fusion);

  auto all_tvs = ir_utils::allTvs(&fusion);
  for (auto tv1 : all_tvs) {
    for (auto tv2 : all_tvs) {
      if (tv1->isFusionInput() || tv2->isFusionInput()) {
        continue;
      }
      for (int i : c10::irange(3)) {
        auto id1 = tv1->axis(i);
        auto id2 = tv2->axis(i);
        TORCH_CHECK(ca_map.areMapped(id1, id2, IdMappingMode::LOOP));
        TORCH_CHECK(ca_map.areMapped(id1, id2, IdMappingMode::PERMISSIVE));
      }
    }
  }
}

TEST_F(NVFuserTest, FusionPrint_CUDA) {
  auto dtypes = {
      at::kFloat,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kLong,
      at::kBool};
  for (auto dtype : dtypes) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeSymbolicTensor(1, aten_to_data_type(dtype));
    fusion->addInput(tv0);
    auto tv1 = print(tv0);
    auto tv2 = sin(tv1);
    fusion->addOutput(tv2);

    // There is no way to check if anything is printed to the console, but we
    // can validate that when print exist, compilation and computation are not
    // broken.
    auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor t0 = at::arange(2, options).to(dtype);

    FusionExecutorCache executor_cache(std::move(fusion));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {t0},
        {t0.sin()},
        __LINE__,
        __FILE__);
  }
}

TEST_F(NVFuserTest, FusionCheckedSymbolicShape_CUDA) {
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor a = at::randn({123, 456}, options);
  at::Tensor b = at::randn({123, 456}, options);
  at::Tensor c = at::randn({321, 654}, options);

  using return_t =
      std::pair<std::unique_ptr<FusionExecutorCache>, std::vector<at::Tensor>>;
  auto matched_add = [](at::Tensor a, at::Tensor b) -> return_t {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* s1 = IrBuilder::create<Int>();
    Val* s2 = IrBuilder::create<Int>();
    auto builder = TensorViewBuilder().shape(std::vector<Val*>{s1, s2});
    TensorView* tv0 = builder.build();
    TensorView* tv1 = builder.build();

    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);

    fusion->addOutput(tv2);

    auto executor_cache =
        std::make_unique<FusionExecutorCache>(std::move(fusion));
    auto cg_outputs = executor_cache->runFusionWithInputs({a, b});
    return {std::move(executor_cache), std::move(cg_outputs)};
  };

  {
    auto ret1 = matched_add(a, b);
    testValidate(
        ret1.first->fusion(), ret1.second, {a, b}, {a + b}, __LINE__, __FILE__);
  }

  {
    EXPECT_THAT(
        [&]() { matched_add(a, c); },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Attempting to bind")));
  }
}

TEST_F(NVFuserTest, FusionSizeDependentData_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* s1 = IrBuilder::create<Int>();
  auto builder = TensorViewBuilder().shape(std::vector<Val*>{s1});
  TensorView* tv0 = builder.build();

  fusion->addInput(tv0);

  auto tv1 = add(tv0, s1);

  fusion->addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor a = at::zeros({123}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({a});

  testValidate(
      executor_cache.fusion(), cg_outputs, {a}, {a + 123}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDependencyCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(1);
  TensorView* tv3 = makeSymbolicTensor(1);

  auto tv4 = add(tv0, tv1);
  auto tv5 = add(tv0, tv2);
  auto tv6 = add(tv0, tv3);

  auto tv7 = add(tv1, tv2);
  auto tv8 = add(tv1, tv3);

  auto tv9 = add(tv2, tv3);

  {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {tv0, tv1}, {tv4, tv5, tv6, tv7, tv8, tv9});
    std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
    std::vector<Val*> results({tv0, tv1, tv4, tv5, tv6, tv7, tv8});
    for (auto result : results) {
      TORCH_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    TORCH_CHECK(all_vals_set.empty());
  }

  auto tv10 = add(tv6, tv7);
  {
    auto all_vals = DependencyCheck::getAllValsBetween({tv0, tv1}, {tv10});
    std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
    std::vector<Val*> results({tv0, tv1, tv6, tv7, tv10});
    for (auto result : results) {
      TORCH_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    TORCH_CHECK(all_vals_set.empty());
  }
}

// Repro for issue #1925
TEST_F(NVFuserTest, FusionScheduleTransposeRepro1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeConcreteTensor({-1, -1, -1, 1});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1, 1, 333, 1}, options);
  at::Tensor input1 = at::randn({1, 1, 333, 1}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = input0 + input1;

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// Repro for issue #1873
TEST_F(NVFuserTest, FusionInlineBroadcastIndexing0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  tv0->computeAt(tv4, 1);

  tv2->split(-1, 8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({123}, options);
  at::Tensor t1 = at::randn({3, 123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  auto outputs = fe.runFusion({t0, t1});

  auto tv_ref = t0 + t1;

  testValidate(&fusion, outputs, {t0, t1}, {tv_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPredicateUnshare_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1926
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion->addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  for (auto tv : {tv1, tv2}) {
    tv->split(0, 4);
    tv->reorder({{1, -1}});
    tv->split(1, 8);
    tv->merge(0);
    tv->split(0, 1);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::Unswitch);
  }
  tv1->merge(2);
  tv2->reorder({{2, 3}});
  tv2->merge(2);
  for (auto tv : {tv1, tv2}) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  inlineMost();

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 5}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto out = cg_outputs[0];

  testValidate(fusion, {out}, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, AsyncCompilation_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(1)); // Group 0
  TensorView* tv4 =
      max(tv3, {0}); // Group 0 (use max instead to avoid numerical issues)
  TensorView* tv5 = add(tv4, tv1); //  Group 0 (Non Broadcast after reduce,
                                   //  keeps normalization scheduler away)
  TensorView* tv6 = add(tv5, tv2); //  Group 1 (Broadcast after reduce)

  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({8, 5}, options);
  at::Tensor t1 = at::randn({5}, options);
  at::Tensor t2 = at::randn({8, 5}, options);

  auto t3 = t0.add(1.0);
  auto t4 = std::get<0>(at::max(t3, 0));
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<IValue> aten_inputs = {t0, t1, t2};

  executor_cache.compileFusionAsync(aten_inputs);

  while (!executor_cache.isCompiled(aten_inputs)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    printf(".");
  }

  auto outputs = executor_cache.runFusionWithInputs(aten_inputs);

  TORCH_CHECK(
      executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation didn't happen");
  TORCH_CHECK(
      executor_cache.getMostRecentKernelRuntime()
              ->fusionSegments()
              ->groups()
              .size() == 2,
      "segmentation didn't happen as expected");

  testValidate(
      executor_cache.fusion(), outputs, aten_inputs, {t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMergeBroadcastingTrivialReduction1_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({1, 1});
  TensorView* tv1 = makeConcreteTensor({-1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = sum(tv0, {1});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  tv0->merge(0);

  MaxRootDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  inlineMost();

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({10}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  testValidate(
      fusion, {out}, {t0, t1}, {t1 + t0.flatten()}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMergeBroadcastingTrivialReduction2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({-1, 1, 1});
  TensorView* tv1 = makeConcreteTensor({-1, -1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = sum(tv0, {1});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  tv2->merge(1);
  tv2->merge(0);

  MaxRootDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  inlineMost();

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 1, 1}, options);
  at::Tensor t1 = at::randn({10, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  testValidate(
      fusion, {out}, {t0, t1}, {t1 + t0.squeeze(-1)}, __LINE__, __FILE__);
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 1, 1});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1, 1}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({0, 1, 9223372036854775807L});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({0, 1, 9223372036854775807L}, options);

  std::vector<IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = TensorViewBuilder().ndims(0).build();
  auto tv1 = TensorViewBuilder().ndims(0).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);
  at::Tensor t1 = at::randn({}, options);

  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0 + t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

TEST_F(NVFuserTest, FusionEmpty_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({10, 10, 10});
  auto tv1 = makeConcreteTensor({10, 10, 10});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 10, 10}, options);
  at::Tensor t1 = at::randn({10, 10, 10}, options);

  std::vector<IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0, t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

TEST_F(NVFuserTest, FusionMappingRelation_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({1, 1});
  TensorView* tv1 = makeConcreteTensor({-1, 1, 1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false, false});
  auto tv4 = add(tv3, tv1);

  fusion->addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-1);

  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  ComputeAtMap ca_map(fusion);

  // FIXME: This is the concerning part that would motivate some
  //  more formalization on concrete/permissive mapping:
  //   exact mapping should ideally imply permissive mapping.
  auto tv4_inner_node = tv4->axis(0)->definition()->input(1)->as<IterDomain>();
  TORCH_CHECK(
      ca_map.areMapped(tv2->axis(0), tv4_inner_node, IdMappingMode::EXACT));
  TORCH_CHECK(!ca_map.areMapped(
      tv2->axis(0), tv4_inner_node, IdMappingMode::PERMISSIVE));

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({2, 1, 1}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  testValidate(
      fusion, {out}, {t0, t1}, {t1 + t0.squeeze(0)}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInlineAt_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  fusion->addOutput(tv2);

  tv1->inlineAt(-1);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto out = cg_outputs[0];

  testValidate(fusion, {out}, {t0}, {t0.sin().cos()}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialInputForwarding_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({-1, -1});
  TensorView* tv1 = makeConcreteTensor({-1, -1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  // Note: tv2 is not needed. Kept it here since previously there was an
  // assertion from sorting in codegen.
  auto tv2 = add(tv1, IrBuilder::create<Double>(3.141));
  fusion->addOutput(tv0);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 4}, options);
  at::Tensor t1 = at::randn({10, 4}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0, t1});

  testValidate(fusion, cg_outputs, {t0, t1}, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  TORCH_CHECK(fec.isCompiled({t0, t1}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0, t1});
  testValidate(fusion, cg_outputs2, {t0, t1}, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialInputForwarding2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(0);
  fusion->addInput(tv0);
  fusion->addOutput(tv0);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0});

  testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  TORCH_CHECK(fec.isCompiled({t0}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0});
  testValidate(fusion, cg_outputs2, {t0}, {t0}, __LINE__, __FILE__);
}

// Simplified repro of issue #2008
TEST_F(NVFuserTest, FusionReplayTrivialReductionAndBroadcast2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({10, 1, 1});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1, 2});
  auto tv3 = broadcast(tv2, {false, true, true});
  fusion.addOutput(tv3);

  tv0->merge(-2, -1)->merge(-2, -1)->split(0, 4);

  MaxRootDomainInfoSpanningTree tree(tv0);
  TransformPropagator tp(tv0);
  tree.traverse(&tp);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  std::vector<IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, {t0 + 1}, __LINE__, __FILE__);
}

namespace {

size_t getVecSizeForPointwise(FusionExecutorCache& fec) {
  auto most_recent_params =
      fec.getMostRecentKernelRuntime()->getMostRecentExecutorLog().params;
  auto params = dynamic_cast<PointwiseParams*>(most_recent_params.get());
  if (params->vectorize) {
    return params->unroll_factor;
  }
  return 1;
}

} // namespace

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity2D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(2).contiguity({false, true}).build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity3D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(3).contiguity({false, true, true}).build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity5D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int>> sizes_and_vec{
      {9, 17, 1}, {9, 10, 2}, {9, 16, 4}};

  for (auto tup : sizes_and_vec) {
    auto size1 = std::get<0>(tup);
    auto size2 = std::get<1>(tup);
    auto vec = std::get<2>(tup);
    at::Tensor t0 = at::randn({4, size1, 12345, size2, 3}, options)
                        .narrow(1, 0, 8)
                        .narrow(3, 0, 4);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguitySelfOverlapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int>> sizes_strides_and_vec{
      {4, 4, 4, 4},
      {4, 4, 2, 2},
      {4, 2, 4, 2},
      {2, 4, 4, 2},
      {4, 4, 1, 1},
      {4, 1, 4, 1},
      {1, 4, 4, 1},
      {2, 2, 2, 2},
      {2, 2, 1, 1},
      {2, 1, 2, 1},
      {1, 2, 2, 1}};

  for (auto tup : sizes_strides_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto vec = std::get<3>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {stride1, stride2 * 12345, stride2, 3, 1};
    at::Tensor t0 = at::empty_strided(shape, stride, options);
    t0.random_();
    auto cg_outputs = fec.runFusionWithInputs({t0});
    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);
    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionSimpleAmperePipeline_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
    return;
  }

  auto tv0 = makeContigTensor(1);

  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  fusion.addOutput(tv1);

  auto tv_cache = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv_cache->setMemoryType(MemoryType::Shared);

  tv1->split(0, 16);
  tv0->computeAt(tv1, 1);

  tv_cache->circularBuffer(10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({255}, options);

  // Add check that the cp async op has an inlined predicate.
  class InlinedCpAsyncPredChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(kir::IfThenElse* ite) final {
      auto prev_within_ite = within_ite_;
      within_ite_ = true;
      kir::IrVisitor::handle(ite);
      within_ite_ = prev_within_ite;
    }

    void handle(LoadStoreOp* ldst) final {
      if (ldst->opType() == LoadStoreOpType::CpAsync) {
        TORCH_INTERNAL_ASSERT(!within_ite_, "CPASYNC predicate not inlined");
        TORCH_INTERNAL_ASSERT(
            ldst->predicate()->hasValue() &&
                !ldst->predicate()->value()->isConst(),
            "CPASYNC predicate is not generated");
      }
    }

   private:
    bool within_ite_ = false;
  } pred_checker;

  // Check that cp async is inlined:
  GpuLower gpulw(&fusion);
  pred_checker.handle(gpulw.kernel()->topLevelExprs());

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1});
  auto cg_outputs = fe.runFusion({input1});

  testValidate(&fusion, cg_outputs, {input1}, {input1}, __LINE__, __FILE__);
}

// Test file size should be up to 10K LoC. Create a new file for more tests.

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
