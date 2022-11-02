#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/swizzle.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

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

  GpuLower gpulw(&fusion);
  auto exprs = gpulw.kernel()->topLevelExprs();
  auto str = ir_utils::toString(exprs);
  TORCH_CHECK(str.find("where") != string::npos);

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

TEST_F(NVFuserTest, FusionSwizzleVectorize_CUDA) {
  // When there is a swizzle, non of the involved dimensions are contiguous, so
  // unable to vectorize.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4, 4});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->swizzle(Swizzle2DType::XOR, 0, 1);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  ASSERT_ANY_THROW(GpuLower lower(&fusion));
}

TEST_F(NVFuserTest, FusionTransposeBankConflictSwizzle1_CUDA) {
  // Both Xor and CyclicShift swizzling should fully remove bank confliction of
  // a 32x32 non-vectorized transpose.
  std::vector<Swizzle2DType> swizzles{
      Swizzle2DType::XOR, Swizzle2DType::CyclicShift};
  for (auto swizzle_type : swizzles) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeConcreteTensor({32, 32});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = transpose(tv1, 0, 1);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->axis(0)->parallelize(ParallelType::TIDy);
    tv1->axis(1)->parallelize(ParallelType::TIDx);
    tv2->axis(0)->parallelize(ParallelType::TIDy);
    tv2->axis(1)->parallelize(ParallelType::TIDx);
    tv3->axis(0)->parallelize(ParallelType::TIDy);
    tv3->axis(1)->parallelize(ParallelType::TIDx);

    // 32-way bank confliction
    auto bank_conflict_info = fusion.bankConflictInfo();
    TORCH_CHECK(!bank_conflict_info.empty());
    for (auto info : bank_conflict_info) {
      std::pair<int, int> expect{32, 0};
      TORCH_CHECK(
          info.second == expect,
          "Expecting 32-way bank conflict, but got ",
          info.second,
          ". Something in our lowering or bank conflict checker must have changed, ",
          "please update them or this test consistently.");
    }

    // no bank confliction after swizzle
    tv1->swizzle(swizzle_type, 0, 1);
    bank_conflict_info = fusion.bankConflictInfo();
    TORCH_CHECK(
        bank_conflict_info.empty(),
        "Expecting no bank conflict after swizzle, but got ",
        bank_conflict_info.size(),
        "bank conflicting expressions.",
        ". Something in our lowering or bank conflict checker must have changed, ",
        "please update them or this test consistently.");
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflictSwizzle2_CUDA) {
  // ZShape should remove half of the bank confliction of a 32x32 non-vectorized
  // transpose.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  // 32-way bank confliction
  auto bank_conflict_info = fusion.bankConflictInfo();
  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{32, 0};
    TORCH_CHECK(
        info.second == expect,
        "Expecting 32-way bank conflict, but got ",
        info.second,
        ". Something in our lowering or bank conflict checker must have changed, ",
        "please update them or this test consistently.");
  }

  // no bank confliction after swizzle
  tv1->swizzle(Swizzle2DType::ZShape, 0, 1);
  bank_conflict_info = fusion.bankConflictInfo();
  TORCH_CHECK(!bank_conflict_info.empty());
  for (auto info : bank_conflict_info) {
    std::pair<int, int> expect{16, 0};
    TORCH_CHECK(
        info.second == expect,
        "Expecting 16-way bank conflict, but got ",
        info.second,
        ". Something in our lowering or bank conflict checker must have changed, ",
        "please update them or this test consistently.");
  }
}

TEST_F(NVFuserTest, FusionDataSwizzleGlobal_CUDA) {
  // Data swizzle is ignored in global indexing, so we should just throw an
  // error if someone wants to do so.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);
  ASSERT_ANY_THROW(tv1->swizzle(Swizzle2DType::XOR, 0, 1));
}

namespace {

// Get the swizzled tensor from input. For example, for ZShape swizzle, if the
// input is
//    1 2 3
//    4 5 6
//    7 8 9
// Then the output will be:
//    1 2 3
//    6 5 4
//    7 8 9
at::Tensor getSwizzledTensor(
    at::Tensor input,
    Swizzle2DType type,
    bool is_unswizzle = false) {
  auto size_x = input.size(0);
  auto size_y = input.size(1);

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  Int* size_x_input = IrBuilder::create<Int>();
  Int* size_y_input = IrBuilder::create<Int>();
  fusion.addInput(size_x_input);
  fusion.addInput(size_y_input);
  auto x = arange(size_x_input);
  auto xx = broadcast(x, {false, true});
  auto y = arange(size_y_input);
  auto yy = broadcast(y, {true, false});
  std::pair<Val*, Val*> swizzle;
  if (is_unswizzle) {
    swizzle = dispatchUnSwizzle(type, xx, yy, size_x_input, size_y_input);
  } else {
    swizzle = dispatchSwizzle(type, xx, yy, size_x_input, size_y_input);
  }
  fusion.addOutput(swizzle.first);
  fusion.addOutput(swizzle.second);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs({size_x, size_y});

  return input.index_put({outputs[0], outputs[1]}, input);
}

} // namespace

TEST_F(NVFuserTest, FusionSwizzleExampleZShape_CUDA) {
  //    1 2 3      1 2 3
  //    4 5 6  =>  6 5 4
  //    7 8 9      7 8 9
  auto options = at::TensorOptions().dtype(kLong).device(at::kCUDA, 0);
  auto input = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, options);
  auto expect = torch::tensor({{1, 2, 3}, {6, 5, 4}, {7, 8, 9}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::ZShape);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::ZShape, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

TEST_F(NVFuserTest, FusionSwizzleExampleXor_CUDA) {
  //    1   2  3  4       1   2   3  4
  //    5   6  7  8       6   5   8  7
  //    9  10 11 12  =>   11  12  9 10
  //    13 14 15 16       16  15 14 13
  auto options = at::TensorOptions().dtype(kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {6, 5, 8, 7}, {11, 12, 9, 10}, {16, 15, 14, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::XOR);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::XOR, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

TEST_F(NVFuserTest, FusionSwizzleExampleCyclicShift_CUDA) {
  //    1   2  3  4       1   2   3   4
  //    5   6  7  8       8   5   6   7
  //    9  10 11 12  =>   11  12  9  10
  //    13 14 15 16       14  15  16 13
  auto options = at::TensorOptions().dtype(kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {8, 5, 6, 7}, {11, 12, 9, 10}, {14, 15, 16, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::CyclicShift);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::CyclicShift, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

// TODO: FusionSwizzleExampleScatter_CUDA
// I need to read more about ld.matrix before I can add that, maybe the
// following link is a good thing to read:
// https://github.com/NVIDIA/cutlass/blob/master/media/docs/implicit_gemm_convolution.md

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
