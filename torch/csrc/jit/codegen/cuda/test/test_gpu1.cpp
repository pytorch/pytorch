#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/grouped_reduction.h>
#include <torch/csrc/jit/codegen/cuda/inlining.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
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

// A few smoke tests for IrGraphGenerator
// (These tests exercise IrGraphGenerator through a non-trivial IR,
//  to make sure that it runs w/o crashing. The actual output is not
//  validated)
TEST_F(NVFuserTest, FusionIrGraphGenerator_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Make sure we can handle empty IRs
  TORCH_CHECK(!IrGraphGenerator::toGraphviz(
                   &fusion, IrGraphGenerator::DetailLevel::Basic)
                   .empty());

  // Construct an interesting IR
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(3.141));
  TensorView* tv3 = broadcast(tv0, {false, true, false, true});
  TensorView* tv4 =
      reductionOp(BinaryOpType::Add, {2}, IrBuilder::create<Double>(0), tv3);
  TensorView* tv5 = clamp(
      tv4, IrBuilder::create<Double>(0.f), IrBuilder::create<Double>(1.f));
  TensorView* tv6 = add(tv2, tv2);

  // Another checkpoint before adding outputs
  TORCH_CHECK(!IrGraphGenerator::toGraphviz(
                   &fusion, IrGraphGenerator::DetailLevel::Explicit)
                   .empty());

  fusion.addOutput(tv6);

  tv4->axis(2)->parallelize(ParallelType::BIDy);
  tv6->merge(0);
  tv6->split(0, 4);
  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv5->reorder({{-1, 0}});
  tv2->computeAt(tv6, 1);

  // Another checkpoint with more node types
  TORCH_CHECK(!IrGraphGenerator::toGraphviz(
                   &fusion, IrGraphGenerator::DetailLevel::ComputeOnly)
                   .empty());

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  // Final IR graph
  TORCH_CHECK(!IrGraphGenerator::toGraphviz(
                   &fusion, IrGraphGenerator::DetailLevel::Verbose)
                   .empty());
}

TEST_F(NVFuserTest, FusionDispatch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Double* f = IrBuilder::create<Double>(2.f);
  std::stringstream ss1, ss2, ss3;
  ss1 << f;
  ss2 << static_cast<Val*>(f);
  ss3 << static_cast<Statement*>(f);
  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0 && ss1.str().compare(ss3.str()) == 0,
      "Error with dispatch system where results differ by passing Double* vs Val* vs Statement*.");
}

// Evaluate basic scalar operations with constant values
TEST_F(NVFuserTest, FusionExprEvalConstants_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator(&fusion);

  auto* a = IrBuilder::create<Int>(7);
  auto* b = IrBuilder::create<Int>(3);

  // Avoid div operation because it casts int operands to float
  checkIntValue(evaluator, neg(a), -7);
  checkIntValue(evaluator, add(a, b), 10);
  checkIntValue(evaluator, neg(mul(sub(a, b), add(a, b))), -40);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
}

TEST_F(NVFuserTest, FusionExprEvalDouble_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto ten = IrBuilder::create<Double>(10);
  auto two = IrBuilder::create<Double>(2);
  auto three = IrBuilder::create<Double>(3);
  auto val = castOp(DataType::Int, ceilDiv(sub(ten, two), three));
  auto reference = static_cast<int64_t>(std::ceil((10.0 - 2.0) / 3.0));
  TORCH_CHECK(reference == val->evaluateInt());
}

// Evaluate basic scalar operations with bound values
TEST_F(NVFuserTest, FusionExprEvalBindings_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator(&fusion);

  auto* a = IrBuilder::create<Int>();
  auto* b = IrBuilder::create<Int>();
  auto* c = add(a, b);
  auto* d = neg(ceilDiv(c, b));
  auto* e = IrBuilder::create<Int>(0);

  // trying to evaluate before binding should give empty results
  TORCH_CHECK(!evaluator.evaluate(a).has_value());
  TORCH_CHECK(!evaluator.evaluate(d).has_value());

  evaluator.bind(a, 7);
  evaluator.bind(b, 3);

  // can't bind to the results of expressions
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(c, 100));

  // can't bind to concrete values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(e, 100));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, sub(a, b), 4);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset evaluation context
  evaluator = ExpressionEvaluator(&fusion);

  evaluator.bind(a, 2);
  evaluator.bind(b, 5);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, sub(a, b), -3);
  checkIntValue(evaluator, mod(a, b), 2);
  checkIntValue(evaluator, ceilDiv(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

// Evaluate expressions in a simple IR
TEST_F(NVFuserTest, FusionExprEvalBasic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  //
  // IMPORTANT:
  // a. The bindings are only as stable as the Vals are in the fusion graph
  // b. You must use the original (rootDomain) extents
  //  (ex. `tv0->getRootDomain()[0]->extent()`
  //   instead of `tv0->axis(0)->extent()`)
  //
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 6);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 128);
  evaluator.bind(tv1->getRootDomain()[0]->extent(), 6);
  evaluator.bind(tv1->getRootDomain()[1]->extent(), 128);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);
}

// Evaluate expressions in a more complex IR
TEST_F(NVFuserTest, FusionExprEvalComplex_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv4 = add(tv2, tv1);
  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv0, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv5->reorder({{-1, 0}});

  tv6->split(0, 5);
  tv5->merge(0);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 129);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 127);

  // Evaluate and check extent values
  TORCH_CHECK(tv0->domain()->nDims() == 2);
  checkIntValue(evaluator, tv0->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv0->axis(1)->extent(), 127);

  TORCH_CHECK(tv3->domain()->nDims() == 2);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 127);

  TORCH_CHECK(tv4->domain()->nDims() == 2);
  checkIntValue(evaluator, tv4->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv4->axis(1)->extent(), 127);

  TORCH_CHECK(tv5->domain()->nDims() == 1);
  checkIntValue(evaluator, tv5->axis(0)->extent(), 16383);

  TORCH_CHECK(tv6->domain()->nDims() == 3);
  checkIntValue(evaluator, tv6->axis(0)->extent(), 26);
  checkIntValue(evaluator, tv6->axis(1)->extent(), 5);
  checkIntValue(evaluator, tv6->axis(2)->extent(), 127);
}

// Evaluate expressions post lowering
TEST_F(NVFuserTest, FusionExprEvalPostLower_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto* bid_x = add(tv3->axis(0)->extent(), IrBuilder::create<Int>(0));
  auto* tid_x = add(tv3->axis(-1)->extent(), IrBuilder::create<Int>(0));

  // Lower
  GpuLower gpulw(&fusion);

  // 1. Create an evaluation context
  ExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 6);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 128);
  evaluator.bind(tv1->getRootDomain()[0]->extent(), 6);
  evaluator.bind(tv1->getRootDomain()[1]->extent(), 128);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);

  checkIntValue(evaluator, bid_x, 2);
  checkIntValue(evaluator, tid_x, 128);
}

// Kernel IR: Evaluate basic scalar operations with constant values
TEST_F(NVFuserTest, FusionKernelExprEvalConstants_CUDA) {
  Fusion fusion;
  kir::Kernel kernel(&fusion);
  FusionGuard fg((&kernel)->as<Fusion>());

  auto a = IrBuilder::create<Int>(7);
  auto b = IrBuilder::create<Int>(3);
  auto c = IrBuilder::subExpr(a, b);
  auto d = IrBuilder::divExpr(a, b);
  auto e = IrBuilder::mulExpr(c, d);

  kir::ExpressionEvaluator evaluator;

  checkIntValue(evaluator, IrBuilder::negExpr(a), -7);
  checkIntValue(evaluator, IrBuilder::addExpr(a, b), 10);
  checkIntValue(evaluator, IrBuilder::negExpr(e), -8);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 1);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 3);
}

// Kernel IR: Evaluate basic scalar operations with bound values
TEST_F(NVFuserTest, FusionKernelExprEvalBindings_CUDA) {
  Fusion fusion;
  kir::Kernel kernel(&fusion);
  FusionGuard fg((&kernel)->as<Fusion>());

  kir::ExpressionEvaluator evaluator;

  auto a = IrBuilder::create<Int>(c10::nullopt);
  auto b = IrBuilder::create<Int>(c10::nullopt);
  auto c = IrBuilder::addExpr(a, b);
  auto d = IrBuilder::negExpr(IrBuilder::ceilDivExpr(c, b));
  auto e = IrBuilder::create<Int>(0);

  // trying to evaluate before binding should give empty results
  TORCH_CHECK(!evaluator.evaluate(a).has_value());
  TORCH_CHECK(!evaluator.evaluate(d).has_value());

  evaluator.bind(a, 7);
  evaluator.bind(b, 3);

  // can't bind to the results of expressions
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(c, 100));

  // can't bind to concrete values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(e, 100));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, IrBuilder::subExpr(a, b), 4);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 1);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset the evaluation context
  evaluator = kir::ExpressionEvaluator();

  evaluator.bind(a, 2);
  evaluator.bind(b, 5);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, IrBuilder::subExpr(a, b), -3);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 2);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

TEST_F(NVFuserTest, FusionClear_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Create a dummy IR

  {
    TensorView* tv0 = makeSymbolicTensor(2);
    TensorView* tv1 = makeSymbolicTensor(2);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
    TensorView* tv3 = add(tv0, tv2);

    fusion.addOutput(tv3);

    tv3->split(0, 4);
    tv0->computeAt(tv3, 1);
    tv1->computeAt(tv3, 1);

    tv3->axis(0)->parallelize(ParallelType::BIDx);
    tv2->axis(1)->parallelize(ParallelType::Unroll);
    tv3->axis(-1)->parallelize(ParallelType::TIDx);
  }

  // 2. Clear the IR

  fusion.clear();

  TORCH_CHECK(fusion.unordered_exprs().empty());
  TORCH_CHECK(fusion.vals().empty());

  TORCH_CHECK(fusion.inputs().empty());
  TORCH_CHECK(fusion.outputs().empty());

  TORCH_CHECK(ir_utils::getReductionOps(&fusion).empty());

  // 3. Rebuild the IR

  {
    TensorView* tv0 = makeSymbolicTensor(3);
    TensorView* tv1 = makeSymbolicTensor(3);
    TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
    TensorView* tv3 = add(tv0, tv2);

    fusion.addInput(tv0);
    fusion.addInput(tv1);
    fusion.addOutput(tv3);

    // tv3 [i0, i1, i2]
    tv3->reorder({{0, 2}, {2, 0}});
    // tv3 [i2, i1, i0]
    tv3->split(-1, 4);
    // tv3 [i2, i1, i0outer, i0inner{4}]
    tv3->reorder({{2, 0}, {3, 1}, {0, 3}});
    // tv3 [i0outer, i0inner{4}, i1, i2]
    tv0->computeAt(tv3, -1);
    tv1->computeAt(tv3, -1);
    tv3->axis(1)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({16, 8, 8}, options);
  at::Tensor input2 = at::randn_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(outputs[0]));
}

TEST_F(NVFuserTest, FusionCopy_CUDA) {
  Fusion original_fusion;

  // Create the test IR
  {
    FusionGuard fg(&original_fusion);

    auto tv0 = makeSymbolicTensor(3);
    auto tv1 = makeSymbolicTensor(3);
    auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
    auto tv3 = sub(add(tv0, mul(tv2, tv2)), tv2);

    original_fusion.addInput(tv0);
    original_fusion.addInput(tv1);
    original_fusion.addOutput(tv3);

    tv3->reorder({{0, 2}, {2, 0}});
    tv3->split(-1, 4);
    tv3->reorder({{2, 0}, {3, 1}, {0, 3}});

    tv0->computeAt(tv3, -1);
    tv1->computeAt(tv3, -1);

    tv3->axis(0)->parallelize(ParallelType::BIDx);
    tv3->axis(-1)->parallelize(ParallelType::TIDx);
  }

  // Test copy before lowering
  Fusion clone = original_fusion;

  // Compare IR dumps
  std::stringstream original_ir;
  std::stringstream clone_ir;
  original_ir << original_fusion;
  clone_ir << clone;
  ASSERT_EQ(original_ir.str(), clone_ir.str());

  // Lower original fusion
  std::string original_kernel;
  {
    // TODO(kir): remove this guard once we implement the cuda codegen visitor
    FusionGuard fg(&original_fusion);
    original_kernel =
        codegen::generateCudaKernel(GpuLower(&original_fusion).kernel());
  }

  // Make sure the "before lowering" clone was not mutated
  // while lowering the original fusion IR
  std::stringstream before_lowering_ir;
  before_lowering_ir << clone;
  ASSERT_EQ(original_ir.str(), before_lowering_ir.str());

  // Test copy after lowering (including assignment operator)
  Fusion before_lowering = clone;
  clone = original_fusion;

  // Compare IR dumps
  std::stringstream original_lowered_ir;
  std::stringstream clone_lowered_ir;
  original_lowered_ir << original_fusion;
  clone_lowered_ir << clone;
  ASSERT_EQ(original_lowered_ir.str(), clone_lowered_ir.str());

  // Lower the "before lowering" and compare kernels
  std::string clone_kernel;
  {
    // TODO(kir): remove this guard once we implement the cuda codegen visitor
    FusionGuard fg(&before_lowering);
    clone_kernel =
        codegen::generateCudaKernel(GpuLower(&before_lowering).kernel());
  }
  ASSERT_EQ(original_kernel, clone_kernel);
}

TEST_F(NVFuserTest, FusionMove_CUDA) {
  Fusion fusion;

  // Create the test IR
  {
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(3);
    auto tv1 = makeSymbolicTensor(3);
    auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
    auto tv3 = sub(add(tv0, mul(tv2, tv2)), tv2);

    fusion.addInput(tv0);
    fusion.addInput(tv1);
    fusion.addOutput(tv3);

    tv3->reorder({{0, 2}, {2, 0}});
    tv3->split(-1, 4);
    tv3->reorder({{2, 0}, {3, 1}, {0, 3}});

    tv0->computeAt(tv3, -1);
    tv1->computeAt(tv3, -1);

    tv3->axis(0)->parallelize(ParallelType::BIDx);
    tv3->axis(-1)->parallelize(ParallelType::TIDx);
  }

  std::stringstream original_ir;
  original_ir << fusion;

  // Test move before lowering
  Fusion another_fusion = std::move(fusion);

  // Check that the original fusion is "empty"
  //
  // IMPORTANT: these checks assume knowledge of the internal
  //    implementation of the move operations. General uses
  //    should only assume that the moved-from object is in
  //    a valid, but unspecified state. This is similar to the
  //    standard library containers:
  //    https://en.cppreference.com/w/cpp/utility/move
  //
  TORCH_CHECK(fusion.unordered_exprs().empty());
  TORCH_CHECK(fusion.vals().empty());
  TORCH_CHECK(fusion.inputs().empty());
  TORCH_CHECK(fusion.outputs().empty());

  // clear() has no pre-conditions so it's valid to call on a moved-from object
  fusion.clear();

  // Compare IR dumps
  std::stringstream another_ir;
  another_ir << another_fusion;
  ASSERT_EQ(original_ir.str(), another_ir.str());

  // Lower the fusion IR
  GpuLower lower(&another_fusion);

  std::stringstream lowered_ir;
  lowered_ir << another_fusion;

  // Test move assignment after lowering
  fusion = std::move(another_fusion);

  // Compare IR dumps
  std::stringstream moved_lowered_ir;
  moved_lowered_ir << fusion;
  ASSERT_EQ(lowered_ir.str(), moved_lowered_ir.str());
}

TEST_F(NVFuserTest, FusionSimpleArith_CUDA) {
  std::stringstream ss1, ss2;

  Fusion fusion;
  FusionGuard fg(&fusion);

  Double* d1 = IrBuilder::create<Double>(1.f);
  Double* d2 = IrBuilder::create<Double>(2.f);
  Double* d3 = IrBuilder::create<Double>();

  // Disrupt the fusion to make sure guard works well
  {
    Fusion fusion2;
    FusionGuard fg(&fusion2);

    Double* d1 = IrBuilder::create<Double>(1.f);
    Double* d2 = IrBuilder::create<Double>(2.f);
    add(d1, d2);
    ss2 << fusion2;
  }

  IrBuilder::create<BinaryOp>(BinaryOpType::Add, d3, d1, d2);
  ss1 << fusion;

  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0,
      "Error where explicit add nodes don't match implicit add nodes.");
}

TEST_F(NVFuserTest, FusionScalarTypePromote_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Bool* b = IrBuilder::create<Bool>(true);
  Double* d = IrBuilder::create<Double>(4.f);
  Int* i = IrBuilder::create<Int>(3);
  ComplexDouble* c =
      IrBuilder::create<ComplexDouble>(c10::complex<double>(1, 2));

  TORCH_CHECK(add(b, b)->getDataType() == DataType::Bool);
  TORCH_CHECK(add(b, d)->getDataType() == DataType::Double);
  TORCH_CHECK(add(b, i)->getDataType() == DataType::Int);
  TORCH_CHECK(add(b, c)->getDataType() == DataType::ComplexDouble);

  TORCH_CHECK(add(d, b)->getDataType() == DataType::Double);
  TORCH_CHECK(add(d, d)->getDataType() == DataType::Double);
  TORCH_CHECK(add(d, i)->getDataType() == DataType::Double);
  TORCH_CHECK(add(d, c)->getDataType() == DataType::ComplexDouble);

  TORCH_CHECK(add(i, b)->getDataType() == DataType::Int);
  TORCH_CHECK(add(i, d)->getDataType() == DataType::Double);
  TORCH_CHECK(add(i, i)->getDataType() == DataType::Int);
  TORCH_CHECK(add(i, c)->getDataType() == DataType::ComplexDouble);

  TORCH_CHECK(add(c, b)->getDataType() == DataType::ComplexDouble);
  TORCH_CHECK(add(c, d)->getDataType() == DataType::ComplexDouble);
  TORCH_CHECK(add(c, i)->getDataType() == DataType::ComplexDouble);
  TORCH_CHECK(add(c, c)->getDataType() == DataType::ComplexDouble);
}

TEST_F(NVFuserTest, FusionComplexAbsTypes_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto tensor_cf = at::randn({4, 4, 4}, options.dtype(at::kComplexFloat));
  auto tensor_cd = at::randn({4, 4, 4}, options.dtype(at::kComplexDouble));

  auto type_cf = TensorType::create(tensor_cf);
  auto tv_cf = IrBuilder::create<TensorView>(type_cf);
  auto type_cd = TensorType::create(tensor_cd);
  auto tv_cd = IrBuilder::create<TensorView>(type_cd);

  TORCH_CHECK(
      tensor_cf.abs().scalar_type() ==
      data_type_to_aten(abs(tv_cf)->getDataType().value()));
  TORCH_CHECK(
      tensor_cd.abs().scalar_type() ==
      data_type_to_aten(abs(tv_cd)->getDataType().value()));
}

TEST_F(NVFuserTest, FusionRegister_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Double* v1 = IrBuilder::create<Double>(1.f);
  Double* v2 = IrBuilder::create<Double>(2.f);
  Val* v3 = binaryOp(BinaryOpType::Add, v1, v2);
  Val* v4 = binaryOp(BinaryOpType::Add, v1, v2);
  TORCH_CHECK(v1->name() + 1 == v2->name());
  TORCH_CHECK(v2->name() + 1 == v3->name());
  TORCH_CHECK(v3->name() + 1 == v4->name());
  TORCH_CHECK(v3->definition()->name() + 1 == v4->definition()->name());
}

// dummy expr with 2 outputs only for toposort test.
struct DummyExpr : public Expr {
  ~DummyExpr() = default;
  DummyExpr(
      IrBuilderPasskey passkey,
      Val* _outlhs,
      Val* _outrhs,
      Val* _lhs,
      Val* _rhs)
      : Expr(passkey, ExprType::UnaryOp) // Not terribly safe...
  {
    addOutput(_outlhs);
    addOutput(_outrhs);
    addInput(_lhs);
    addInput(_rhs);
  }
  DummyExpr(const DummyExpr& other) = delete;
  DummyExpr& operator=(const DummyExpr& other) = delete;
  DummyExpr(DummyExpr&& other) = delete;
  DummyExpr& operator=(DummyExpr&& other) = delete;
  Expr* shallowCopy() const override {
    return nullptr;
  }
};

TEST_F(NVFuserTest, FusionTopoSort_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // e0: v3, v2 = dummy(v1, v0)
  // e1: v4     =   add(v3, v2)
  // e2: v5     =   add(v2, v4)
  // e3: v6     =   add(v5, v5)
  Double* v0 = IrBuilder::create<Double>();
  Double* v1 = IrBuilder::create<Double>();
  Double* v2 = IrBuilder::create<Double>();
  Double* v3 = IrBuilder::create<Double>();
  Double* v4 = IrBuilder::create<Double>();
  Double* v5 = IrBuilder::create<Double>();
  Double* v6 = IrBuilder::create<Double>();

  std::vector<Val*> inputs = {v0, v1};
  for (auto val : inputs) {
    fusion.addInput(val);
  }

  Expr* e0 = IrBuilder::create<DummyExpr>(v3, v2, v1, v0);
  Expr* e1 = IrBuilder::create<BinaryOp>(BinaryOpType::Add, v4, v3, v2);
  Expr* e2 = IrBuilder::create<BinaryOp>(BinaryOpType::Add, v5, v2, v4);
  Expr* e3 = IrBuilder::create<BinaryOp>(BinaryOpType::Add, v6, v5, v5);

  fusion.addOutput(v2);
  fusion.addOutput(v3);
  auto exprs = fusion.exprs();
  TORCH_CHECK(exprs.size() == 1, "Found ", exprs.size(), " but expecting 1");
  TORCH_CHECK(exprs[0] == e0);

  fusion.addOutput(v5);
  exprs = fusion.exprs();
  TORCH_CHECK(exprs.size() == 3, "Found ", exprs.size(), " but expecting 3");
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v4);
  exprs = fusion.exprs();
  TORCH_CHECK(exprs.size() == 3, "Found ", exprs.size(), " but expecting 3");
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v6);
  exprs = fusion.exprs();
  TORCH_CHECK(exprs.size() == 4, "Found ", exprs.size(), " but expecting 4");
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  TORCH_CHECK(v2->definition()->name() == 0);
  TORCH_CHECK(v3->definition()->name() == 0);
  TORCH_CHECK(v4->definition()->name() == 1);
  TORCH_CHECK(v5->definition()->name() == 2);
  TORCH_CHECK(v6->definition()->name() == 3);
}

TEST_F(NVFuserTest, FusionTensor_CUDA) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  {
    auto tensor = at::randn({2, 3, 4, 5}, options);
    auto tensor_type = TensorType::create(tensor);
    auto fuser_tensor = IrBuilder::create<TensorView>(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (const auto i : c10::irange(fuser_tensor->nDims())) {
      // size 1 dimension are makred as broadcast
      TORCH_CHECK(
          fuser_tensor->axis(i)->isBroadcast() == (tensor.sizes()[i] == 1));
      // check contiguity information;
      TORCH_CHECK(fuser_tensor->domain()->contiguity()[i]);
    }
  }

  // TensorType::create fills stride_properties, which helps us to mark
  // IterDomain properly
  // Note: implementation could change, depending on how much we want to invest
  // in our home-brew contiguity coalescing. For now let's make sure that we
  // properly test what we are using.
  {
    auto tensor = at::randn({4, 4, 4}, options);
    auto sliced_tensor = tensor.slice(1, 0, -1, 2);

    auto tensor_type = TensorType::create(sliced_tensor);
    auto fuser_tensor = IrBuilder::create<TensorView>(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (const auto i : c10::irange(fuser_tensor->nDims())) {
      // size 1 dimension are makred as broadcast
      TORCH_CHECK(fuser_tensor->axis(i)->isBroadcast() == false);
    }
    TORCH_CHECK(fuser_tensor->domain()->contiguity()[0]);
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[1]);
    TORCH_CHECK(fuser_tensor->domain()->contiguity()[2]);
  }

  {
    auto tensor = at::randn({2, 3, 4, 5}, options);
    auto permuted_tensor = tensor.permute({0, 3, 1, 2});
    auto tensor_type = TensorType::create(permuted_tensor);
    auto fuser_tensor = IrBuilder::create<TensorView>(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (const auto i : c10::irange(fuser_tensor->nDims())) {
      // size 1 dimension are makred as broadcast
      TORCH_CHECK(fuser_tensor->axis(i)->isBroadcast() == false);
    }
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[0]);
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[1]);
    TORCH_CHECK(fuser_tensor->domain()->contiguity()[2]);
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[3]);
  }
}

TEST_F(NVFuserTest, FusionFilterVals_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(1);
  auto scalar0 = IrBuilder::create<Double>(0);
  auto scalar1 = IrBuilder::create<Int>(0);
  auto scalar2 = IrBuilder::create<Int>(1);

  const std::vector<Val*> vals = {tv0, scalar0, tv1, scalar1, scalar2};

  std::vector<TensorView*> tvs(
      ir_utils::filterByType<TensorView>(vals).begin(),
      ir_utils::filterByType<TensorView>(vals).end());
  TORCH_CHECK(tvs.size() == 2);
  TORCH_CHECK(tvs[0] == tv0);
  TORCH_CHECK(tvs[1] == tv1);

  std::vector<Double*> floats(
      ir_utils::filterByType<Double>(vals).begin(),
      ir_utils::filterByType<Double>(vals).end());
  TORCH_CHECK(floats.size() == 1);
  TORCH_CHECK(floats[0] == scalar0);

  std::vector<Int*> ints(
      ir_utils::filterByType<Int>(vals).begin(),
      ir_utils::filterByType<Int>(vals).end());
  TORCH_CHECK(ints.size() == 2);
  TORCH_CHECK(ints[0] == scalar1);
  TORCH_CHECK(ints[1] == scalar2);

  TORCH_CHECK(
      ir_utils::filterByType<Expr>(vals).begin() ==
          ir_utils::filterByType<Expr>(vals).end(),
      "Not expecting any results");
}

TEST_F(NVFuserTest, FusionTVSplit_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeSymbolicTensor(3);

  tv = tv->split(2, 2);
  TORCH_CHECK(tv->nDims() == 4);
  Expr* outer = tv->axis(2)->extent()->definition();

  TORCH_CHECK(
      outer->getExprType().value() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(outer)->getBinaryOpType() ==
          BinaryOpType::CeilDiv &&
      static_cast<BinaryOp*>(outer)->lhs()->sameAs(
          tv->getRootDomain()[2]->extent()) &&
      static_cast<Int*>(static_cast<BinaryOp*>(outer)->rhs())
          ->sameAs(IrBuilder::create<Int>(2)));

  IterDomain* inner = static_cast<IterDomain*>(tv->axis(3));
  TORCH_CHECK(
      inner->extent()->isScalar() &&
      static_cast<Int*>(inner->extent())->isConst() &&
      static_cast<Int*>(inner->extent())->value().value() == 2);
}

TEST_F(NVFuserTest, FusionTVMerge_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeSymbolicTensor(3);

  tv = tv->merge(1);
  Expr* axisOp = tv->axis(1)->extent()->definition();

  TORCH_CHECK(
      tv->nDims() == 2 && axisOp->getExprType() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(axisOp)->getBinaryOpType() == BinaryOpType::Mul &&
      static_cast<BinaryOp*>(axisOp)->lhs() ==
          tv->getRootDomain()[1]->extent() &&
      static_cast<BinaryOp*>(axisOp)->rhs() ==
          tv->getRootDomain()[2]->extent());
}

TEST_F(NVFuserTest, FusionTVReorder_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::unordered_map<int, int> shift_right{{-1, 0}};

  std::unordered_map<int, int> shift_left{{0, -1}};

  std::unordered_map<int, int> shift_left_2{{0, -1}, {1, 0}, {2, 1}};

  std::unordered_map<int, int> swap{{0, 2}, {2, 0}};

  auto tv = makeSymbolicTensor(3);
  std::vector<IterDomain*> ref;
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_left);
  for (const auto i : c10::irange(tv->nDims())) {
    TORCH_CHECK(ref[i]->sameAs(tv->axis(i - 1)));
  }

  tv = makeSymbolicTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_left);
  for (const auto i : c10::irange(tv->nDims())) {
    TORCH_CHECK(ref[i]->sameAs(tv->axis(i - 1)));
  }

  tv = makeSymbolicTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_right);
  TORCH_CHECK(ref[ref.size() - 1]->sameAs(tv->axis(0)));
  for (const auto i : c10::irange(1, tv->nDims())) {
    TORCH_CHECK(ref[i - 1]->sameAs(tv->axis(i)));
  }

  tv = makeSymbolicTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());
  tv->reorder(swap);
  TORCH_CHECK(ref[0]->sameAs(tv->axis(2)));
  TORCH_CHECK(ref[2]->sameAs(tv->axis(0)));
  TORCH_CHECK(ref[1]->sameAs(tv->axis(1)));
}

TEST_F(NVFuserTest, FusionEquality_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Double* fval1 = IrBuilder::create<Double>();
  Double* fval1_copy = fval1;
  Double* fval2 = IrBuilder::create<Double>();
  Double* fone = IrBuilder::create<Double>(1.0);

  TORCH_CHECK(fval1->sameAs(fval1_copy));
  TORCH_CHECK(!fval1->sameAs(fval2));
  TORCH_CHECK(!fone->sameAs(fval1));
  TORCH_CHECK(fone->sameAs(IrBuilder::create<Double>(1.0)));

  Int* ival1 = IrBuilder::create<Int>();
  Int* ival1_copy = ival1;
  Int* ival2 = IrBuilder::create<Int>();
  Int* ione = IrBuilder::create<Int>(1);

  TORCH_CHECK(ival1->sameAs(ival1_copy));
  TORCH_CHECK(!ival1->sameAs(ival2));
  TORCH_CHECK(!ione->sameAs(ival1));
  TORCH_CHECK(ione->sameAs(IrBuilder::create<Int>(1)));

  BinaryOp* add1 = IrBuilder::create<BinaryOp>(
      BinaryOpType::Add, IrBuilder::create<Double>(), fval1, ival1);
  BinaryOp* add1_copy = IrBuilder::create<BinaryOp>(
      BinaryOpType::Add, IrBuilder::create<Double>(), fval1, ival1);
  BinaryOp* sub1 = IrBuilder::create<BinaryOp>(
      BinaryOpType::Sub, IrBuilder::create<Double>(), fval1, ival1);

  UnaryOp* neg1 = IrBuilder::create<UnaryOp>(
      UnaryOpType::Neg, IrBuilder::create<Double>(), fval1);
  UnaryOp* neg2 = IrBuilder::create<UnaryOp>(
      UnaryOpType::Neg, IrBuilder::create<Double>(), fval2);
  UnaryOp* neg1_copy = IrBuilder::create<UnaryOp>(
      UnaryOpType::Neg, IrBuilder::create<Double>(), fval1);

  TORCH_CHECK(add1->sameAs(add1_copy));
  TORCH_CHECK(!add1->sameAs(sub1));

  TORCH_CHECK(neg1->sameAs(neg1_copy));
  TORCH_CHECK(!static_cast<Expr*>(neg1)->sameAs(add1));
  TORCH_CHECK(!neg1->sameAs(neg2));
}

TEST_F(NVFuserTest, FusionDependency_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Double* d0 = IrBuilder::create<Double>(0.f);
  Double* d1 = IrBuilder::create<Double>(1.f);
  auto d2 = add(d0, d1);

  auto d3 = add(d2, d2);

  Double* d4 = IrBuilder::create<Double>(4.f);
  Double* d5 = IrBuilder::create<Double>(5.f);
  auto d6 = add(d4, d5);

  Double* d7 = IrBuilder::create<Double>(7.f);
  Double* d8 = IrBuilder::create<Double>(8.f);
  auto d9 = add(d7, d8);

  auto d10 = add(d6, d9);

  auto d11 = add(d3, d10);

  TORCH_CHECK(DependencyCheck::isDependencyOf(d0, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d1, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d2, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d3, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d6, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d9, d11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d0, d2));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d2, d3));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d4, d6));
  TORCH_CHECK(DependencyCheck::isDependencyOf(d8, d10));

  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d1));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d3));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d11, d5));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d2, d0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d3, d2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d6, d4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(d10, d8));

  auto dep_chain = DependencyCheck::getSingleDependencyChain(d0, d11);
  TORCH_CHECK(dep_chain.back() == d11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == d3);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == d2);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(d6, d11);
  TORCH_CHECK(dep_chain.back() == d11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == d10);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(d4, d11);
  TORCH_CHECK(dep_chain.back() == d11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == d10);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == d6);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(d11, d2);
  TORCH_CHECK(dep_chain.empty());
}

TEST_F(NVFuserTest, FusionParser_CUDA) {
  // This test may not pass if using a custom block sync as there may
  // be additional calls. Skip the test as it's not specifically
  // relevant with block synchronizatin.
  if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
    return;
  }
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
    graph(%0 : Float(2, strides=[1]),
          %1 : Float(2, strides=[1])):
      %c0 : Float(2, strides=[1]) = aten::mul(%0, %1)
      %d0 : Float(2, strides=[1]) = aten::mul(%c0, %0)
      return (%d0))IR";
  parseIR(graph0_string, g.get());

  // strides are not yet supported in the irparser.
  for (auto val : g->block()->inputs()) {
    if (val->isCompleteTensor())
      val->setType(val->type()->castRaw<TensorType>()->contiguous());
  }
  for (auto node : g->block()->nodes()) {
    for (auto val : node->outputs()) {
      if (val->isCompleteTensor())
        val->setType(val->type()->castRaw<TensorType>()->contiguous());
    }
  }

  auto fusion = parseJitIR(g);
  FusionGuard fg(fusion.get());
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // Avoid vectorization here as those kernels can't be lowered twice at the
  // moment
  at::Tensor input1 = at::randn({16}, options);
  at::Tensor input2 = at::randn({16}, options);
  auto lparams = schedulePointwise(fusion.get(), {input1, input2});

  // CONSIDER:
  // 1. this can be moved to a dedicated "golden" file
  // 2. use a fuzzy compare (ignore non-significant whitespaces for example)
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 1> T0, Tensor<float, 1> T1, Tensor<float, 1> T3) {
  int64_t i50;
  i50 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i50 < T0.size[0])) {
    float T5[1];
    T5[0] = 0;
    T5[0]
       = T1[i50];
    float T4[1];
    T4[0] = 0;
    T4[0]
       = T0[i50];
    float T2[1];
    T2[0]
      = T4[0]
      * T5[0];
    float T6[1];
    T6[0]
      = T2[0]
      * T4[0];
    T3[i50]
       = T6[0];
  }
}
)";

  const std::string actual_kernel =
      "\n" + codegen::generateCudaKernel(GpuLower(fusion.get()).kernel());
  if (expected_kernel.size() != actual_kernel.size() ||
      expected_kernel.compare(actual_kernel) != 0) {
    std::cerr
        << " Codegen mismatch, codegen possibly changed, or is incorrect. "
        << " \n ========= EXPECTED ========= \n"
        << expected_kernel << "\n========= ACTUAL ========== \n"
        << actual_kernel << "\n=================" << std::endl;
    auto it = std::mismatch(
        expected_kernel.begin(),
        expected_kernel.end(),
        actual_kernel.begin(),
        actual_kernel.end());
    std::string actual_mismatched_snippet(it.second, actual_kernel.end());
    actual_mismatched_snippet = actual_mismatched_snippet.substr(0, 10);
    std::string expected_mismatched_snippet(it.first, expected_kernel.end());
    expected_mismatched_snippet = expected_mismatched_snippet.substr(0, 10);
    std::cerr << "First mismatch found at: " << actual_mismatched_snippet
              << ", expected: " << expected_mismatched_snippet << std::endl;
    TORCH_CHECK(false);
  }

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1, input2}, lparams);
  auto outputs = fe.runFusion({input1, input2}, lparams);
  at::Tensor output_ref = input1 * input2 * input1;
  TORCH_CHECK(output_ref.equal(outputs[0]));
}

TEST_F(NVFuserTest, FusionOuterSplit_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(3);

  IrBuilder::create<BinaryOp>(
      BinaryOpType::Add,
      tv0,
      IrBuilder::create<Double>(0.0),
      IrBuilder::create<Double>(1.0));
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(3.0));
  fusion.addOutput(tv2);

  //[I0, I1, I2]
  tv2->split(-1, 4, false);
  //[I0, I1, I2o{4}, I2i]
  tv2->merge(0);
  tv2->merge(0);
  //[I0*I1*I2o{4}, I2i]
  tv2->split(0, 2);
  //[I0*I1*I2o{4}o, I0*I1*I2o{4}i{2}, I2i]
  tv2->reorder({{0, 1}, {1, 0}});
  // I0*I1*I2o{4}i{2}, [I0*I1*I2o{4}o, I2i]

  tv0->computeAt(tv2, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor output = at::empty({2, 6, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({}, {output});

  at::Tensor output_ref = at::zeros_like(output, options);
  output_ref = output_ref + 0.0 + 1.0 + 2.0 + 3.0;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionCodeGen_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(3);

  IrBuilder::create<BinaryOp>(
      BinaryOpType::Add,
      tv0,
      IrBuilder::create<Double>(0.0),
      IrBuilder::create<Double>(1.0));
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(3.0));
  fusion.addOutput(tv2);

  //[I0, I1, I2]
  tv2 = tv2->split(0, 4);
  //[I0o, I0i{4}, I1, I2]
  tv2 = tv2->merge(1);
  //[I0o, I0i{4}*I1, I2]
  tv2 = tv2->split(-1, 2);
  //[I0o, I0i{4}*I1, I2o, I2i{2}]
  tv2 = tv2->reorder({{0, 1}, {1, 0}, {3, 2}});
  //[I0i{4}*I1, I0o, I2i{2}, I2o]

  tv0->computeAt(tv2, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor output = at::empty({16, 8, 8}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({}, {output});

  at::Tensor output_ref = at::zeros_like(output, options);
  output_ref = output_ref + 0.0 + 1.0 + 2.0 + 3.0;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionCodeGen2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(3);
  TensorView* tv1 = makeSymbolicTensor(3);
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv3);

  //[I0, I1, I2]
  tv3->reorder({{0, 2}, {2, 0}});
  //[I2, I1, I0]
  tv3->split(-1, 4);
  //[I2, I1, I0o, I0i{4}]
  tv3->reorder({{2, 0}, {3, 1}, {0, 3}});
  // I0o, I0i{4}, I1, I2]

  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({16, 8, 8}, options);
  at::Tensor input2 = at::randn_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(outputs[0]));
}

TEST_F(NVFuserTest, FusionSimplePWise_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);
  tv3->merge(0);

  // Split by n_threads
  tv3->split(0, 128);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::Unroll);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 2, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionSimplePWiseDtypeComplex_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims, DataType::ComplexFloat);
  TensorView* tv1 = makeContigTensor(nDims, DataType::ComplexFloat);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  c10::complex<double> scalar1(2.0, 3.0);
  TensorView* tv2 = add(tv1, IrBuilder::create<ComplexDouble>(scalar1));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);
  tv3->merge(0);

  // Split by n_threads
  tv3->split(0, 128);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::Unroll);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(at::kComplexFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 2, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + scalar1;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionExecKernel_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(0, 128);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::ones({1, 128}, options);
  at::Tensor input2 = at::ones_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor check = at::full({1, 128}, 4, options);
  ;
  TORCH_CHECK(outputs[0].equal(check));
}

int ceilDiv_(int a, int b) {
  return (a + b - 1) / b;
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt1_CUDA) {
  // Case 1
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 + 3
  // tv4 = tv1 * 2
  // tv5 = tv3 + tv2
  // tv6 = tv5 + tv4
  // tv7 = tv1 + tv4
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3.0));
  TensorView* tv4 = mul(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv5 = add(tv3, tv2);

  TensorView* tv6 = add(tv5, tv4);
  TensorView* tv7 = add(tv1, tv4);

  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  // Lets setup to actually run
  tv7->merge(0);
  tv7->split(0, 128);
  tv7->split(0, 4);

  tv7->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv7, 1);

  ComputeAtMap ca_map(&fusion);

  // The this-position of the last tensor should be zero.
  TORCH_CHECK(
      tv7->nDims() == 3 && tv7->getComputeAtPosition() == 0 &&
      tv7->getMaxProducerPosition() == 1);
  TORCH_CHECK(
      tv7->nDims() == 3 && tv6->getComputeAtPosition() == 0 &&
      tv6->getMaxProducerPosition() == 1);
  // The position of every other tensor should be 1.
  for (auto tv : {tv1, tv2, tv3, tv4, tv5}) {
    TORCH_CHECK(tv->nDims() == 3 && tv->getComputeAtPosition() == 1);

    TORCH_CHECK(
        ca_map.areMapped(tv7->axis(0), tv->axis(0), IdMappingMode::PERMISSIVE));
  }

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({129, 127}, options);

  auto t1 = aten_input.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t1.add({3.0});
  auto t4 = t1.mul({2.0});
  auto t5 = t3.add(t2);
  auto t6 = t5.add(t4);
  auto t7 = t1.add(t4);

  std::vector<at::Tensor> aten_outputs = {t6, t7};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options), at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt2_CUDA) {
  // Case 2
  // tv1 = tv0 * -1
  // tv2 = tv0 + 3
  // tv3 = tv0 * 2
  // tv4 = tv2 + tv1
  // tv5 = tv4 + tv3
  // tv6 = tv5 + tv3
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv4 = add(tv2, tv1);

  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv5, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  // Lets setup to actually run
  tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv6, 1);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({129, 127}, options);

  auto t1 = input.mul({-1.0});
  auto t2 = input.add({3.0});
  auto t3 = input.mul({2.0});
  auto t4 = t2.add(t1);
  auto t5 = t4.add(t3);
  auto t6 = t5.add(t3);

  std::vector<at::Tensor> aten_outputs = {t5, t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt3_CUDA) {
  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(.979361));
  TensorView* tv3 = mul(tv2, tv0);

  fusion.addOutput(tv3);

  // Lets setup to actually run
  while (tv3->nDims() > 1)
    tv3->merge(0);
  tv3->split(0, 128);
  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t1.mul({0.979361});
  auto aten_output = t2.mul(t0);

  std::vector<IValue> aten_inputs = {t0, t1};

  at::Tensor cg_output = at::empty_like(t0, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt4_CUDA) {
  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  TensorView* tv2 = makeSymbolicTensor(4);
  fusion.addInput(tv2);

  TensorView* tv3 = makeSymbolicTensor(4);
  fusion.addInput(tv3);

  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);

  fusion.addOutput(tv6);

  // Lets setup to actually run
  while (tv6->nDims() > 1)
    tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);

  tv0->computeAt(tv6, 1);
  tv1->computeAt(tv6, 1);
  tv2->computeAt(tv6, 1);
  tv3->computeAt(tv6, 1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);
  at::Tensor t2 = at::rand_like(t0, options);
  at::Tensor t3 = at::rand_like(t0, options);

  auto t4 = t2.sub(t3);
  auto t5 = t1.add(t4);
  auto aten_output = t5.sub(t0);

  std::vector<IValue> aten_inputs = {t0, t1, t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt5_CUDA) {
  // Case 5
  // tv2 = tv0 + 2.0
  // tv3 = tv1 * tv2
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(-1, 8);
  tv3->split(-1, 4);

  tv2->computeAt(tv3, 1);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t0.add(2.0);
  auto aten_output = t1.mul(t2);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(-1, 8);
  tv2->split(-1, 4);
  tv3->merge(0);
  tv3->split(-1, 8);

  tv2->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t0.add(2.0);
  auto aten_output = t1.mul(t2);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = add(tv2, IrBuilder::create<Double>(3.0));

  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);

  auto tv5 = broadcast(tv1, {false, true});

  auto tv6 = makeSymbolicTensor(2);
  fusion.addInput(tv6);

  auto tv7 = mul(tv5, tv6);

  fusion.addOutput(tv7);

  tv7->split(1, 2);
  tv7->merge(0);
  tv7->split(0, 4);
  tv7->split(0, 128);

  tv7->axis(0)->parallelize(ParallelType::BIDx);
  tv7->axis(1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv7, 1);
  auto tv5_domain = tv5->domain()->domain();

  // These computeAt transformations should not affect the TV5 domain
  tv0->computeAt(tv4, -1);
  tv2->computeAt(tv4, -1);

  auto tv5_domain_current = tv5->domain()->domain();
  TORCH_CHECK(tv5_domain == tv5_domain_current, "Invalid TV5 domain");

  const int numel_x = 100;
  const int numel_y = 200;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({numel_x}, options);
  auto t2 = at::randn({numel_x}, options);
  auto t6 = at::randn({numel_x, numel_y}, options);

  auto t1 = t0.add(1.0);
  auto t3 = t2.add(3.0);
  auto t4 = t1.add(t3);
  auto t5 = t1.unsqueeze(1);
  auto t7 = t5.mul(t6);

  std::vector<IValue> aten_inputs = {t0, t2, t6};
  std::vector<at::Tensor> aten_outputs = {t4, t7};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAt8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = add(tv2, IrBuilder::create<Double>(3.0));

  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);

  auto tv5 = broadcast(tv1, {false, true});

  auto tv6 = makeSymbolicTensor(2);
  fusion.addInput(tv6);

  auto tv7 = mul(tv5, tv6);

  fusion.addOutput(tv7);

  tv7->split(1, 2);
  tv7->merge(0);
  tv7->split(0, 128, false);
  tv7->split(0, 4, false);

  tv7->axis(0)->parallelize(ParallelType::BIDx);
  tv7->axis(1)->parallelize(ParallelType::TIDx);

  // Reverse computeAt structure from previous test
  tv0->computeAt(tv4, -1);
  tv2->computeAt(tv4, -1);
  tv0->computeAt(tv7, -1);

  const int numel_x = 100;
  const int numel_y = 200;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({numel_x}, options);
  auto t2 = at::randn({numel_x}, options);
  auto t6 = at::randn({numel_x, numel_y}, options);

  auto t1 = t0.add(1.0);
  auto t3 = t2.add(3.0);
  auto t4 = t1.add(t3);
  auto t5 = t1.unsqueeze(1);
  auto t7 = t5.mul(t6);

  std::vector<IValue> aten_inputs = {t0, t2, t6};
  std::vector<at::Tensor> aten_outputs = {t4, t7};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith1_CUDA) {
  // Case 1
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 + 3
  // tv4 = tv1 * 2
  // tv5 = tv3 + tv2
  // tv6 = tv5 + tv4
  // tv7 = tv1 + tv4
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3.0));
  TensorView* tv4 = mul(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv5 = add(tv3, tv2);

  TensorView* tv6 = add(tv5, tv4);
  TensorView* tv7 = add(tv1, tv4);

  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  // Lets setup to actually run
  tv0->merge(0);
  tv0->split(0, 128);
  tv0->split(0, 4);

  tv0->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeWith(tv7, 1);

  GpuLower gpulw(&fusion);

  // The this-position of the last tensor should be zero.
  TORCH_CHECK(
      tv7->nDims() == 3 && tv7->getComputeAtPosition() == 0 &&
      tv7->getMaxProducerPosition() == 1);
  TORCH_CHECK(
      tv7->nDims() == 3 && tv6->getComputeAtPosition() == 0 &&
      tv6->getMaxProducerPosition() == 1);

  ComputeAtMap ca_map(&fusion);

  // The position of every other tensor should be 1.
  for (auto tv : {tv1, tv2, tv3, tv4, tv5}) {
    TORCH_CHECK(tv->nDims() == 3 && tv->getComputeAtPosition() == 1);
    TORCH_CHECK(
        ca_map.areMapped(tv7->axis(0), tv->axis(0), IdMappingMode::PERMISSIVE));
  }

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({129, 127}, options);

  auto t1 = aten_input.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t1.add({3.0});
  auto t4 = t1.mul({2.0});
  auto t5 = t3.add(t2);
  auto t6 = t5.add(t4);
  auto t7 = t1.add(t4);

  std::vector<at::Tensor> aten_outputs = {t6, t7};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options), at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith2_CUDA) {
  // Case 2
  // tv1 = tv0 * -1
  // tv2 = tv0 + 3
  // tv3 = tv0 * 2
  // tv4 = tv2 + tv1
  // tv5 = tv4 + tv3
  // tv6 = tv5 + tv3
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv4 = add(tv2, tv1);

  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv5, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  // Lets setup to actually run
  tv0->merge(0);
  tv0->split(0, 128);
  tv0->split(0, 4);

  tv0->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeWith(tv6, 1);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({129, 127}, options);

  auto t1 = input.mul({-1.0});
  auto t2 = input.add({3.0});
  auto t3 = input.mul({2.0});
  auto t4 = t2.add(t1);
  auto t5 = t4.add(t3);
  auto t6 = t5.add(t3);

  std::vector<at::Tensor> aten_outputs = {t5, t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith3_CUDA) {
  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(.979361));
  TensorView* tv3 = mul(tv2, tv0);

  fusion.addOutput(tv3);

  // Lets setup to actually run
  while (tv0->nDims() > 1)
    tv0->merge(0);
  tv0->split(0, 128);
  tv0->split(0, 4);

  while (tv1->nDims() > 1)
    tv1->merge(0);
  tv1->split(0, 128);
  tv1->split(0, 4);

  tv0->computeWith(tv3, 1);
  tv1->computeWith(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t1.mul({0.979361});
  auto aten_output = t2.mul(t0);

  std::vector<IValue> aten_inputs = {t0, t1};

  at::Tensor cg_output = at::empty_like(t0, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith4_CUDA) {
  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  TensorView* tv2 = makeSymbolicTensor(4);
  fusion.addInput(tv2);

  TensorView* tv3 = makeSymbolicTensor(4);
  fusion.addInput(tv3);

  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);

  fusion.addOutput(tv6);
  std::vector<TensorView*> tvs = {tv0, tv1, tv2};
  for (auto tv : tvs) {
    // Lets setup to actually run
    while (tv->nDims() > 1) {
      tv->merge(0);
    }
    tv->split(0, 128);
    tv->split(0, 4);
    tv->computeWith(tv6, 1);
  }

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);
  at::Tensor t2 = at::rand_like(t0, options);
  at::Tensor t3 = at::rand_like(t0, options);

  auto t4 = t2.sub(t3);
  auto t5 = t1.add(t4);
  auto aten_output = t5.sub(t0);

  std::vector<IValue> aten_inputs = {t0, t1, t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith5_CUDA) {
  // Case 5
  // tv2 = tv0 + 2.0
  // tv3 = tv1 * tv2
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(-1, 8);
  tv2->split(-1, 4);

  tv2->computeWith(tv3, 1);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t0.add(2.0);
  auto aten_output = t1.mul(t2);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeWith6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(-1, 8);
  tv2->split(-1, 4);
  tv3->merge(0);
  tv3->split(-1, 8);

  tv2->computeWith(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t0.add(2.0);
  auto aten_output = t1.mul(t2);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeAtMultiConsumers_CUDA) {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -2
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = mul(tv1, IrBuilder::create<Double>(-2.0));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  // This computeAt will affect tv2 as well, even though tv2 is not in
  // the data-flow path between tv1 and tv3. The reason is that tv1 is
  // now computed at tv3, so tv2 must also be computed at the same
  // location. Overall, what will happen is basically we merge
  // expressions of all tensors and compute them in a single loop
  // nest.
  TensorView* computeAtTarget = tv3;
  computeAtTarget->split(0, 128);
  tv1->computeAt(computeAtTarget, 1);

  TensorView* affected_tensors[] = {tv1, tv2, tv3};
  for (auto tv : affected_tensors) {
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  GpuLower gpulw(&fusion);

  TORCH_CHECK(tv1->getComputeAtPosition() == 1);
  TORCH_CHECK(
      tv2->getComputeAtPosition() == 0 && tv2->getMaxProducerPosition() == 1);
  TORCH_CHECK(
      tv3->getComputeAtPosition() == 0 && tv3->getMaxProducerPosition() == 1);

  ComputeAtMap ca_map(&fusion);

  // Note that tv2 is also computed at tv3.
  for (auto tv : {tv1, tv2}) {
    TORCH_CHECK(ca_map.areMapped(
        tv->axis(0), computeAtTarget->axis(0), IdMappingMode::PERMISSIVE));
  }

  TORCH_CHECK(tv3->getComputeAtPosition() == 0);

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);
  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({1000}, options);

  auto t1 = aten_input * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;

  std::vector<at::Tensor> aten_outputs = {t2, t3};

  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options), at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

// Similar to ComputeAtMultiConsumers, but with a common consumer.
TEST_F(NVFuserTest, FusionComputeAtCommonConsumer1_CUDA) {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -2
  // tv4 = tv2 + tv3
  // tv5 = tv4 * 5
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = mul(tv1, IrBuilder::create<Double>(-2.0));
  TensorView* tv4 = add(tv2, tv3);
  TensorView* tv5 = mul(tv4, IrBuilder::create<Double>(5.0));
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  // Computing tv1 at tv3. This will affect tv2 as discussed in
  // ComplexComputeAt1. Additionally, in this case, notice that tv4 is
  // the common consumer of tv2 and tv3, so they are computed at
  // tv4. The indirect propagation of the computeAt should stop at the
  // common consumer, and no further change should occur. More
  // specifically, the computeAT position of tv4 and tv5 should be zero.
  TensorView* computeAtTarget = tv3;
  computeAtTarget->split(0, 128);
  tv1->computeAt(computeAtTarget, 1);

  TensorView* affected_tensors[] = {tv1, tv2, tv3, tv4};
  for (auto tv : affected_tensors) {
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  TORCH_CHECK(tv1->getComputeAtPosition() == 1);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv3->getComputeAtPosition() == 1);
  TORCH_CHECK(tv4->getComputeAtPosition() == 0);
  TORCH_CHECK(tv5->getComputeAtPosition() == 0);

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  // Transform tv5 to make it look like the rest
  tv5->split(0, 128);
  tv5->axis(1)->parallelize(ParallelType::TIDx);
  tv5->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({1000}, options);

  auto t1 = aten_input * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;
  auto t4 = t2 + t3;
  auto t5 = t4 * 5.0;

  std::vector<at::Tensor> aten_outputs = {t3, t4, t5};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeAtCommonConsumer2_CUDA) {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -1
  // tv4 = tv1 + 4
  // tv5 = tv3 + tv4
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = mul(tv2, IrBuilder::create<Double>(-1.0));
  TensorView* tv4 = add(tv1, IrBuilder::create<Double>(4.0));
  TensorView* tv5 = add(tv3, tv4);

  fusion.addOutput(tv5);

  TensorView* computeAtTarget = tv3;

  computeAtTarget->merge(0);
  computeAtTarget->split(0, 128);
  computeAtTarget->split(0, 4);

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  // This computeAt will affect all tensors including tv3, tv4 and
  // tv5, even though it appears to impact only tv1 and tv2. The
  // reason is that tv1 is now computed at tv3, so tv4 must also be
  // computed at the same location. Similarly, the consumer of tv4,
  // tv5, must also be computed at the same location. Overall, what
  // will happen is basically we merge expressions of all tensors and
  // compute them in a single loop nest. Internally, this will be
  // realized by making all tensors, except for those in the path
  // between tv1 and tv3, computed at tv5, which we call the common
  // consumer.
  tv1->computeAt(computeAtTarget, 1);

  // All tensors should have the same dimenionality as the target
  for (Val* val : fusion.vals()) {
    if (val->isFusionInput() ||
        val->getValType().value() != ValType::TensorView) {
      continue;
    }
    TensorView* tv = val->as<TensorView>();
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
    if (tv == tv5) {
      TORCH_CHECK(tv->getComputeAtPosition() == 0);
    } else {
      TORCH_CHECK(tv->getComputeAtPosition() == 1);
    }
  }

  for (auto tv : ir_utils::filterByType<TensorView>(fusion.vals())) {
    if (!tv->isFusionInput()) {
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({129, 127}, options);

  auto t1 = aten_input.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t2.mul({-1.0});
  auto t4 = t1.add({4.0});
  auto aten_output = t3 + t4;

  at::Tensor cg_output = at::empty_like(aten_input, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

// Similar to the above common consumer test but adds an additional
// tensor that has no common consumer with the other tensors.
TEST_F(NVFuserTest, FusionComputeAtCommonConsumer3_CUDA) {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -1
  // tv4 = tv1 + 4
  // tv5 = tv2 + tv3
  // tv6 = tv1 + 6
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = mul(tv2, IrBuilder::create<Double>(-1.0));
  TensorView* tv4 = add(tv1, IrBuilder::create<Double>(4.0));
  TensorView* tv5 = add(tv3, tv4);
  TensorView* tv6 = add(tv1, IrBuilder::create<Double>(6.0));

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  TensorView* computeAtTarget = tv3;

  computeAtTarget->merge(0);
  computeAtTarget->split(0, 128);
  computeAtTarget->split(0, 4);

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  // This will have the same impact on the tensors except for tv5 and
  // tv6. tv6 does not have any common consumer with the computeAt
  // target, but since it uses tv1, it must be also computed at the
  // same location as the other impacted tensors. We can either make
  // tv5 computed at tv6 or tv6 computed at tv5. In this case, tv5
  // should be computed at tv6 just because the current implementation
  // orders the computeAt relationship based on the order in which
  // tensors are specified as outputs.

  tv1->computeAt(computeAtTarget, 1);

  // All tensors should have the same dimenionality as the target
  for (auto tv : ir_utils::filterByType<TensorView>(fusion.vals())) {
    if (tv->isFusionInput()) {
      continue;
    }
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
    if (tv == tv5 || tv == tv6) {
      TORCH_CHECK(tv->getComputeAtPosition() == 0);
      TORCH_CHECK(tv->getMaxProducerPosition() == 1);
    } else {
      TORCH_CHECK(tv->getComputeAtPosition() == 1);
    }
  }

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({129, 127}, options);

  auto t1 = aten_input.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t2.mul({-1.0});
  auto t4 = t1.add({4.0});
  auto t5 = t3 + t4;
  auto t6 = t1.add({6.0});

  std::vector<at::Tensor> aten_outputs = {t5, t6};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options), at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

// Similar to ComputeAtCommonConsumer1 but with an addtiona ltensor
// that does not have data dependency with the consumer.
TEST_F(NVFuserTest, FusionComputeAtNoCommonConsumer_CUDA) {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 * -2
  // tv4 = tv2 + tv3
  // tv5 = tv4 * 5
  // tv6 = tv1 * 6
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(-1.0));
  TensorView* tv3 = mul(tv1, IrBuilder::create<Double>(-2.0));
  TensorView* tv4 = add(tv2, tv3);
  TensorView* tv5 = mul(tv4, IrBuilder::create<Double>(5.0));
  // Notice that tv6 is not a consumer of tv4.
  TensorView* tv6 = mul(tv1, IrBuilder::create<Double>(6.0));
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  TensorView* computeAtTarget = tv3;
  computeAtTarget->split(0, 128);
  tv1->computeAt(computeAtTarget, 1);

  TensorView* affected_tensors[] = {tv1, tv2, tv3, tv4, tv5, tv6};
  for (auto tv : affected_tensors) {
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
    if (tv == tv6 || tv == tv5) {
      TORCH_CHECK(tv->getComputeAtPosition() == 0);
    } else {
      TORCH_CHECK(tv->getComputeAtPosition() == 1);
    }
  }

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({1000}, options);

  auto t1 = aten_input * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;
  auto t4 = t2 + t3;
  auto t5 = t4 * 5.0;
  auto t6 = t1 * 6.0;

  std::vector<at::Tensor> aten_outputs = {t3, t4, t5, t6};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

namespace {

void checkIdMapped(
    ComputeAtRootDomainMap& root_map,
    TensorView* v0,
    IterDomain* id0,
    TensorView* v1,
    IterDomain* id1,
    bool should_map) {
  if (should_map) {
    TORCH_CHECK(
        root_map.canMap(v0->domain(), id0, v1->domain(), id1),
        "Should be mappable: ",
        id0,
        " of ",
        v0,
        " and ",
        id1,
        " of ",
        v1);
  } else {
    TORCH_CHECK(
        !root_map.canMap(v0->domain(), id0, v1->domain(), id1),
        "Should not be mappable: ",
        id0,
        " of ",
        v0,
        " and ",
        id1,
        " of ",
        v1);
  }
}

void checkIdMapped(
    TensorView* v0,
    const std::vector<IterDomain*>& root0,
    const std::vector<bool> should_map0,
    TensorView* v1,
    const std::vector<IterDomain*>& root1,
    const std::vector<bool> should_map1) {
  ComputeAtRootDomainMap map;
  map.build();
  TORCH_INTERNAL_ASSERT(root0.size() == should_map0.size());
  TORCH_INTERNAL_ASSERT(root1.size() == should_map1.size());
  size_t idx0 = 0;
  for (const auto i : c10::irange(root0.size())) {
    size_t idx1 = 0;
    for (const auto j : c10::irange(root1.size())) {
      if (should_map0[i] && should_map1[j] && idx0 == idx1) {
        checkIdMapped(map, v0, root0[i], v1, root1[j], true);
      } else {
        checkIdMapped(map, v0, root0[i], v1, root1[j], false);
      }
      if (should_map1[j])
        ++idx1;
    }
    if (should_map0[i])
      ++idx0;
  }
}

void checkIdMapped(
    TensorView* v0,
    const std::vector<IterDomain*>& root0,
    TensorView* v1,
    const std::vector<IterDomain*>& root1) {
  checkIdMapped(
      v0,
      root0,
      std::vector<bool>(root0.size(), true),
      v1,
      root1,
      std::vector<bool>(root1.size(), true));
}

} // namespace

TEST_F(NVFuserTest, FusionRootMappingBasic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv3 = broadcast(tv0, {true, false, false});
  auto tv4 = broadcast(tv1, {false, true, false});
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {false, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, false, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {false, true},
      tv1,
      tv1->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {false, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, false, true});
  checkIdMapped(tv3, tv3->getRootDomain(), tv4, tv4->getRootDomain());
  checkIdMapped(tv3, tv3->getRootDomain(), tv5, tv5->getRootDomain());
  checkIdMapped(tv4, tv4->getRootDomain(), tv5, tv5->getRootDomain());
}

TEST_F(NVFuserTest, FusionRootMappingRfactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [I,I]
  TensorView* tv0 = makeSymbolicTensor(2);
  // [I,I,I]
  TensorView* tv1 = makeSymbolicTensor(3);

  //[I,I,R]
  auto tv2 = sum(tv1, {2});
  auto tv3 = add(tv2, tv0);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv3);

  // scheduling:
  //[B,I,R0,R1=128], root = [B,I,R]
  tv2->split(2, 128);

  // root=[B,I,Irf], rfactor=[B,I,Irf,Rrf]
  auto tv4 = tv2->rFactor({3});

  checkIdMapped(tv1, tv1->getRootDomain(), tv4, tv4->getRootDomain());
  checkIdMapped(
      tv4,
      tv4->getRFactorDomain(),
      {true, true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true, false},
      tv3,
      tv3->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, true, false},
      tv3,
      tv3->getRootDomain(),
      {true, true});
  checkIdMapped(tv0, tv0->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv1,
      tv1->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv2,
      tv2->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRFactorDomain(),
      {true, true, false, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true, false});
}

TEST_F(NVFuserTest, FusionRootMappingReductionDependency1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  // The second dimension cannot be mapped as it would require recomputation.
  checkIdMapped(tv0, tv0->getRootDomain(), tv1, tv1->getRootDomain());
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

TEST_F(NVFuserTest, FusionRootMappingReductionDependency2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(tv2, tv2->getRootDomain(), tv3, tv3->getRootDomain());
}

TEST_F(NVFuserTest, FusionRootMappingReductionDependency3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  auto tv3 = tv1->rFactor({-2});

  checkIdMapped(tv0, tv0->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv3,
      tv3->getMaybeRFactorDomain(),
      {true, false, true},
      tv1,
      tv1->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

TEST_F(NVFuserTest, FusionRootMappingReductionDependency4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  tv1->split(-1, 4);
  auto tv4 = tv1->rFactor({-2});

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getMaybeRFactorDomain(),
      {true, false, true},
      tv1,
      tv1->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(tv2, tv2->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

// Reproducer of issue #749
TEST_F(NVFuserTest, FusionRootMappingReductionDependency5_CUDA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, true});
}

// Similar to RootMappingReductionDependency5 but with rFactor
TEST_F(NVFuserTest, FusionRootMappingReductionDependency6_CUDA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  tv2->split(1, 4);
  auto tv6 = tv2->rFactor({-1});

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv6,
      tv6->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv6,
      tv6->getMaybeRFactorDomain(),
      {true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, true});
}

TEST_F(
    NVFuserTest,
    FusionRootMappingMultipleBroadcastWithNoCommonConsumer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv0, {true, false});
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  // If there is no common consumer, there is no recomputation constraint.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {false, true});
}

TEST_F(NVFuserTest, FusionRootMappingBroadcastNonUniqueSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);
  auto tv3 = broadcast(tv0, {false, true});
  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);
  auto tv5 = add(tv2, tv3);
  fusion.addOutput(tv5);

  // Broadcast domains can be used with multiple domains with
  // different sizes. In this test, the broadcast domain of tv3 has
  // two consumers, tv4 and tv5, which may have different sizes. Each
  // of the consumers is used with the broadcast domain of tv3, but
  // the two consumers may not have the same size, it is not possible
  // to map those domains.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, false},
      tv5,
      tv5->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, false},
      tv5,
      tv5->getRootDomain(),
      {true, false});
}

TEST_F(NVFuserTest, FusionRootMappingBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  // tv0[I0]
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {true, false});
  // tv1[B1, I0]
  auto tv2 = broadcast(tv1, {true, false, false});
  // tv2[B2, B1, I0]
  fusion.addOutput(tv2);

  // In this case, tv1 and tv2 has one and two broadcast domains,
  // respectively. It is the second broadcast domain that is mapped to
  // the broadcast of tv1.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv2,
      tv2->getRootDomain(),
      {false, true, true}); // Not {true, false, true}
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {false, false, true});
}

// Reproducer of issue #723
TEST_F(NVFuserTest, FusionRootMappingTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = sum(tv2, {0});
  auto tv4 = add(tv2, tv1);

  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  ComputeAtRootDomainMap map;
  map.build();

  checkIdMapped(
      map, tv2, tv2->getRootDomain()[0], tv4, tv4->getRootDomain()[0], true);
  checkIdMapped(
      map, tv2, tv2->getRootDomain()[0], tv3, tv3->getRootDomain()[0], true);

  tv2->computeAt(tv4, -1);

  const int x = 11;
  const int y = 12;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t1 = at::randn({y, x}, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t3 = t0;
  auto t4 = t0.unsqueeze(0).expand({y, x}) + t1;

  testValidate(&fusion, outputs, aten_inputs, {t3, t4}, __LINE__, __FILE__);
}

// Repro of issue #1950
TEST_F(NVFuserTest, FusionRootMappingRepro1950_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);
  auto tv2 = makeSymbolicTensor(3);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = set(tv0);
  auto tv4 = mul(tv1, tv3);
  auto tv5 = mul(tv1, tv2);
  auto tv6 = mul(tv5, tv3);
  auto tv7 = sum(tv6, {2});
  auto tv8 = broadcast(tv7, {false, false, true});
  auto tv9 = mul(tv3, tv8);

  // Issue #1950 was caused by a particular traversal ordering based
  // on the output tensor ordering as below
  fusion.addOutput(tv9);
  fusion.addOutput(tv5);
  fusion.addOutput(tv4);

  ComputeAtRootDomainMap root_map;
  root_map.build();

  checkIdMapped(root_map, tv4, tv4->axis(-1), tv9, tv9->axis(-1), false);
}

TEST_F(NVFuserTest, FusionDetectSelfMappedDomains_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  // [I1]
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  // [B2, I2]
  auto tv2 = broadcast(tv1, {true, false});
  // [I3, B3]
  auto tv3 = broadcast(tv1, {false, true});
  // [I4, I5]
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // IterDomainGraph maps B2, I3 and I4 together, and similarly I2,
  // B3 and I5. The problem is I1 is mapped with both of the ID
  // groups, so eventually all of the IDs are mapped
  // together. IterDomainGraph should throw an exception as this
  // pattern of domain mappings is not supported.

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW({ IterDomainGraph id_graph(&fusion); });
}

TEST_F(NVFuserTest, FusionScalarInputs_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  Double* d0 = IrBuilder::create<Double>();
  fusion.addInput(d0);
  Double* d1 = IrBuilder::create<Double>();
  fusion.addInput(d1);
  Double* d2 = IrBuilder::create<Double>();
  fusion.addInput(d2);
  Double* d3 = IrBuilder::create<Double>();
  fusion.addInput(d3);
  Val* d4 = mul(d0, d1);
  Val* d5 = sub(d2, d3);

  TensorView* tv2 = sub(tv1, d4);
  TensorView* tv3 = add(tv0, d5);
  TensorView* tv4 = mul(tv3, tv2);

  fusion.addOutput(tv4);

  // Lets setup to actually run
  while (tv4->nDims() > 1)
    tv4->merge(0);
  tv4->split(0, 128);
  tv4->split(0, 4);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  // d4 = d0 * d1
  // d5 = d2 - d3
  // t2 = t1 - d4
  // t3 = t0 + d5
  // t4 = t3 * t2

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  float fl0 = 0.1;
  float fl1 = -0.2;
  float fl2 = 0.3;
  float fl3 = -0.4;
  float fl4 = fl0 * fl1;
  float fl5 = fl2 - fl3;

  at::Tensor t0 = at::randn({129, 127}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t1.sub(fl4);
  auto t3 = t0.add(fl5);
  auto aten_output = t3.mul(t2);

  at::Tensor cg_output = at::empty_like(t0, options);

  at::Scalar test(fl0);

  std::vector<IValue> aten_inputs = {
      t0,
      t1,
      at::Scalar(fl0),
      at::Scalar(fl1),
      at::Scalar(fl2),
      at::Scalar(fl3)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLoopUnroll_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3);
  TensorView* tv1 = makeSymbolicTensor(3);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  int block_size = 16;

  tv3->merge(0, 1);
  tv3->merge(0, 1);

  tv3->split(0, block_size);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  // Parallelize
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn({129, 13, 3}, options);
  at::Tensor input1 = at::randn({129, 13, 3}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1});
  auto outputs = fe.runFusion({input0, input1});

  TORCH_CHECK(outputs[0].equal(input0.add(input1.add(2.0))));
}

/*
 * Helper function for single op testing that generates a codegen operand
 */

Val* gen_jit_operand(std::pair<ValType, DataType> desc) {
  if (desc.first == ValType::TensorView) {
    return makeSymbolicTensor(2, desc.second);
  } else if (desc.first == ValType::Scalar) {
    if (desc.second == DataType::Float) {
      return IrBuilder::create<Double>();
    } else if (desc.second == DataType::Double) {
      return IrBuilder::create<Double>();
    } else if (desc.second == DataType::ComplexFloat) {
      return IrBuilder::create<ComplexDouble>();
    } else if (desc.second == DataType::ComplexDouble) {
      return IrBuilder::create<ComplexDouble>();
    } else if (desc.second == DataType::Int) {
      return IrBuilder::create<Int>();
    } else {
      TORCH_CHECK(false, "Not currently supported type: ", desc.first);
    }
  } else {
    TORCH_CHECK(false, "Not currently supported type: ", desc.first);
  }
  return nullptr;
}

/*
 * Helper function for single op testing that generates an ATen operand
 */

IValue gen_aten_operand(
    std::pair<ValType, DataType> desc,
    int blocks,
    int threads,
    bool rand) {
  if (desc.first == ValType::TensorView) {
    if (desc.second == DataType::Double || desc.second == DataType::Float ||
        desc.second == DataType::ComplexDouble ||
        desc.second == DataType::ComplexFloat ||
        desc.second == DataType::Half || desc.second == DataType::BFloat16) {
      auto options = at::TensorOptions()
                         .dtype(data_type_to_aten(desc.second))
                         .device(at::kCUDA, 0);
      if (rand) {
        return IValue(at::rand({blocks, threads}, options));
      } else {
        return IValue(at::empty({blocks, threads}, options));
      }
    } else if (desc.second == DataType::Int || desc.second == DataType::Int32) {
      auto dtype = desc.second == DataType::Int32 ? at::kInt : at::kLong;
      if (rand) {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        return IValue(at::randn({blocks, threads}, options).mul(5).to(dtype));
      } else {
        auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
        return IValue(at::empty({blocks, threads}, options));
      }
    } else if (desc.second == DataType::Bool) {
      if (rand) {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        return IValue(
            at::rand({blocks, threads}, options).round().to(at::kBool));
      } else {
        auto options =
            at::TensorOptions().dtype(at::kBool).device(at::kCUDA, 0);
        return IValue(at::empty({blocks, threads}, options));
      }
    } else {
      TORCH_CHECK(false, "Not currently supported type: ", desc.second)
    }
  } else if (desc.first == ValType::Scalar) {
    // IValue scalars can only be double int64 or bool
    if (desc.second == DataType::ComplexDouble ||
        desc.second == DataType::ComplexFloat) {
      return IValue(at::Scalar(c10::complex<double>(1.0, 0.0)));
    } else if (
        desc.second == DataType::Double || desc.second == DataType::Float ||
        desc.second == DataType::Half || desc.second == DataType::BFloat16) {
      return IValue(at::Scalar(1.0));
    } else if (desc.second == DataType::Int) {
      return IValue(at::Scalar(1));
    } else {
      TORCH_CHECK(false, "Not currently supported type: ", desc.first);
    }
  } else {
    TORCH_CHECK(false, "Not currently supported type: ", desc.first);
  }
  return nullptr;
}

/*
 * Templatized Helper Function To generate single Op comparison between the
 * JIT codegen for Cuda and the ATen Library.
 */

using OutputPair = std::pair<ValType, DataType>;
template <
    typename AtenFunc,
    typename JitFunc,
    typename InputTuple,
    size_t... NumInputs>
void test_op(
    int blocks,
    int threads,
    std::string op_str,
    AtenFunc af,
    JitFunc jf,
    OutputPair op,
    InputTuple it,
    std::index_sequence<NumInputs...>) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Generate Input JIT function Inputs and add them as Inputs to the Fusion
  // Graph
  std::array<Val*, sizeof...(NumInputs)> jit_inputs = {
      gen_jit_operand(std::get<NumInputs>(it))...};
  std::for_each(jit_inputs.begin(), jit_inputs.end(), [&fusion](Val* v) {
    fusion.addInput(v);
  });
  TensorView* out =
      static_cast<TensorView*>(jf(std::get<NumInputs>(jit_inputs)...));
  fusion.addOutput(out);

  std::for_each(jit_inputs.begin(), jit_inputs.end(), [out](Val* v) {
    if (v->getValType() == ValType::TensorView)
      static_cast<TensorView*>(v)->computeAt(out, -1);
  });
  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(-1)->parallelize(ParallelType::TIDx);

  std::array<IValue, sizeof...(NumInputs)> aten_inputs = {gen_aten_operand(
      std::get<NumInputs>(it), blocks, threads, /*rand*/ true)...};
  const at::ArrayRef<IValue> aten_inputs_ivalues(aten_inputs);

  at::Tensor cg_output =
      gen_aten_operand(op, blocks, threads, /*rand*/ false).toTensor();
  std::vector<at::Tensor> output_vect = {cg_output};
  cudaDeviceSynchronize();
  if (fusion.isStochastic())
    at::manual_seed(0);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs_ivalues);
  fe.runFusion(aten_inputs_ivalues, output_vect);
  cudaDeviceSynchronize();

  if (fusion.isStochastic())
    at::manual_seed(0);
  at::Tensor aten_output = af(aten_inputs);
  cudaDeviceSynchronize(); // This sync shouldn't be necessary;

  std::string op_msg = "Operation " + op_str;

  testValidate(
      &fusion,
      {cg_output},
      aten_inputs,
      {aten_output},
      __LINE__,
      __FILE__,
      op_msg);
}

/*
 *  Templatized Helper Function that uses variadic templates to
 *  process a variable length Input Tuple of different Operand Type.
 */
template <typename AtenFunc, typename JitFunc, typename InputTuple>
void test_op(
    int blocks,
    int threads,
    std::string op_str,
    AtenFunc af,
    JitFunc jf,
    OutputPair op,
    InputTuple it) {
  static constexpr auto size = std::tuple_size<InputTuple>::value;
  test_op(
      blocks,
      threads,
      op_str,
      af,
      jf,
      op,
      it,
      std::make_index_sequence<size>{});
}

TEST_F(NVFuserTest, FusionUnaryOps_CUDA) {
  using OpTuple =
      std::tuple<at::Tensor (*)(const at::Tensor&), UnaryOpType, std::string>;

  // [Note: explicit tuple type for uniform initialization list]
  // Tuple type must be explicitly specified for each uniform initialization
  // list within the vector to make this code compatible with some old env
  // which we still need to support. eg. gcc 5.4 + cuda 9.2.
  std::vector<OpTuple> ops{
      OpTuple{at::acos, UnaryOpType::Acos, "acos"},
      OpTuple{at::asin, UnaryOpType::Asin, "asin"},
      OpTuple{at::atan, UnaryOpType::Atan, "atan"},
      // There does not appear to be an appropriate ATen function for atanh
      // OpTuple{at::atanh,      UnaryOpType::Atanh,      "atanh"      },
      OpTuple{at::cos, UnaryOpType::Cos, "cos"},
      OpTuple{at::cosh, UnaryOpType::Cosh, "cosh"},
      OpTuple{at::exp, UnaryOpType::Exp, "exp"},
      // OpTuple{at::gelu, UnaryOpType::Gelu, "gelu"},
      OpTuple{at::log, UnaryOpType::Log, "log"},
      OpTuple{at::log10, UnaryOpType::Log10, "log10"},
      OpTuple{at::neg, UnaryOpType::Neg, "neg"},
      OpTuple{at::reciprocal, UnaryOpType::Reciprocal, "reciprocal"},
      OpTuple{at::sigmoid, UnaryOpType::Sigmoid, "sigmoid"},
      OpTuple{at::sin, UnaryOpType::Sin, "sin"},
      OpTuple{at::sinh, UnaryOpType::Sinh, "sinh"},
      OpTuple{at::sqrt, UnaryOpType::Sqrt, "sqrt"},
      OpTuple{at::tan, UnaryOpType::Tan, "tan"},
      OpTuple{at::tanh, UnaryOpType::Tanh, "tanh"},
      OpTuple{at::isfinite, UnaryOpType::IsFinite, "isfinite"},
      OpTuple{at::isinf, UnaryOpType::IsInf, "isinf"},
      OpTuple{at::isnan, UnaryOpType::IsNan, "isnan"},
      OpTuple{at::isreal, UnaryOpType::IsReal, "isreal"},
  };

  // The following ops has no complex support in eager mode
  std::vector<OpTuple> ops_without_complex{
      OpTuple{at::ceil, UnaryOpType::Ceil, "ceil"},
      OpTuple{at::floor, UnaryOpType::Floor, "floor"},
      OpTuple{at::frac, UnaryOpType::Frac, "frac"},
      OpTuple{at::trunc, UnaryOpType::Trunc, "trunc"},
      OpTuple{at::round, UnaryOpType::Round, "round"},
      OpTuple{at::relu, UnaryOpType::Relu, "relu"},
      OpTuple{at::expm1, UnaryOpType::Expm1, "expm1"},
      OpTuple{at::log1p, UnaryOpType::Log1p, "log1p"},
      OpTuple{at::lgamma, UnaryOpType::Lgamma, "lgamma"},
      OpTuple{at::erf, UnaryOpType::Erf, "erf"},
      OpTuple{at::erfc, UnaryOpType::Erfc, "erfc"},
      OpTuple{at::isneginf, UnaryOpType::IsNegInf, "isneginf"},
      OpTuple{at::isposinf, UnaryOpType::IsPosInf, "isposinf"},
  };

  // The following ops only supports complex
  std::vector<OpTuple> ops_complex_only{
      // real is supported via UnaryOpType::Set for non-complex types, and
      // UnaryOpType::Real requires input to be complex
      OpTuple{at::real, UnaryOpType::Real, "real"},
      OpTuple{at::imag, UnaryOpType::Imag, "imag"},
  };

  // Complex support for the following op is not working in nvFuser yet
  std::vector<OpTuple> ops_skip_complex{
      // TODO: abs is actually supported in nvFuser, but it has bug!!!
      // In eager mode, abs(complex_tensor) returns floating point tensor
      // but in nvFuser, it wrongly returns complex tensor!
      // We need to:
      //  1. change our type promotion logic to make a special case for abs
      //  2. why this bug is not detected here? we should bump up test coverage
      OpTuple{at::abs, UnaryOpType::Abs, "abs"},
      // TODO: the following two ops fails with compilation error like
      // "undefined function rsqrt(complex)", we could implement them in
      // helpers.cu, but I think it is better to check with Jiterator first,
      // because Jiterator uses the same string for complex support.
      OpTuple{at::rsqrt, UnaryOpType::Rsqrt, "rsqrt"},
      OpTuple{at::log2, UnaryOpType::Log2, "log2"}};

  std::vector<DataType> dtypes = {
      DataType::Float,
      DataType::Double,
      DataType::ComplexFloat,
      DataType::ComplexDouble};

  for (auto dtype : dtypes) {
    auto ops_to_test = ops;
    if (dtype != DataType::ComplexFloat && dtype != DataType::ComplexDouble) {
      ops_to_test.insert(
          ops_to_test.end(),
          ops_without_complex.begin(),
          ops_without_complex.end());
      ops_to_test.insert(
          ops_to_test.end(), ops_skip_complex.begin(), ops_skip_complex.end());
    } else {
      ops_to_test.insert(
          ops_to_test.end(), ops_complex_only.begin(), ops_complex_only.end());
    }
    std::for_each(ops.begin(), ops.end(), [&](OpTuple& op) {
      test_op(
          /*blocks*/ 640,
          /*threads*/ 64,
          /*name*/ std::get<2>(op),
          /*Aten Func   */
          [&op](std::array<IValue, 1>& vals) {
            return std::get<0>(op)(vals[0].toTensor());
          },
          /*JIT  Func   */
          [&op](Val* in1) -> Val* { return unaryOp(std::get<1>(op), in1); },
          /*Output      */ std::make_pair(ValType::TensorView, dtype),
          /*Inputs Tuple*/
          std::make_tuple(std::make_pair(ValType::TensorView, dtype)));
    });
  }

  dtypes = {DataType::Int, DataType::Int32, DataType::Bool};
  for (auto dtype : dtypes) {
    test_op(
        /*blocks*/ 128,
        /*threads*/ 64,
        /*name*/ "bitwise_not",
        /*Aten Func   */
        [](std::array<IValue, 1>& vals) {
          return at::bitwise_not(vals[0].toTensor());
        },
        /*JIT  Func   */
        [](Val* in1) -> Val* { return unaryOp(UnaryOpType::Not, in1); },
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(std::make_pair(ValType::TensorView, dtype)));
  }
}

TEST_F(NVFuserTest, FusionBinaryOps_CUDA) {
  using AtenFuncSig = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
  using OpTuple = std::tuple<AtenFuncSig, BinaryOpType, std::string>;

  std::vector<DataType> dtypes = {
      DataType::Double,
      DataType::Float,
      DataType::ComplexFloat,
      DataType::ComplexDouble};

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> equal_ops{
      OpTuple{at::eq, BinaryOpType::Eq, "eq"},
      OpTuple{at::ne, BinaryOpType::NE, "ne"}};

  // Complex numbers are not ordered
  std::vector<OpTuple> order_ops{
      OpTuple{at::ge, BinaryOpType::GE, "ge"},
      OpTuple{at::gt, BinaryOpType::GT, "gt"},
      OpTuple{at::le, BinaryOpType::LE, "le"},
      OpTuple{at::lt, BinaryOpType::LT, "lt"}};

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> math_ops{
      OpTuple{at::div, BinaryOpType::Div, "div"},
      OpTuple{at::mul, BinaryOpType::Mul, "mul"},
      OpTuple{at::pow, BinaryOpType::Pow, "pow"}};

  // The following ops has no complex support in eager mode
  std::vector<OpTuple> math_ops_without_complex{
      OpTuple{at::atan2, BinaryOpType::Atan2, "atan2"},
      OpTuple{at::max, BinaryOpType::Max, "max"},
      OpTuple{at::min, BinaryOpType::Min, "min"},
      OpTuple{at::fmod, BinaryOpType::Fmod, "fmod"},
      // NOTE: Remainder does not match the Aten impl exactly
      // despite using an identical function.
      OpTuple{at::remainder, BinaryOpType::Remainder, "remainder"}};

  for (auto dtype : dtypes) {
    auto logic_ops = equal_ops;
    if (dtype != DataType::ComplexFloat && dtype != DataType::ComplexDouble) {
      logic_ops.insert(logic_ops.end(), order_ops.begin(), order_ops.end());
    }
    std::for_each(logic_ops.begin(), logic_ops.end(), [&](OpTuple& op) {
      test_op(
          /*blocks*/ 640,
          /*threads*/ 64,
          /*name*/ std::get<2>(op),
          /*Aten Func   */
          [&op](std::array<IValue, 2>& vals) {
            return std::get<0>(op)(vals[0].toTensor(), vals[1].toTensor());
          },
          /*JIT  Func   */
          [&op](Val* in1, Val* in2) -> Val* {
            return binaryOp(std::get<1>(op), in1, in2);
          },
          /*Output      */ std::make_pair(ValType::TensorView, DataType::Bool),
          /*Inputs Tuple*/
          std::make_tuple(
              std::make_pair(ValType::TensorView, dtype),
              std::make_pair(ValType::TensorView, dtype)));
    });

    auto enabled_math_ops = math_ops;
    if (dtype != DataType::ComplexFloat && dtype != DataType::ComplexDouble) {
      enabled_math_ops.insert(
          enabled_math_ops.end(),
          math_ops_without_complex.begin(),
          math_ops_without_complex.end());
    }
    std::for_each(
        enabled_math_ops.begin(), enabled_math_ops.end(), [&](OpTuple& op) {
          test_op(
              /*blocks*/ 640,
              /*threads*/ 64,
              /*name*/ std::get<2>(op),
              /*Aten Func   */
              [&op](std::array<IValue, 2>& vals) {
                return std::get<0>(op)(vals[0].toTensor(), vals[1].toTensor());
              },
              /*JIT  Func   */
              [&op](Val* in1, Val* in2) -> Val* {
                return binaryOp(std::get<1>(op), in1, in2);
              },
              /*Output      */ std::make_pair(ValType::TensorView, dtype),
              /*Inputs Tuple*/
              std::make_tuple(
                  std::make_pair(ValType::TensorView, dtype),
                  std::make_pair(ValType::TensorView, dtype)));
        });

    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ "add_alpha",
        /*Aten Func   */
        [](std::array<IValue, 3>& vals) {
          return at::add(
              vals[0].toTensor(), vals[1].toTensor(), vals[2].toScalar());
        },
        /*JIT  Func   */ static_cast<Val* (*)(Val*, Val*, Val*)>(&add_alpha),
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::Scalar, dtype)));

    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ "sub_alpha",
        /*Aten Func   */
        [](std::array<IValue, 3>& vals) {
          return at::sub(
              vals[0].toTensor(), vals[1].toTensor(), vals[2].toScalar());
        },
        /*JIT  Func   */ static_cast<Val* (*)(Val*, Val*, Val*)>(&sub_alpha),
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::Scalar, dtype)));
  }
}

TEST_F(NVFuserTest, FusionTernaryOps_CUDA) {
  std::vector<DataType> dtypes = {
      DataType::Double,
      DataType::Float,
      DataType::ComplexFloat,
      DataType::ComplexDouble};

  for (auto dtype : dtypes) {
    // clamp and threshold are not supported for complex on eager mode
    if (dtype != DataType::ComplexFloat && dtype != DataType::ComplexDouble) {
      test_op(
          /*blocks*/ 640,
          /*threads*/ 64,
          /*name*/ "clamp",
          /*Aten Func   */
          [](std::array<IValue, 1>& vals) {
            return at::clamp(vals[0].toTensor(), 0.f, 1.f);
          },
          /*JIT  Func   */
          [&](Val* in1) -> Val* {
            if (dtype == DataType::Float) {
              return clamp(
                  in1,
                  IrBuilder::create<Double>(0.f),
                  IrBuilder::create<Double>(1.f));
            } else {
              return clamp(
                  in1,
                  IrBuilder::create<Double>(0.f),
                  IrBuilder::create<Double>(1.f));
            }
          },
          /*Output      */ std::make_pair(ValType::TensorView, dtype),
          /*Inputs Tuple*/
          std::make_tuple(std::make_pair(ValType::TensorView, dtype)));
      test_op(
          /*blocks*/ 640,
          /*threads*/ 64,
          /*name*/ "threshold",
          /*Aten Func   */
          [](std::array<IValue, 1>& vals) {
            return at::threshold(vals[0].toTensor(), 0.f, 1.f);
          },
          /*JIT  Func   */
          [&](Val* in1) -> Val* {
            if (dtype == DataType::Float) {
              return threshold(
                  in1,
                  IrBuilder::create<Double>(0.f),
                  IrBuilder::create<Double>(1.f));
            } else {
              return threshold(
                  in1,
                  IrBuilder::create<Double>(0.f),
                  IrBuilder::create<Double>(1.f));
            }
          },
          /*Output      */ std::make_pair(ValType::TensorView, dtype),
          /*Inputs Tuple*/
          std::make_tuple(std::make_pair(ValType::TensorView, dtype)));
    }
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ "where",
        /*Aten Func   */
        [](std::array<IValue, 3>& vals) {
          return at::where(
              vals[0].toTensor(), vals[1].toTensor(), vals[2].toTensor());
        },
        /*JIT  Func   */ static_cast<Val* (*)(Val*, Val*, Val*)>(&where),
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, DataType::Bool),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype)));
  }
}

TEST_F(NVFuserTest, FusionCompoundOps_CUDA) {
  std::vector<DataType> dtypes = {
      DataType::Double,
      DataType::Float,
      DataType::ComplexFloat,
      DataType::ComplexDouble};

  for (auto dtype : dtypes) {
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ "lerp",
        /*Aten Func   */
        [](std::array<IValue, 3>& vals) {
          return at::lerp(
              vals[0].toTensor(), vals[1].toTensor(), vals[2].toTensor());
        },
        /*JIT  Func   */ static_cast<Val* (*)(Val*, Val*, Val*)>(&lerp),
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype)));
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ "addcmul",
        /*Aten Func   */
        [](std::array<IValue, 4>& vals) {
          return at::addcmul(
              vals[0].toTensor(),
              vals[1].toTensor(),
              vals[2].toTensor(),
              vals[3].toScalar());
        },
        /*JIT  Func   */
        static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(&addcmul),
        /*Output      */ std::make_pair(ValType::TensorView, dtype),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::TensorView, dtype),
            std::make_pair(ValType::Scalar, dtype)));
  }
}

TEST_F(NVFuserTest, FusionCastOps_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2, DataType::Half);

  TensorView* intrm1 = castOp(DataType::Float, tv0);
  TensorView* out = castOp(DataType::Half, intrm1);

  fusion.addInput(tv0);
  fusion.addOutput(out);
  tv0->computeAt(out, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({1, 4}, options);
  at::Tensor ref_output = at::empty_like(input1);

  std::array<IValue, 1> inputs = {input1};
  const at::ArrayRef<IValue> input_ivalues(inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, input_ivalues);
  auto outputs = fe.runFusion(input_ivalues);

  ref_output = at::_cast_Half(at::_cast_Double(input1));

  TORCH_CHECK(
      outputs[0].equal(ref_output),
      "\nOp Type: -- ",
      "cast FP16->FP32->FP16",
      " -- had a mismatch.\n",
      "\nABS MAX DIFF: ",
      outputs[0].sub(ref_output).abs().max(),
      "\n");
}

// Start off simple, block on the outer dim
// block stride + thread all reduce + unrolling on inner dim
TEST_F(NVFuserTest, FusionReduction1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, 128);
  // tv1[I0, R1o, R1i{128}] = tv0[I0, I1]
  tv1->split(1, 4);
  // tv1[I0, R1oo, R1oi{4}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}] = tv0[I0, I1]
  // tv1[I0,        R1oi{4},  R1i{128}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}]

  TensorView* tv3 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}] = tv0[I0, I1]
  // tv3[I0,        R1oi{4}, Ir1i{128}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}]
  // tv1[I0,                  R1i{128}] = tv3[I0,        R1oi{4}, Ir1i{128}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv3, 1);
  tv3->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv2->axis(2)->parallelize(ParallelType::Unroll);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 65000;
  int numel_y = 1025;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);

  fusion.addOutput(tv1);

  // switches to try some different scenarios. maybe we should iterate on all
  // permutations.
  bool bind_bidx = true;
  bool bind_tidx = true;
  bool bind_tidy = true;
  bool bind_unroll = true;

  int numel_x = 1025; // Cannot exceed block dim max size / tidy
  int numel_y = 129;
  int tidx = 16;
  int tidy = 8;
  int unroll_factor = 4;

  tv1->split(1, tidx);
  // tv1[I0, R1o, R1i{tidx}] = tv0[I0, I1]

  tv1->split(1, unroll_factor);
  // tv1[I0, R1oo, R1oi{unroll}, R1i{tidx}] = tv0[I0, I1]

  tv1->split(0, tidy);

  TensorView* tv2 = tv1->rFactor({-3});
  // tv2[I0,             >R1oo<, Ir1oi{unroll}, Ir1i{tidx}]
  // tv1[I0o, I0i{tidy},          R1oi{unroll},  R1i{tidx}]

  TensorView* tv3 = tv1->rFactor({-2});
  // tv2[I0,             >R1oo<, Ir1oi{unroll}, Ir1i{tidx}]
  // tv3[I0,                      R1oi{unroll}, Ir1i{tidx}]
  // tv1[I0o, I0i{tidy},                         R1i{tidx}]

  tv0->computeAt(tv1, -2);

  if (bind_unroll)
    tv2->axis(-2)->parallelize(ParallelType::Unroll);
  if (bind_bidx)
    tv1->axis(0)->parallelize(ParallelType::BIDx);
  if (bind_tidy)
    tv1->axis(1)->parallelize(ParallelType::TIDy);

  if (bind_tidx) {
    tv2->axis(-1)->parallelize(ParallelType::TIDx);
    tv3->axis(-1)->parallelize(ParallelType::TIDx);
    tv1->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduction3_CUDA) {
  // What if Z participates in the reduction with X?
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);

  fusion.addOutput(tv1);

  int numel_x = 1025; // Cannot exceed block dim max size / tidy
  int numel_y = 129;
  int tidx = 16;
  int tidz = 8;

  tv1->split(1, tidz);
  // tv1[I0, R1o, R1i{tidz}] = tv0[I0, I1]

  tv1->split(1, tidx);
  // tv1[I0, R1oo, R1oi{tidx}, R1i{tidz}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({-3});
  // tv2[I0,  >R1oo<, Ir1oi{tidx}, Ir1i{tidz}]
  // tv1[I0o,          R1oi{tidx},  R1i{tidz}]

  tv0->computeAt(tv1, -3);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDz);

  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDz);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  auto aten_output = aten_input.to(at::kDouble).sum({1});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduction4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  TensorView* tv2 = add(tv0, tv1);
  // tv2[I0, I1] = tv0[I0, I1] + tv1[I0, I1]

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv3 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv2);
  // tv3[I0, R1] = tv2[I0, I1]

  TensorView* tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);

  // tv5[I0] = tv3[I0, R1] * tv4[I0]
  TensorView* tv5 = mul(tv3, tv4);
  fusion.addOutput(tv5);

  int tidx = 16;

  // RFactor the reduction
  tv3->split(1, tidx);
  // tv3[I0, R1o, R1i{tidx}] = tv2[I0, I1]

  TensorView* tv6 = tv3->rFactor({-2});
  // tv6[I0, R1o, iR1i{tidx}] = tv2[I0, I1]
  // tv3[I0,       R1i{tidx}] = tv3[I0, I1]
  tv2->computeAt(tv6, 2);

  // Compute at inline with tv5 (only 1D)
  tv6->computeAt(tv3, 1);
  tv3->computeAt(tv5, 1);

  tv5->axis(0)->parallelize(ParallelType::BIDx);

  // Intermediate tensors only need this, but doesn't hurt to do on inputs
  // tv0, 1, 4
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 1025;
  int numel_y = 129;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t1 = at::randn({numel_x, numel_y}, options);
  at::Tensor t4 = at::randn({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t4});
  auto cg_outputs = fe.runFusion({t0, t1, t4});

  auto t2 = t0.add(t1);
  auto t3 = t2.to(at::kDouble).sum({1});
  auto aten_output = t3.mul(t4);

  testValidate(
      &fusion, cg_outputs, {t0, t1, t4}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduction5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3);

  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);

  fusion.addOutput(tv1);

  int bidy = 2;
  int tidy = 4;
  int tidx = 5;

  int dim1 = 11;

  tv1->split(-2, tidy);

  TensorView* tv2 = tv1->rFactor({-3});

  tv0->computeAt(tv1, 1);
  tv1->axis(0)->parallelize(ParallelType::BIDy);

  for (auto* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      val->as<TensorView>()->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv1->axis(-2)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({bidy, dim1, tidx}, options);

  at::Tensor cg_output = at::empty({bidy, tidx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduction6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 64;
  const int bdimy = 8;

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // tv1[I0, R1, R2] = tv0[I0, I1, I2]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1, 2}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(2, bdimx);
  // tv1[I0, R1, R2o, R2i{128}] = tv0[I0, I1, I2]
  tv1->split(1, bdimy);
  // tv1[I0, R1o, R1i{8}, R2o, R2i{128}] = tv0[I0, I1, I2]

  TensorView* tv2 = tv1->rFactor({3});
  // tv2[I0, I1o, I1i{8}, R2o, I2i{128}] = tv0[I0, I1, I2]
  // tv1[I0, R1o, R1i{8},      R2i{128}] = tv2[I0, I1o, I1i{8}, R2o, I2i{128}]

  TensorView* tv3 = tv1->rFactor({1});
  // tv2[I0, I1o, I1i{8}, R2o, I2i{128}] = tv0[I0, I1, I2]
  // tv3[I0, R1o, I1i{8},      I2i{128}] = tv2[I0, I1o, I1i{8}, R2o, I2i{128}]
  // tv1[I0,      R1i{8},      R2i{128}] = tv3[I0, R1o, I1i{8},      I2i{128}]

  tv3->computeAt(tv1, 1);
  tv2->computeAt(tv3, 2);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-3)->parallelize(ParallelType::TIDy);

  int numel_x = 650;
  int numel_y = 1000;
  int numel_z = 4;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y, numel_z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({1, 2});
  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMultiGridReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = max(tv0, {0});
  TensorView* tv2 = sum(tv0, {0});

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  int numel_x = 4;
  int numel_y = 2;

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  std::vector<at::Tensor> aten_outputs = {
      std::get<0>(input.to(at::kDouble).max(0)), input.to(at::kDouble).sum(0)};
  testValidate(&fusion, cg_outputs, {input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMultiGridReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(0)->parallelize(ParallelType::BIDy);

  FusionExecutor fe;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionReductionTFT_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);

  fusion.addOutput(tv1);

  int numel_x = 1025;
  int numel_y = 129;
  int tidx = 16;
  int tidy = 8;
  int tidz = 8;

  tv1->split(1, tidx);
  // tv1[I0, R1o, R1i{tidx}]

  tv1->split(1, tidz);
  // tv1[I0, R1oo, R1Oi{tidz}, R1R1i{tidx}]

  tv1->split(0, tidy);
  // tv1[I0o, I0i, R1oo, R1Oi{tidz}, R1R1i{tidx}]

  TensorView* tv2 = tv1->rFactor({2});
  // tv2[I0o, I0i, R1oo, I1Oi{tidz}, I11i{tidx}]
  // tv1[I0o, I0i,       R1Oi{tidz}, R1R1i{tidx}]

  tv2->computeAt(tv1, 2);

  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(-2)->parallelize(ParallelType::TIDz);
  tv2->axis(-2)->parallelize(ParallelType::TIDz);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionOuterSplit_CUDA) {
  // based off FusionReduction4
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  TensorView* tv2 = add(tv0, tv1);
  // tv2[I0, I1] = tv0[I0, I1] + tv1[I0, I1]

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv3 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv2);
  // tv3[I0, R1] = tv2[I0, I1]

  TensorView* tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);

  // tv5[I0] = tv3[I0, R1] * tv4[I0]
  TensorView* tv5 = mul(tv3, tv4);
  fusion.addOutput(tv5);

  // RFactor the reduction
  tv3->split(1, 16, false);
  // tv3[I0, R1o{16}, R1i{tidx}] = tv2[I0, I1]

  TensorView* tv6 = tv3->rFactor({-2});
  // tv6[I0, R1o{16}, iR1i{tidx}] = tv2[I0, I1]
  // tv3[I0,           R1i{tidx}] = tv3[I0, I1]
  tv2->computeAt(tv6, 2);

  // Compute at inline with tv5 (only 1D)
  tv6->computeAt(tv3, 1);
  tv3->computeAt(tv5, 1);

  tv5->axis(0)->parallelize(ParallelType::BIDx);

  // Intermediate tensors only need this, but doesn't hurt to do on inputs
  // tv0, 1, 4
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 1025;
  int numel_y = 129;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t1 = at::randn({numel_x, numel_y}, options);
  at::Tensor t4 = at::randn({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t4});
  auto cg_outputs = fe.runFusion({t0, t1, t4});

  auto t2 = t0.add(t1);
  auto t3 = t2.to(at::kDouble).sum({1});
  auto aten_output = t3.mul(t4);

  testValidate(
      &fusion, cg_outputs, {t0, t1, t4}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBranches_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);
  TensorView* tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv3, tv2);
  auto tv6 = add(tv4, tv5);

  fusion.addOutput(tv6);

  constexpr int x = 63, y = 33;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);
  at::Tensor t2 = at::randn({x, y}, options);

  FusionExecutor fe;
  tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv6, 1);
  tv1->computeAt(tv6, 1);
  tv2->computeAt(tv6, 1);

  tv3->axis(-2)->parallelize(ParallelType::Unroll);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-2)->parallelize(ParallelType::Unroll);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-2)->parallelize(ParallelType::Unroll);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  std::vector<IValue> aten_inputs = {t0, t1, t2};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t3 = t0.add(1.0);
  auto t4 = t3.add(t1);
  auto t5 = t3.add(t2);
  auto aten_output = t4.add(t5);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleBCast1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1.5));

  TensorView* tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);
  TensorView* tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  TensorView* tv4 = sub(tv2, tv3);

  TensorView* tv5 = broadcast(tv1, {false, false, true});
  TensorView* tv6 = broadcast(tv4, {true, false, false});

  TensorView* tv7 = add(tv5, tv6);
  fusion.addOutput(tv7);

  tv7->split(-1, 4);
  tv7->split(0, 8);

  tv0->computeAt(tv7, -1);
  tv2->computeAt(tv7, -1);

  tv7->axis(0)->parallelize(ParallelType::BIDx);
  tv7->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int x = 63, y = 33, z = 15;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t1 = t0.add(1.5);

  at::Tensor t2 = at::randn({y, z}, options);
  at::Tensor t3 = at::randn({y, z}, options);

  at::Tensor t4 = t2.sub(t3);
  at::Tensor t5 = t1.unsqueeze(-1).expand({x, y, z});

  at::Tensor t6 = t4.expand({x, y, z});

  at::Tensor aten_output = t5.add(t6);

  std::vector<IValue> aten_inputs = {t0, t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleBCast2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  TensorView* tv3 = broadcast(tv2, {false, false, true});

  TensorView* tv4 = makeSymbolicTensor(2);
  fusion.addInput(tv4);

  TensorView* tv5 = sub(tv4, IrBuilder::create<Double>(0.1));

  TensorView* tv6 = broadcast(tv5, {true, false, false});

  TensorView* tv7 = add(tv3, tv6);

  fusion.addOutput(tv7);

  tv7->merge(0, 1);

  tv0->computeAt(tv7, -1);
  tv4->computeAt(tv7, -1);

  tv7->axis(0)->parallelize(ParallelType::BIDx);
  tv7->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int x = 63, y = 33, z = 15;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);
  at::Tensor t2 = t0.add(t1);
  at::Tensor t3 = t2.unsqueeze(-1).expand({x, y, z});

  at::Tensor t4 = at::randn({y, z}, options);
  at::Tensor t5 = t4.sub(0.1);
  at::Tensor t6 = t5.expand({x, y, z});
  at::Tensor aten_output = t3.add(t6);

  at::Tensor cg_output = at::empty({x, y, z}, options);

  std::vector<IValue> aten_inputs = {t0, t1, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleBCast3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up input tensor views
  // tv0[I1, B{1}]
  TensorView* tv0 = makeConcreteTensor({-1, 1});
  fusion.addInput(tv0);

  // tv1[I0, I1, I2]
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->merge(0);

  tv0->computeAt(tv3, -1);
  tv2->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  constexpr int x = 2, y = 3, z = 4;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({y, 1}, options);
  at::Tensor t2 = at::randn({x, y, z}, options);
  auto aten_output = t0.add(t2);

  std::vector<IValue> aten_inputs = {t0, t2};
  at::Tensor cg_output = at::empty({x, y, z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleBCast4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({1, -1});

  TensorView* tv1 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv3 = add(tv0, tv1);

  tv3->merge(0);
  tv3->merge(0);
  tv3->split(0, 128);
  tv3->split(0, 4);

  fusion.addOutput(tv3);

  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-2)->parallelize(ParallelType::Unroll);

  constexpr int x = 63, y = 33, z = 15;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({1, z}, options);
  at::Tensor t1 = at::randn({x, y, z}, options);

  auto aten_output = t0.add(t1);

  at::Tensor cg_output = at::empty({x, y, z}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleBCast5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int m = 2, k = 3, n = 4;
  auto tv0 = makeConcreteTensor({m, k});
  auto tv1 = makeConcreteTensor({k, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = broadcast(tv0, {false, false, true});
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  TensorView* tv4 = add(tv2, tv3);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);

  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({m, k}, options);
  at::Tensor t1 = at::randn({k, n}, options);

  auto t2 = t0.unsqueeze(-1).expand({m, k, n});
  auto t3 = t1.expand({m, k, n});
  auto aten_output = t2.add(t3);

  at::Tensor cg_output = at::empty({m, k, n}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComplexBCast1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int x = 2, y = 3, z = 4;

  auto tv0 = makeConcreteTensor({y});
  auto tv1 = div(tv0, IrBuilder::create<Double>(2.0));
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = makeConcreteTensor({y, z});
  auto tv4 = mul(tv2, tv3);
  auto tv5 = broadcast(tv4, {true, false, false});
  auto tv6 = makeConcreteTensor({x, y, z});
  auto tv7 = add(tv5, tv6);

  // tv0[    i1    ] = input
  // tv1[    i1    ] = tv0/2.0
  // tv2[    i1, b2] = bcast(tv1)
  // tv3[    i1, i2] = input
  // tv4[    i1, i2] = tv2 * tv3
  // tv5[b0, i1, i2] = bcast(tv4)
  // tv6[i0, i1, i2] = input
  // tv7[i0, i1, i2] = tv5 + tv6

  // tv4 = bcast(tv1) * tv3
  // tv7 = bcast(tv4) + tv6

  fusion.addInput(tv0);
  fusion.addInput(tv3);
  fusion.addInput(tv6);

  fusion.addOutput(tv7);

  tv7->merge(0);
  tv7->merge(0);
  tv0->computeAt(tv7, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({y}, options);
  at::Tensor t3 = at::randn({y, z}, options);
  at::Tensor t6 = at::randn({x, y, z}, options);

  auto t4 = t0.div(2.0).unsqueeze(-1).expand({y, z}) * t3;
  auto aten_output = t4.unsqueeze(0).expand({x, y, z}) + t6;

  std::vector<IValue> aten_inputs = {t0, t3, t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComplexBCast2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int x = 2, y = 3, z = 4;

  auto tv0 = makeConcreteTensor({y, z});
  auto tv1 = div(tv0, IrBuilder::create<Double>(2.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = makeConcreteTensor({x, y});
  auto tv5 = add(tv3, tv4);

  // tv0[    i1, i2] = input
  // tv1[    i1, i2] = tv0/2.0
  // tv2[    i1    ] = sum(tv1, 1)
  // tv3[b0, i1    ] = bcast(tv2)
  // tv4[i0, i1    ] = input
  // tv5[i0, i1    ] = tv3 + tv4

  // tv2 = sum(tv0/2.0, 1)
  // tv5 = bcast(tv2) + tv4

  fusion.addInput(tv0);
  fusion.addInput(tv4);

  fusion.addOutput(tv5);

  tv5->merge(0);
  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv2, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({y, z}, options);
  at::Tensor t4 = at::randn({x, y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t4});
  auto cg_outputs = fe.runFusion({t0, t4});

  auto t1 = t0.div(2.0);
  auto t2 = t1.to(at::kDouble).sum(1);
  auto t3 = t2.unsqueeze(0).expand({x, y});
  auto aten_output = t3.add(t4);

  testValidate(
      &fusion, {cg_outputs}, {t0, t4}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto aten_output = t3.add(t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-2);
  tv4->merge(-2);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto aten_output = t3.add(t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.add(t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  TensorView* tv1 = makeConcreteTensor({4, 4, 8});
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv3 = broadcast(tv2, {true, false, false});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 8}, options);
  at::Tensor t1 = at::randn({4, 4, 8}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.add(t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(3);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv3 = broadcast(tv2, {true, false, true});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv3->merge(0)->merge(0)->split(0, 2)->split(0, 3);
  tv4->merge(0)->merge(0)->split(0, 2)->split(0, 3);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({7}, options);
  at::Tensor t1 = at::randn({5, 7, 11}, options);

  auto t2 = t0.add(1.0);
  auto aten_output = t2.unsqueeze(-1).add(t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> tensor0_shape{7, 4, 7};
  std::vector<int64_t> tensor1_shape{4, 7};

  TensorView* tv0 = makeSymbolicTensor(tensor0_shape.size());
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(tensor1_shape.size());
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  TensorView* tv3 = sum(tv2, {0, 1});
  fusion.addOutput(tv3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn(tensor0_shape, options);
  at::Tensor input1 = at::randn(tensor1_shape, options);

  std::vector<int64_t> reduction_axes{0, 1};
  auto reduction_params = getReductionHeuristics(&fusion, {input0, input1});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, reduction_params->lparams);
  auto cg_outputs = fe.runFusion({input0, input1}, reduction_params->lparams);

  auto aten_output = input0.add(input1).to(at::kDouble).sum(reduction_axes);

  testValidate(
      &fusion,
      cg_outputs,
      {input0, input1},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      reduction_params->lparams);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing7_CUDA) {
  // Might be able to use this one without 6 as the heuristics in 6 may change
  // and this test is to cover the same issue.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128);
  tv4->split(0, 4);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing8_CUDA) {
  // Same as 7 but with outer splits instead of inner
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128, false);
  tv4->split(0, 4, false);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing9_CUDA) {
  // Same as 7 but with outer splits instead of inner
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);

  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  const int numel_x = 200;
  const int numel_y = 300;
  const int numel_z = 400;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_y}, options);
  auto at_t3 = at::randn({numel_x, numel_y, numel_z}, options);
  std::vector<IValue> aten_inputs = {at_t0, at_t3};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  auto at_t1 = at_t0.unsqueeze(-1);
  auto at_t2 = at_t1.mul(2.0);

  auto at_t4 = at_t3.add(at_t2);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedIndexing10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  std::vector<TensorView*> tvs = {tv0_cache, tv1_cache, tv2, tv3};

  for (auto tv : tvs) {
    tv->split(1, 2, false);
    tv->split(1, 1);
    tv->split(-1, 4);
    // [I0, 2, 1, I1/2/4, 4]
    tv->reorder({{1, 2}, {2, 3}, {3, 1}});
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionAdvancedIndexing11_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = broadcast(tv2, {true, false, true, true});
  auto tv4 = add(tv3, tv0);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(1);

  tv4->split(1, 32);
  tv4->split(0, 1);

  tv4->reorder({{2, 1}});

  tv2->computeAt(tv4, 3);

  tv2->setMemoryType(MemoryType::Global);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDy);
  tv4->axis(2)->parallelize(ParallelType::Unswitch);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({w, x, y, z}, options);
  at::Tensor t1 = at::randn({x}, options);

  auto t3 = t1.add(1.0).unsqueeze(-1).unsqueeze(-1);
  auto aten_output = t3.add(t0);

  std::vector<IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

// Intended to stress the lowering of our code generator
TEST_F(NVFuserTest, FusionAdvancedLowering1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({9, 5});
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3));
  TensorView* tv4 = sum(tv3, {1});

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  auto tv5 = tv4->rFactor({2});

  tv1->computeAt(tv5, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor aten_input = at::randn({9, 5}, options);

  auto t1 = aten_input.add(1.0);
  auto t2 = t1.add(2.0);
  auto t3 = t1.add(3.0);
  auto t4 = t3.sum(1);

  std::vector<at::Tensor> aten_outputs = {t2, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedLowering2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Progressively broadcast tensors
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv4 = broadcast(tv3, {false, true});
  TensorView* tv5 = add(tv4, tv1);
  TensorView* tv6 = add(tv5, tv2);

  fusion.addOutput(tv6);

  // Split inner dimension
  tv6->split(1, 4);
  // Merge middle dims with outer dimensions
  tv6->merge(2);
  tv6->merge(0);

  // tv6[I0*I1o, I1i*I2]

  // Compute everything inline
  tv0->computeAt(tv6, -1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9, z = 5;
  at::Tensor t0 = at::randn({y}, options);
  at::Tensor t1 = at::randn({y, z}, options);
  at::Tensor t2 = at::randn({x, y, z}, options);

  auto t3 = t0.add(1.0);
  auto t4 = t3.unsqueeze(-1);
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);

  std::vector<IValue> aten_inputs = {t0, t1, t2};
  std::vector<at::Tensor> aten_outputs = {t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

// TODO: Complete test
TEST_F(NVFuserTest, FusionAdvancedLowering3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1, -1});
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [b0, i1]
  auto tv2 = add(tv0, IrBuilder::create<Double>(2.0));

  // [i0, i1]
  auto tv3 = add(tv1, IrBuilder::create<Double>(3.0));

  // [b0, i1]
  auto tv4 = add(tv2, IrBuilder::create<Double>(4.0));

  // [io, i1]
  auto tv5 = add(tv2, tv3);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv0->computeAt(tv4, -1);

  tv3->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9;
  at::Tensor t0 = at::randn({1, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);

  auto t4 = t0 + 2 + 4;
  auto t5 = t0 + 2 + t1 + 3;

  std::vector<IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t4, t5};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

// This excercises indexing with broadcast root axes. Non-broadcast
// axes need to be preferred when propagating index exprs to root
// axes. See, e.g., Index::getConsumerIndex_impl.
TEST_F(NVFuserTest, FusionAdvancedLowering4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->merge(1)->merge(0);
  tv4->split(0, 8);
  tv0->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 10;
  const int by = 20;
  const int bz = 30;
  at::Tensor t0 = at::randn({bx}, options);
  at::Tensor t3 = at::randn({bx, by, bz}, options);
  std::vector<IValue> aten_inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output =
      t0.unsqueeze(-1).expand({bx, by}).unsqueeze(-1).expand({bx, by, bz}) + t3;

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedLowering5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);

  TensorView* tv1 = makeConcreteTensor({5, 3});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true, false});

  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv2->merge(0);
  tv1->computeAt(tv2, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({5, 3}, options);
  auto t2 = t1.unsqueeze(1);
  auto t3 = t0 + t2;

  std::vector<IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedLowering6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({4});
  fusion.addInput(tv1);
  auto tv2 = unaryOp(UnaryOpType::Set, tv0);
  auto tv3 = unaryOp(UnaryOpType::Set, tv1);

  auto tv4 = sum(tv2, {0, 2});
  auto tv5 = add(tv4, tv3);
  fusion.addOutput(tv5);

  auto tv6 = broadcast(tv3, {true, false, true});
  auto tv7 = add(tv2, tv6);
  fusion.addOutput(tv7);

  tv2->computeAt(tv4, -1, ComputeAtMode::BestEffort);
  tv3->computeAt(tv7, -1, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(1);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({4}, options);

  auto t2 = t0;
  auto t3 = t1;

  std::vector<int64_t> reduction_axes{0, 2};
  auto t4 = t2.sum(reduction_axes);
  auto t5 = add(t4, t3);
  auto t6 = t3.unsqueeze(0).unsqueeze(-1);
  auto t7 = t2.add(t6);

  std::vector<IValue> aten_inputs = {t0, t1};
  std::vector<at::Tensor> aten_outputs = {t5, t7};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

// Test a simple Gemm but also play around with fusion executor features
TEST_F(NVFuserTest, FusionSimpleGemm_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2); // M, K
  TensorView* tv1 = makeSymbolicTensor(2); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // tv2[I0, I1, B] = tv0[I0, I1]

  TensorView* tv3 = broadcast(tv1, {true, false, false});
  // tv3[B, I1, I2] = tv1[I1, I2]

  // tv4[I0, I1, I2] = tv2[I0, I1, B] * tv3[B, I1, I2]
  TensorView* tv4 = mul(tv2, tv3);
  // tv5[I0, R1, I2] = tv4[I0, I1, I2]
  TensorView* tv5 = sum(tv4, {1});
  fusion.addOutput(tv5);

  tv5->split(1, 32);
  // tv5[I0, R1o, R1i{32}, I2]

  auto tv6 = tv5->rFactor({1});
  // tv6[I0, R1o, I1i{32}, I2] = tv4[I0, I1, I2]
  // tv5[I0,    , R1i{32}, I2] = tv6[I0, R1o, I1i{32}, I2]

  tv5->split(0, 4);
  tv5->split(-1, 4);
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]

  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  // tv6[I0o, I0i{4}, R1o, I1i{32}, I2o, I2i{4}]
  // tv5[I0o, I0i{4},    , R1i{32}, I2o, I2i{4}]
  //--> (line symbolizes compute at location)
  // tv4[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, I1o]
  // tv6[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, R1o]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv0->computeAt(tv6, -1);
  tv1->computeAt(tv6, -1);
  // tv4[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, I1o |]
  // tv6[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, R1o |]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::TIDz);

  tv5->axis(-2)->parallelize(ParallelType::BIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);

  tv5->axis(2)->parallelize(ParallelType::TIDx);
  tv6->axis(2)->parallelize(ParallelType::TIDx);

  constexpr int M = 65, K = 33, N = 17;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(1, -1, -1, 32, 4, 4));
  // Lets specify a few bounds in launch params to make sure it works
  fe.runFusion({t0, t1}, LaunchParams(1, -1, -1, 32, 4, 4));

  // Make sure bad launch params throws
  // TODO: Re-enable once we have parallelization validation in.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // ASSERT_ANY_THROW(fe.runFusion({t0, t1}, LaunchParams(1, 2, 3, 4, 5, 6)));

  // Don't specify any launch params
  auto cg_outputs = fe.runFusion({t0, t1});

  auto aten_output = t0.to(at::kDouble).matmul(t1.to(at::kDouble));

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

// Softmax with a 1D tensor. Parallelized only with a single thread block.
TEST_F(NVFuserTest, FusionSoftmax1D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 128;
  const int dimx = 1000;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(1);
  fusion.addInput(input_tv0);

  TensorView* exp_tv1 = unaryOp(UnaryOpType::Exp, input_tv0);
  TensorView* sum_exp_tv2 = sum(exp_tv1, {-1});
  TensorView* bcast_sum_tv3 = broadcast(sum_exp_tv2, {true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* exp_tv1_copy = unaryOp(UnaryOpType::Exp, input_tv0);

  TensorView* output_tv4 = div(exp_tv1_copy, bcast_sum_tv3);

  fusion.addOutput(output_tv4);

  bcast_sum_tv3->split(0, tidx);

  sum_exp_tv2->split(-1, tidx);
  TensorView* sum_exp_rf_tv5 = sum_exp_tv2->rFactor({-2});

  output_tv4->split(-1, tidx);

  exp_tv1->computeAt(sum_exp_rf_tv5, -1);
  exp_tv1_copy->computeAt(output_tv4, -1);

  TensorView* tensors_to_parallelize[] = {
      sum_exp_tv2, bcast_sum_tv3, output_tv4, sum_exp_rf_tv5};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dimx}, options);
  at::Tensor cg_output = at::empty({dimx}, options);
  at::Tensor t3_output = at::empty_like(cg_output, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  fe.runFusion({t0}, {cg_output});

  auto aten_output = at::_softmax(t0.to(at::kDouble), -1, false);

  testValidate(&fusion, {cg_output}, {t0}, {aten_output}, __LINE__, __FILE__);
}

// Softmax with a 1D tensor with input normalization.
TEST_F(NVFuserTest, FusionSoftmax1DNormalized_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 128;
  const int dimx = 1000;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(1);
  fusion.addInput(input_tv0);

  // Normalize with the max value before computing exp.
  TensorView* max_val_tv1 = reductionOp(
      BinaryOpType::Max, {-1}, IrBuilder::create<Double>(0), input_tv0);
  TensorView* bcast_max_tv2 = broadcast(max_val_tv1, {true});
  TensorView* sub_tv3 = sub(input_tv0, bcast_max_tv2);
  TensorView* exp_tv4 = unaryOp(UnaryOpType::Exp, sub_tv3);
  TensorView* sum_exp_tv5 = sum(exp_tv4, {-1});
  TensorView* bcast_sum_tv6 = broadcast(sum_exp_tv5, {true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* sub_tv3_copy = sub(input_tv0, bcast_max_tv2);
  TensorView* exp_tv4_copy = unaryOp(UnaryOpType::Exp, sub_tv3_copy);

  TensorView* output_tv7 = div(exp_tv4_copy, bcast_sum_tv6);

  fusion.addOutput(output_tv7);
  bcast_max_tv2->split(0, tidx);
  bcast_sum_tv6->split(0, tidx);

  max_val_tv1->split(-1, tidx);
  TensorView* max_val_rf_tv8 = max_val_tv1->rFactor({-2});

  sum_exp_tv5->split(-1, tidx);
  TensorView* sum_exp_rf_tv9 = sum_exp_tv5->rFactor({-2});

  output_tv7->split(-1, tidx);

  sub_tv3->computeAt(sum_exp_rf_tv9, -1);
  sub_tv3_copy->computeAt(output_tv7, -1);

  TensorView* tensors_to_parallelize[] = {
      max_val_tv1,
      bcast_max_tv2,
      sum_exp_tv5,
      bcast_sum_tv6,
      output_tv7,
      max_val_rf_tv8,
      sum_exp_rf_tv9};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({dimx}, options);
  at::Tensor t3_output = at::empty({dimx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = at::_softmax(input.to(at::kDouble), -1, false);

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// Softmax with a 3D tensor, where the inner-most 3rd dimension is
// normalized. Pallelized with multiple thread blocks.
TEST_F(NVFuserTest, FusionSoftmax3D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(3);
  fusion.addInput(input_tv0);

  TensorView* exp_tv1 = unaryOp(UnaryOpType::Exp, input_tv0);
  TensorView* sum_exp_tv2 = sum(exp_tv1, {-1});
  TensorView* bcast_sum_tv3 = broadcast(sum_exp_tv2, {false, false, true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* exp_tv1_copy = unaryOp(UnaryOpType::Exp, input_tv0);

  TensorView* output_tv4 = div(exp_tv1_copy, bcast_sum_tv3);

  fusion.addOutput(output_tv4);

  bcast_sum_tv3->split(-1, tidx);

  sum_exp_tv2->split(-1, tidx);
  TensorView* sum_exp_rf_tv5 = sum_exp_tv2->rFactor({-2});

  output_tv4->split(-1, tidx);

  exp_tv1->computeAt(sum_exp_rf_tv5, -1);
  exp_tv1_copy->computeAt(output_tv4, -1);

  TensorView* tensors_to_parallelize[] = {
      sum_exp_tv2, bcast_sum_tv3, output_tv4, sum_exp_rf_tv5};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({dimx, dimy, dimz}, options);

  at::Tensor cg_output = at::empty({dimx, dimy, dimz}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = at::_softmax(input.to(at::kDouble), -1, false);

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

// Softmax with a 3D tensor with input normalization.
TEST_F(NVFuserTest, FusionSoftmax3DNormalized_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(3);
  fusion.addInput(input_tv0);

  // Normalize with the max value before computing exp.
  TensorView* max_val_tv1 = reductionOp(
      BinaryOpType::Max, {-1}, IrBuilder::create<Double>(0), input_tv0);
  TensorView* bcast_max_tv2 = broadcast(max_val_tv1, {false, false, true});
  TensorView* sub_tv3 = sub(input_tv0, bcast_max_tv2);
  TensorView* exp_tv4 = unaryOp(UnaryOpType::Exp, sub_tv3);
  TensorView* sum_exp_tv5 = sum(exp_tv4, {-1});
  TensorView* bcast_sum_tv6 = broadcast(sum_exp_tv5, {false, false, true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* sub_tv3_copy = sub(input_tv0, bcast_max_tv2);
  TensorView* exp_tv4_copy = unaryOp(UnaryOpType::Exp, sub_tv3_copy);

  TensorView* output_tv7 = div(exp_tv4_copy, bcast_sum_tv6);

  fusion.addOutput(output_tv7);

  bcast_max_tv2->split(-1, tidx);
  bcast_sum_tv6->split(-1, tidx);

  max_val_tv1->split(-1, tidx);
  TensorView* max_val_rf_tv8 = max_val_tv1->rFactor({-2});

  sum_exp_tv5->split(-1, tidx);
  TensorView* sum_exp_rf_tv9 = sum_exp_tv5->rFactor({-2});

  output_tv7->split(-1, tidx);

  sub_tv3->computeAt(sum_exp_rf_tv9, -1);
  sub_tv3_copy->computeAt(output_tv7, -1);

  TensorView* tensors_to_parallelize[] = {
      max_val_tv1,
      bcast_max_tv2,
      sum_exp_tv5,
      bcast_sum_tv6,
      output_tv7,
      max_val_rf_tv8,
      sum_exp_rf_tv9};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({dimx, dimy, dimz}, options);
  at::Tensor t3_output = at::empty({dimx, dimy, dimz}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = at::_softmax(input.to(at::kDouble), -1, false);

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSoftmaxComputeAt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});

  auto tv3 = add(tv0, IrBuilder::create<Double>(1.0));

  auto tv4 = mul(tv2, tv3);

  auto tv5 = sum(tv4, {1});
  auto tv6 = broadcast(tv5, {false, true});

  auto tv7 = sub(tv6, tv4);
  fusion.addOutput(tv7);

  tv1->computeAt(tv7, 1);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv1->computeAt(tv7, -1));
}

// Similar to FusionReduction but uses grid reduction
TEST_F(NVFuserTest, FusionGridReduction1_CUDA) {
  const int gdimx = 32;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, bdimx);
  // tv1[I0, R1o, R1i{128}] = tv0[I0, I1]
  tv1->split(1, gdimx);
  // tv1[I0, R1oo, R1oi{32}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}] = tv0[I0, I1]
  // tv1[I0,        R1oi{32},  R1i{128}] = tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::BIDx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 10000;
  int numel_y = 65000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

// Same test as the above but uses BIDy and TIDx for reduction
TEST_F(NVFuserTest, FusionGridReduction2_CUDA) {
  const int gdimy = 32;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, bdimx);
  // tv1[I0, R1o, R1i{128}] = tv0[I0, I1]
  tv1->split(1, gdimy);
  // tv1[I0, R1oo, R1oi{32}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}] = tv0[I0, I1]
  // tv1[I0,        R1oi{32},  R1i{128}] = tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::BIDy);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 10000;
  int numel_y = 65000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// Same test but uses BIDy and BIDz for reduction. No TID used.
TEST_F(NVFuserTest, FusionGridReduction3dim1_CUDA) {
  // Grid reductions when there aren't any threads are serial reductions
  // keep these numbers low so our error isn't too high compared to normal cuda
  // reductions
  const int gdimz = 15;
  const int gdimy = 9;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, gdimy);
  // tv1[I0, R1o, R1i{128}] = tv0[I0, I1]
  tv1->split(1, gdimz);
  // tv1[I0, R1oo, R1oi{32}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}] = tv0[I0, I1]
  // tv1[I0,        R1oi{32},  R1i{128}] = tv2[I0, R1oo, Ir1oi{32}, Ir1i{128}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDz);
  tv2->axis(2)->parallelize(ParallelType::BIDz);
  tv1->axis(-1)->parallelize(ParallelType::BIDy);
  tv2->axis(-1)->parallelize(ParallelType::BIDy);

  int numel_x = 100;
  int numel_y = 6500;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

// Same as testGPU_FusionGridReduction3dim1 but reduces dimension 0
TEST_F(NVFuserTest, FusionGridReduction3dim0_CUDA) {
  // Grid reductions when there aren't any threads are serial reductions
  // keep these numbers low so our error isn't too high compared to normal cuda
  // reductions
  const int gdimz = 15;
  const int gdimy = 9;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[R0, I1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {0}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(0, gdimy);
  // tv1[R0o, R0i{128}, I1] = tv0[I0, I1]
  tv1->split(0, gdimz);
  // tv1[R0oo, R0oi{32}, R0i{128}, I1] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({0});
  // tv2[R0oo, I0oi{32}, I0i{128}, I1] = tv0[I0, I1]
  // tv1[      R0oi{32}, R0i{128}, I1] = tv2[R0oo, I0oi{32}, I0i{128}, I1]

  // Note that computeAt isn't going to make anything better as there
  // is no dynamically sized dimension.

  // Map parallelism as [Serial, BIDz, BIDy, BIDx]
  tv1->axis(-1)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::BIDx);
  tv1->axis(-2)->parallelize(ParallelType::BIDy);
  tv2->axis(-2)->parallelize(ParallelType::BIDy);
  tv1->axis(-3)->parallelize(ParallelType::BIDz);
  tv2->axis(-3)->parallelize(ParallelType::BIDz);

  int numel_x = 6500;
  int numel_y = 100;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({0});

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// This is similar to the FusionReduction, but swaps BIDx and TIDx
TEST_F(NVFuserTest, FusionGridReduction4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 128;
  const int gdimx = 1024;

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, gdimx);
  // tv1[I0, R1o, R1i{1024}] = tv0[I0, I1]
  tv1->split(1, 4);
  // tv1[I0, R1oo, R1oi{4}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{1024}] = tv0[I0, I1]
  // tv1[I0,        R1oi{4},  R1i{1024}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{1024}]

  TensorView* tv3 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{1024}] = tv0[I0, I1]
  // tv3[I0,        R1oi{4}, Ir1i{1024}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{1024}]
  // tv1[I0,                  R1i{1024}] = tv3[I0,        R1oi{4}, Ir1i{1024}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv3, 1);
  tv3->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv2->axis(2)->parallelize(ParallelType::Unroll);
  tv1->axis(0)->parallelize(ParallelType::TIDx);

  tv1->axis(-1)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::BIDx);

  int numel_x = bdimx;
  int numel_y = 65000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

// Grid reduction with 2D thread blocks but only TIDx and BIDx are
// mapped to a reduction dim
TEST_F(NVFuserTest, FusionGridReduction5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 64;
  const int bdimy = 16;
  const int gdimx = 4;

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  tv1->split(1, bdimx);
  // tv1[I0, R1o, R1i{64}] = tv0[I0, I1]
  tv1->split(1, gdimx);
  // tv1[I0, R1oo, R1oi{4}, R1i{64}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{64}] = tv0[I0, I1]
  // tv1[I0,        R1oi{4},  R1i{64}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{64}]

  tv0->computeAt(tv1, 1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv2->axis(-2)->parallelize(ParallelType::BIDx);

  tv1->axis(0)->parallelize(ParallelType::TIDy);

  int numel_x = bdimy;
  int numel_y = 6500;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// Similar to FusionGridReduction1 but with 3D tensors
TEST_F(NVFuserTest, FusionGridReduction6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  // tv1[I0, R1, R2] = tv0[I0, I1, I2]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1, 2}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion).size(),
      "Could not detect reduction in fusion.");

  // Splitting for TID
  tv1->split(2, 128);
  // tv1[I0, R1, R2o, R2i{128}] = tv0[I0, I1, I2]

  // Splitting for BID
  tv1->split(1, 128);

  // tv1[I0, R1o, R1i{128}, R2o, R2i{128}] = tv0[I0, I1, I2]

  TensorView* tv2 = tv1->rFactor({3});
  // tv2[I0, I1o, I1i{128}, R2o, I2i{128}]
  // tv1[I0, R1o, R1i{128},      R2i{128}]

  TensorView* tv3 = tv1->rFactor({1});
  // tv2[I0, I1o, I1i{128}, R2o, I2i{128}]
  // tv3[I0, R1o, I1i{128},      I2i{128}]
  // tv1[I0,      R1i{128},      R2i{128}]

  tv3->computeAt(tv1, 1);
  tv2->computeAt(tv3, 3);

  tv1->axis(0)->parallelize(ParallelType::BIDy);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::BIDx);

  int numel_x = 6500;
  int numel_y = 200;
  int numel_z = numel_y;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y, numel_z}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({1, 2});

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

// See issue #1049
TEST_F(NVFuserTest, FusionGridReduction7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  tv1->split(0, 1000);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);

  const int numel_x = 1;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = input.sum({0});

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReduction8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  const int numel_x = 2;
  const int numel_y = 4;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = input.sum({0});

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReduction9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv1->split(1, 2);

  tv1->axis(1)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::BIDy);

  tv1->computeAt(tv3, 1);

  const int numel_x = 4;
  const int numel_y = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t2 = at::randn({numel_x}, options);

  std::vector<IValue> aten_inputs = {t0, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_output = fe.runFusion(aten_inputs);

  auto aten_output = t0.sum({1}).add(t2);

  testValidate(&fusion, cg_output, {t0, t2}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReduction10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {-1});
  auto tv2 = sum(tv1, {-1});
  auto tv3 = sum(tv2, {-1});

  fusion.addOutput(tv3);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(3)->parallelize(ParallelType::TIDz);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv3, 1);

  const int numel_w = 2;
  const int numel_x = 3;
  const int numel_y = 4;
  const int numel_z = 5;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_w, numel_x, numel_y, numel_z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_output = fe.runFusion({t0});

  auto aten_output = t0.sum({1, 2, 3});

  testValidate(&fusion, cg_output, {t0}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionNonRedAxisBind_CUDA) {
  int bid_x = 3;
  int tid_x = 2;
  int red_dim = 0;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, {red_dim}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  tv1->split(-1, tid_x);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({16, bid_x * tid_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = input.to(at::kDouble).sum({red_dim});

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSplitBCast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(3);
  TensorView* input_tv1 = makeSymbolicTensor(3);
  fusion.addInput(input_tv0);
  fusion.addInput(input_tv1);

  TensorView* sum_tv2 = reductionOp(
      BinaryOpType::Add, {2}, IrBuilder::create<Double>(0), input_tv0);
  TensorView* bcast_tv3 = broadcast(sum_tv2, {false, false, true});
  TensorView* output_tv4 = div(input_tv1, bcast_tv3);

  sum_tv2->split(-1, 32);
  TensorView* sum_rf_tv5 = sum_tv2->rFactor({-2});

  bcast_tv3->split(-1, 32);
  output_tv4->split(-1, 32);

  sum_rf_tv5->axis(0)->parallelize(ParallelType::BIDx);
  sum_tv2->axis(0)->parallelize(ParallelType::BIDx);
  bcast_tv3->axis(0)->parallelize(ParallelType::BIDx);
  output_tv4->axis(0)->parallelize(ParallelType::BIDx);

  sum_rf_tv5->axis(1)->parallelize(ParallelType::BIDy);
  sum_tv2->axis(1)->parallelize(ParallelType::BIDy);
  bcast_tv3->axis(1)->parallelize(ParallelType::BIDy);
  output_tv4->axis(1)->parallelize(ParallelType::BIDy);

  sum_rf_tv5->axis(-1)->parallelize(ParallelType::TIDx);
  sum_tv2->axis(-1)->parallelize(ParallelType::TIDx);
  bcast_tv3->axis(-1)->parallelize(ParallelType::TIDx);
  output_tv4->axis(-1)->parallelize(ParallelType::TIDx);

  fusion.addOutput(output_tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32, 128}, options);
  at::Tensor t1 = at::randn({32, 32, 128}, options);
  at::Tensor cg_output = at::empty({32, 32, 128}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  fe.runFusion({t0, t1}, {cg_output});
}

TEST_F(NVFuserTest, FusionBCastInnerDim_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // reduce then broadcast
  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {false, true});

  TORCH_CHECK(!tv2->axis(0)->isReduction() && tv2->axis(1)->isBroadcast());
}

TEST_F(NVFuserTest, FusionBCastReduce_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);

  auto tv1 = broadcast(tv0, {true, false, false});
  auto tv2 = sum(tv1, {1});
  TORCH_CHECK(
      tv2->axis(0)->isBroadcast() && tv2->axis(1)->isReduction() &&
      !tv2->axis(2)->isBroadcast() && !tv2->axis(2)->isReduction());
}

// Multiple consumer reduction with computeAt
// https://github.com/csarofeen/pytorch/issues/110
TEST_F(NVFuserTest, FusionReductionMultiConsumer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = unaryOp(UnaryOpType::Exp, tv0);
  auto tv2 =
      reductionOp(BinaryOpType::Max, {-1}, IrBuilder::create<Double>(0), tv1);
  auto tv3 =
      reductionOp(BinaryOpType::Min, {-1}, IrBuilder::create<Double>(0), tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);
  tv1->computeAt(tv2, -1, ComputeAtMode::BestEffort);

  TORCH_CHECK(tv1->getComputeAtPosition() == 2);
}

TEST_F(NVFuserTest, FusionComputeAtExprOrder1_CUDA) {
  for (const auto i : c10::irange(2)) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeSymbolicTensor(1);
    fusion.addInput(tv0);

    auto tv1 = add(tv0, IrBuilder::create<Double>(1));
    auto tv2 = add(tv0, IrBuilder::create<Double>(1));
    TensorView* tv3 = add(tv1, tv2);
    // Set outputs tv2 or tv1 and then tv3
    if (i == 0) {
      fusion.addOutput(tv2);
    } else {
      fusion.addOutput(tv1);
    }
    fusion.addOutput(tv3);

    if (i == 0) {
      tv1->computeAt(tv3, -1);
    } else {
      tv2->computeAt(tv3, -1);
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor aten_input = at::randn({100}, options);
    std::vector<at::Tensor> aten_outputs = {
        aten_input + 1, (aten_input + 1) * 2};

    FusionExecutor fe;
    fe.compileFusion(&fusion, {aten_input});
    auto cg_outputs = fe.runFusion({aten_input});

    testValidate(
        &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionComputeAtExprOrder2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->split(-1, 32);

  tv1->computeAt(tv3, -1);
  tv2->computeAt(tv3, -2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100, 100}, options);
  auto aten_output = (aten_input + 1) * 2;

  at::Tensor cg_output = at::empty_like(aten_input, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeAtExprOrder3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t dimx = 13;
  const int64_t dimy = 15;

  TensorView* tv0 = makeConcreteTensor({dimx, dimy});
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv2, IrBuilder::create<Double>(3));
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));
  TensorView* tv5 = mul(tv2, tv4);
  fusion.addOutput(tv5);

  tv1->computeAt(tv2, 2);
  tv3->computeAt(tv4, 1);
  tv4->computeAt(tv5, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({dimx, dimy}, options);
  auto t1 = aten_input.add(1.);
  auto t2 = t1.add(2.);
  auto t3 = t2.add(3.);
  auto t4 = t3.add(4.);
  auto aten_output = t2.mul(t4);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionZeroDimComputeAt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);
  TORCH_CHECK(tv2->nDims() == 0);
  tv1->computeAt(tv2, 0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);
  auto aten_output = aten_input.to(at::kDouble).sum() + 1;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionZeroDimBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, true});
  TORCH_CHECK(tv1->nDims() == 2);

  TensorView* tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv3->computeAt(tv4, -1);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);
  at::Tensor t1 = at::randn({10, 10}, options);

  auto aten_output = (t0.unsqueeze(-1).unsqueeze(-1).expand({10, 10}) + t1)
                         .to(at::kDouble)
                         .sum();

  std::vector<IValue> aten_inputs = {t0, t1};
  at::Tensor cg_output = at::empty({}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, {cg_output});

  testValidate(
      &fusion, {cg_output}, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionZeroDimReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 32;
  const int gdimx = 32;

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  tv1->split(0, bdimx);
  tv1->split(0, gdimx);
  auto tv2 = tv1->rFactor({0});

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv2->axis(-2)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({1000}, options);
  auto aten_output = aten_input.to(at::kDouble).sum();

  at::Tensor cg_output = at::empty({}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBCastAfterReduce_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  const int tidx = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});

  tv1->split(1, tidx);
  auto tv3 = tv1->rFactor({-2});

  TensorView* tv4 = makeSymbolicTensor(2);
  fusion.addInput(tv4);

  auto tv5 = add(tv2, tv4);
  fusion.addOutput(tv5);
  tv5->split(1, tidx);

  tv3->computeAt(tv5, 1);

  tv2->split(1, tidx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  tv5->axis(0)->parallelize(ParallelType::BIDx);

  int x = 63, y = 200;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t4 = at::randn({x, y}, options);

  auto t3 = t0.to(at::kDouble).sum({1}).unsqueeze(-1).expand({x, y});
  auto aten_output = t3.add(t4);

  std::vector<IValue> aten_inputs = {t0, t4};
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t4});
  auto cg_outputs = fe.runFusion({t0, t4});

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionOutputBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({2, 3});
  fusion.addInput(tv0);

  TensorView* tv1 = broadcast(tv0, {true, false, true, false, true});

  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({2, 3}, options);
  auto aten_output = aten_input.unsqueeze(2).unsqueeze(1).unsqueeze(0);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionKeepDimBasic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({2, 3, 4, 5, 6});
  fusion.addInput(tv0);

  TensorView* tv1 = sum(tv0, {0, 2, -1}, /*keep_dim=*/true);

  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({2, 3, 4, 5, 6}, options);
  auto aten_output =
      aten_input.to(at::kDouble).sum({0, 2, -1}, /*keepdim=*/true);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionKeepDimScheduler_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x});
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add,
      {red_dim},
      IrBuilder::create<Double>(0),
      tv0,
      /*keep_dim=*/true);

  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({bid_x, tid_x}, options);
  auto aten_output =
      aten_input.to(at::kDouble).sum({red_dim}, /*keepdim=*/true);

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionSumTo_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> tensor_shape{2, 3, 4, 5, 6};
  std::vector<int64_t> sum_to_shape{1, 5, 6};

  std::vector<int64_t> tensor_shape_ref{2, 3, 4, 5, 6};
  std::vector<int64_t> sum_to_shape_ref{1, 5, 6};

  std::vector<Int*> sum_to_symb;
  std::transform(
      sum_to_shape.begin(),
      sum_to_shape.end(),
      std::back_inserter(sum_to_symb),
      [](int s) -> Int* { return IrBuilder::create<Int>(s); });

  TensorView* tv0 = makeConcreteTensor(tensor_shape);
  fusion.addInput(tv0);

  TensorView* tv1 = sum_to(tv0, sum_to_symb);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn(tensor_shape_ref, options);
  auto aten_output = at::sum_to(aten_input.to(at::kDouble), sum_to_shape_ref);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  TORCH_CHECK(
      cg_outputs[0].dim() == static_cast<int64_t>(sum_to_shape.size()),
      "sum_to not keeping the final dimension");

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSumToNoop_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> tensor_shape{4, 5, 6};
  std::vector<int64_t> sum_to_shape{4, 5, 6};

  std::vector<int64_t> tensor_shape_ref{4, 5, 6};
  std::vector<int64_t> sum_to_shape_ref{4, 5, 6};

  std::vector<Int*> sum_to_symb;
  std::transform(
      sum_to_shape.begin(),
      sum_to_shape.end(),
      std::back_inserter(sum_to_symb),
      [](int s) -> Int* { return IrBuilder::create<Int>(s); });

  TensorView* tv0 = makeConcreteTensor(tensor_shape);
  fusion.addInput(tv0);

  TensorView* tv1 = sum_to(tv0, sum_to_symb);

  // Dummy operator to avoid tv0 both input and output
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(0));
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn(tensor_shape_ref, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});
  auto aten_output = at::sum_to(aten_input.to(at::kDouble), sum_to_shape_ref);

  TORCH_CHECK(
      cg_outputs[0].dim() == static_cast<int64_t>(sum_to_shape.size()),
      "sum_to not keeping the final dimension");

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionScheduler_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, {red_dim}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({bid_x, tid_x}, options);
  auto aten_output = aten_input.to(at::kDouble).sum({red_dim});

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

// This test checks if our system could correctly handles the case where both
// reduction and trivial reduction exist in the fusion. Trivial reduction
// deserve testing because trivial reduction is handled more like a broadcasting
// rather than a reduction.
TEST_F(NVFuserTest, FusionReductionWithTrivialReduction_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;

  std::vector<std::vector<int64_t>> shapes = {
      {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1}};

  for (auto shape : shapes) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr;
    FusionGuard fg(&fusion);

    std::vector<std::vector<int64_t>> reduction_dims = {
        {0},
        {1},
        {2},
        {0, 1},
        {0, 2},
        {1, 2},
        {0, 1, 2},
    };

    // Set up your input tensor views
    TensorView* tv0 = makeConcreteTensor(shape);
    fusion.addInput(tv0);

    for (auto rdims : reduction_dims) {
      std::vector<int> rdims_(rdims.begin(), rdims.end());
      auto tv = sum(tv0, rdims_);
      fusion.addOutput(tv);
    }

    const auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    auto concrete_shape = shape;
    std::deque<int64_t> concrete_values = {bid_x, tid_x};
    for (auto& s : concrete_shape) {
      if (s == -1) {
        s = concrete_values.front();
        concrete_values.pop_front();
      }
    }

    at::Tensor aten_input = at::randn(concrete_shape, options);
    std::vector<at::Tensor> aten_outputs;
    for (auto rdims : reduction_dims) {
      aten_outputs.push_back(aten_input.sum(rdims));
    }

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({aten_input});

    testValidate(
        &fusion,
        cg_outputs,
        {aten_input},
        aten_outputs,
        __LINE__,
        __FILE__,
        "");
  }
}

// Simple reduction parallelized on a symbolic size.
TEST_F(NVFuserTest, FusionSymbolicReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  // Interface should just be a direct split with a Parallel type. We can
  // include the parallelize call if we do this.
  tv1->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
  // tv1[I0, R1o, R1i{BIDx}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{BIDx}] = tv0[I0, I1]
  // tv1[I0,        R1oi{4},  R1i{BIDx}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{BIDx}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv1, 1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 65000;
  int numel_y = 1025;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);
  auto aten_output = aten_input.to(at::kDouble).sum({1});

  // How many threads to use for the block reduction
  int runtime_threadIdx_dim = 128;

  LaunchParams lparams(-1, -1, -1, runtime_threadIdx_dim, -1, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionReductionSchedulerMultiDimNonFastest_CUDA) {
  const std::vector<int> red_dims = {0, 2};
  // Copy is because CodeGen requires int and Pytorch requires int64_t
  // for a vector of reduction dimensions
  const std::vector<int64_t> red_dims64 = {0, 2};
  const std::vector<int64_t> tensor_dims_in = {5, 10, 15, 20};
  const std::vector<int64_t> tensor_dims_out = {10, 20};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, red_dims, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(tensor_dims_in, options);
  auto aten_output = aten_input.to(at::kDouble).sum(red_dims64);
  at::Tensor cg_output = at::empty(tensor_dims_out, options);

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  fe.runFusion({aten_input}, {cg_output}, lparams);

  testValidate(
      &fusion,
      {cg_output},
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionReductionSchedulerMultiDimFastest_CUDA) {
  const std::vector<int> red_dims = {1, 3};
  // Copy is because CodeGen requires int and Pytorch requires int64_t
  // for a vector of reduction dimensions
  const std::vector<int64_t> red_dims64 = {1, 3};
  const std::vector<int64_t> tensor_dims_in = {5, 10, 15, 20};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, red_dims, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(tensor_dims_in, options);
  auto aten_output = aten_input.to(at::kDouble).sum(red_dims64);

  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionReductionSchedulerNoODimShmoo_CUDA) {
  std::vector<DataType> dtypes = {
      DataType::Double, DataType::Float, DataType::Half};
  // TODO: add test for complex. Currently complex fails with the following
  // NVRTC compilation error message:
  //   error: no suitable user-defined conversion from
  //   "CudaCodeGen::std::complex<double>" to "CudaCodeGen::std::complex<float>"
  //   exists
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (at::cuda::getDeviceProperties(0)->major >= 8) {
    dtypes.insert(dtypes.end(), DataType::BFloat16);
  }
#endif

  std::vector<int> red_dims;

  // Tried to cut down the number iterations with just
  // doing every other power of 2.
  for (int i = 1; i <= 1024 * 1024; i <<= 2) {
    red_dims.push_back(i);
  }

  for (auto dtype : dtypes) {
    at::ScalarType aten_dtype = data_type_to_aten(dtype);
    for (auto& rdim : red_dims) {
      Fusion fusion;
      FusionGuard fg(&fusion);

      bool is_fp16 = dtype == DataType::Half;
      bool is_bf16 = dtype == DataType::BFloat16;

      TensorView* tv0 = makeSymbolicTensor(1, dtype);
      fusion.addInput(tv0);

      TensorView* tv0_cast = tv0;
      if (is_fp16 || is_bf16) {
        tv0_cast = castOp(DataType::Float, tv0);
      }

      TensorView* tv1 = sum(tv0_cast, {0});

      TensorView* tv1_cast = tv1;
      if (is_fp16) {
        tv1_cast = castOp(DataType::Half, tv1);
      }
      if (is_bf16) {
        tv1_cast = castOp(DataType::BFloat16, tv1);
      }

      fusion.addOutput(tv1_cast);

      auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);

      at::Tensor aten_input = at::randn({rdim}, options);
      auto aten_output = aten_input.to(at::kDouble).sum({0});

      auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
      TORCH_CHECK(reduction_params != nullptr, "Reduction is not found!");
      scheduleReduction(&fusion, *reduction_params);
      auto lparams = reduction_params->lparams;

      FusionExecutor fe;
      fe.compileFusion(&fusion, {aten_input}, lparams);
      auto cg_outputs = fe.runFusion({aten_input}, lparams);

      testValidate(
          &fusion,
          cg_outputs,
          {aten_input},
          {aten_output},
          __LINE__,
          __FILE__,
          "",
          lparams);
    }
  }
}

TEST_F(NVFuserTest, FusionReductionSchedulerDimShmoo_CUDA) {
  std::vector<DataType> dtypes = {
      DataType::Double, DataType::Float, DataType::Half};
  // TODO: add complex support. Currently, complex fails with the following
  // NVRTC compilation error:
  //   error: no instance of overloaded function "__shfl_xor_sync" matches the
  //   argument list
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (at::cuda::getDeviceProperties(0)->major >= 8) {
    dtypes.insert(dtypes.end(), DataType::BFloat16);
  }
#endif

  std::vector<int> red_axis = {1, 0};
  std::vector<int> output_dims = {160, 320};
  std::vector<int> red_dims;

  // Tried to cut down the number iterations with just
  // doing every other power of 2.
  for (int i = 1; i <= 1024 * 1024; i <<= 2) {
    red_dims.push_back(i);
  }

  for (auto dtype : dtypes) {
    at::ScalarType aten_dtype = data_type_to_aten(dtype);
    for (auto& axis : red_axis) {
      for (auto& odim : output_dims) {
        for (auto& rdim : red_dims) {
          Fusion fusion;
          FusionGuard fg(&fusion);

          bool is_fp16 = dtype == DataType::Half;
          bool is_bf16 = dtype == DataType::BFloat16;

          TensorView* tv0 = makeSymbolicTensor(2, dtype);
          fusion.addInput(tv0);

          TensorView* tv0_cast = tv0;
          if (is_fp16 || is_bf16) {
            tv0_cast = castOp(DataType::Float, tv0);
          }

          TensorView* tv1 = sum(tv0_cast, {axis});

          TensorView* tv1_cast = tv1;
          if (is_fp16) {
            tv1_cast = castOp(DataType::Half, tv1);
          }
          if (is_bf16) {
            tv1_cast = castOp(DataType::BFloat16, tv1);
          }
          fusion.addOutput(tv1_cast);

          auto options =
              at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);

          at::Tensor aten_input =
              (axis ? at::randn({odim, rdim}, options)
                    : at::randn({rdim, odim}, options));

          auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
          TORCH_CHECK(reduction_params != nullptr, "Reduction is not found!");
          scheduleReduction(&fusion, *reduction_params);
          auto lparams = reduction_params->lparams;

          FusionExecutor fe;
          fe.compileFusion(&fusion, {aten_input}, lparams);
          auto cg_outputs = fe.runFusion({aten_input}, lparams);
          auto aten_output = aten_input.to(at::kDouble).sum({axis});
          testValidate(
              &fusion,
              cg_outputs,
              {aten_input},
              {aten_output},
              __LINE__,
              __FILE__,
              "",
              lparams);
        }
      }
    }
  }
}

TEST_F(NVFuserTest, FusionCacheBefore_CUDA) {
  // TVM Cache Write
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(3.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  // Before: TV2 = TV1 * 3
  // After:  TV3 = TV1 * 3;
  //         TV2 = TV3;
  TensorView* tv3 = tv2->cacheBefore();

  constexpr int BSX = 32;
  tv2->split(-1, BSX);
  tv0->computeAt(tv2, -1);

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 750;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({M, N}, options);
  at::Tensor aten_output = (aten_input + 1.0) * 3.0;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheAfter_CUDA) {
  // TVM Cache Read
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(3.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  // Before: TV1 = TV0 + 1
  // After:  TV3 = TV0;
  //         TV1 = TV3 + 1
  TensorView* tv3 = tv0->cacheAfter();

  constexpr int BSX = 32;
  tv2->split(-1, BSX);
  tv0->computeAt(tv2, -1);

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 457;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({M, N}, options);
  at::Tensor aten_output = (aten_input + 1.0) * 3.0;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheFork_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Double>(3.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  // Before:  TV1 = TV0 + 1
  //          TV2 = TV1 * 1
  // Output:  TV1, TV2

  // After:   TV1 = TV0 + 1
  //          TV3 = TV1
  //          TV2 = TV1 * 1
  // Output:  TV3, TV2

  // cacheFork !!does not!! automatically apply ComputeAt to the cache
  auto tv3 = tv1->cacheFork();

  constexpr int BSX = 32;
  tv2->split(-1, BSX);
  tv0->computeAt(tv2, -1);

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 457;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({M, N}, options);
  at::Tensor aten_output1 = aten_input + 1.0;
  at::Tensor aten_output2 = aten_output1 * 3.0;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output1, aten_output2},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionCacheIndirect_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);
  TensorView* tv2 = makeSymbolicTensor(2);
  TensorView* tv3 = makeSymbolicTensor(2);
  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addOutput(tv6);
  // t6 = ((t1 + (t2 - t3)) - t0)

  tv5->cacheAfter();
  tv5->cacheBefore();

  // cacheAfter on inputs placed before schedule
  constexpr int BSX = 32;
  tv6->split(-1, BSX);
  tv2->computeAt(tv6, -1);

  // Thread and Block binding
  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 810;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t1 = at::randn({M, N}, options);
  at::Tensor t2 = at::randn({M, N}, options);
  at::Tensor t3 = at::randn({M, N}, options);

  std::vector<IValue> aten_inputs = {t0, t1, t2, t3};
  at::Tensor aten_output = (t1 + (t2 - t3)) - t0;

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheBcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(1); // (M, 1)
  TensorView* tv1 = broadcast(tv0, {false, true});
  TensorView* tv2 = makeSymbolicTensor(1); // (1, N)
  TensorView* tv3 = broadcast(tv2, {true, false});
  TensorView* tv4 = mul(tv1, tv3);
  fusion.addInput(tv0);
  fusion.addInput(tv2);
  fusion.addOutput(tv4);

  // Case 1
  tv0->cacheAfter();

  // Case 2
  tv1->cacheBefore();

  // Case 3
  tv1->cacheAfter();

  // Case 4
  TensorView* tv8 = tv4->cacheBefore();

  constexpr int BSX = 128;
  tv4->split(0, BSX);
  tv4->split(-1, BSX);
  tv4->reorder({{0, 0}, {1, 2}, {2, 1}, {3, 3}});
  // M/BSX, N/BSY, BSX, BSY
  tv0->computeAt(tv4, 2);
  tv2->computeAt(tv4, 2);
  // 0, 1 | 2, 3, 4

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Replay on TV3
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv8->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 92, N = 500;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M}, options);
  at::Tensor t1 = at::randn({N}, options);
  std::vector<IValue> aten_inputs = {t0, t1};
  at::Tensor aten_output =
      t0.to(at::kDouble).unsqueeze(1).matmul(t1.to(at::kDouble).unsqueeze(0));

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheMultiConsumer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(2));

  fusion.addInput(tv0);
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  auto tv5 = tv1->cacheBefore();
  auto tv6 = tv3->cacheBefore();
  tv5->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);

  tv1->computeAt(tv2, -1);
  tv3->computeAt(tv4, -1);

  // Fails because tensor must be recomputed twice
  // auto tv7 = tv0->cacheAfter();

  constexpr int N = 800;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({N}, options);
  auto aten_output = (aten_input + 1) + 2;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output, aten_output},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(2); // (M, N)
  TensorView* tv1 = makeSymbolicTensor(2); // (M, N)
  TensorView* tv2 = mul(tv0, tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv2);

  // Schedule
  TensorView* tv3 = tv0->cacheAfter();
  TensorView* tv4 = tv1->cacheAfter();
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);

  constexpr int BSY = 32;
  constexpr int BSX = 128;
  tv2->split(0, BSY);
  tv2->split(2, BSX);
  // M/BSX, BSX, N/BSX, BSX
  tv2->reorder({{0, 0}, {1, 2}, {2, 1}, {3, 3}});
  // M/BSX, N/BSX, BSX, BSX

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Binding
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 128, N = 10240;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t1 = at::randn({M, N}, options);
  at::Tensor aten_output = mul(t0, t1);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemReduce_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(3); // M, K, N
  TensorView* tv1 = sum(tv0, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter();
  tv2->setMemoryType(MemoryType::Shared);

  // Schedule
  constexpr int BSX = 32;
  tv1->split(2, BSX);
  tv1->split(1, 128);
  tv1->split(0, BSX);
  // M/BSX, BSX, K/BSX, BSX, N/BSX, BSX
  tv1->reorder({{0, 0}, {1, 2}, {2, 4}, {3, 5}, {4, 1}, {5, 3}});
  TensorView* tv3 = tv1->rFactor({-2});

  tv0->computeAt(tv1, -2);
  tv0->computeAt(tv3, -2);

  // Thread and Block binding
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Binding
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({M, K, N}, options);
  at::Tensor aten_output = sum(aten_input.to(at::kDouble), {1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemBlockGemm_CUDA) {
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

  // Schedule
  constexpr int BSX = 16;
  tv5->split(2, BSX - 1);
  tv5->split(1, BSX);
  tv5->split(0, BSX + 1);
  // M/BSX, BSX, K/BSX, BSX, N/BSX, BSX
  tv5->reorder({{0, 0}, {1, 3}, {2, 2}, {3, 5}, {4, 1}, {5, 4}});
  // M/BSX, N/BSX, K/BSX, MSX, NSX, KSX
  TensorView* tv6 = tv5->rFactor({-1});

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv5, 3);
  tv1->computeAt(tv5, 3);

  // Thread and Block binding
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-2)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Binding
  tv2->axis(-3)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-3)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-3)->parallelize(ParallelType::TIDy);
  tv6->axis(-2)->parallelize(ParallelType::TIDx);

  // Make sure BIDx is makred as exact (see issue #1119)
  GpuLower gpulw(&fusion);
  TORCH_CHECK(gpulw.parallelDimensionMap().isExact(ParallelType::BIDx));

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  std::vector<IValue> aten_inputs = {t0, t1};
  at::Tensor aten_output = matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemBlockGemmCache_CUDA) {
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

  // Schedule
  // Remove reduction axis from tv5
  // tv6 = (M, R, N)
  // tv5 = (M, N)
  TensorView* tv6 = tv5->cacheBefore();

  constexpr int BSX = 16;
  tv5->split(1, BSX);
  tv5->split(0, BSX);
  // M/BSX, BSX, N/BSX, BSX
  tv5->reorder({{0, 0}, {1, 2}, {2, 1}, {3, 3}});
  // tv5 = M/BSX, N/BSX, MSX, NSX

  tv6->computeAt(tv5, 2);
  tv6->computeAt(tv5, 2);

  tv6->split(-1, BSX);
  // M/BSX, BSX, K/BSX, BSX, N/BSX, BSX
  tv6->reorder({{0, 0}, {1, 1}, {2, 3}, {3, 4}, {4, 2}, {5, 5}});
  // M/BSX, N/BSX, K/BSX, MSX, NSX, KSX
  TensorView* tv7 = tv6->rFactor({-1});
  // tv7 = M/BSX, N/BSX, K/BSXrf, MSX, NSX, KSXr
  // tv6 = M/BSX, N/BSX, K/BSXr, MSX, NSX

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv0->computeAt(tv7, 3);
  tv1->computeAt(tv7, 3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);
  tv7->setMemoryType(MemoryType::Shared);
  // Memory Type

  // Thread and Block binding
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-2)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Binding
  tv2->axis(-3)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-3)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  tv7->axis(-3)->parallelize(ParallelType::TIDy);
  tv7->axis(-2)->parallelize(ParallelType::TIDx);

  tv6->axis(-2)->parallelize(ParallelType::TIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

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

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemDynamicPersistentSoftmax2D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(2);
  fusion.addInput(x);
  TensorView* max_val = reductionOp(
      BinaryOpType::Max,
      {-1},
      IrBuilder::create<Double>(std::numeric_limits<float>::lowest()),
      x); // (M)
  TensorView* bcast_max = broadcast(max_val, {false, true}); // (M, B)
  TensorView* x_max_sub = sub(x, bcast_max); // (M, N)
  TensorView* exp = unaryOp(UnaryOpType::Exp, x_max_sub); // (M, N)
  TensorView* sum_exp = sum(exp, {-1}); // (M, R)
  TensorView* bcast_sum = broadcast(sum_exp, {false, true}); // (M, B)
  TensorView* softmax = div(exp, bcast_sum); // (M, N)
  fusion.addOutput(softmax);

  // Read Input into Shared Memory
  // Load Input + Pwise into shared memory
  auto cache_x = x->cacheAfter();
  cache_x->setMemoryType(MemoryType::Shared);
  exp->setMemoryType(MemoryType::Shared);

  std::vector<TensorView*> all_tensors(
      {x,
       cache_x,
       max_val,
       bcast_max,
       x_max_sub,
       exp,
       sum_exp,
       bcast_sum,
       softmax});

  auto tidx = IrBuilder::create<Int>();
  fusion.addInput(tidx);

  for (auto tensor : all_tensors) {
    tensor->split(-1, tidx);
  }

  auto sum_exp_rf = sum_exp->rFactor({1});
  all_tensors.push_back(sum_exp_rf);

  // computeAt
  x->computeAt(x_max_sub, 1);
  exp->computeAt(softmax, 1);
  x_max_sub->computeAt(exp, 2);

  softmax->axis(0)->parallelize(ParallelType::BIDx);
  for (auto tensor : all_tensors) {
    tensor->axis(-1)->parallelize(ParallelType::TIDx);
  }

  const int64_t dimx = 1024;
  const int64_t dimy = 4096;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({dimx, dimy}, options);
  auto aten_output = at::_softmax(aten_input.to(at::kDouble), -1, false);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input, 128});
  auto cg_outputs = fe.runFusion({aten_input, 128});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, 128},
      {aten_output},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionMagicSchedulerSoftmax_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int kReductionAxis = 3;
  std::vector<int64_t> input_shape{10, 10, 10, 67};
  TensorView* input = makeSymbolicTensor(input_shape.size());
  fusion.addInput(input);

  auto output = softmax(input, kReductionAxis);

  fusion.addOutput(output);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  auto aten_output =
      at::_softmax(aten_input.to(at::kDouble), kReductionAxis, false);

  auto reduction_params = getPersistentHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  schedulePersistentKernel(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionTestMaskSoftmax_CUDA) {
  // This test is testing the usage of all padding tokens
  // with softmax like Bert might might use in a full padding
  // sequence.
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int kReductionAxis = 3;
  std::vector<int64_t> input_shape{256, 16, 128, 128};
  TensorView* input = makeSymbolicTensor(input_shape.size());
  TensorView* mask = makeSymbolicTensor(input_shape.size());
  fusion.addInput(input);
  fusion.addInput(mask);

  auto out1 = add(input, mask);
  auto output = softmax(out1, kReductionAxis);

  fusion.addOutput(output);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  at::Tensor aten_mask = at::ones(input_shape, options);
  // -10,000 is used here as a magic number because the padding
  // tokens need to be a value that gives a value close to zero
  // as to not influence softmax.  Bert, in particular, does
  // not use -Infinity because sometimes it will have a
  // softmax of all padding tokkens that can result a divide by
  // zero that creates NaN result.
  aten_mask = aten_mask * -10000.0;
  auto aten_out1 = aten_input + aten_mask;
  auto aten_output = at::_softmax(aten_out1, kReductionAxis, false);

  auto reduction_params =
      getPersistentHeuristics(&fusion, {aten_input, aten_mask});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  schedulePersistentKernel(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input, aten_mask}, lparams);
  auto cg_outputs = fe.runFusion({aten_input, aten_mask}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, aten_mask},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionMagicSchedulerLayerNormBackward_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  const size_t kM = shape.size();
  const size_t kN = norm_shape.size();
  const size_t kOuterNumDims = kM - kN;

  std::vector<int64_t> outer_shape;
  for (const auto idx : c10::irange(kOuterNumDims)) {
    outer_shape.push_back(shape[idx]);
  }
  for (const auto idx : c10::irange(kOuterNumDims, kM)) {
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

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> aten_inputs = {
      aten_grad_out, aten_input, aten_mean, aten_rstd, aten_weight, aten_bias};
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto aten_gradients = at::native_layer_norm_backward(
      aten_grad_out.to(at::kDouble),
      aten_input.to(at::kDouble),
      norm_shape,
      aten_mean.to(at::kDouble),
      aten_rstd.to(at::kDouble),
      c10::optional<at::Tensor>(aten_weight.to(at::kDouble)),
      c10::optional<at::Tensor>(aten_bias.to(at::kDouble)),
      {true, true, true});

  testValidate(
      &fusion,
      cg_outputs,
      aten_inputs,
      {std::get<0>(aten_gradients),
       std::get<1>(aten_gradients),
       std::get<2>(aten_gradients)},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionMagicSchedulerRMSNormBackward_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const int64_t NORM_SIZE = 1024;
  std::vector<int64_t> shape{8, 56, NORM_SIZE};
  std::vector<int64_t> norm_shape{NORM_SIZE};

  const size_t kM = shape.size();
  const size_t kN = norm_shape.size();
  const size_t kOuterNumDims = kM - kN;

  std::vector<int64_t> outer_shape;
  for (const auto idx : c10::irange(kOuterNumDims)) {
    outer_shape.push_back(shape[idx]);
  }
  for (const auto idx : c10::irange(kOuterNumDims, kM)) {
    outer_shape.push_back(1);
  }

  auto grad_out = makeContigTensor(shape.size());
  auto input = makeContigTensor(shape.size());
  auto rstd = makeConcreteTensor(outer_shape);
  auto weight = makeContigTensor(norm_shape.size());
  fusion.addInput(grad_out);
  fusion.addInput(input);
  fusion.addInput(rstd);
  fusion.addInput(weight);

  auto grads = rms_norm_backward(
      grad_out, input, norm_shape, rstd, weight, {true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_grad_out = at::randn(shape, options);
  at::Tensor aten_input = at::randn(shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  auto at_weight = c10::optional<at::Tensor>(aten_weight);

  const float kEps = 1e-6;
  auto pow2 = at::pow(aten_input, 2);
  auto sum = at::sum(pow2, -1, true);
  auto var = at::mul(sum, 1.0 / NORM_SIZE);
  auto aten_rstd = at::pow(at::add(var, kEps), -0.5);

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> aten_inputs = {
      aten_grad_out, aten_input, aten_rstd, aten_weight};
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto in_mul_rstd = at::mul(aten_input, aten_rstd);
  auto grad_out_mul = at::mul(aten_grad_out, in_mul_rstd);
  auto aten_grad_weight = at::sum(grad_out_mul, c10::IntArrayRef{0, 1});
  auto sum_loss1 = at::sum(at::mul(aten_grad_out, aten_weight), -1, true);
  auto sum_loss2 = at::sum(
      at::mul(
          at::mul(at::mul(aten_grad_out, aten_weight), aten_input), aten_rstd),
      -1,
      true);

  const float fH = NORM_SIZE;
  auto term1 = at::mul(aten_rstd, 1.0 / fH);
  auto aten_grad_input = at::mul(at::mul(aten_grad_out, fH), aten_weight);
  aten_grad_input = at::sub(aten_grad_input, sum_loss1);
  aten_grad_input = at::sub(
      aten_grad_input, at::mul(at::mul(aten_input, aten_rstd), sum_loss2));
  aten_grad_input = at::mul(aten_grad_input, term1);
  testValidate(
      &fusion,
      cg_outputs,
      aten_inputs,
      {aten_grad_input, aten_grad_weight},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionMagicSchedulerLayerNormalization_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const float kEps = 1e-5;
  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  std::vector<int64_t> input_shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  auto input = makeSymbolicTensor(input_shape.size());
  fusion.addInput(input);

  auto result = layer_norm(input, norm_shape, nullptr, nullptr, eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = c10::nullopt;
  c10::optional<at::Tensor> aten_bias = c10::nullopt;
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  // Check reduction axis is same for all reductions
  // Generate Launch Parameters
  auto reduction_params = getPersistentHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({aten_input});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionMagicSchedulerRMSNormalization_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int64_t NORM_SIZE = 1024;
  const float kEps = 1e-6;
  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  std::vector<int64_t> input_shape{8, 56, NORM_SIZE};
  std::vector<int64_t> norm_shape{NORM_SIZE};

  auto input = makeContigTensor(input_shape.size());
  fusion.addInput(input);
  auto result = rms_norm(input, norm_shape, nullptr, eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = c10::nullopt;

  auto pow2 = at::pow(aten_input, 2);

  auto sum = at::sum(pow2, -1, true);
  auto var = at::mul(sum, 1.0 / NORM_SIZE);
  auto invstd = at::pow(at::add(var, kEps), -0.5);
  auto output = at::mul(aten_input, invstd);
  //// Check reduction axis is same for all reductions
  //// Generate Launch Parameters
  auto reduction_params = getPersistentHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({aten_input});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {output, invstd},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionMagicSchedulerBatchNormalization_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const bool kTraining = true;
  std::vector<int64_t> input_shape{20, 100, 35, 45};

  auto input = makeSymbolicTensor(input_shape.size());
  auto weight = makeSymbolicTensor(1);
  auto bias = makeSymbolicTensor(1);
  auto running_mean = makeSymbolicTensor(1);
  auto running_var = makeSymbolicTensor(1);
  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);

  Double* momentum = IrBuilder::create<Double>(kMomentum);
  Double* eps = IrBuilder::create<Double>(kEps);

  auto result = batch_norm(
      input, weight, bias, running_mean, running_var, kTraining, momentum, eps);

  fusion->addOutput(result.output);
  fusion->addOutput(result.mean);
  fusion->addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_input = at::randn(input_shape, options);
  auto at_weight = at::ones({input_shape[1]}, options);
  auto at_bias = at::zeros({input_shape[1]}, options);
  auto at_run_mean = at::zeros({input_shape[1]}, options);
  auto at_run_var = at::ones({input_shape[1]}, options);

  std::vector<IValue> aten_inputs = {
      at_input, at_weight, at_bias, at_run_mean, at_run_var};

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto aten_outputs = at::native_batch_norm(
      at_input,
      c10::optional<at::Tensor>(at_weight),
      c10::optional<at::Tensor>(at_bias),
      c10::optional<at::Tensor>(at_run_mean),
      c10::optional<at::Tensor>(at_run_var),
      kTraining,
      kMomentum,
      kEps);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionMagicSchedulerInstanceNormalization_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const bool kUseInputStats = true;
  std::vector<int64_t> input_shape{20, 100, 35, 45};

  auto input = makeSymbolicTensor(input_shape.size());
  auto weight = makeSymbolicTensor(1);
  auto bias = makeSymbolicTensor(1);
  auto running_mean = makeSymbolicTensor(1);
  auto running_var = makeSymbolicTensor(1);
  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);

  Double* momentum = IrBuilder::create<Double>(kMomentum);
  Double* eps = IrBuilder::create<Double>(kEps);

  auto result = instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kUseInputStats,
      momentum,
      eps);

  fusion->addOutput(result.output);
  // fusion->addOutput(result.mean);
  // fusion->addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_input = at::randn(input_shape, options);
  auto at_weight = at::ones({input_shape[1]}, options);
  auto at_bias = at::zeros({input_shape[1]}, options);
  auto at_run_mean = at::zeros({input_shape[1]}, options);
  auto at_run_var = at::ones({input_shape[1]}, options);

  std::vector<IValue> aten_inputs = {
      at_input, at_weight, at_bias, at_run_mean, at_run_var};

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  auto cg_outputs_full = {at_run_mean, at_run_var, cg_outputs[0]};

  auto aten_outputs = at::instance_norm(
      at_input,
      c10::optional<at::Tensor>(at_weight),
      c10::optional<at::Tensor>(at_bias),
      c10::optional<at::Tensor>(at_run_mean),
      c10::optional<at::Tensor>(at_run_var),
      kUseInputStats,
      kMomentum,
      kEps,
      false);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      // TODO: can run_mean/run_var be checked here?
      // fusion_outputs.size() == aten_outputs.size() && aten_outputs.size() ==
      // fusion->outputs().size() - output_alias_indices.size()
      {aten_outputs},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionMagicSchedulerInstanceNormalizationBackward_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  auto fusion_forward = std::make_unique<Fusion>();
  FusionGuard fg_forward(fusion_forward.get());

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const bool kUseInputStats = true;
  const bool channels_last = true;
  const int B = 2;
  const int C = 5;
  const int S = 3;
  std::vector<int64_t> input_shape{B, C, S, S, S};
  // explicit channels-last for NVFuser
  std::vector<int64_t> nvfuser_input_shape{B, S, S, S, C};

  auto input = makeContigTensor(input_shape.size());
  auto weight = makeContigTensor(1);
  auto bias = makeContigTensor(1);
  fusion_forward->addInput(input);
  fusion_forward->addInput(weight);
  fusion_forward->addInput(bias);

  Double* momentum = IrBuilder::create<Double>(kMomentum);
  Double* eps = IrBuilder::create<Double>(kEps);
  auto result_forward = instance_norm(
      input,
      weight,
      bias,
      nullptr,
      nullptr,
      kUseInputStats,
      momentum,
      eps,
      channels_last);
  fusion_forward->addOutput(result_forward.output);
  fusion_forward->addOutput(result_forward.mean);
  fusion_forward->addOutput(result_forward.invstd);

  FusionExecutorCache executor_cache_forward(std::move(fusion_forward));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_input = at::randn(input_shape, options)
                      .to(at::MemoryFormat::ChannelsLast3d)
                      .set_requires_grad(true);
  auto at_input_nvfuser = at_input.clone().detach().permute({0, 2, 3, 4, 1});
  auto at_weight = at::ones({input_shape[1]}, options).set_requires_grad(true);
  auto at_weight_nvfuser = at_weight.clone().detach();
  auto at_bias = at::zeros({input_shape[1]}, options).set_requires_grad(true);
  auto at_bias_nvfuser = at_bias.clone().detach();
  std::vector<torch::jit::IValue> aten_inputs_forward = {
      at_input_nvfuser, at_weight_nvfuser, at_bias_nvfuser};
  // out, mean, invstd
  auto outputs_forward =
      executor_cache_forward.runFusionWithInputs(aten_inputs_forward);
  auto at_out = at::instance_norm(
      at_input,
      c10::optional<at::Tensor>(at_weight),
      c10::optional<at::Tensor>(at_bias),
      c10::optional<at::Tensor>(c10::nullopt),
      c10::optional<at::Tensor>(c10::nullopt),
      kUseInputStats,
      kMomentum,
      kEps,
      false);
  auto at_grad =
      at::randn(input_shape, options).to(at::MemoryFormat::ChannelsLast3d);
  auto at_grad_nvfuser = at_grad.clone().detach().permute({0, 2, 3, 4, 1});
  at_out.backward(at_grad);
  auto fusion_backward = std::make_unique<Fusion>();
  FusionGuard fg_backward(fusion_backward.get());

  input = makeContigTensor(input_shape.size());
  auto grad_output = makeContigTensor(input_shape.size());
  weight = makeContigTensor(1);
  auto save_mean = makeContigTensor(2);
  auto save_invstd = makeContigTensor(2);
  auto dummy = makeContigTensor(0);

  fusion_backward->addInput(input);
  fusion_backward->addInput(grad_output);
  fusion_backward->addInput(weight);
  fusion_backward->addInput(dummy); // dummy for run_mean
  fusion_backward->addInput(dummy); // dummy for run_var
  fusion_backward->addInput(save_mean);
  fusion_backward->addInput(save_invstd);

  auto result_backward = instance_norm_backward(
      input,
      grad_output,
      weight,
      nullptr,
      nullptr,
      save_mean,
      save_invstd,
      kUseInputStats,
      eps,
      {true, true, true},
      channels_last);

  fusion_backward->addOutput(result_backward.grad_input);
  fusion_backward->addOutput(result_backward.grad_weight);
  fusion_backward->addOutput(result_backward.grad_bias);

  FusionExecutorCache executor_cache_backward(std::move(fusion_backward));
  std::vector<torch::jit::IValue> aten_inputs_backward = {
      at_input_nvfuser,
      at_grad_nvfuser,
      at_weight_nvfuser,
      at::empty({}),
      at::empty({}),
      outputs_forward[1],
      outputs_forward[2]};
  auto outputs_backward =
      executor_cache_backward.runFusionWithInputs(aten_inputs_backward);
  outputs_backward[0] = outputs_backward[0].permute({0, 4, 1, 2, 3});
  testValidate(
      executor_cache_backward.fusion(),
      outputs_backward,
      aten_inputs_backward,
      {at_input.grad(), at_weight.grad(), at_bias.grad()},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionPersistentSoftmaxLocalShared_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int pixels_per_thread = 64;
  const int TIDX = 128;
  const int static_size = pixels_per_thread * TIDX;

  TensorView* sx = makeConcreteTensor({-1, static_size});
  TensorView* dx = makeSymbolicTensor(2);
  fusion.addInput(sx);
  fusion.addInput(dx);

  TensorView* max_sx = reductionOp(
      BinaryOpType::Max,
      {-1},
      IrBuilder::create<Double>(std::numeric_limits<float>::lowest()),
      sx); // (M)
  TensorView* max_dx = reductionOp(
      BinaryOpType::Max,
      {-1},
      IrBuilder::create<Double>(std::numeric_limits<float>::lowest()),
      dx); // (M)

  // Reduction => merge local and shared memory TensorViews
  TensorView* max_val = binaryOp(BinaryOpType::Max, max_sx, max_dx);
  TensorView* bcast_max = broadcast(max_val, {false, true}); // (M, B)

  TensorView* sx_max_sub = sub(sx, bcast_max); // (M, N)
  TensorView* dx_max_sub = sub(dx, bcast_max); // (M, N)

  TensorView* sx_exp = unaryOp(UnaryOpType::Exp, sx_max_sub); // (M, N)
  TensorView* dx_exp = unaryOp(UnaryOpType::Exp, dx_max_sub); // (M, N)

  TensorView* sx_sum_exp = sum(sx_exp, {-1}); // (M, R)
  TensorView* dx_sum_exp = sum(dx_exp, {-1}); // (M, R)

  // Reduction => merge local and shared memory TensorViews
  TensorView* sum_exp = binaryOp(BinaryOpType::Add, sx_sum_exp, dx_sum_exp);
  TensorView* bcast_sum = broadcast(sum_exp, {false, true}); // (M, B)

  TensorView* sx_softmax = div(sx_exp, bcast_sum); // (M, N)
  TensorView* dx_softmax = div(dx_exp, bcast_sum); // (M, N)
  fusion.addOutput(sx_softmax);
  fusion.addOutput(dx_softmax);

  auto sx_cache = sx->cacheAfter();
  auto dx_cache = dx->cacheAfter();
  dx_cache->setMemoryType(MemoryType::Shared);
  dx_exp->setMemoryType(MemoryType::Shared);

  // Reduction and Broadcast Tensors common to both memory TVs
  std::vector<TensorView*> common_tensors(
      {max_val, sum_exp, bcast_max, bcast_sum});

  // Static Local Memory TVs
  std::vector<TensorView*> static_tensors(
      {sx, sx_cache, max_sx, sx_max_sub, sx_exp, sx_sum_exp, sx_softmax});

  // Dynamic Local Memory TVs
  std::vector<TensorView*> dynamic_tensors(
      {dx, dx_cache, max_dx, dx_max_sub, dx_exp, dx_sum_exp, dx_softmax});

  std::vector<TensorView*> all_tensors;
  all_tensors.insert(
      all_tensors.end(), common_tensors.begin(), common_tensors.end());
  all_tensors.insert(
      all_tensors.end(), static_tensors.begin(), static_tensors.end());
  all_tensors.insert(
      all_tensors.end(), dynamic_tensors.begin(), dynamic_tensors.end());

  // M => M
  // M, N => M, N/128, 128
  for (auto tensor : all_tensors) {
    if (tensor->nDims() > 1) {
      tensor->split(-1, TIDX);
    }
  }

  auto sx_sum_exp_rf = sx_sum_exp->rFactor({1});
  auto dx_sum_exp_rf = dx_sum_exp->rFactor({1});
  all_tensors.push_back(sx_sum_exp_rf);
  all_tensors.push_back(dx_sum_exp_rf);

  // computeAt
  sx->computeAt(sx_max_sub, 1);
  dx->computeAt(dx_max_sub, 1);

  sx_exp->computeAt(sx_softmax, 1);
  dx_exp->computeAt(dx_softmax, 1);

  sx_max_sub->computeAt(sx_exp, 2);
  dx_max_sub->computeAt(dx_exp, 2);

  sx_softmax->axis(0)->parallelize(ParallelType::BIDx);
  dx_softmax->axis(0)->parallelize(ParallelType::BIDx);
  for (auto tensor : all_tensors) {
    if (tensor->nDims() > 1) {
      tensor->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  const int64_t dimx = 1024;
  const int64_t dimy = 16384;

  auto properties = at::cuda::getDeviceProperties(0);
  const size_t required_smem_size =
      (dimy - static_size) * sizeof(float) + TIDX * sizeof(float);
  if (properties->sharedMemPerBlockOptin < required_smem_size) {
    GTEST_SKIP() << "not enough shared memory space on device to run test: "
                 << properties->sharedMemPerBlock;
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({dimx, dimy}, options);
  at::Tensor aten_static_in = aten_input.narrow(1, 0, static_size);
  at::Tensor aten_dynamic_in =
      aten_input.narrow(1, static_size, dimy - static_size);

  at::Tensor out = at::zeros({dimx, dimy}, options);
  at::Tensor cg_static_out = out.narrow(1, 0, static_size);
  at::Tensor cg_dynamic_out = out.narrow(1, static_size, dimy - static_size);

  std::vector<at::Tensor> aten_outputs;

  auto aten_output = at::_softmax(aten_input.to(at::kDouble), -1, false);
  at::Tensor aten_static_out = aten_output.narrow(1, 0, static_size);
  at::Tensor aten_dynamic_out =
      aten_output.narrow(1, static_size, dimy - static_size);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_static_in, aten_dynamic_in});
  fe.runFusion(
      {aten_static_in, aten_dynamic_in}, {cg_static_out, cg_dynamic_out});

  testValidate(
      &fusion,
      {cg_static_out, cg_dynamic_out},
      {aten_static_in, aten_dynamic_in},
      {cg_static_out, cg_dynamic_out},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionPersistentNormLocalShared_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int pixels_per_thread = 64;
  const int TIDX = 128;
  const int static_size = pixels_per_thread * TIDX;

  TensorView* sx = makeConcreteTensor({-1, static_size});
  TensorView* dx = makeSymbolicTensor(2);
  fusion.addInput(sx);
  fusion.addInput(dx);

  Double* gamma = IrBuilder::create<Double>();
  Double* beta = IrBuilder::create<Double>();
  Double* eps = IrBuilder::create<Double>();
  Int* N = IrBuilder::create<Int>();
  fusion.addInput(gamma);
  fusion.addInput(beta);
  fusion.addInput(eps);
  fusion.addInput(N);

  // Reduction
  auto sx_sum = sum(sx, {-1}); // (M, R)
  auto dx_sum = sum(dx, {-1}); // (M, R)
  // Reduction => merge local and shared memory TensorViews
  auto x_sum = binaryOp(BinaryOpType::Add, sx_sum, dx_sum);

  // Broadcast
  auto x_sum_bcast = broadcast(x_sum, {false, true}); // (M, B)
  // Pwise
  auto x_mean = div(x_sum_bcast, N); // (M, B)

  auto sx_mean_sub = sub(sx, x_mean); // (M, N)
  auto dx_mean_sub = sub(dx, x_mean); // (M, N)

  auto sx_mean_sub_pow = mul(sx_mean_sub, sx_mean_sub); // (M, N)
  auto dx_mean_sub_pow = mul(dx_mean_sub, dx_mean_sub); // (M, N)

  // Reduction
  auto sx_var_sum = sum(sx_mean_sub_pow, {-1}); // (M, R)
  auto dx_var_sum = sum(dx_mean_sub_pow, {-1}); // (M, R)
  // Reduction => merge local and shared memory TensorViews
  auto var_sum = binaryOp(BinaryOpType::Add, sx_var_sum, dx_var_sum);

  // Broadcast
  auto var_sum_bcast = broadcast(var_sum, {false, true}); // (M, B)
  // Pwise
  auto var = div(var_sum_bcast, N); // (M, B)
  auto var_eps = add(var, eps); // (M, B)
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps); // (M, B)

  auto sx_norm = mul(sx_mean_sub, rvar);
  auto dx_norm = mul(dx_mean_sub, rvar);

  auto sx_norm_gamma = mul(sx_norm, gamma);
  auto dx_norm_gamma = mul(dx_norm, gamma);

  auto sx_norm_gamma_beta = add(sx_norm_gamma, beta);
  auto dx_norm_gamma_beta = add(dx_norm_gamma, beta);

  fusion.addOutput(sx_norm_gamma_beta);
  fusion.addOutput(dx_norm_gamma_beta);

  sx_norm_gamma_beta->setContiguity(false);
  dx_norm_gamma_beta->setContiguity(false);

  // Read Input into Shared Memory
  // Read Input minus Input_Mean into Shared Memory
  auto sx_cache = sx->cacheAfter();
  auto dx_cache = dx->cacheAfter();
  dx_cache->setMemoryType(MemoryType::Shared);
  dx_mean_sub->setMemoryType(MemoryType::Shared);

  std::vector<TensorView*> common_tensors(
      {x_sum, x_sum_bcast, x_mean, var_sum, var_sum_bcast, var, var_eps, rvar});

  std::vector<TensorView*> static_tensors(
      {sx,
       sx_cache,
       sx_sum,
       sx_mean_sub,
       sx_mean_sub_pow,
       sx_var_sum,
       sx_norm,
       sx_norm_gamma,
       sx_norm_gamma_beta});

  std::vector<TensorView*> dynamic_tensors(
      {dx,
       dx_cache,
       dx_sum,
       dx_mean_sub,
       dx_mean_sub_pow,
       dx_var_sum,
       dx_norm,
       dx_norm_gamma,
       dx_norm_gamma_beta});

  std::vector<TensorView*> all_tensors;
  all_tensors.insert(
      all_tensors.end(), common_tensors.begin(), common_tensors.end());
  all_tensors.insert(
      all_tensors.end(), static_tensors.begin(), static_tensors.end());
  all_tensors.insert(
      all_tensors.end(), dynamic_tensors.begin(), dynamic_tensors.end());

  // M => M
  // M, N => M, N/128, 128
  for (auto tensor : all_tensors) {
    if (tensor->nDims() > 1) {
      tensor->split(-1, TIDX);
    }
  }

  // Local Sum => Block Broadcast
  TensorView* sx_sum_rf = sx_sum->rFactor({1});
  TensorView* sx_var_sum_rf = sx_var_sum->rFactor({1});
  TensorView* dx_sum_rf = dx_sum->rFactor({1});
  TensorView* dx_var_sum_rf = dx_var_sum->rFactor({1});
  all_tensors.push_back(sx_sum_rf);
  all_tensors.push_back(sx_var_sum_rf);
  all_tensors.push_back(dx_sum_rf);
  all_tensors.push_back(dx_var_sum_rf);

  // ComputeAt
  sx->computeAt(sx_mean_sub_pow, 1);
  dx->computeAt(dx_mean_sub_pow, 1);

  var_sum->computeAt(rvar, 1);

  sx_mean_sub_pow->computeAt(sx_var_sum_rf, 2);
  dx_mean_sub_pow->computeAt(dx_var_sum_rf, 2);

  sx_norm->computeAt(sx_norm_gamma_beta, 2);
  dx_norm->computeAt(dx_norm_gamma_beta, 2);

  sx_norm_gamma_beta->axis(0)->parallelize(ParallelType::BIDx);
  dx_norm_gamma_beta->axis(0)->parallelize(ParallelType::BIDx);
  for (auto tensor : all_tensors) {
    if (tensor->nDims() > 1) {
      tensor->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  const int dimx = 1024;
  const int dimy = 16384;
  const float kGamma = 1.0f;
  const float kBeta = 0.0f;
  const float kEps = 1e-5;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto properties = at::cuda::getDeviceProperties(0);
  const size_t required_smem_size =
      (dimy - static_size) * sizeof(float) + TIDX * sizeof(float);
  if (properties->sharedMemPerBlockOptin < required_smem_size) {
    GTEST_SKIP() << "not enough shared memory space on device to run test: "
                 << properties->sharedMemPerBlock;
  }

  at::Tensor aten_input = at::randn({dimx, dimy}, options);
  at::Tensor aten_static_in = aten_input.narrow(1, 0, static_size);
  at::Tensor aten_dynamic_in =
      aten_input.narrow(1, static_size, dimy - static_size);

  at::Tensor out = at::zeros({dimx, dimy}, options);
  at::Tensor cg_static_out = out.narrow(1, 0, static_size);
  at::Tensor cg_dynamic_out = out.narrow(1, static_size, dimy - static_size);

  std::vector<IValue> aten_inputs = {
      aten_static_in, aten_dynamic_in, kGamma, kBeta, kEps, dimy};

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  fe.runFusion(aten_inputs, {cg_static_out, cg_dynamic_out});

  auto at_mu = at::mean(aten_input.to(at::kDouble), -1).unsqueeze(1);
  auto at_var = at::var(aten_input.to(at::kDouble), -1, false).unsqueeze(1);
  auto at_rvar = at::rsqrt(at::add(at_var, kEps));
  auto at_norm = at::mul(at::sub(aten_input, at_mu), at_rvar);
  auto aten_output = at::add(at::mul(at_norm, kGamma), kBeta);
  at::Tensor aten_static_out = aten_output.narrow(1, 0, static_size);
  at::Tensor aten_dynamic_out =
      aten_output.narrow(1, static_size, dimy - static_size);

  testValidate(
      &fusion,
      {cg_static_out, cg_dynamic_out},
      aten_inputs,
      {aten_static_out, aten_dynamic_out},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionSmemDynamicPersistentNorm_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  auto x = makeSymbolicTensor(2);
  Double* gamma = IrBuilder::create<Double>();
  Double* beta = IrBuilder::create<Double>();
  Double* eps = IrBuilder::create<Double>();
  Int* N = IrBuilder::create<Int>();
  fusion.addInput(x);
  fusion.addInput(gamma);
  fusion.addInput(beta);
  fusion.addInput(eps);
  fusion.addInput(N);

  // Reduction
  auto x_sum = sum(x, {-1}); // (M, R)
  // Broadcast
  auto x_sum_bcast = broadcast(x_sum, {false, true}); // (M, B)
  // Pwise
  auto x_mean = div(x_sum_bcast, N); // (M, B)
  auto x_mean_sub = sub(x, x_mean); // (M, N)
  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub); // (M, N)
  // Reduction
  auto var_sum = sum(x_mean_sub_pow, {-1}); // (M, R)
  // Broadcast
  auto var_sum_bcast = broadcast(var_sum, {false, true}); // (M, B)
  // Pwise
  auto var = div(var_sum_bcast, N); // (M, B)
  auto var_eps = add(var, eps); // (M, B)
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps); // (M, B)
  auto norm = mul(x_mean_sub, rvar);
  auto norm_gamma = mul(norm, gamma);
  auto norm_gamma_beta = add(norm_gamma, beta);
  fusion.addOutput(norm_gamma_beta);

  // Read Input into Shared Memory
  // Read Input minus Input_Mean into Shared Memory
  auto cache_x = x->cacheAfter();
  cache_x->setMemoryType(MemoryType::Shared);
  x_mean_sub->setMemoryType(MemoryType::Shared);

  std::vector<TensorView*> all_tensors(
      {x_sum,
       x_mean,
       cache_x,
       x_sum_bcast,
       x_mean_sub,
       x_mean_sub_pow,
       var_sum,
       var_sum_bcast,
       var,
       var_eps,
       rvar,
       norm,
       norm_gamma,
       norm_gamma_beta});

  auto tidx = IrBuilder::create<Int>();
  fusion.addInput(tidx);

  for (auto tensor : all_tensors) {
    tensor->split(-1, tidx);
  }

  // Local Sum => Block Broadcast
  TensorView* x_sum_rf = x_sum->rFactor({1});
  TensorView* var_sum_rf = var_sum->rFactor({1});
  all_tensors.push_back(x_sum_rf);
  all_tensors.push_back(var_sum_rf);

  // ComputeAt
  x->computeAt(x_mean_sub_pow, 1);
  var_sum->computeAt(rvar, 1);
  x_mean_sub_pow->computeAt(var_sum_rf, 2);
  norm->computeAt(norm_gamma_beta, 2);

  for (auto tv : all_tensors) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  const int dimx = 128;
  const int dimy = 2048;
  const float kGamma = 1.0f;
  const float kBeta = 0.0f;
  const float kEps = 1e-5;
  const int TIDX = 128;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({dimx, dimy}, options);
  auto at_mu = at::mean(aten_input.to(at::kDouble), -1).unsqueeze(1);
  auto at_var = at::var(aten_input.to(at::kDouble), -1).unsqueeze(1);
  auto at_rvar = at::rsqrt(at::add(at_var, kEps));
  auto at_norm = at::mul(at::sub(aten_input, at_mu), at_rvar);
  auto aten_output = at::add(at::mul(at_norm, kGamma), kBeta);

  std::vector<IValue> aten_inputs = {
      aten_input, kGamma, kBeta, kEps, dimy, TIDX};

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSmemDynamicReductionSymbolic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  // tv1[I0, R1] = tv0[I0, I1]

  // Interface should just be a direct split with a Parallel type. We can
  // include the parallelize call if we do this.
  tv1->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
  // tv1[I0, R1o, R1i{BIDx}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({2});
  tv2->setMemoryType(MemoryType::Shared);
  // tv2[I0, R1oo, Ir1i{BIDx}] = tv0[I0, I1]
  // tv1[I0,        R1i{BIDx}] = tv2[I0, R1oo, Ir1i{BIDx}]

  tv0->computeAt(tv1, 1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  constexpr int numel_x = 65000, numel_y = 1024;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);
  auto aten_output = aten_input.to(at::kDouble).sum({1});

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  LaunchParams lparams(-1, -1, -1, runtime_threadIdx_dim, -1, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemDynamicReductionSymbolicArg_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  Int* sym_bsx = IrBuilder::create<Int>();
  TensorView* tv0 = makeSymbolicTensor(3); // M, K, N
  fusion.addInput(tv0);
  fusion.addInput(sym_bsx);

  TensorView* tv1 = sum(tv0, {1}); // M, R, N
  fusion.addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter();
  tv2->setMemoryType(MemoryType::Shared);

  // Schedule
  constexpr int BSX = 32;
  tv1->split(2, BSX);
  tv1->split(1, sym_bsx);
  tv1->split(0, BSX);
  // M/BSX, BSX, K/BSX, BSX, N/BSX, BSX
  tv1->reorder({{0, 0}, {1, 2}, {2, 4}, {3, 5}, {4, 1}, {5, 3}});
  TensorView* tv3 = tv1->rFactor({-2});

  tv0->computeAt(tv1, -2);
  tv0->computeAt(tv3, -2);

  // Thread and Block binding
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  // Manual Binding
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({M, K, N}, options);
  at::Tensor aten_output = aten_input.to(at::kDouble).sum({1});

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  auto lparams = LaunchParams(-1, -1, -1, runtime_threadIdx_dim, -1, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input, runtime_threadIdx_dim}, lparams);
  auto cg_outputs = fe.runFusion({aten_input, runtime_threadIdx_dim}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, runtime_threadIdx_dim},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 0);
}

TEST_F(NVFuserTest, FusionSmemDynamicPwiseMulSymbolicArgWAR_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Int* sym_bsx = IrBuilder::create<Int>();
  TensorView* tv0 = makeSymbolicTensor(2); // (M, K)
  TensorView* tv1 = makeSymbolicTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(sym_bsx);
  fusion.addOutput(tv4);
  // Algorithm

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  constexpr int BSX = 32;
  tv4->split(2, BSX);
  tv4->split(1, sym_bsx);
  tv4->split(0, BSX);
  // M/BSX, BSX, K/BSX, BSX, N/BSX, BSX
  tv4->reorder({{0, 0}, {1, 3}, {2, 1}, {3, 4}, {4, 2}, {5, 5}});
  // M/BSX, K/BSX, N/BSX, MSX, KSX, NSX

  tv0->computeAt(tv4, 3);
  tv1->computeAt(tv4, 3);
  // Schedule

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::BIDy);
  // Manual Binding
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  // Thread and Block binding

  constexpr int M = 128, K = 457, N = 1024;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = mul(t0.unsqueeze(2), t1.unsqueeze(0));
  std::vector<IValue> aten_inputs = {t0, t1, BSX};

  LaunchParams lparams(-1, -1, -1, BSX, -1, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      aten_inputs,
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 1);
}

TEST_F(NVFuserTest, FusionSmemDynamicTiledGemm_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Int* symbolic_m_tile_dim = IrBuilder::create<Int>(); // bound to threadIdx.z
  Int* symbolic_split_k_tile_dim =
      IrBuilder::create<Int>(); // bound to blockIdx.x
  Int* symbolic_block_k_tile_dim =
      IrBuilder::create<Int>(); // bound to threadIdx.x
  // Compile-time integer for tiling
  int n_smem_tile = 8; // bound to threadIdx.y

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Turn the K-dimension of tv4 into a reduction dimension
  TensorView* tv5 = sum(tv4, {1});

  // Register inputs and outputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  // Register runtime tile dims as inputs
  fusion.addInput(symbolic_m_tile_dim);
  fusion.addInput(symbolic_split_k_tile_dim);
  fusion.addInput(symbolic_block_k_tile_dim);

  // Make a 3D tile, mix of symbolic and constant, do in reverse order because
  // dims are inserted
  // [M, K, N]
  tv5->split(2, n_smem_tile);
  tv5->split(1, symbolic_block_k_tile_dim);
  tv5->split(1, symbolic_split_k_tile_dim);
  tv5->split(0, symbolic_m_tile_dim);
  // [Mo, Mi, Koo, Koi, Ki, No, Ni]

  // Reorder so all outer tiles are in the leftmost 3 positions
  tv5->reorder({{1, 5}, {5, 1}});
  // [Mo, No, Koo, Koi, Ki, Mi, Ni]

  // Factor out the outer reduction IterDomain, then run the inter-cta
  // reduction, and intra-cta reduction
  auto tv6 = tv5->rFactor({2});
  // [Mo, No, rKoo, rKoi, rKi, Mi, Ni]
  // [Mo, No,       rKoi, rKi, Mi, Ni]

  // Scope computations
  tv6->computeAt(tv5, 2);
  // [Mo, No, rKoo,  Koi,  Ki, Mi, Ni]
  // [Mo, No,       rKoi, rKi, Mi, Ni]

  // Setup compute at schedule
  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);
  tv4->computeAt(tv6, -1);
  //
  // T2[Mo,  bNo, Koo, Koi,  Kii,  Mi, bNi] CA(4, 3)
  // T3[bMo,  No, Koo, Koi,  Kii, bMi,  Ni] CA(4, 3)
  // T4[ Mo,  No, Koo, Koi,  Kii,  Mi,  Ni]
  // T6[ Mo,  No, rKoo, Koi, Kii,  Mi,  Ni]
  // T5[ Mo,  No,      rKoi, rKii, Mi,  Ni]

  // Cache smem tiles
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Local);
  tv6->setMemoryType(MemoryType::Local);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  std::vector<TensorView*> tv_list = {tv2, tv3, tv4, tv5, tv6};
  for (auto tv : tv_list) {
    tv->axis(-2)->parallelize(ParallelType::TIDz);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }
  tv2->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::TIDx);
  tv6->axis(3)->parallelize(ParallelType::TIDx);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(4)->parallelize(ParallelType::BIDx);
  tv3->axis(4)->parallelize(ParallelType::BIDx);
  tv4->axis(4)->parallelize(ParallelType::BIDx);
  tv6->axis(4)->parallelize(ParallelType::BIDx);
  tv5->axis(3)->parallelize(ParallelType::BIDx);

  constexpr int M = 31, K = 65, N = 33;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  // Runtime tiling
  int m_tile = 4; // bound to threadIdx.z
  int split_k = 7; // bound to blockIdx.x
  int intra_cta = 8; // bound to threadIdx.x

  std::vector<IValue> aten_inputs = {t0, t1, m_tile, split_k, intra_cta};
  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  FusionExecutor fe;
  // Generate CUDA and compile with nvRTC
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);

  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count == 1);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
