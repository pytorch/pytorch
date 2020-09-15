#if defined(USE_CUDA)

#include <test/cpp/jit/test_base.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include "torch/csrc/jit/ir/irparser.h"

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

namespace {

TensorView* makeContigTensor(int nDims, DataType dtype = DataType::Float) {
  std::vector<IterDomain*> dom;
  for (int i = 0; i < nDims; i++)
    dom.push_back(new IterDomain(new Int(0), new Int()));
  std::vector<bool> contig(dom.size(), true);
  return new TensorView(new TensorDomain(dom, contig), dtype);
}

TensorView* makeDummyTensor(int nDims, DataType dtype = DataType::Float) {
  // We can uncomment the below statement to test all tests with contiguous
  // tensors. return makeContigTensor(nDims, dtype);
  std::vector<IterDomain*> dom;
  for (int i = 0; i < nDims; i++)
    dom.push_back(new IterDomain(new Int(0), new Int()));
  return new TensorView(new TensorDomain(dom), dtype);
}

TensorView* makeConcreteTensor(
    std::vector<int> sizes,
    DataType dtype = DataType::Float) {
  // We can uncomment the below statement to test all tests with contiguous
  // tensors. return makeContigTensor(nDims, dtype);
  std::vector<IterDomain*> dom;
  for (size_t i = 0; i < sizes.size(); i++)
    dom.push_back(new IterDomain(new Int(0), new Int(sizes[i])));
  return new TensorView(new TensorDomain(dom), dtype);
}

TensorView* makeTensorWithContig(
    int nDims,
    std::vector<bool> contig_info,
    DataType dtype = DataType::Float) {
  std::vector<IterDomain*> dom;
  for (int i = 0; i < nDims; i++)
    dom.push_back(new IterDomain(new Int(0), new Int()));
  return new TensorView(new TensorDomain(dom, contig_info), dtype);
}

void checkIntValue(
    StatefulExpressionEvaluator& evaluator,
    Val* val,
    Int::ScalarType expected_value) {
  TORCH_CHECK(val->isAnInt());
  const auto actual_value = evaluator.inferValue(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

} // namespace

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

// A few smoke tests for IrGraphGenerator
// (These tests exercise IrGraphGenerator through a non-trivial IR,
//  to make sure that it runs w/o crashing. The actual output is not
//  validated)
void testGPU_IrGraphGenerator() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Make sure we can handle empty IRs
  TORCH_CHECK(!IrGraphGenerator::toGraphviz(
                   &fusion, IrGraphGenerator::DetailLevel::Basic)
                   .empty());

  // Construct an interesting IR
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv2 = add(tv0, new Float(3.141));
  TensorView* tv3 = broadcast(tv0, {false, true, false, true});
  TensorView* tv4 = reductionOp(BinaryOpType::Add, {2}, new Float(0), tv3);
  TensorView* tv5 = clamp(tv4, new Float(0.f), new Float(1.f));
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
    if (!fusion.hasInput(val) &&
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

void testGPU_FusionDispatch() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f = new Float{2.f};
  std::stringstream ss1, ss2, ss3;
  ss1 << f;
  ss2 << static_cast<Val*>(f);
  ss3 << static_cast<Statement*>(f);
  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0 && ss1.str().compare(ss3.str()) == 0,
      "Error with dispatch system where results differ by passing Float* vs Val* vs Statement*.");
}

// Evaluate basic scalar operations with constant values
void testGPU_FusionExprEvalConstants() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  StatefulExpressionEvaluator evaluator(&fusion);

  auto* a = new Int(7);
  auto* b = new Int(3);

  checkIntValue(evaluator, neg(a), -7);
  checkIntValue(evaluator, add(a, b), 10);
  checkIntValue(evaluator, neg(mul(sub(a, b), div(a, b))), -8);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
}

// Evaluate basic scalar operations with bound values
void testGPU_FusionExprEvalBindings() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  StatefulExpressionEvaluator evaluator(&fusion);

  auto* a = new Int();
  auto* b = new Int();
  auto* c = add(a, b);
  auto* d = neg(ceilDiv(c, b));
  auto* e = new Int(0);

  // trying to evaluate before binding should give empty results
  TORCH_CHECK(!evaluator.inferValue(a).has_value());
  TORCH_CHECK(!evaluator.inferValue(d).has_value());

  evaluator.safeBind(a, 7);
  evaluator.safeBind(b, 3);

  // can't bind to the results of expressions
  ASSERT_ANY_THROW(evaluator.safeBind(c, 100));

  // can't bind to concrete values
  ASSERT_ANY_THROW(evaluator.safeBind(e, 100));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, sub(a, b), 4);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset evaluation context
  evaluator = StatefulExpressionEvaluator(&fusion);

  evaluator.safeBind(a, 2);
  evaluator.safeBind(b, 5);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, sub(a, b), -3);
  checkIntValue(evaluator, mod(a, b), 2);
  checkIntValue(evaluator, ceilDiv(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

// Evaluate expressions in a simple IR
void testGPU_FusionExprEvalBasic() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, new Float(2.0));
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
  StatefulExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  //
  // IMPORTANT:
  // a. The bindings are only as stable as the Vals are in the fusion graph
  // b. You must use the original (rootDomain) extents
  //  (ex. `tv0->getRootDomain()[0]->extent()`
  //   instead of `tv0->axis(0)->extent()`)
  //
  evaluator.safeBind(tv0->getRootDomain()[0]->extent(), 6);
  evaluator.safeBind(tv0->getRootDomain()[1]->extent(), 128);
  evaluator.safeBind(tv1->getRootDomain()[0]->extent(), 6);
  evaluator.safeBind(tv1->getRootDomain()[1]->extent(), 128);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->rawExtent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->rawExtent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->rawExtent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->rawExtent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->rawExtent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->rawExtent(), 128);
}

// Evaluate expressions in a more complex IR
void testGPU_FusionExprEvalComplex() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(-1.0));
  TensorView* tv2 = add(tv0, new Float(3.0));
  TensorView* tv3 = mul(tv0, new Float(2.0));
  TensorView* tv4 = add(tv2, tv1);
  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv0, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv5->reorder({{-1, 0}});

  tv6->split(0, 5);
  tv5->merge(0);

  // 1. Create an evaluator
  StatefulExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  evaluator.safeBind(tv0->getRootDomain()[0]->extent(), 129);
  evaluator.safeBind(tv0->getRootDomain()[1]->extent(), 127);

  // Evaluate and check extent values
  TORCH_CHECK(tv0->domain()->nDims() == 2);
  checkIntValue(evaluator, tv0->axis(0)->rawExtent(), 129);
  checkIntValue(evaluator, tv0->axis(1)->rawExtent(), 127);

  TORCH_CHECK(tv3->domain()->nDims() == 2);
  checkIntValue(evaluator, tv3->axis(0)->rawExtent(), 129);
  checkIntValue(evaluator, tv3->axis(1)->rawExtent(), 127);

  TORCH_CHECK(tv4->domain()->nDims() == 2);
  checkIntValue(evaluator, tv4->axis(0)->rawExtent(), 129);
  checkIntValue(evaluator, tv4->axis(1)->rawExtent(), 127);

  TORCH_CHECK(tv5->domain()->nDims() == 1);
  checkIntValue(evaluator, tv5->axis(0)->rawExtent(), 16383);

  TORCH_CHECK(tv6->domain()->nDims() == 3);
  checkIntValue(evaluator, tv6->axis(0)->rawExtent(), 26);
  checkIntValue(evaluator, tv6->axis(1)->rawExtent(), 5);
  checkIntValue(evaluator, tv6->axis(2)->rawExtent(), 127);
}

// Evaluate expressions post lowering
void testGPU_FusionExprEvalPostLower() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, new Float(2.0));
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

  auto* bid_x = add(tv3->axis(0)->rawExtent(), new Int(0));
  auto* tid_x = add(tv3->axis(-1)->rawExtent(), new Int(0));

  // Lower
  GpuLower gpulw(&fusion);

  // 1. Create an evaluation context
  StatefulExpressionEvaluator evaluator(&fusion);

  // 2. Bind values
  evaluator.safeBind(tv0->getRootDomain()[0]->extent(), 6);
  evaluator.safeBind(tv0->getRootDomain()[1]->extent(), 128);
  evaluator.safeBind(tv1->getRootDomain()[0]->extent(), 6);
  evaluator.safeBind(tv1->getRootDomain()[1]->extent(), 128);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->rawExtent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->rawExtent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->rawExtent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->rawExtent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->rawExtent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->rawExtent(), 128);

  checkIntValue(evaluator, bid_x, 2);
  checkIntValue(evaluator, tid_x, 128);
}

void testGPU_FusionClear() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Create a dummy IR

  {
    TensorView* tv0 = makeDummyTensor(2);
    TensorView* tv1 = makeDummyTensor(2);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    TensorView* tv2 = add(tv1, new Float(2.0));
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

  TORCH_CHECK(fusion.exprs().empty());
  TORCH_CHECK(fusion.vals().empty());

  TORCH_CHECK(fusion.inputs().empty());
  TORCH_CHECK(fusion.outputs().empty());

  TORCH_CHECK(!fusion.hasReduction());
  TORCH_CHECK(!fusion.hasBlockReduction());
  TORCH_CHECK(!fusion.hasGridReduction());

  // 3. Rebuild the IR

  {
    TensorView* tv0 = makeDummyTensor(3);
    TensorView* tv1 = makeDummyTensor(3);
    TensorView* tv2 = add(tv1, new Float(2.0));
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(outputs[0]));
}

void testGPU_FusionCopy() {
  Fusion original_fusion;

  // Create the test IR
  {
    FusionGuard fg(&original_fusion);

    auto tv0 = makeDummyTensor(3);
    auto tv1 = makeDummyTensor(3);
    auto tv2 = add(tv1, new Float(2.0));
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

void testGPU_FusionMove() {
  Fusion fusion;

  // Create the test IR
  {
    FusionGuard fg(&fusion);

    auto tv0 = makeDummyTensor(3);
    auto tv1 = makeDummyTensor(3);
    auto tv2 = add(tv1, new Float(2.0));
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
  TORCH_CHECK(fusion.exprs().empty());
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

void testGPU_FusionSimpleArith() {
  std::stringstream ss1, ss2;

  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();

  // Disrupt the fusion to make sure guard works well
  {
    Fusion fusion2;
    FusionGuard fg(&fusion2);

    Float* f1 = new Float(1.f);
    Float* f2 = new Float(2.f);
    add(f1, f2);
    ss2 << fusion2;
  }

  new BinaryOp(BinaryOpType::Add, f3, f1, f2);
  ss1 << fusion;

  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0,
      "Error where explicit add nodes don't match implicit add nodes.");
}

void testGPU_FusionSimpleTypePromote() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);

  TORCH_CHECK(f5->getDataType() == DataType::Float);
}

class ZeroMutator : public OptOutMutator {
 public:
  Statement* mutate(Float* f) {
    if (f->isConst() && *(f->value()) == 1.0)
      return new Float(0.0);
    return f;
  }
  void mutate(Fusion* f) {
    OptOutMutator::mutate(f);
  }
};

void testGPU_FusionMutator() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  Val* f5 = add(f4, i1);
  ZeroMutator mutator;
  mutator.mutate(&fusion);
  Val* lhs = static_cast<BinaryOp*>(fusion.origin(f5))->lhs();
  TORCH_CHECK(
      lhs->getValType().value() == ValType::Scalar &&
      lhs->getDataType().value() == DataType::Float);
  Float* flhs = static_cast<Float*>(lhs);

  TORCH_CHECK(flhs->value().value() == 0.f);
}

void testGPU_FusionRegister() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Float* v1 = new Float{1.f};
  Float* v2 = new Float{2.f};
  Val* v3 = binaryOp(BinaryOpType::Add, v1, v2);
  Val* v4 = binaryOp(BinaryOpType::Add, v1, v2);
  TORCH_CHECK(v1->name() + 1 == v2->name());
  TORCH_CHECK(v2->name() + 1 == v3->name());
  TORCH_CHECK(v3->name() + 1 == v4->name());
  TORCH_CHECK(fusion.origin(v3)->name() + 1 == fusion.origin(v4)->name());
}

// dummy expr with 2 outputs only for toposort test.
struct DummyExpr : public Expr {
  ~DummyExpr() = default;
  DummyExpr(Val* _outlhs, Val* _outrhs, Val* _lhs, Val* _rhs)
      : Expr(ExprType::UnaryOp) // Not terribly safe...
  {
    addOutput(_outlhs);
    addOutput(_outrhs);
    addInput(_lhs);
    addInput(_rhs);
    this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  }
  DummyExpr(const DummyExpr& other) = delete;
  DummyExpr& operator=(const DummyExpr& other) = delete;
  DummyExpr(DummyExpr&& other) = delete;
  DummyExpr& operator=(DummyExpr&& other) = delete;
};

void testGPU_FusionTopoSort() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // e0: v3, v2 = dummy(v1, v0)
  // e1: v4     =   add(v3, v2)
  // e2: v5     =   add(v2, v4)
  // e3: v6     =   add(v5, v5)
  Float* v0 = new Float{1.f};
  Float* v1 = new Float{2.f};
  Float* v2 = new Float();
  Float* v3 = new Float();
  Float* v4 = new Float();
  Float* v5 = new Float();
  Float* v6 = new Float();

  Expr* e0 = new DummyExpr(v3, v2, v1, v0);
  Expr* e1 = new BinaryOp(BinaryOpType::Add, v4, v3, v2);
  Expr* e2 = new BinaryOp(BinaryOpType::Add, v5, v2, v4);
  Expr* e3 = new BinaryOp(BinaryOpType::Add, v6, v5, v5);

  std::vector<Expr*> exprs = fusion.exprs();

  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  fusion.addOutput(v2);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 1);
  TORCH_CHECK(exprs[0] == e0);

  fusion.addOutput(v5);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v4);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v3);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v6);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  TORCH_CHECK(fusion.origin(v2)->name() == 0);
  TORCH_CHECK(fusion.origin(v3)->name() == 0);
  TORCH_CHECK(fusion.origin(v4)->name() == 1);
  TORCH_CHECK(fusion.origin(v5)->name() == 2);
  TORCH_CHECK(fusion.origin(v6)->name() == 3);
}

void testGPU_FusionTensor() {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  {
    auto tensor = at::randn({2, 3, 4, 5}, options);
    auto tensor_type = TensorType::create(tensor);
    auto fuser_tensor = new TensorView(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (int i = 0; i < static_cast<int>(fuser_tensor->nDims()); i++) {
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
    auto fuser_tensor = new TensorView(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (int i = 0; i < static_cast<int>(fuser_tensor->nDims()); i++) {
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
    auto fuser_tensor = new TensorView(tensor_type);
    TORCH_CHECK((int64_t)fuser_tensor->nDims() == tensor.dim());
    TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
    TORCH_CHECK(fuser_tensor->domain() != nullptr);
    for (int i = 0; i < static_cast<int>(fuser_tensor->nDims()); i++) {
      // size 1 dimension are makred as broadcast
      TORCH_CHECK(fuser_tensor->axis(i)->isBroadcast() == false);
    }
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[0]);
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[1]);
    TORCH_CHECK(fuser_tensor->domain()->contiguity()[2]);
    TORCH_CHECK(!fuser_tensor->domain()->contiguity()[3]);
  }
}

void testGPU_FusionFilterVals() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeDummyTensor(1);
  auto tv1 = makeDummyTensor(1);
  auto scalar0 = new Float(0);
  auto scalar1 = new Int(0);
  auto scalar2 = new Int(1);

  const std::vector<Val*> vals = {tv0, scalar0, tv1, scalar1, scalar2};

  std::vector<TensorView*> tvs(
      ir_utils::filterByType<TensorView>(vals).begin(),
      ir_utils::filterByType<TensorView>(vals).end());
  TORCH_CHECK(tvs.size() == 2);
  TORCH_CHECK(tvs[0] == tv0);
  TORCH_CHECK(tvs[1] == tv1);

  std::vector<Float*> floats(
      ir_utils::filterByType<Float>(vals).begin(),
      ir_utils::filterByType<Float>(vals).end());
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

void testGPU_FusionTVSplit() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeDummyTensor(3);

  tv = tv->split(2, 2);
  TORCH_CHECK(tv->nDims() == 4);
  Expr* outer = tv->axis(2)->extent()->getOrigin();

  TORCH_CHECK(
      outer->getExprType().value() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(outer)->getBinaryOpType() ==
          BinaryOpType::CeilDiv &&
      static_cast<BinaryOp*>(outer)->lhs()->sameAs(
          tv->getRootDomain()[2]->extent()) &&
      static_cast<Int*>(static_cast<BinaryOp*>(outer)->rhs())
          ->sameAs(new Int(2)));

  IterDomain* inner = static_cast<IterDomain*>(tv->axis(3));
  TORCH_CHECK(
      inner->extent()->isScalar() &&
      static_cast<Int*>(inner->extent())->isConst() &&
      static_cast<Int*>(inner->extent())->value().value() == 2);
}

void testGPU_FusionTVMerge() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeDummyTensor(3);

  tv = tv->merge(1);
  Expr* axisOp = tv->axis(1)->extent()->getOrigin();

  TORCH_CHECK(
      tv->nDims() == 2 && axisOp->getExprType() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(axisOp)->getBinaryOpType() == BinaryOpType::Mul &&
      static_cast<BinaryOp*>(axisOp)->lhs() ==
          tv->getRootDomain()[1]->extent() &&
      static_cast<BinaryOp*>(axisOp)->rhs() ==
          tv->getRootDomain()[2]->extent());
}

void testGPU_FusionTVReorder() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::unordered_map<int, int> shift_right{{-1, 0}};

  std::unordered_map<int, int> shift_left{{0, -1}};

  std::unordered_map<int, int> shift_left_2{{0, -1}, {1, 0}, {2, 1}};

  std::unordered_map<int, int> swap{{0, 2}, {2, 0}};

  auto tv = makeDummyTensor(3);
  std::vector<IterDomain*> ref;
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_left);
  for (int i = 0; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref[i]->sameAs(tv->axis(i - 1)));

  tv = makeDummyTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_left);
  for (int i = 0; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref[i]->sameAs(tv->axis(i - 1)));

  tv = makeDummyTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());

  tv->reorder(shift_right);
  TORCH_CHECK(ref[ref.size() - 1]->sameAs(tv->axis(0)));
  for (int i = 1; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref[i - 1]->sameAs(tv->axis(i)));

  tv = makeDummyTensor(3);
  ref = std::vector<IterDomain*>(
      tv->domain()->domain().begin(), tv->domain()->domain().end());
  tv->reorder(swap);
  TORCH_CHECK(ref[0]->sameAs(tv->axis(2)));
  TORCH_CHECK(ref[2]->sameAs(tv->axis(0)));
  TORCH_CHECK(ref[1]->sameAs(tv->axis(1)));
}

void testGPU_FusionEquality() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* fval1 = new Float();
  Float* fval1_copy = fval1;
  Float* fval2 = new Float();
  Float* fone = new Float(1.0);

  TORCH_CHECK(fval1->sameAs(fval1_copy));
  TORCH_CHECK(!fval1->sameAs(fval2));
  TORCH_CHECK(!fone->sameAs(fval1));
  TORCH_CHECK(fone->sameAs(new Float(1.0)));

  Int* ival1 = new Int();
  Int* ival1_copy = ival1;
  Int* ival2 = new Int();
  Int* ione = new Int(1);

  TORCH_CHECK(ival1->sameAs(ival1_copy));
  TORCH_CHECK(!ival1->sameAs(ival2));
  TORCH_CHECK(!ione->sameAs(ival1));
  TORCH_CHECK(ione->sameAs(new Int(1)));

  BinaryOp* add1 = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* add1_copy =
      new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* sub1 = new BinaryOp(BinaryOpType::Sub, new Float(), fval1, ival1);

  UnaryOp* neg1 = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);
  UnaryOp* neg2 = new UnaryOp(UnaryOpType::Neg, new Float(), fval2);
  UnaryOp* neg1_copy = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);

  TORCH_CHECK(add1->sameAs(add1_copy));
  TORCH_CHECK(!add1->sameAs(sub1));

  TORCH_CHECK(neg1->sameAs(neg1_copy));
  TORCH_CHECK(!static_cast<Expr*>(neg1)->sameAs(add1));
  TORCH_CHECK(!neg1->sameAs(neg2));
}

void testGPU_FusionDependency() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f0 = new Float(0.f);
  Float* f1 = new Float(1.f);
  auto f2 = add(f0, f1);

  auto f3 = add(f2, f2);

  Float* f4 = new Float(4.f);
  Float* f5 = new Float(5.f);
  auto f6 = add(f4, f5);

  Float* f7 = new Float(7.f);
  Float* f8 = new Float(8.f);
  auto f9 = add(f7, f8);

  auto f10 = add(f6, f9);

  auto f11 = add(f3, f10);

  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f1, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f3, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f6, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f9, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f2));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f3));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f4, f6));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f8, f10));

  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f1));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f3));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f5));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f2, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f3, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f6, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f10, f8));

  auto dep_chain = DependencyCheck::getSingleDependencyChain(f0, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f3);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f2);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f6, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f10);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f4, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f10);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f6);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f11, f2);
  TORCH_CHECK(dep_chain.empty());
}

void testGPU_FusionParser() {
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
    graph(%0 : Float(2:1),
          %1 : Float(2:1)):
      %c0 : Float(2:1) = aten::mul(%0, %1)
      %d0 : Float(2:1) = aten::mul(%c0, %0)
      return (%d0))IR";
  torch::jit::parseIR(graph0_string, g.get());

  // strides are not yet supported in the irparser.
  for (auto val : g->block()->inputs()) {
    if (val->isCompleteTensor())
      val->setType(val->type()->cast<TensorType>()->contiguous());
  }
  for (auto node : g->block()->nodes()) {
    for (auto val : node->outputs()) {
      if (val->isCompleteTensor())
        val->setType(val->type()->cast<TensorType>()->contiguous());
    }
  }

  auto fusion = fuser::cuda::parseJitIR(g);
  FusionGuard fg(fusion.get());
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16}, options);
  at::Tensor input2 = at::randn({16}, options);
  fuser::cuda::scheduleFusion(fusion.get(), {input1, input2});

  // CONSIDER:
  // 1. this can be moved to a dedicated "golden" file
  // 2. use a fuzzy compare (ignore non-significant whitespaces for example)
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 1> T0, Tensor<float, 1> T1, Tensor<float, 1> T3){
  float T2[1];
  if ( ( ( ( ( ( blockIdx.x * 1 ) + ( 1 - 1 ) ) * 128 ) + threadIdx.x ) < T0.size[0] ) ) {
    for(size_t i6 = 0; i6 < 1; ++i6 ) {
      T2[ i6 ]
         = T0[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ]
         * T1[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ];
      T3[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ]
         = T2[ i6 ]
         * T0[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ];
    }
  } else {
    for(size_t i6 = 0; i6 < 1; ++i6 ) {
      if ( ( ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) < T0.size[0] ) ) {
        T2[ i6 ]
           = T0[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ]
           * T1[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ];
      }
      if ( ( ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) < T0.size[0] ) ) {
        T3[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ]
           = T2[ i6 ]
           * T0[ ( ( ( ( blockIdx.x * 1 ) + i6 ) * 128 ) + threadIdx.x ) ];
      }
    }
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
    TORCH_CHECK(false);
  }

  cuda::FusionExecutor fe;
  fe.compileFusion(fusion.get());
  auto outputs = fe.runFusion({input1, input2});
  at::Tensor output_ref = input1 * input2 * input1;
  TORCH_CHECK(output_ref.equal(outputs[0]));
}

void testGPU_FusionForLoop() {
// TODO(kir): re-enable this test
//  due to the current "GpuLower guard" approach, we can only create
//  kernel IR during GpuLower::lower()
#if 0
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto TV0 = new TensorView(
      new TensorDomain({new IterDomain(new Int(0), new Int(16))}),
      DataType::Float);
  const auto TV1 = new TensorView(
      new TensorDomain({new IterDomain(new Int(0), new Int(16))}),
      DataType::Float);

  fusion.addInput(TV0);
  fusion.addInput(TV1);

  auto ID0 = new kir::IterDomain(new IterDomain(new Int(0), new Int(8)));

  TensorView* TV2 = add(TV0, TV1);
  BinaryOp* op = static_cast<BinaryOp*>(TV2->getOrigin());
  fusion.addOutput(TV2);

  auto fl = new kir::ForLoop(new kir::Int(c10::nullopt), ID0, {op});

  std::stringstream result;
  std::stringstream ref;
  result << fl;
  ref << "for(size_t i3{0}; i3 < iS{8}; ++i3 ) {\nT2[ iS{16} ] = T0[ iS{16} ] + T1[ iS{16} ]\n}";

  if (result.str().compare(ref.str()) == 0) {
    std::stringstream err_msg;
    err_msg << "ForLoop printing has changed or something has gone wrong. "
            << result.str() << "\n does not match reference: " << ref.str()
            << std::endl;
    TORCH_CHECK(false, err_msg.str());
  }
#endif
}

void testGPU_FusionCodeGen() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(3);

  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));
  TensorView* tv1 = add(tv0, new Float(2.0));
  TensorView* tv2 = add(tv1, new Float(3.0));
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({}, {output});

  at::Tensor output_ref = at::zeros_like(output, options);
  output_ref = output_ref + 0.0 + 1.0 + 2.0 + 3.0;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionCodeGen2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(3);
  TensorView* tv1 = makeDummyTensor(3);
  TensorView* tv2 = add(tv1, new Float(2.0));
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(outputs[0]));
}

void testGPU_FusionSimplePWise() {
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
  TensorView* tv2 = add(tv1, new Float(2.0));
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionExecKernel() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, new Float(2.0));
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor check = at::full({1, 128}, 4, options);
  ;
  TORCH_CHECK(outputs[0].equal(check));
}

int ceilDiv_(int a, int b) {
  return (a + b - 1) / b;
}

void testGPU_FusionAdvancedComputeAt() {
  // Case 1
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 + 3
  // tv4 = tv1 * 2
  // tv5 = tv3 + tv2
  // tv6 = tv5 + tv4
  // tv7 = tv1 + tv4
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    TensorView* tv1 = mul(tv0, new Float(0.5));
    TensorView* tv2 = mul(tv1, new Float(-1.0));
    TensorView* tv3 = add(tv1, new Float(3.0));
    TensorView* tv4 = mul(tv1, new Float(2.0));
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

    TORCH_CHECK(tv1->hasComputeAt() && tv1->nDims() == 3);
    TORCH_CHECK(tv2->getComputeAtView() == tv5 && tv2->nDims() == 3);
    TORCH_CHECK(tv3->getComputeAtView() == tv5 && tv3->nDims() == 3);
    TORCH_CHECK(tv4->hasComputeAt() && tv4->nDims() == 3);
    TORCH_CHECK(tv5->getComputeAtView() == tv6 && tv5->nDims() == 3);
    TORCH_CHECK(tv6->getComputeAtView() == tv7 && tv6->nDims() == 3);
    TORCH_CHECK(!tv7->hasComputeAt());

    for (Val* val : fusion.vals()) {
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);
        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor t0 = at::randn({129, 127}, options);

    auto t1 = t0.mul({0.5});
    auto t2 = t1.mul({-1.0});
    auto t3 = t1.add({3.0});
    auto t4 = t1.mul({2.0});
    auto t5 = t3.add(t2);
    auto t6 = t5.add(t4);
    auto t7 = t1.add(t4);

    at::Tensor kernel_tv6 = at::empty_like(t0, options);
    at::Tensor kernel_tv7 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0}, {kernel_tv6, kernel_tv7});

    TORCH_CHECK(at::allclose(kernel_tv6, t6));
    TORCH_CHECK(at::allclose(kernel_tv7, t7));
  }

  // Case 2
  // tv1 = tv0 * -1
  // tv2 = tv0 + 3
  // tv3 = tv0 * 2
  // tv4 = tv2 + tv1
  // tv5 = tv4 + tv3
  // tv6 = tv5 + tv3
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    TensorView* tv1 = mul(tv0, new Float(-1.0));
    TensorView* tv2 = add(tv0, new Float(3.0));
    TensorView* tv3 = mul(tv0, new Float(2.0));
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
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);

        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({129, 127}, options);

    auto t1 = t0.mul({-1.0});
    auto t2 = t0.add({3.0});
    auto t3 = t0.mul({2.0});
    auto t4 = t2.add(t1);
    auto t5 = t4.add(t3);
    auto t6 = t5.add(t3);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0});

    TORCH_CHECK(at::allclose(outputs[0], t5));
    TORCH_CHECK(at::allclose(outputs[1], t6));
  }

  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(4);
    fusion.addInput(tv0);

    TensorView* tv1 = makeDummyTensor(4);
    fusion.addInput(tv1);

    TensorView* tv2 = mul(tv1, new Float(.979361));
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
      if (!fusion.hasInput(val) &&
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
    auto t3 = t2.mul(t0);

    at::Tensor kernel_tv3 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0, t1}, {kernel_tv3});

    TORCH_CHECK(at::allclose(kernel_tv3, t3));
  }

  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(4);
    fusion.addInput(tv0);

    TensorView* tv1 = makeDummyTensor(4);
    fusion.addInput(tv1);

    TensorView* tv2 = makeDummyTensor(4);
    fusion.addInput(tv2);

    TensorView* tv3 = makeDummyTensor(4);
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
      if (!fusion.hasInput(val) &&
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
    auto t6 = t5.sub(t0);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1, t2, t3});

    TORCH_CHECK(at::allclose(outputs[0], t6));
  }

  // Case 5
  // tv2 = tv0 + 2.0
  // tv3 = tv1 * tv2
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);
    TensorView* tv1 = makeDummyTensor(2);
    fusion.addInput(tv1);
    TensorView* tv2 = add(tv0, new Float(2.0));
    TensorView* tv3 = mul(tv1, tv2);
    fusion.addOutput(tv3);

    tv3->merge(0);
    tv3->split(-1, 8);
    tv3->split(-1, 4);

    tv2->computeAt(tv3, 1);
    tv2->split(-1, 4); // Kernel will break without this split
    tv3->axis(0)->parallelize(ParallelType::BIDx);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({63, 65}, options);
    at::Tensor t1 = at::rand_like(t0, options);

    auto t2 = t0.add(2.0);
    auto t3 = t1.mul(t2);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1});

    TORCH_CHECK(at::allclose(outputs[0], t3));
  }
}

void testGPU_FusionComputeAtMultiConsumers() {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -2
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = mul(tv1, new Float(-1.0));
  TensorView* tv3 = mul(tv1, new Float(-2.0));
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

  // Note that tv2 is also computed at tv3.
  TORCH_CHECK(tv1->getComputeAtView() == computeAtTarget);
  TORCH_CHECK(tv2->getComputeAtView() == tv3);
  TORCH_CHECK(!tv3->hasComputeAt());

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);
  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({1000}, options);

  auto t1 = t0 * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;

  at::Tensor kernel_tv2 = at::empty_like(t0, options);
  at::Tensor kernel_tv3 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {kernel_tv2, kernel_tv3});

  TORCH_CHECK(at::allclose(kernel_tv2, t2));
  TORCH_CHECK(at::allclose(kernel_tv3, t3));
}

// Similar to ComputeAtMultiConsumers, but with a common consumer.
void testGPU_FusionComputeAtCommonConsumer1() {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -2
  // tv4 = tv2 + tv3
  // tv5 = tv4 * 5
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = mul(tv1, new Float(-1.0));
  TensorView* tv3 = mul(tv1, new Float(-2.0));
  TensorView* tv4 = add(tv2, tv3);
  TensorView* tv5 = mul(tv4, new Float(5.0));
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  // Computing tv1 at tv3. This will affect tv2 as discussed in
  // ComplexComputeAt1. Additionally, in this case, notice that tv4 is
  // the common consumer of tv2 and tv3, so they are computed at
  // tv4. The indirect propagation of the computeAt should stop at the
  // common consumer, and no further change should occur. More
  // specifically, tv4 and tv5 should not have a computeAt tensor.
  TensorView* computeAtTarget = tv3;
  computeAtTarget->split(0, 128);
  tv1->computeAt(computeAtTarget, 1);

  TensorView* affected_tensors[] = {tv1, tv2, tv3, tv4};
  for (auto tv : affected_tensors) {
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  TORCH_CHECK(tv1->getComputeAtView() == computeAtTarget);
  TORCH_CHECK(tv2->getComputeAtView() == tv4);
  TORCH_CHECK(tv3->getComputeAtView() == tv4);
  TORCH_CHECK(!tv4->hasComputeAt());
  TORCH_CHECK(!tv5->hasComputeAt());

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({1000}, options);

  auto t1 = t0 * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;
  auto t4 = t2 + t3;
  auto t5 = t4 * 5.0;

  at::Tensor kernel_tv3 = at::empty_like(t0, options);
  at::Tensor kernel_tv4 = at::empty_like(t0, options);
  at::Tensor kernel_tv5 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {kernel_tv3, kernel_tv4, kernel_tv5});

  TORCH_CHECK(at::allclose(kernel_tv3, t3));
  TORCH_CHECK(at::allclose(kernel_tv4, t4));
  TORCH_CHECK(at::allclose(kernel_tv5, t5));
}

void testGPU_FusionComputeAtCommonConsumer2() {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -1
  // tv4 = tv1 + 4
  // tv5 = tv3 + tv4
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = mul(tv1, new Float(-1.0));
  TensorView* tv3 = mul(tv2, new Float(-1.0));
  TensorView* tv4 = add(tv1, new Float(4.0));
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
    if (fusion.hasInput(val) ||
        val->getValType().value() != ValType::TensorView) {
      continue;
    }
    TensorView* tv = val->as<TensorView>();
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  TORCH_CHECK(tv1->getComputeAtView() == tv2);
  TORCH_CHECK(tv2->getComputeAtView() == tv3);
  // tv3 and tv4 are computed at tv5
  TORCH_CHECK(tv3->getComputeAtView() == tv5);
  TORCH_CHECK(tv4->getComputeAtView() == tv5);
  TORCH_CHECK(!tv5->hasComputeAt());

  for (Val* val : fusion.vals()) {
    if (!fusion.hasInput(val) &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({129, 127}, options);

  auto t1 = t0.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t2.mul({-1.0});
  auto t4 = t1.add({4.0});
  auto t5 = t3 + t4;

  at::Tensor kernel_tv5 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {kernel_tv5});

  TORCH_CHECK(at::allclose(kernel_tv5, t5));
}

// Similar to the above common consumer test but adds an additional
// tensor that has no common consumer with the other tensors.
void testGPU_FusionComputeAtCommonConsumer3() {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv2 * -1
  // tv4 = tv1 + 4
  // tv5 = tv2 + tv3
  // tv6 = tv1 + 6
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = mul(tv1, new Float(-1.0));
  TensorView* tv3 = mul(tv2, new Float(-1.0));
  TensorView* tv4 = add(tv1, new Float(4.0));
  TensorView* tv5 = add(tv3, tv4);
  TensorView* tv6 = add(tv1, new Float(6.0));

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
  for (Val* val : fusion.vals()) {
    if (fusion.hasInput(val) ||
        val->getValType().value() != ValType::TensorView) {
      continue;
    }
    TensorView* tv = val->as<TensorView>();
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  TORCH_CHECK(tv1->getComputeAtView() == tv2);
  TORCH_CHECK(tv2->getComputeAtView() == tv3);

  // tv3 and tv4 are computed at tv5
  TORCH_CHECK(tv3->getComputeAtView() == tv5);
  TORCH_CHECK(tv4->getComputeAtView() == tv5);

  // tv5 should be computed at tv6 since tv5 is added as an output
  // before tv6. If we call fusion.addOutput(tv6) first, tv6 should be
  // computed at tv5.
  TORCH_CHECK(tv5->getComputeAtView() == tv6);
  TORCH_CHECK(!tv6->hasComputeAt());

  for (Val* val : fusion.vals()) {
    if (!fusion.hasInput(val) &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({129, 127}, options);

  auto t1 = t0.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t2.mul({-1.0});
  auto t4 = t1.add({4.0});
  auto t5 = t3 + t4;
  auto t6 = t1.add({6.0});

  at::Tensor kernel_tv5 = at::empty_like(t0, options);
  at::Tensor kernel_tv6 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {kernel_tv5, kernel_tv6});

  TORCH_CHECK(at::allclose(kernel_tv5, t5));
  TORCH_CHECK(at::allclose(kernel_tv6, t6));
}

// Similar to ComputeAtCommonConsumer1 but with an addtiona ltensor
// that does not have data dependency with the consumer.
void testGPU_FusionComputeAtNoCommonConsumer() {
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 * -2
  // tv4 = tv2 + tv3
  // tv5 = tv4 * 5
  // tv6 = tv1 * 6
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = mul(tv1, new Float(-1.0));
  TensorView* tv3 = mul(tv1, new Float(-2.0));
  TensorView* tv4 = add(tv2, tv3);
  TensorView* tv5 = mul(tv4, new Float(5.0));
  // Notice that tv6 is not a consumer of tv4.
  TensorView* tv6 = mul(tv1, new Float(6.0));
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  TensorView* computeAtTarget = tv3;
  computeAtTarget->split(0, 128);
  tv1->computeAt(computeAtTarget, 1);

  TensorView* affected_tensors[] = {tv1, tv2, tv3, tv4, tv6};
  for (auto tv : affected_tensors) {
    TORCH_CHECK(tv->nDims() == computeAtTarget->nDims());
  }

  TORCH_CHECK(tv1->getComputeAtView() == computeAtTarget);
  TORCH_CHECK(tv2->getComputeAtView() == tv4);
  TORCH_CHECK(tv3->getComputeAtView() == tv4);
  TORCH_CHECK(tv4->getComputeAtView() == tv5);
  TORCH_CHECK(tv5->getComputeAtView() == tv6);
  TORCH_CHECK(!tv6->hasComputeAt());

  computeAtTarget->axis(0)->parallelize(ParallelType::BIDx);

  for (auto tv : affected_tensors) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({1000}, options);

  auto t1 = t0 * 0.5;
  auto t2 = t1 * -1.0;
  auto t3 = t1 * -2.0;
  auto t4 = t2 + t3;
  auto t5 = t4 * 5.0;
  auto t6 = t1 * 6.0;

  at::Tensor kernel_tv3 = at::empty_like(t0, options);
  at::Tensor kernel_tv4 = at::empty_like(t0, options);
  at::Tensor kernel_tv5 = at::empty_like(t0, options);
  at::Tensor kernel_tv6 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {kernel_tv3, kernel_tv4, kernel_tv5, kernel_tv6});

  TORCH_CHECK(at::allclose(kernel_tv3, t3));
  TORCH_CHECK(at::allclose(kernel_tv4, t4));
  TORCH_CHECK(at::allclose(kernel_tv5, t5));
  TORCH_CHECK(at::allclose(kernel_tv6, t6));
}

void testGPU_FusionScalarInputs() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeDummyTensor(2);
  fusion.addInput(tv1);

  Float* f0 = new Float();
  fusion.addInput(f0);
  Float* f1 = new Float();
  fusion.addInput(f1);
  Float* f2 = new Float();
  fusion.addInput(f2);
  Float* f3 = new Float();
  fusion.addInput(f3);
  Val* f4 = mul(f0, f1);
  Val* f5 = sub(f2, f3);

  TensorView* tv2 = sub(tv1, f4);
  TensorView* tv3 = add(tv0, f5);
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
    if (!fusion.hasInput(val) &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  // f4 = f0 * f1
  // f5 = f2 - f3
  // t2 = t1 - f4
  // t3 = t0 + f5
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
  auto t4 = t3.mul(t2);

  at::Tensor kernel_tv4 = at::empty_like(t0, options);

  at::Scalar test(fl0);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion(
      {t0,
       t1,
       at::Scalar(fl0),
       at::Scalar(fl1),
       at::Scalar(fl2),
       at::Scalar(fl3)},
      {kernel_tv4});

  TORCH_CHECK(at::allclose(kernel_tv4, t4));
}

void testGPU_FusionLoopUnroll() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(3);
  TensorView* tv1 = makeDummyTensor(3);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, new Float(2.0));
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

  at::Tensor input0 = at::rand({129, 13, 3}, options);
  at::Tensor input1 = at::rand({129, 13, 3}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input0, input1});

  TORCH_CHECK(outputs[0].equal(input0.add(input1.add(2.0))));
}

/*
 * Helper function for single op testing that generates a codegen operand
 */

Val* gen_jit_operand(std::pair<ValType, DataType> desc) {
  if (desc.first == ValType::TensorView) {
    return makeDummyTensor(2, desc.second);
  } else if (desc.first == ValType::Scalar) {
    if (desc.second == DataType::Float)
      return new Float();
    else if (desc.second == DataType::Int)
      return new Int();
    else
      TORCH_CHECK("Not currently supported type", desc.first);
  } else {
    TORCH_CHECK("Not currently supported type", desc.first);
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
    if (desc.second == DataType::Float) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      if (rand)
        return IValue(at::rand({blocks, threads}, options));
      else
        return IValue(at::empty({blocks, threads}, options));
    } else if (desc.second == DataType::Half) {
      auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
      if (rand)
        return IValue(at::rand({blocks, threads}, options));
      else
        return IValue(at::empty({blocks, threads}, options));
    } else if (desc.second == DataType::Bool) {
      if (rand) {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        return IValue(at::rand({blocks, threads}, options).to(at::kBool));
      } else {
        auto options =
            at::TensorOptions().dtype(at::kBool).device(at::kCUDA, 0);
        return IValue(at::empty({blocks, threads}, options));
      }
    } else {
      TORCH_CHECK("Not currently supported type", desc.second)
    }
  } else if (desc.first == ValType::Scalar) {
    if (desc.second == DataType::Float)
      return IValue(at::Scalar(1.f));
    else if (desc.second == DataType::Int)
      return IValue(at::Scalar(1));
    else
      TORCH_CHECK("Not currently supported type", desc.first);
  } else {
    TORCH_CHECK("Not currently supported type", desc.first);
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

  at::Tensor output =
      gen_aten_operand(op, blocks, threads, /*rand*/ false).toTensor();
  std::vector<at::Tensor> output_vect = {output};
  cudaDeviceSynchronize();
  if (fusion.hasRNG())
    at::manual_seed(0);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion(aten_inputs_ivalues, output_vect);
  cudaDeviceSynchronize();

  if (fusion.hasRNG())
    at::manual_seed(0);
  at::Tensor ref_output = af(aten_inputs);
  cudaDeviceSynchronize(); // This sync shouldn't be necessary;

  std::function<std::string()> aten_inputs_to_str =
      [&aten_inputs]() -> std::string {
    int input_cnt = 1;
    std::stringstream ss;
    std::for_each(
        aten_inputs.begin(), aten_inputs.end(), [&input_cnt, &ss](IValue& iv) {
          ss << "\nINPUT" << input_cnt++ << ": " << iv.toTensor();
        });
    return ss.str();
  };

  at::Tensor diff;
  if (output.scalar_type() == at::kBool) {
    diff = at::eq(output, ref_output);
  } else {
    diff = at::sub(output, ref_output);
  }

  TORCH_CHECK(
      (output.scalar_type() == at::kBool
           ? output.equal(ref_output)
           :
           // The absolute Tolerance was raised to 1e-07 from 1e-08 to allow
           // allow for the remainder function to pass.
           output.allclose(ref_output, /*rtol*/ 1e-05, /*atol*/ 1e-07)),
      "\nOp Type: -- ",
      op_str,
      " -- had a mismatch.",
      aten_inputs_to_str(),
      "\nJIT: ",
      output,
      "\nREF: ",
      ref_output,
      "\nDIFF: ",
      diff,
      "\n");
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

void testGPU_FusionUnaryOps() {
  using OpTuple =
      std::tuple<at::Tensor (*)(const at::Tensor&), UnaryOpType, std::string>;

  // [Note: explicit tuple type for uniform initialization list]
  // Tuple type must be explicitly specified for each uniform initialization
  // list within the vector to make this code compatible with some old env
  // which we still need to support. eg. gcc 5.4 + cuda 9.2.
  std::vector<OpTuple> ops{
      OpTuple{at::abs, UnaryOpType::Abs, "abs"},
      OpTuple{at::acos, UnaryOpType::Acos, "acos"},
      OpTuple{at::asin, UnaryOpType::Asin, "asin"},
      OpTuple{at::atan, UnaryOpType::Atan, "atan"},
      // There does not appear to be an appropriate ATen function for atanh
      // OpTuple{at::atanh,      UnaryOpType::Atanh,      "atanh"      },
      OpTuple{at::ceil, UnaryOpType::Ceil, "ceil"},
      OpTuple{at::cos, UnaryOpType::Cos, "cos"},
      OpTuple{at::cosh, UnaryOpType::Cosh, "cosh"},
      OpTuple{at::erf, UnaryOpType::Erf, "erf"},
      OpTuple{at::erfc, UnaryOpType::Erfc, "erfc"},
      OpTuple{at::exp, UnaryOpType::Exp, "exp"},
      OpTuple{at::expm1, UnaryOpType::Expm1, "expm1"},
      OpTuple{at::floor, UnaryOpType::Floor, "floor"},
      OpTuple{at::frac, UnaryOpType::Frac, "frac"},
      OpTuple{at::gelu, UnaryOpType::Gelu, "gelu"},
      OpTuple{at::lgamma, UnaryOpType::Lgamma, "lgamma"},
      OpTuple{at::log, UnaryOpType::Log, "log"},
      OpTuple{at::log10, UnaryOpType::Log10, "log10"},
      OpTuple{at::log1p, UnaryOpType::Log1p, "log1p"},
      OpTuple{at::log2, UnaryOpType::Log2, "log2"},
      OpTuple{at::neg, UnaryOpType::Neg, "neg"},
      OpTuple{at::reciprocal, UnaryOpType::Reciprocal, "reciprocal"},
      OpTuple{at::relu, UnaryOpType::Relu, "relu"},
      OpTuple{at::round, UnaryOpType::Round, "round"},
      OpTuple{at::rsqrt, UnaryOpType::Rsqrt, "rsqrt"},
      OpTuple{at::sigmoid, UnaryOpType::Sigmoid, "sigmoid"},
      OpTuple{at::sin, UnaryOpType::Sin, "sin"},
      OpTuple{at::sinh, UnaryOpType::Sinh, "sinh"},
      OpTuple{at::sqrt, UnaryOpType::Sqrt, "sqrt"},
      OpTuple{at::tan, UnaryOpType::Tan, "tan"},
      OpTuple{at::tanh, UnaryOpType::Tanh, "tanh"},
      OpTuple{at::trunc, UnaryOpType::Trunc, "trunc"}};

  std::for_each(ops.begin(), ops.end(), [](OpTuple& op) {
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
        /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
        /*Inputs Tuple*/
        std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
  });

  test_op(
      /*blocks*/ 128,
      /*threads*/ 64,
      /*name*/ "rand_like",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::rand_like(vals[0].toTensor());
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* { return unaryOp(UnaryOpType::RandLike, in1); },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
}

void testGPU_FusionBinaryOps() {
  using AtenFuncSig = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
  using OpTuple = std::tuple<AtenFuncSig, BinaryOpType, std::string>;

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> logic_ops{OpTuple{at::eq, BinaryOpType::Eq, "eq"},
                                 OpTuple{at::ge, BinaryOpType::GE, "ge"},
                                 OpTuple{at::gt, BinaryOpType::GT, "gt"},
                                 OpTuple{at::le, BinaryOpType::LE, "le"},
                                 OpTuple{at::lt, BinaryOpType::LT, "lt"},
                                 OpTuple{at::ne, BinaryOpType::NE, "ne"}};

  std::for_each(logic_ops.begin(), logic_ops.end(), [](OpTuple& op) {
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
            std::make_pair(ValType::TensorView, DataType::Float),
            std::make_pair(ValType::TensorView, DataType::Float)));
  });

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> math_ops{
      OpTuple{at::atan2, BinaryOpType::Atan2, "atan2"},
      OpTuple{at::div, BinaryOpType::Div, "div"},
      OpTuple{at::fmod, BinaryOpType::Fmod, "fmod"},
      OpTuple{at::max, BinaryOpType::Max, "max"},
      OpTuple{at::min, BinaryOpType::Min, "min"},
      OpTuple{at::mul, BinaryOpType::Mul, "mul"},
      OpTuple{at::pow, BinaryOpType::Pow, "pow"},
      // NOTE: Remainder does not match the Aten impl exactly
      // despite using an identical function.
      OpTuple{at::remainder, BinaryOpType::Remainder, "remainder"},
  };

  std::for_each(math_ops.begin(), math_ops.end(), [](OpTuple& op) {
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
        /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, DataType::Float),
            std::make_pair(ValType::TensorView, DataType::Float)));
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
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
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
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
}

void testGPU_FusionTernaryOps() {
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "clamp",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::clamp(vals[0].toTensor(), 0.f, 1.f);
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* {
        return clamp(in1, new Float(0.f), new Float(1.f));
      },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "threshold",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::threshold(vals[0].toTensor(), 0.f, 1.f);
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* {
        return threshold(in1, new Float(0.f), new Float(1.f));
      },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
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
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Bool),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float)));
}

void testGPU_FusionCompoundOps() {
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
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float)));
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
      /*JIT  Func   */ static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(&addcmul),
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
}

void testGPU_FusionCastOps() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2, DataType::Half);

  TensorView* intrm1 = castOp(DataType::Float, tv0);
  TensorView* out = castOp(DataType::Half, intrm1);

  fusion.addInput(tv0);
  fusion.addOutput(out);
  tv0->computeAt(out, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor input1 = at::rand({1, 4}, options);
  at::Tensor ref_output = at::empty_like(input1);

  std::array<IValue, 1> inputs = {input1};
  const at::ArrayRef<IValue> input_ivalues(inputs);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(input_ivalues);

  ref_output = at::_cast_Half(at::_cast_Float(input1));

  TORCH_CHECK(
      outputs[0].equal(ref_output),
      "\nOp Type: -- ",
      "cast FP16->FP32->FP16",
      " -- had a mismatch.\n",
      "IN1 : ",
      input1,
      "\n",
      "JIT: ",
      outputs[0],
      "\n",
      "REF: ",
      ref_output,
      "\n");
}

// We want split/merge/reorder all tested both on and off rfactor domains, also
// want compute at into the rfactor domain, and into its consumer
void testGPU_FusionRFactorReplay() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);

  // Register your inputs
  fusion.addInput(tv0);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv1 = sum(tv0, {1});
  // tv1[I0, R1]
  tv1->split(0, 32);
  // tv1[I0o, I0i{32}, R1]
  tv1->split(0, 16);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1]
  tv1->split(-1, 8);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1o, R1i{8}]
  tv1->split(-2, 4);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1oo, R1oi{4}, R1i{8}]
  tv1->reorder({{0, -2}, {2, -1}, {-3, 0}, {-1, 1}});
  // tv1[R1oo, R1i{8}, I0oi{16}, R1oi{4}, I0oo, I0i{32}]

  tv1->merge(0);
  tv1->merge(-2);

  // tv1[R1oo*R1i{8}, I0oi{16}, R1oi{4}, I0oo*I0i{32}]
  TensorDomain* new_domain = TransformRFactor::runReplay(tv1->domain(), {0});
  // new_domain[r(R1oo*R1i{8})rf, I0oi{16}, ir1oi{4}rf, I0oo*I0i{32}]

  TensorDomain* new_domain2 = TransformRFactor::runReplay2(tv1->domain(), {0});
  // new_domain2[                 I0oi{16},           , I0oo*I0i{32}, R1oi{4}]

  // Move rfactor axis to end, keep iter rfactor axis
  new_domain->reorder({{0, -1}, {2, 2}});

  // Replay casp, replay new_domain2 as new_domain
  // reordered_new_domain[I0oi{16}, I0oo*I0i{32}, ir1oi{4}rf, R(R1oo*R1i{8})rf]
  auto replay_casp = TransformReplay::replayCasP(new_domain2, new_domain, 2);
  TensorDomain* casp = replay_casp.first;
  // new_domain[I0oi{16}, I0oo*I0i{32}, ir1oi{4}rf, R(R1oo*R1i{8})rf]
  //       casp[I0oi{16}, I0oo*I0i{32},  R1oi{4}]

  casp->split(1, new Int(2));
  // casp      [I0oi{16}, (I0oo*I0i{32})o, I(Ioo*I0i)i{2}, ir1oi{4} ]
  // new_domain[I0oi{16},  I0oo*I0i{32}  ,                 ir1oi{4}rf,
  // R(R1oo*R1i{8})rf]

  auto replay_pasc = TransformReplay::replayPasC(new_domain, casp, 2);
  TensorDomain* pasc = replay_pasc.first;
  // pasc      [I0oi{16}, (I0oo*I0i{32})o, I(Ioo*I0i)i{2}, ir1oi{4}rf,
  // R(R1oo*R1i{8})rf]

  TORCH_CHECK(
      new_domain->nDims() - 1 == new_domain2->nDims(),
      casp->nDims() == new_domain2->nDims() + 1,
      pasc->nDims() == new_domain->nDims() + 1,
      "Error in rfactor, number of dimensions is not correct.");

  TORCH_CHECK(
      !casp->sameAs(new_domain2) && !pasc->sameAs(new_domain) &&
          !new_domain->sameAs(new_domain2) &&
          !tv1->domain()->sameAs(new_domain) &&
          !tv1->domain()->sameAs(new_domain2),
      "Error in rfactor, number of dimensions is not correct.");

  auto dom = new_domain->getRootDomain();
  TORCH_CHECK(
      !dom[0]->isReduction() &&
          std::any_of(
              dom.begin(),
              dom.end(),
              [](IterDomain* id) { return id->isReduction(); }) &&
          std::any_of(
              dom.begin(),
              dom.end(),
              [](IterDomain* id) { return id->isRFactorProduct(); }),
      "Error in rFactor, there seems to be something wrong in root domain.");

  auto dom2 = new_domain2->getRootDomain();
  TORCH_CHECK(
      !dom2[0]->isReduction() &&
          std::any_of(
              dom2.begin(),
              dom2.end(),
              [](IterDomain* id) { return id->isReduction(); }),
      "Error in rFactor, there seems to be something wrong in root domain.");
}

// Start off simple, block on the outer dim
// block stride + thread all reduce + unrolling on inner dim
void testGPU_FusionReduction() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

void testGPU_FusionReduction2() {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    // tv1[I0, R1] = tv0[I0, I1]
    TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);

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
    at::Tensor input = at::rand({numel_x, numel_y}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({input});

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto aten_output = input.sum({1});
    TORCH_CHECK(aten_output.allclose(outputs[0]));
  }

  {
    // What if Z participates in the reduction with X?
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    // tv1[I0, R1] = tv0[I0, I1]
    TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);

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
    at::Tensor input = at::rand({numel_x, numel_y}, options);
    at::Tensor cg_output = at::empty({numel_x}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({input}, {cg_output});

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto aten_output = input.sum({1});
    TORCH_CHECK(aten_output.allclose(cg_output));
  }
}

void testGPU_FusionReduction3() {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    TensorView* tv1 = makeDummyTensor(2);

    TensorView* tv2 = add(tv0, tv1);
    // tv2[I0, I1] = tv0[I0, I1] + tv1[I0, I1]

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    TensorView* tv3 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv2);
    // tv3[I0, R1] = tv2[I0, I1]

    TensorView* tv4 = makeDummyTensor(1);
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
    at::Tensor t0 = at::rand({numel_x, numel_y}, options);
    at::Tensor t1 = at::rand({numel_x, numel_y}, options);
    auto t2 = t0.add(t1);
    auto t3 = t2.sum({1});
    at::Tensor t4 = at::rand({numel_x}, options);
    auto t5 = t3.mul(t4);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1, t4});

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    TORCH_CHECK(
        t5.allclose(outputs[0]), "Error of: ", t5.sub(outputs[0]).abs().max());
  }
}

void testGPU_FusionReduction4() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(3);

  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);

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
    if (!fusion.hasInput(val) &&
        val->getValType().value() == ValType::TensorView) {
      val->as<TensorView>()->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv1->axis(-2)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({bidy, dim1, tidx}, options);

  at::Tensor cg_output = at::empty({bidy, tidx}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(
      aten_output.allclose(cg_output, 1e-5, 1e-7),
      "Error of: ",
      aten_output.sub(cg_output).abs().max());
}

void testGPU_FusionReduction5() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 64;
  const int bdimy = 8;

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(3);
  fusion.addInput(tv0);

  // tv1[I0, R1, R2] = tv0[I0, I1, I2]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1, 2}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y, numel_z}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum({1, 2});
  TORCH_CHECK(aten_output.allclose(outputs[0]));
}

void testGPU_FusionReductionTFT() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

void testGPU_FusionBranches() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);
  TensorView* tv2 = makeDummyTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = add(tv0, new Float(1.0));
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv3, tv2);
  auto tv6 = add(tv4, tv5);

  fusion.addOutput(tv6);

  constexpr int x = 63, y = 33;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);
  at::Tensor t2 = at::randn({x, y}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
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

  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t1, t2});

  auto t3 = t0.add(1.0);
  auto t4 = t3.add(t1);
  auto t5 = t3.add(t2);
  auto t6 = t4.add(t5);

  TORCH_CHECK(t6.allclose(outputs[0]));
}

void testGPU_FusionSimpleBCast() {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);
    TensorView* tv1 = add(tv0, new Float(1.5));

    TensorView* tv2 = makeDummyTensor(2);
    fusion.addInput(tv2);
    TensorView* tv3 = makeDummyTensor(2);
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
    at::Tensor t7 = t5.add(t6);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t2, t3});

    TORCH_CHECK(t7.allclose(outputs[0]));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);
    TensorView* tv1 = makeDummyTensor(2);
    fusion.addInput(tv1);

    TensorView* tv2 = add(tv0, tv1);

    TensorView* tv3 = broadcast(tv2, {false, false, true});

    TensorView* tv4 = makeDummyTensor(2);
    fusion.addInput(tv4);

    TensorView* tv5 = sub(tv4, new Float(0.1));

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
    at::Tensor t7 = t3.add(t6);

    at::Tensor cg_output = at::empty({x, y, z}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0, t1, t4}, {cg_output});

    TORCH_CHECK(t7.allclose(cg_output));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    std::vector<IterDomain*> dom;
    dom.push_back(new IterDomain(new Int(0), new Int()));
    dom.push_back(new IterDomain(
        new Int(0),
        new Int(1),
        ParallelType::Serial,
        IterType::BroadcastWithStride));

    // tv0[I1, B{1}]
    TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);
    fusion.addInput(tv0);

    // tv1[I0, I1, I2]
    TensorView* tv2 = makeDummyTensor(3);
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
    auto t3 = t0.add(t2);

    at::Tensor cg_output = at::empty({x, y, z}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0, t2}, {cg_output});

    TORCH_CHECK(t3.allclose(cg_output));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    std::vector<IterDomain*> dom;
    dom.push_back(new IterDomain(
        new Int(0),
        new Int(1),
        ParallelType::Serial,
        IterType::BroadcastWithStride));
    dom.push_back(new IterDomain(new Int(0), new Int()));
    TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);

    TensorView* tv1 = makeDummyTensor(3);
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

    at::Tensor cg_output = at::empty({x, y, z}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0, t1}, {cg_output});

    auto t3 = t0.add(t1);

    TORCH_CHECK(t3.allclose(cg_output));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    constexpr int m = 2, k = 3, n = 4;

    auto zero = new Int(0);
    auto M = new IterDomain(zero, new Int(m));
    auto K = new IterDomain(zero, new Int(k));
    auto N = new IterDomain(zero, new Int(n));

    // Set up your input tensor views
    TensorView* tv0 =
        new TensorView(new TensorDomain({M, K}, {true, true}), DataType::Float);
    TensorView* tv1 =
        new TensorView(new TensorDomain({K, N}, {true, true}), DataType::Float);

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

    at::Tensor cg_output = at::empty({m, k, n}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({t0, t1}, {cg_output});

    auto t2 = t0.unsqueeze(-1).expand({m, k, n});
    auto t3 = t1.expand({m, k, n});
    auto t4 = t2.add(t3);

    TORCH_CHECK(t4.allclose(cg_output));
  }
}

void testGPU_FusionComplexBCast() {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    int x = 2, y = 3, z = 4;

    auto tv0 = makeConcreteTensor({y});
    auto tv1 = div(tv0, new Float(2.0));
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
    auto t7 = t4.unsqueeze(0).expand({x, y, z}) + t6;

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t3, t6});

    TORCH_CHECK(t7.allclose(outputs[0]));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    int x = 2, y = 3, z = 4;

    auto tv0 = makeConcreteTensor({y, z});
    auto tv1 = div(tv0, new Float(2.0));
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
    auto t1 = t0.div(2.0);
    auto t2 = t1.sum(1);
    auto t3 = t2.unsqueeze(0).expand({x, y});
    at::Tensor t4 = at::randn({x, y}, options);
    auto t5 = t3.add(t4);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t4});

    TORCH_CHECK(t5.allclose(outputs[0]));
  }
}

void testGPU_FusionAdvancedIndexing() {
  // Merging left to right is still broken in some instances. Indexing can't
  // complete because we assume we can simply traverse consumer->producer in the
  // index/extent map, but this case breaks this assumption.
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    int w = 3, x = 4, y = 7, z = 8;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    auto tv0 = makeDummyTensor(3);
    auto tv1 = makeDummyTensor(4);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = add(tv0, new Float(1.0));
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

    torch::jit::fuser::cuda::FusionExecutor fe;

    at::Tensor t0 = at::randn({x, y, z}, options);
    at::Tensor t1 = at::randn({w, x, y, z}, options);

    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1});

    auto t3 = t0.add(1.0);
    auto t4 = t3.add(t1);

    TORCH_CHECK(t4.allclose(outputs[0]));
  }

  // Merging right to left actually does work.
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    int w = 3, x = 4, y = 7, z = 8;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    auto tv0 = makeDummyTensor(3);
    auto tv1 = makeDummyTensor(4);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = add(tv0, new Float(1.0));
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

    torch::jit::fuser::cuda::FusionExecutor fe;

    at::Tensor t0 = at::randn({x, y, z}, options);
    at::Tensor t1 = at::randn({w, x, y, z}, options);

    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1});

    auto t3 = t0.add(1.0);
    auto t4 = t3.add(t1);

    TORCH_CHECK(t4.allclose(outputs[0]));
  }
  // Same issue as the first one in this section
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    int w = 3, x = 4, y = 7, z = 8;

    auto tv0 = makeDummyTensor(3);
    auto tv1 = makeDummyTensor(4);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = add(tv0, new Float(1.0));
    auto tv3 = add(tv2, tv1);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({x, y, z}, options);
    at::Tensor t1 = at::randn({w, x, y, z}, options);

    fuser::cuda::scheduleFusion(&fusion, {t0, t1});

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1});

    auto t2 = t0.add(1.0);
    auto t3 = t2.add(t1);

    TORCH_CHECK(t3.allclose(outputs[0]));
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeConcreteTensor({10, 20});
    fusion.addInput(tv0);
    TensorView* tv1 = makeConcreteTensor({10, 10, 20});
    fusion.addInput(tv1);

    TensorView* tv2 = add(tv0, new Float(1));
    TensorView* tv3 = broadcast(tv2, {true, false, false});
    TensorView* tv4 = add(tv3, tv1);
    fusion.addOutput(tv4);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({10, 20}, options);
    at::Tensor t1 = at::randn({10, 10, 20}, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    auto outputs = fe.runFusion({t0, t1});

    auto t2 = t0.add(1.0);
    auto t3 = t2.add(t1);

    TORCH_CHECK(t3.allclose(outputs[0]));
  }
}

// Test a simple Gemm but also play around with fusion executor features
void testGPU_FusionSimpleGemm() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2); // M, K
  TensorView* tv1 = makeDummyTensor(2); // K, N
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  // Lets specify a few bounds in launch params to make sure it works
  fe.runFusion(
      {t0, t1}, torch::jit::fuser::cuda::LaunchParams(1, -1, -1, 32, 4, 4));

  // Make sure bad launch params throws
  ASSERT_ANY_THROW(fe.runFusion(
      {t0, t1}, torch::jit::fuser::cuda::LaunchParams(1, 2, 3, 4, 5, 6)));

  // Don't specify any launch params
  auto outputs = fe.runFusion({t0, t1});

  auto t2 = t0.matmul(t1);
  TORCH_CHECK(
      t2.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      t2.sub(outputs[0]).abs().max());
}

// Softmax with a 1D tensor. Parallelized only with a single thread block.
void testGPU_FusionSoftmax1D() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 128;
  const int dimx = 1000;

  // Set up your input tensor views
  TensorView* input_tv0 = makeDummyTensor(1);
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {cg_output});

  auto t2 = at::_softmax(t0, -1, false);
  TORCH_CHECK(
      t2.allclose(cg_output, 1e-5, 1e-5),
      "Error of: ",
      t2.sub(cg_output).abs().max());
}

// Softmax with a 1D tensor with input normalization.
void testGPU_FusionSoftmax1DNormalized() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 128;
  const int dimx = 1000;

  // Set up your input tensor views
  TensorView* input_tv0 = makeDummyTensor(1);
  fusion.addInput(input_tv0);

  // Normalize with the max value before computing exp.
  TensorView* max_val_tv1 =
      reductionOp(BinaryOpType::Max, {-1}, new Float(0), input_tv0);
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

  TensorView* tensors_to_parallelize[] = {max_val_tv1,
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
  at::Tensor t0 = at::randn({dimx}, options);
  at::Tensor t3_output = at::empty({dimx}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0});

  auto t2 = at::_softmax(t0, -1, false);
  TORCH_CHECK(
      t2.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      t2.sub(outputs[0]).abs().max());
}

// Softmax with a 3D tensor, where the inner-most 3rd dimension is
// normalized. Pallelized with multiple thread blocks.
void testGPU_FusionSoftmax3D() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeDummyTensor(3);
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
  at::Tensor t0 = at::randn({dimx, dimy, dimz}, options);
  at::Tensor cg_output = at::empty({dimx, dimy, dimz}, options);
  at::Tensor t3_output = at::empty_like(cg_output, options);
  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0}, {cg_output});

  auto t2 = at::_softmax(t0, -1, false);
  TORCH_CHECK(
      t2.allclose(cg_output, 1e-5, 1e-5),
      "Error of: ",
      t2.sub(cg_output).abs().max());
}

// Softmax with a 3D tensor with input normalization.
void testGPU_FusionSoftmax3DNormalized() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeDummyTensor(3);
  fusion.addInput(input_tv0);

  // Normalize with the max value before computing exp.
  TensorView* max_val_tv1 =
      reductionOp(BinaryOpType::Max, {-1}, new Float(0), input_tv0);
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

  TensorView* tensors_to_parallelize[] = {max_val_tv1,
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
  at::Tensor t0 = at::randn({dimx, dimy, dimz}, options);
  at::Tensor t3_output = at::empty({dimx, dimy, dimz}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0});

  auto t2 = at::_softmax(t0, -1, false);
  TORCH_CHECK(
      t2.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      t2.sub(outputs[0]).abs().max());
}

void testGPU_FusionSoftmaxComputeAt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});

  auto tv3 = add(tv0, new Float(1.0));

  auto tv4 = mul(tv2, tv3);

  auto tv5 = sum(tv4, {1});
  auto tv6 = broadcast(tv5, {false, true});

  auto tv7 = sub(tv6, tv4);
  fusion.addOutput(tv7);

  tv1->computeAt(tv7, 1);
  ASSERT_ANY_THROW(tv1->computeAt(tv7, -1));
}

// Similar to FusionReduction but uses grid reduction
void testGPU_FusionGridReduction1() {
  const int gdimx = 32;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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

  // fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

// Same test as the above but uses BIDy and TIDx for reduction
void testGPU_FusionGridReduction2() {
  const int gdimy = 32;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(outputs[0]));
}

// Same test but uses BIDy and BIDz for reduction. No TID used.
void testGPU_FusionGridReduction3dim1() {
  const int gdimz = 32;
  const int gdimy = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

// Same as testGPU_FusionGridReduction3dim1 but reduces dimension 0
void testGPU_FusionGridReduction3dim0() {
  const int rdim = 0;
  const int gdimy = 128;
  const int gdimz = 32;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[R0, I1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {rdim}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

  tv1->split(rdim, gdimy);
  // tv1[R0o, R0i{128}, I1] = tv0[I0, I1]
  tv1->split(rdim, gdimz);
  // tv1[R0oo, R0oi{32}, R0i{128}, I1] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({rdim});
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
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum({0});
  TORCH_CHECK(aten_output.allclose(outputs[0]));
}

// This is similar to the FusionReduction, but swaps BIDx and TIDx
void testGPU_FusionGridReduction4() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 128;
  const int gdimx = 1024;

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

// Grid reduction with 2D thread blocks but only TIDx and BIDx are
// mapped to a reduction dim
void testGPU_FusionGridReduction5() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 64;
  const int bdimy = 16;
  const int gdimx = 4;

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(outputs[0]));
}

// Similar to FusionGridReduction1 but with 3D tensors
void testGPU_FusionGridReduction6() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(3);
  fusion.addInput(tv0);

  // tv1[I0, R1, R2] = tv0[I0, I1, I2]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1, 2}, new Float(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

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
  at::Tensor input = at::rand({numel_x, numel_y, numel_z}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.sum({1, 2});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

void testGPU_FusionNonRedAxisBind() {
  int bid_x = 3;
  int tid_x = 2;
  int red_dim = 0;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {red_dim}, new Float(0), tv0);
  fusion.addOutput(tv1);

  tv1->split(-1, tid_x);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({16, bid_x * tid_x}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum({red_dim});

  TORCH_CHECK(
      aten_output.allclose(outputs[0]),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSplitBCast() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* input_tv0 = makeDummyTensor(3);
  TensorView* input_tv1 = makeDummyTensor(3);
  fusion.addInput(input_tv0);
  fusion.addInput(input_tv1);

  TensorView* sum_tv2 =
      reductionOp(BinaryOpType::Add, {2}, new Float(0), input_tv0);
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({t0, t1}, {cg_output});
}

void testGPU_FusionBCastInnerDim() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // reduce then broadcast
  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {false, true});

  TORCH_CHECK(!tv2->axis(0)->isReduction() && tv2->axis(1)->isBroadcast());
}

void testGPU_FusionBCastReduce() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);

  auto tv1 = broadcast(tv0, {true, false, false});
  auto tv2 = sum(tv1, {1});
  TORCH_CHECK(
      tv2->axis(0)->isBroadcast() && tv2->axis(1)->isReduction() &&
      !tv2->axis(2)->isBroadcast() && !tv2->axis(2)->isReduction());
}

// Multiple consumer reduction with computeAt
// https://github.com/csarofeen/pytorch/issues/110
void testGPU_FusionReductionMultiConsumer() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);
  auto tv1 = unaryOp(UnaryOpType::Exp, tv0);
  auto tv2 = reductionOp(BinaryOpType::Max, {-1}, new Float(0), tv1);
  auto tv3 = reductionOp(BinaryOpType::Min, {-1}, new Float(0), tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);
  tv1->computeAt(tv2, -1);

  TORCH_CHECK(
      (tv1->getComputeAtView() == tv2 || tv1->getComputeAtView() == tv3) &&
      tv1->getThisComputeAtAxis() == 2 && tv1->getRelativeComputeAtAxis() == 2);
}

void testGPU_FusionComputeAtExprOrder() {
  {
    for (int i = 0; i < 2; ++i) {
      Fusion fusion;
      FusionGuard fg(&fusion);

      // Set up your input tensor views
      TensorView* tv0 = makeDummyTensor(1);
      fusion.addInput(tv0);

      auto tv1 = add(tv0, new Float(1));
      auto tv2 = add(tv0, new Float(1));
      TensorView* tv3 = add(tv1, tv2);
      if (i == 0) {
        tv1->computeAt(tv3, -1);
        fusion.addOutput(tv2);
      } else {
        tv2->computeAt(tv3, -1);
        fusion.addOutput(tv1);
      }
      fusion.addOutput(tv3);

      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor input = at::rand({100}, options);

      torch::jit::fuser::cuda::FusionExecutor fe;
      fe.compileFusion(&fusion);
      auto outputs = fe.runFusion({input});

      auto aten_output = (input + 1) * 2;
      TORCH_CHECK(
          aten_output.allclose(outputs[1]),
          "Error of: ",
          aten_output.sub(outputs[1]).abs().max());
    }
  }
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    auto tv1 = add(tv0, new Float(1));
    auto tv2 = add(tv0, new Float(1));
    TensorView* tv3 = add(tv1, tv2);
    fusion.addOutput(tv3);

    tv3->split(-1, 32);

    tv1->computeAt(tv3, -1);
    tv2->computeAt(tv3, -2);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::rand({100, 100}, options);
    at::Tensor output = at::empty_like(input, options);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);
    fe.runFusion({input}, {output});

    auto aten_output = (input + 1) * 2;
    TORCH_CHECK(
        aten_output.allclose(output),
        "Error of: ",
        aten_output.sub(output).abs().max());
  }
}

void testGPU_FusionZeroDimComputeAt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, new Float(1));
  fusion.addOutput(tv2);
  TORCH_CHECK(tv2->nDims() == 0);
  tv1->computeAt(tv2, 0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({100}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum() + 1;
  TORCH_CHECK(
      aten_output.allclose(outputs[0]),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionZeroDimBroadcast() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(0);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, true});
  TORCH_CHECK(tv1->nDims() == 2);

  TensorView* tv2 = makeDummyTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv3->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::rand({}, options);
  at::Tensor input2 = at::rand({10, 10}, options);
  at::Tensor output = at::empty({}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input1, input2}, {output});

  auto aten_output =
      (input1.unsqueeze(-1).unsqueeze(-1).expand({10, 10}) + input2).sum();
  TORCH_CHECK(
      aten_output.allclose(output),
      "Error of: ",
      aten_output.sub(output).abs().max());
}

void testGPU_FusionZeroDimReduction() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int bdimx = 32;
  const int gdimx = 32;

  TensorView* tv0 = makeDummyTensor(1);
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
  at::Tensor input = at::rand({1000}, options);
  at::Tensor output = at::empty({}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {output});

  auto aten_output = input.sum();
  TORCH_CHECK(
      aten_output.allclose(output),
      "Error of: ",
      aten_output.sub(output).abs().max());
}

void testGPU_FusionBCastAfterReduce() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  const int tidx = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});

  tv1->split(1, tidx);
  auto tv3 = tv1->rFactor({-2});

  TensorView* tv4 = makeDummyTensor(2);
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t4});

  auto t3 = t0.sum({1}).unsqueeze(-1).expand({x, y});
  auto t5 = t3.add(t4);

  // Error is larger than the default threshold
  TORCH_CHECK(t5.allclose(outputs[0], 1e-5, 1e-5));
}

void testGPU_FusionReductionScheduler() {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {red_dim}, new Float(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({bid_x, tid_x}, options);

  // Apply reduction heuristic
  const at::ArrayRef<c10::IValue> inputs({input});

  const auto rparams = cuda::getReductionHeuristics(&fusion, inputs, tv1);
  TORCH_CHECK(rparams.has_value(), "Reduction heuristics was not generated!");
  cuda::scheduleReduction(&fusion, rparams.value(), tv1, {});

  cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  // no broadcasting needed, omitting the last optional argument;
  auto outputs = fe.runFusion({input});
  auto aten_output = input.sum({red_dim});

  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-04, 1e-04),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

// Simple reduction parallelized on a symbolic size.
void testGPU_FusionSymbolicReduction() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
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
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  // How many threads to use for the block reduction
  int runtime_threadIdx_dim = 128;

  torch::jit::fuser::cuda::FusionExecutor executor;
  executor.compileFusion(&fusion);
  auto outputs = executor.runFusion(
      {input},
      torch::jit::fuser::cuda::LaunchParams(
          -1, -1, -1, runtime_threadIdx_dim, -1, -1));

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(outputs[0]));
}

void testGPU_FusionReductionSchedulerMultiDimNonFastest() {
  const std::vector<int> red_dims = {0, 2};
  // Copy is because CodeGen requires int and Pytorch requires int64_t
  // for a vector of reduction dimensions
  const std::vector<int64_t> red_dims64 = {0, 2};
  const std::vector<int64_t> tensor_dims_in = {5, 10, 15, 20};
  const std::vector<int64_t> tensor_dims_out = {10, 20};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, red_dims, new Float(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(tensor_dims_in, options);
  at::Tensor cg_output = at::empty(tensor_dims_out, options);

  // Apply reduction heuristic
  const at::ArrayRef<c10::IValue> inputs({input});

  const auto rparams = cuda::getReductionHeuristics(&fusion, inputs, tv1);
  TORCH_CHECK(rparams.has_value(), "Reduction heuristics was not generated!");
  cuda::scheduleReduction(&fusion, rparams.value(), tv1, {});

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum(red_dims64);

  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-04, 1e-04),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionReductionSchedulerMultiDimFastest() {
  const std::vector<int> red_dims = {1, 3};
  // Copy is because CodeGen requires int and Pytorch requires int64_t
  // for a vector of reduction dimensions
  const std::vector<int64_t> red_dims64 = {1, 3};
  const std::vector<int64_t> tensor_dims_in = {5, 10, 15, 20};
  const std::vector<int64_t> tensor_dims_out = {5, 15};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, red_dims, new Float(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(tensor_dims_in, options);

  const auto rparams = cuda::getReductionHeuristics(&fusion, {input}, tv1);
  TORCH_CHECK(rparams.has_value(), "Reduction heuristics was not generated!");
  cuda::scheduleReduction(&fusion, rparams.value(), tv1, {});

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = input.sum(red_dims64);

  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-05, 1e-05),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionReductionSchedulerDimShmoo() {
  std::vector<bool> fp16_usage = {true, false};
  std::vector<int> red_axis = {1, 0};
  std::vector<int> output_dims = {320, 640};
  std::vector<int> red_dims;

  // Tried to cut down the number iterations with just
  // doing every other power of 2.
  for (int i = 1; i <= 1024 * 1024; i <<= 2) {
    red_dims.push_back(i);
  }

  for (auto fp16 : fp16_usage) {
    for (auto& axis : red_axis) {
      for (auto& odim : output_dims) {
        for (auto& rdim : red_dims) {
          Fusion fusion;
          FusionGuard fg(&fusion);

          TensorView* tv0 =
              makeDummyTensor(2, (fp16 ? DataType::Half : DataType::Float));
          fusion.addInput(tv0);

          torch::jit::fuser::Val* tv0_cast = nullptr;
          if (fp16) {
            tv0_cast = castOp(DataType::Float, tv0);
          }

          TensorView* tv1 = reductionOp(
              BinaryOpType::Add,
              {axis},
              new Float(0),
              (fp16 ? tv0_cast->as<TensorView>() : tv0));

          TensorView* tv1_cast = nullptr;
          if (fp16) {
            tv1_cast = castOp(DataType::Half, tv1);
          }

          fusion.addOutput((fp16 ? tv1_cast : tv1));

          auto options = at::TensorOptions()
                             .dtype((fp16 ? at::kHalf : at::kFloat))
                             .device(at::kCUDA, 0);
          at::Tensor input =
              (axis ? at::randn({odim, rdim}, options)
                    : at::randn({rdim, odim}, options));

          const at::ArrayRef<c10::IValue> inputs({input});
          std::vector<TensorView*> outputs_of_red;
          if (fp16) {
            outputs_of_red.push_back(tv1_cast);
          }
          const auto rparams =
              cuda::getReductionHeuristics(&fusion, inputs, tv1);
          TORCH_CHECK(
              rparams.has_value(), "Reduction heuristics was not generated!");
          cuda::scheduleReduction(
              &fusion, rparams.value(), tv1, outputs_of_red);

          torch::jit::fuser::cuda::FusionExecutor fe;
          fe.compileFusion(&fusion);

          auto cg_output = fe.runFusion({input});
          auto aten_output = input.sum({axis});

          TORCH_CHECK(
              aten_output.allclose(cg_output[0], 1e-03, 1e-03),
              "Error of: ",
              aten_output.sub(cg_output[0]).abs().max());
        }
      }
    }
  }
}

void testGPU_FusionCacheBefore() {
  // TVM Cache Write
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = add(tv0, new Float(1.0));
  TensorView* tv2 = mul(tv1, new Float(3.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);
  // Before: TV2 = TV1 * 3
  // After:  TV3 = TV1 * 3;
  //         TV2 = TV3;

  constexpr int BSX = 32;
  tv2->split(-1, BSX);
  tv0->computeAt(tv2, -1);

  // cache_before automatically applies ComputeAt to the cache TensorView
  tv2->cache_before();

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 750;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({M, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  at::Tensor aten_output = (input + 1.0) * 3.0;
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
}

void testGPU_FusionCacheAfter() {
  // TVM Cache Read
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = add(tv0, new Float(1.0));
  TensorView* tv2 = mul(tv1, new Float(3.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);
  // Before: TV1 = TV0 + 1
  // After:  TV3 = TV0;
  //         TV1 = TV3 + 1

  constexpr int BSX = 32;
  tv2->split(-1, BSX);
  tv0->computeAt(tv2, -1);

  // cache_after automatically applies ComputeAt to the cache TensorView
  tv0->cache_after();

  // Thread and Block binding
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 457;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({M, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  at::Tensor aten_output = (input + 1.0) * 3.0;
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
}

void testGPU_FusionCacheIndirect() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);
  TensorView* tv2 = makeDummyTensor(2);
  TensorView* tv3 = makeDummyTensor(2);
  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addOutput(tv6);
  // t6 = ((t1 + (t2 - t3)) - t0)

  // cache_after on inputs placed before schedule
  constexpr int BSX = 32;
  tv6->split(-1, BSX);
  tv2->computeAt(tv6, -1);

  tv5->cache_after();
  tv5->cache_before();

  // Thread and Block binding
  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 32, N = 810;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in0 = at::rand({M, N}, options);
  at::Tensor in1 = at::rand({M, N}, options);
  at::Tensor in2 = at::rand({M, N}, options);
  at::Tensor in3 = at::rand({M, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({in0, in1, in2, in3});

  at::Tensor aten_output = (in1 + (in2 - in3)) - in0;
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
}

void testGPU_FusionCacheBcast() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeDummyTensor(1); // (M, 1)
  TensorView* tv1 = broadcast(tv0, {false, true});
  TensorView* tv2 = makeDummyTensor(1); // (1, N)
  TensorView* tv3 = broadcast(tv2, {true, false});
  TensorView* tv4 = mul(tv1, tv3);
  fusion.addInput(tv0);
  fusion.addInput(tv2);
  fusion.addOutput(tv4);

  constexpr int BSX = 128;
  tv4->split(0, BSX);
  tv4->split(-1, BSX);
  tv4->reorder({{0, 0}, {1, 2}, {2, 1}, {3, 3}});
  // M/BSX, N/BSY, BSX, BSY
  tv0->computeAt(tv4, 2);
  tv2->computeAt(tv4, 2);
  // 0, 1 | 2, 3, 4

  // Case 1
  tv0->cache_after();

  // Case 2
  tv1->cache_before();

  // Case 3
  tv1->cache_after();

  // Case 4
  TensorView* tv8 = tv4->cache_before();

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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t1});

  at::Tensor aten_output = t0.unsqueeze(1).matmul(t1.unsqueeze(0));
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionCacheComplex() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2); // (N, N)
  TensorView* tv1 = makeDummyTensor(1); // (N)
  TensorView* tv2 = sum(tv0, {1}); // (N)
  TensorView* tv3 = broadcast(tv2, {false, true}); // (N, 1)
  TensorView* tv4 = broadcast(tv1, {true, false}); // (1, N)
  TensorView* tv5 = mul(tv3, tv4); // (N, N)
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  // Exception: Cache-Before on reduction Op
  // TensorView* tv9 = tv2->cache_before();

  constexpr int BSX = 128;
  tv5->split(0, BSX);
  tv5->split(-1, BSX);
  // M/BSX, BSX, N/BSX, BSX
  tv5->reorder({{0, 0}, {1, 2}, {2, 1}, {3, 3}});
  // M/BSX, N/BSY, BSX, BSY
  tv0->computeAt(tv5, 2);
  tv1->computeAt(tv5, 2);
  // 0, 1 | 2, 3, 4

  tv2->cache_after();
  TensorView* tv7 = tv5->cache_before();

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv7->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int N = 800;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::rand({N, N}, options);
  at::Tensor input2 = at::rand({N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input1, input2});

  at::Tensor aten_output =
      matmul(sum(input1, 1).unsqueeze(1), input2.unsqueeze(0));
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
}

void testGPU_FusionCacheMultiConsumer() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv1, new Float(2));
  TensorView* tv3 = add(tv0, new Float(1));
  TensorView* tv4 = add(tv3, new Float(2));

  fusion.addInput(tv0);
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  tv1->computeAt(tv2, -1);
  tv3->computeAt(tv4, -1);

  auto tv5 = tv1->cache_before();
  auto tv6 = tv3->cache_before();
  tv5->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);

  // Fails because tensor must be recomputed twice
  // auto tv7 = tv0->cache_after();

  constexpr int N = 800;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = (input + 1) + 2;
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
  TORCH_CHECK(
      aten_output.allclose(outputs[1], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[1]).abs().sum());
}

void testGPU_FusionSmem() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeDummyTensor(2); // (M, N)
  TensorView* tv1 = makeDummyTensor(2); // (M, N)
  TensorView* tv2 = mul(tv0, tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv2);

  // Schedule
  TensorView* tv3 = tv0->cache_after();
  TensorView* tv4 = tv1->cache_after();
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

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t1});

  at::Tensor aten_output = mul(t0, t1);
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemReduce() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeDummyTensor(3); // M, K, N
  TensorView* tv1 = sum(tv0, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  TensorView* tv2 = tv0->cache_after();
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
  at::Tensor t0 = at::randn({M, K, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0});

  at::Tensor aten_output = sum(t0, {1});
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemBlockGemm() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeDummyTensor(2); // (M, K)
  TensorView* tv1 = makeDummyTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  // Schedule
  constexpr int BSX = 16;
  tv5->split(2, BSX);
  tv5->split(1, BSX);
  tv5->split(0, BSX);
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
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-3)->parallelize(ParallelType::TIDy);
  tv6->axis(-2)->parallelize(ParallelType::TIDx);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t1});

  at::Tensor aten_output = matmul(t0, t1);
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemBlockGemmCache() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeDummyTensor(2); // (M, K)
  TensorView* tv1 = makeDummyTensor(2); // (K, N)
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
  TensorView* tv6 = tv5->cache_before();

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
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  tv7->axis(-3)->parallelize(ParallelType::TIDy);
  tv7->axis(-2)->parallelize(ParallelType::TIDx);

  tv6->axis(-2)->parallelize(ParallelType::TIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t0, t1});

  at::Tensor aten_output = matmul(t0, t1);
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemDynamicReductionSymbolic() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
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
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  torch::jit::fuser::cuda::FusionExecutor executor;
  executor.compileFusion(&fusion);
  auto outputs = executor.runFusion(
      {input},
      torch::jit::fuser::cuda::LaunchParams(
          -1, -1, -1, runtime_threadIdx_dim, -1, -1));

  auto aten_output = input.sum({1});
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemDynamicReductionSymbolicArg() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  Int* sym_bsx = new Int();
  TensorView* tv0 = makeDummyTensor(3); // M, K, N
  fusion.addInput(tv0);
  fusion.addInput(sym_bsx);

  TensorView* tv1 = sum(tv0, {1}); // M, R, N
  fusion.addOutput(tv1);

  TensorView* tv2 = tv0->cache_after();
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
  at::Tensor t0 = at::randn({M, K, N}, options);

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  torch::jit::fuser::cuda::FusionExecutor executor;
  executor.compileFusion(&fusion);
  auto outputs = executor.runFusion(
      {t0, runtime_threadIdx_dim},
      torch::jit::fuser::cuda::LaunchParams(
          -1, -1, -1, runtime_threadIdx_dim, -1, -1));

  at::Tensor aten_output = sum(t0, {1});
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionSmemDynamicPwiseMulSymbolicArg() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Int* symbolic_m_tile_dim = new Int(); // bound to threadIdx.z
  Int* symbolic_split_k_tile_dim = new Int(); // bound to blockIdx.x
  Int* symbolic_block_k_tile_dim = new Int(); // bound to threadIdx.x
  // Compile-time integer for tiling
  int n_smem_tile = 8; // bound to threadIdx.y

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);

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
  tv5->split(2, n_smem_tile);
  tv5->split(1, symbolic_block_k_tile_dim);
  tv5->split(1, symbolic_split_k_tile_dim);
  tv5->split(0, symbolic_m_tile_dim);

  // Reorder so all outer tiles are in the leftmost 3 positions
  tv5->reorder({{1, 5}, {5, 1}});

  // Factor out the outer reduction IterDomain, then run the inter-cta
  // reduction, and intra-cta reduction
  auto tv6 = tv5->rFactor({2});

  // Scope computations
  tv6->computeAt(tv5, 2);

  // RFactor moves reduction axes around, reorder to match ordering of tv5
  tv6->reorder({
      {2, -2},
      {3, -1},
      {4, 2},
      {5, 3},
      {6, 4},
  });

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

  fusion.printMath();
  fusion.printKernel();

  constexpr int M = 31, K = 65, N = 33;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor A = at::randn({M, K}, options);
  at::Tensor B = at::randn({K, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  // Generate CUDA and compile with nvRTC
  fe.compileFusion(&fusion);

  // Runtime tiling
  int m_tile = 4; // bound to threadIdx.z
  int split_k = 7; // bound to blockIdx.x
  int intra_cta = 8; // bound to threadIdx.x

  auto fuser_outputs = fe.runFusion({A, B, m_tile, split_k, intra_cta});
  auto C_fuser = fuser_outputs[0];

  at::Tensor aten_C = mul(A.unsqueeze(2), B.unsqueeze(0)).sum(1);
  // TODO: re-enable after fixing #380
#if 0
  TORCH_CHECK(
      aten_C.allclose(C_fuser, 1e-5, 1e-5),
      "Error of: ",
      aten_C.sub(C_fuser).abs().max());
#endif
}

void testGPU_FusionGlobalIntermediate() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  // tv1[I0, R1] = tv0[I0, I1]

  // Interface should just be a direct split with a Parallel type. We can
  // include the parallelize call if we do this.
  tv1->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
  // tv1[I0, R1o, R1i{BIDx}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({2});
  tv2->setMemoryType(MemoryType::Global);
  // tv2[I0, R1oo, Ir1i{BIDx}] = tv0[I0, I1]
  // tv1[I0,        R1i{BIDx}] = tv2[I0, R1oo, Ir1i{BIDx}]

  tv0->computeAt(tv1, 1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  constexpr int numel_x = 65000, numel_y = 1024;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({numel_x, numel_y}, options);

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  torch::jit::fuser::cuda::FusionExecutor executor;
  executor.compileFusion(&fusion);
  auto outputs = executor.runFusion(
      {input},
      torch::jit::fuser::cuda::LaunchParams(
          -1, -1, -1, runtime_threadIdx_dim, -1, -1));

  auto aten_output = input.sum({1});
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionGlobalIntermediateDefaultSchedule() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);
  TensorView* tv2 = makeDummyTensor(2);
  TensorView* tv3 = makeDummyTensor(2);
  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addOutput(tv6);
  // t6 = ((t1 + (t2 - t3)) - t0)

  tv4->setMemoryType(MemoryType::Global);
  tv5->setMemoryType(MemoryType::Global);
  tv6->setMemoryType(MemoryType::Global);

  constexpr int M = 32, N = 810;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in0 = at::rand({M, N}, options);
  at::Tensor in1 = at::rand({M, N}, options);
  at::Tensor in2 = at::rand({M, N}, options);
  at::Tensor in3 = at::rand({M, N}, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({in0, in1, in2, in3});

  at::Tensor aten_output = (in1 + (in2 - in3)) - in0;
  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-5, 1e-5),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().sum());
}

void testGPU_FusionConstCheck() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto one = new Int(1);
  TORCH_CHECK(one->isConstScalar());

  auto one_x2 = mul(one, one);
  TORCH_CHECK(one_x2->isConstScalar());

  auto one_x3 = mul(one_x2, one);
  TORCH_CHECK(one_x3->isConstScalar());

  auto one_x4 = mul(one_x3, one);
  TORCH_CHECK(one_x4->isConstScalar());
}

void testGPU_FusionUnrollWithAlloc() {
  const std::vector<int64_t> tensor_dims_in = {128, 128};
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, new Float(0));
  TensorView* tv2 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand(tensor_dims_in, options);
  at::Tensor cg_output = at::empty({tensor_dims_in[0]}, options);

  // const at::ArrayRef<c10::IValue> inputs({input});

  // Schedule
  tv2->split(1, 32);
  tv2->split(1, 4); // unroll

  auto tv2_rf = tv2->rFactor({-3, -2});

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv2_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv2_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv2_rf->axis(-2)->parallelize(ParallelType::Unroll);

  tv1->computeAt(tv2_rf, -1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto aten_output = (input + 0).sum(1);

  TORCH_CHECK(
      aten_output.allclose(outputs[0]),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

// Test isZeroInt
void testGPU_FusionIsZeroInt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Int* x = new Int(0);
  Int* y = new Int(1);
  Val* z = mul(x, y);
  TORCH_CHECK(x->isZeroInt());
  TORCH_CHECK(!y->isZeroInt());
  TORCH_CHECK(!z->isZeroInt());
}

// Test isOneInt
void testGPU_FusionIsOneInt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Int* x = new Int(1);
  Int* y = new Int(1);
  Val* z = mul(x, y);
  TORCH_CHECK(x->isOneInt());
  TORCH_CHECK(y->isOneInt());
  TORCH_CHECK(!z->isOneInt());
}

// This is to verify no cycle of computeAt is created. A more complex
// variation of this pattern appears in one of the Python tests
// (test_random_topo).
void testGPU_FusionComputeAtNonterminatingOutput() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  // Common intermediate tensor
  auto tv1 = add(tv0, new Float(1));
  // tv1 -> tv2
  auto tv2 = add(tv1, new Float(2));
  // tv1 -> tv3 -> tv4
  auto tv3 = add(tv1, new Float(3));
  auto tv4 = add(tv3, new Float(4));

  // NOTE: This should no longer occur as of PR #201.
  // The order of adding outputs matters. If tv3 is added before tv4,
  // it should be fine. However, if tv4 is added before tv3, there
  // will be a cycle of tv3->tv4 and tv4->tv3. tv3->tv4 is created
  // first, and then tv4->tv3 is created at the final phase of
  // computeAt (ComputeAt::setupOutputs).
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);
  fusion.addOutput(tv3);

  tv0->computeAt(tv2, -1);

  TORCH_CHECK(
      !(tv3->getComputeAtView() == tv4 && tv4->getComputeAtView() == tv3),
      "ComputeAt cycle detected between tv3 and tv4");

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand(100, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto& output_tv2 = outputs[0];
  auto& output_tv4 = outputs[1];
  auto& output_tv3 = outputs[2];

  auto aten_t1 = input + 1;
  auto aten_t2 = aten_t1 + 2;
  auto aten_t3 = aten_t1 + 3;
  auto aten_t4 = aten_t3 + 4;

  TORCH_CHECK(
      aten_t2.allclose(output_tv2),
      "Error of: ",
      aten_t2.sub(output_tv2).abs().max());
  TORCH_CHECK(
      aten_t3.allclose(output_tv3),
      "Error of: ",
      aten_t3.sub(output_tv3).abs().max());
  TORCH_CHECK(
      aten_t4.allclose(output_tv4),
      "Error of: ",
      aten_t4.sub(output_tv4).abs().max());

  return;
}

void testGPU_FusionTraversalOrder1() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv0, new Float(2));
  TensorView* tv3 = add(tv1, new Float(3));
  TensorView* tv4 = add(tv1, new Float(4));

  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  tv1->computeAt(tv3, -1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({10, 10}, options);
  at::Tensor cg_output_tv2 = at::empty_like(input, options);
  at::Tensor cg_output_tv3 = at::empty_like(input, options);
  at::Tensor cg_output_tv4 = at::empty_like(input, options);
  fe.runFusion({input}, {cg_output_tv2, cg_output_tv3, cg_output_tv4});

  auto t1 = input + 1;
  auto t2 = input + 2;
  auto t3 = t1 + 3;
  auto t4 = t1 + 4;

  TORCH_CHECK(
      t2.allclose(cg_output_tv2),
      "tv2 error of: ",
      t2.sub(cg_output_tv2).abs().max());
  TORCH_CHECK(
      t3.allclose(cg_output_tv3),
      "tv5 error of: ",
      t3.sub(cg_output_tv3).abs().max());
  TORCH_CHECK(
      t4.allclose(cg_output_tv4),
      "tv4 error of: ",
      t4.sub(cg_output_tv4).abs().max());
}

void testGPU_FusionTraversalOrder2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv1, new Float(2));

  TensorView* tv3 = add(tv0, new Float(3));
  TensorView* tv4 = add(tv3, new Float(4));

  TensorView* tv5 = add(tv1, tv3);

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv1->computeAt(tv5, -1);
  tv3->computeAt(tv5, -1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({10, 10}, options);
  at::Tensor cg_output_tv2 = at::empty_like(input, options);
  at::Tensor cg_output_tv4 = at::empty_like(input, options);
  at::Tensor cg_output_tv5 = at::empty_like(input, options);
  fe.runFusion({input}, {cg_output_tv2, cg_output_tv4, cg_output_tv5});

  auto t1 = input + 1;
  auto t2 = t1 + 2;
  auto t3 = input + 3;
  auto t4 = t3 + 4;
  auto t5 = t1 + t3;

  TORCH_CHECK(
      t2.allclose(cg_output_tv2),
      "tv2 error of: ",
      t2.sub(cg_output_tv2).abs().max());
  TORCH_CHECK(
      t4.allclose(cg_output_tv4),
      "tv4 error of: ",
      t4.sub(cg_output_tv4).abs().max());
  TORCH_CHECK(
      t5.allclose(cg_output_tv5),
      "tv5 error of: ",
      t5.sub(cg_output_tv5).abs().max());
}

void testGPU_FusionTraversalOrder3() {
  for (int i = 0; i < 2; ++i) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(1);
    fusion.addInput(tv0);

    TensorView* tv1 = add(tv0, new Float(1));
    TensorView* tv2 = add(tv1, new Float(2));

    TensorView* tv3 = add(tv0, new Float(3));
    TensorView* tv4 = add(tv3, new Float(4));

    TensorView* tv5 = add(tv1, tv3);

    fusion.addOutput(tv2);
    fusion.addOutput(tv4);
    fusion.addOutput(tv5);

    const int tile = 32;

    tv1->split(-1, tile);
    tv2->split(-1, tile);
    tv3->split(-1, tile);
    tv4->split(-1, tile);
    tv5->split(-1, tile);

    auto compute_at_outer = tv1;
    auto compute_at_inner = tv3;
    if (i == 1) {
      std::swap(compute_at_inner, compute_at_outer);
    }

    compute_at_outer->computeAt(tv5, -2);
    compute_at_inner->computeAt(tv5, -1);

    torch::jit::fuser::cuda::FusionExecutor fe;
    fe.compileFusion(&fusion);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::rand({100}, options);
    at::Tensor cg_output_tv2 = at::empty_like(input, options);
    at::Tensor cg_output_tv4 = at::empty_like(input, options);
    at::Tensor cg_output_tv5 = at::empty_like(input, options);
    fe.runFusion({input}, {cg_output_tv2, cg_output_tv4, cg_output_tv5});

    auto t1 = input + 1;
    auto t2 = t1 + 2;
    auto t3 = input + 3;
    auto t4 = t3 + 4;
    auto t5 = t1 + t3;

    TORCH_CHECK(
        t2.allclose(cg_output_tv2),
        "tv2 error of: ",
        t2.sub(cg_output_tv2).abs().max());
    TORCH_CHECK(
        t4.allclose(cg_output_tv4),
        "tv4 error of: ",
        t4.sub(cg_output_tv4).abs().max());
    TORCH_CHECK(
        t5.allclose(cg_output_tv5),
        "tv5 error of: ",
        t5.sub(cg_output_tv5).abs().max());
  }
}

void testGPU_FusionTraversalOrder4() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // First tree
  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv1, new Float(2));
  TensorView* tv3 = add(tv1, new Float(3));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  // Second tree
  TensorView* tv4 = makeDummyTensor(1);
  fusion.addInput(tv4);
  TensorView* tv5 = add(tv4, new Float(5));
  TensorView* tv6 = add(tv5, new Float(6));
  TensorView* tv7 = add(tv5, new Float(7));
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  tv1->computeAt(tv2, -1);
  tv5->computeAt(tv6, -1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::rand({100}, options);
  at::Tensor t4 = at::rand_like(t0, options);
  at::Tensor cg_output_tv2 = at::empty_like(t0, options);
  at::Tensor cg_output_tv3 = at::empty_like(t0, options);
  at::Tensor cg_output_tv6 = at::empty_like(t0, options);
  at::Tensor cg_output_tv7 = at::empty_like(t0, options);

  fe.runFusion(
      {t0, t4}, {cg_output_tv2, cg_output_tv3, cg_output_tv6, cg_output_tv7});

  auto t1 = t0 + 1;
  auto t2 = t1 + 2;
  auto t3 = t1 + 3;
  auto t5 = t4 + 5;
  auto t6 = t5 + 6;
  auto t7 = t5 + 7;

  TORCH_CHECK(
      t2.allclose(cg_output_tv2),
      "tv2 error of: ",
      t2.sub(cg_output_tv2).abs().max());
  TORCH_CHECK(
      t3.allclose(cg_output_tv3),
      "tv3 error of: ",
      t3.sub(cg_output_tv3).abs().max());
  TORCH_CHECK(
      t6.allclose(cg_output_tv6),
      "tv6 error of: ",
      t6.sub(cg_output_tv6).abs().max());
  TORCH_CHECK(
      t7.allclose(cg_output_tv7),
      "tv7 error of: ",
      t7.sub(cg_output_tv7).abs().max());
}

void testGPU_FusionTraversalOrder5() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv1, new Float(2));
  TensorView* tv3 = add(tv0, new Float(3));
  TensorView* tv4 = add(tv3, new Float(4));
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv1);
  fusion.addOutput(tv3);
  fusion.addOutput(tv5);

  tv2->computeAt(tv5, -1);
  tv4->computeAt(tv5, -1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::rand({100}, options);
  at::Tensor cg_output_tv1 = at::empty_like(t0, options);
  at::Tensor cg_output_tv3 = at::empty_like(t0, options);
  at::Tensor cg_output_tv5 = at::empty_like(t0, options);

  fe.runFusion({t0}, {cg_output_tv1, cg_output_tv3, cg_output_tv5});

  auto t1 = t0 + 1;
  auto t2 = t1 + 2;
  auto t3 = t0 + 3;
  auto t4 = t3 + 4;
  auto t5 = t2 + t4;

  TORCH_CHECK(
      t1.allclose(cg_output_tv1),
      "tv1 error of: ",
      t1.sub(cg_output_tv1).abs().max());
  TORCH_CHECK(
      t3.allclose(cg_output_tv3),
      "tv3 error of: ",
      t3.sub(cg_output_tv3).abs().max());
  TORCH_CHECK(
      t5.allclose(cg_output_tv5),
      "tv5 error of: ",
      t5.sub(cg_output_tv5).abs().max());
}

void testGPU_FusionTraversalOrder6() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv0, new Float(2));
  TensorView* tv3 = add(tv1, tv2);
  TensorView* tv4 = add(tv3, new Float(4));

  fusion.addOutput(tv4);

  tv1->split(0, 32);
  tv2->split(0, 32);
  tv3->split(0, 32);
  tv4->split(0, 32);

  tv3->computeAt(tv4, -2);
  tv1->computeAt(tv3, -1);
  tv2->computeAt(tv3, -2);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::rand({100}, options);
  at::Tensor cg_output_tv4 = at::empty_like(t0, options);

  fe.runFusion({t0}, {cg_output_tv4});

  auto t1 = t0 + 1;
  auto t2 = t0 + 2;
  auto t3 = t1 + t2;
  auto t4 = t3 + 4;

  TORCH_CHECK(
      t4.allclose(cg_output_tv4),
      "tv4 error of: ",
      t4.sub(cg_output_tv4).abs().max());
}

void testGPU_FusionTraversalOrder7() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, new Float(1));
  TensorView* tv2 = add(tv1, new Float(2));
  TensorView* tv3 = add(tv0, new Float(3));
  TensorView* tv4 = add(tv3, new Float(4));
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv5);

  TensorView* tvs[] = {tv1, tv2, tv3, tv4, tv5};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->split(0, 4);
    tv->split(0, 8);
  }

  // computeAt into inner loop nests
  tv1->computeAt(tv2, -1);
  tv3->computeAt(tv4, -2);

  tv2->computeAt(tv5, -4);
  tv4->computeAt(tv5, -3);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::rand({100}, options);
  at::Tensor cg_output_tv5 = at::empty_like(t0, options);
  fe.runFusion({t0}, {cg_output_tv5});

  auto t1 = t0 + 1;
  auto t2 = t1 + 2;
  auto t3 = t0 + 3;
  auto t4 = t3 + 4;
  auto t5 = t2 + t4;

  TORCH_CHECK(
      t5.allclose(cg_output_tv5),
      "tv5 error of: ",
      t5.sub(cg_output_tv5).abs().max());
}

// Test predication of grid reduction
void testGPU_FusionThreadPredicate() {
  const int gdimx = 4;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0);
  TensorView* tv2 = unaryOp(UnaryOpType::Neg, tv1);
  TensorView* tv3 = add(tv0, new Float(2));

  fusion.addOutput(tv3);
  fusion.addOutput(tv2);

  tv1->split(1, bdimx);
  tv1->split(1, gdimx);
  tv3->split(1, bdimx);
  tv3->split(1, gdimx);

  TensorView* tv1_rf = tv1->rFactor({1});

  tv1->computeAt(tv2, -1);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1_rf->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(2)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDy);

  int numel_x = 100;
  int numel_y = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output_tv2 = at::empty({numel_x}, options);
  at::Tensor cg_output_tv3 = at::empty_like(input, options);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.runFusion({input}, {cg_output_tv3, cg_output_tv2});

  auto aten_output_tv2 = -input.sum({1});
  TORCH_CHECK(aten_output_tv2.allclose(cg_output_tv2));
  auto aten_output_tv3 = input + 2.0;
  TORCH_CHECK(aten_output_tv3.allclose(cg_output_tv3));
}

void testGPU_FusionLSTMCell() {
  const int hidden_features = 512;
  const int batch_size = 64;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tvs[16];
  for (size_t i = 0; i < 16; i++) {
    tvs[i] = makeDummyTensor(2);
    fusion.addInput(tvs[i]);
  }

  auto ingate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[0], tvs[1]), tvs[2]), tvs[3]));

  auto forgetgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[4], tvs[5]), tvs[6]), tvs[7]));

  auto cellgate = unaryOp(
      UnaryOpType::Tanh, add(add(add(tvs[8], tvs[9]), tvs[10]), tvs[11]));

  auto outgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[12], tvs[13]), tvs[14]), tvs[15]));

  auto cx = makeContigTensor(2);
  fusion.addInput(cx);

  auto cy = add(mul(forgetgate, cx), mul(ingate, cellgate));

  auto hy = mul(outgate, unaryOp(UnaryOpType::Tanh, cy));

  fusion.addOutput(cy);
  fusion.addOutput(hy);

  std::vector<c10::IValue> inputs;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor large_tensor0 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor1 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor2 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor3 =
      at::randn({batch_size, hidden_features * 4}, options);

  auto chunked0 = large_tensor0.chunk(4, 1);
  auto chunked1 = large_tensor1.chunk(4, 1);
  auto chunked2 = large_tensor2.chunk(4, 1);
  auto chunked3 = large_tensor3.chunk(4, 1);

  inputs.insert(inputs.end(), chunked0.begin(), chunked0.end());
  inputs.insert(inputs.end(), chunked1.begin(), chunked1.end());
  inputs.insert(inputs.end(), chunked2.begin(), chunked2.end());
  inputs.insert(inputs.end(), chunked3.begin(), chunked3.end());

  auto at_ingate =
      chunked0[0].add(chunked0[1]).add(chunked0[2]).add(chunked0[3]).sigmoid();
  auto at_forgetgate =
      chunked1[0].add(chunked1[1]).add(chunked1[2]).add(chunked1[3]).sigmoid();
  auto at_cellgate =
      chunked2[0].add(chunked2[1]).add(chunked2[2]).add(chunked2[3]).tanh();
  auto at_outgate =
      chunked3[0].add(chunked3[1]).add(chunked3[2]).add(chunked3[3]).sigmoid();

  auto at_cx = at::randn({batch_size, hidden_features}, options);
  inputs.push_back(at_cx);
  auto at_cy = at_forgetgate.mul(at_cx).add(at_ingate.mul(at_cellgate));
  auto at_hy = at_outgate.mul(at_cy.tanh());

  fuser::cuda::scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion(c10::ArrayRef<c10::IValue>(inputs));

  TORCH_CHECK(at_cy.allclose(outputs[0], 1e-4, 1e-7));
  TORCH_CHECK(at_hy.allclose(outputs[1], 1e-4, 1e-7));
}

void testGPU_FusionComputeAtMultiBCast() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, new Float(0.5));
  TensorView* tv2 = broadcast(tv1, {true, false});
  TensorView* tv3 = broadcast(tv1, {false, true});
  TensorView* tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // This is not supported and should throw an exception.
  ASSERT_ANY_THROW(tv1->computeAt(tv3, -1));
}

void testGPU_FusionReductionHalf() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(3, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = add(tv1, new Float(1.0));
  auto tv3 = sum(tv2, {2});
  auto tv4 = castOp(DataType::Half, tv3);

  fusion.addOutput(tv4);

  const auto options =
      at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 8, 16}, options);

  const auto rparams = cuda::getReductionHeuristics(&fusion, {input}, tv3);
  TORCH_CHECK(rparams.has_value(), "Reduction heuristics was not generated!");
  cuda::scheduleReduction(&fusion, rparams.value(), tv3, {tv4});

  cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);
  // no broadcasting needed, omitting the last optional argument;
  auto outputs = fe.runFusion({input});

  auto aten_output = input.to(c10::ScalarType::Float)
                         .add(1.0)
                         .sum({2})
                         .to(c10::ScalarType::Half);

  TORCH_CHECK(
      aten_output.allclose(outputs[0], 1e-04, 1e-04),
      "Error of: ",
      aten_output.sub(outputs[0]).abs().max());
}

void testGPU_FusionInputsIdLookup() {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 8, 8}, options);
  at::Tensor t1 = at::randn({8, 8}, options);
  at::Tensor t2 = at::randn({6, 4}, options);

  // create a cache with max size 2;
  auto inputs_id_lookup = torch::jit::fuser::cuda::InputsIdLookup(2);

  // testing basic function, same encoding for identical inputs
  auto id_0 = inputs_id_lookup.lookupId({t0, t1, 5.0});
  auto id_0_lookup = inputs_id_lookup.lookupId({t0, t1, 2.5});
  TORCH_CHECK(id_0.id == id_0_lookup.id);
  TORCH_CHECK(inputs_id_lookup.size() == 1);
  TORCH_CHECK(id_0.eviction == false);

  // new input (even tho same shape, but we have different signature because of
  // missing scalar input
  auto id_1 = inputs_id_lookup.lookupId({t0, t1});
  auto id_1_lookup = inputs_id_lookup.lookupId({t0, t1});
  TORCH_CHECK(id_1.id == id_1_lookup.id);
  TORCH_CHECK(inputs_id_lookup.size() == 2);
  TORCH_CHECK(id_1.eviction == false);

  // eviction should happen at this point
  auto id_2 = inputs_id_lookup.lookupId({t2, t1});
  TORCH_CHECK(id_2.id != id_0.id);
  TORCH_CHECK(id_2.id != id_1.id);
  TORCH_CHECK(inputs_id_lookup.size() == 2);
  TORCH_CHECK(id_2.eviction == true);
  TORCH_CHECK(id_2.evict_id == id_0.id);

  // look at input 1 again
  auto id_1_relook = inputs_id_lookup.lookupId({t0, t1});
  TORCH_CHECK(id_1_relook.id == id_1.id);
  TORCH_CHECK(id_1_relook.eviction == false);
}

} // namespace jit
} // namespace torch

#endif // #if defined(USE_CUDA)
