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
#include <torch/csrc/jit/codegen/cuda/inline_propagator.h>
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

namespace {

TensorView* loweredTv(TensorView* tv, GpuLower& gpulw) {
  auto used_tvs = ir_utils::allTvs(gpulw.kernel()->as<Fusion>());
  TensorView* matching_tv = nullptr;
  for (auto lowered_tv : used_tvs) {
    if (lowered_tv->name() == tv->name()) {
      matching_tv = lowered_tv;
    }
  }
  TORCH_INTERNAL_ASSERT(matching_tv != nullptr);
  return matching_tv;
}

class PredicatedChecker : public kir::IrVisitor {
 public:
  // Checks if the provided tv is written to within a non-trivial conditional
  static bool isPredicated(TensorView* tv, GpuLower& gpulw) {
    PredicatedChecker checker(
        loweredTv(tv, gpulw), gpulw.kernel()->topLevelExprs());
    return checker.is_predicated_;
  }

 private:
  PredicatedChecker() = delete;

  PredicatedChecker(TensorView* tv, std::vector<Expr*> exprs) : tv_(tv) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool is_predicated_ = false;
  bool predicated_ite_ = false;
  TensorView* tv_ = nullptr;

  void handle(kir::IfThenElse* ite) final {
    auto prev_ite = predicated_ite_;
    predicated_ite_ = !ite->predicate()->value()->isConstScalar();
    kir::IrVisitor::handle(ite);
    predicated_ite_ = prev_ite;
  }

  void handle(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view() == tv_) {
        is_predicated_ = is_predicated_ | predicated_ite_;
        if (expr->predicate() != nullptr &&
            !expr->predicate()->value()->isConst()) {
          is_predicated_ = true;
        }
      }
    }
    kir::IrVisitor::handle(expr);
  }
};

class UnswitchInElseChecker : public kir::IrVisitor {
 public:
  // Checks if there are any unswitched for loops within an else clause
  static bool check(GpuLower& gpulw) {
    UnswitchInElseChecker checker(gpulw.kernel()->topLevelExprs());
    return checker.found_in_else_;
  }

 private:
  UnswitchInElseChecker() = delete;
  UnswitchInElseChecker(std::vector<Expr*> exprs) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool within_else_ = false;
  bool found_in_else_ = false;

  void handle(kir::IfThenElse* ite) final {
    auto prev_within_else = within_else_;
    within_else_ = true;
    kir::IrVisitor::handle(ite->elseBody().exprs());
    within_else_ = prev_within_else;
  }

  void handle(kir::ForLoop* for_loop) final {
    if (for_loop->iter_domain()->getParallelType() == ParallelType::Unswitch) {
      found_in_else_ = found_in_else_ || within_else_;
    }
    kir::IrVisitor::handle(for_loop);
  }
};

class PredicateMagicZeroChecker : public kir::IrVisitor {
 public:
  // Checks if all predicated domains of the provided tv are protected with
  // magic zero
  static bool isProtected(TensorView* tv, GpuLower& gpulw) {
    PredicateMagicZeroChecker checker(
        loweredTv(tv, gpulw), gpulw.kernel()->topLevelExprs());
    return checker.is_protected_;
  }

 private:
  using kir::IrVisitor::handle;

  PredicateMagicZeroChecker(TensorView* tv, std::vector<Expr*> exprs)
      : tv_(tv) {
    handle(exprs);
  }

  void handle(kir::IfThenElse* ite) final {
    auto prev_predicate = predicate_;
    predicate_ = ite->predicate()->value();
    kir::IrVisitor::handle(ite);
    predicate_ = prev_predicate;
  }

  void handle(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view() == tv_) {
        is_protected_ = checkPredicateOfTensor(predicate_);
        return;
      }
    }

    if (expr->isA<kir::ForLoop>()) {
      handle(expr->as<kir::ForLoop>());
    } else if (expr->isA<kir::IfThenElse>()) {
      handle(expr->as<kir::IfThenElse>());
    } else {
      for (auto input : expr->inputs()) {
        handle(input);
      }
    }
  }

  // Return true If all predicated domains are protected
  bool checkPredicateOfTensor(Val* predicate) {
    auto id_predicates = decomposeCompoundPredicate(predicate);
    for (auto id_predicate : id_predicates) {
      // Just check if nvfuser_zero is used. Not perfect but probably
      // good enough.
      is_magic_zero_found_ = false;
      handle(id_predicate);
      if (!is_magic_zero_found_) {
        return false;
      }
    }
    return true;
  }

  // Decompose "X && Y" to a vector of {X, Y}.
  std::vector<Val*> decomposeCompoundPredicate(Val* predicate) {
    if (auto binary_op = dynamic_cast<BinaryOp*>(predicate->definition())) {
      if (binary_op->getBinaryOpType() == BinaryOpType::And) {
        auto pred = decomposeCompoundPredicate(binary_op->lhs());
        auto rhs_pred = decomposeCompoundPredicate(binary_op->rhs());
        pred.insert(pred.end(), rhs_pred.begin(), rhs_pred.end());
        return pred;
      }
    }

    return {predicate};
  }

  void handle(Val* val) final {
    if (isMagicZero(val)) {
      is_magic_zero_found_ = true;
      return;
    }

    auto def = val->definition();
    if (def != nullptr) {
      handle(def);
    }
  }

 private:
  bool is_protected_ = false;
  Val* predicate_ = nullptr;
  TensorView* tv_ = nullptr;
  bool is_magic_zero_found_ = false;
};

// Basically just TransformPropagator, except that it checks the consistency
// replayPasC with getMatchedLeafPosWithoutReplayPasC, replayCasP with
// getMatchedLeafPosWithoutReplayCasP, and fullSelfReplay with fullSelfMatching:
// - After replayPasC, getMatchedLeafPosWithoutReplayPasC should return the same
//   replayed position
// - After replayCasP, getMatchedLeafPosWithoutReplayCasP should return the same
//   replayed position
// - After fullSelfReplay, fullSelfMatching should return true
struct TransformPropagatorWithCheck : public TransformPropagator {
 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateC2P(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(
            to, from, from_pos) == to_pos);
  }
  virtual void propagateP2C(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateP2C(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(
            to, from, from_pos) == to_pos);
  }
  virtual void propagateSibling(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateSibling(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(from_pos == to_pos);
    TORCH_CHECK(TransformReplay::fullSelfMatching(from, to));
  }
  using TransformPropagator::TransformPropagator;
};

} // namespace

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1});
  auto cg_outputs = fe.runFusion({input1});

  testValidate(&fusion, cg_outputs, {input1}, {input1}, __LINE__, __FILE__);
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

TEST_F(NVFuserTest, FusionRootMappingMultipleBroadcast_CUDA) {
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    GTEST_SKIP() << "Somehow it fails on sm_80+ GPUs"
                 << " See https://github.com/pytorch/pytorch/issues/86717";
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  // tv0 cannot be mapped with the consumers as it would mean its only
  // domain would be mapped to both the first and second domains of
  // the two consumers, thus computing tv0 at both corresponding loops.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {false},
      tv1,
      tv1->getRootDomain(),
      {false, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {false},
      tv2,
      tv2->getRootDomain(),
      {false, false});
  checkIdMapped(tv1, tv1->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(tv2, tv2->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {false},
      tv3,
      tv3->getRootDomain(),
      {false, false});
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
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    GTEST_SKIP() << "Somehow it does not throw on sm_80+"
                 << " See https://github.com/pytorch/pytorch/issues/86714";
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // computeAt should fail as there is no valid root mapping.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv1->computeAt(tv4, 1));
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

TEST_F(NVFuserTest, FusionPersistentSoftmaxLocalSmem_CUDA) {
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
  // Require 70KB of smem to run test
  const size_t required_smem_size = 70 << 10;
  if (properties->sharedMemPerBlockOptin < required_smem_size) {
    GTEST_SKIP() << "not enough shared memory space on device to run test";
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

  auto properties = at::cuda::getDeviceProperties(0);
  // Require 70KB of smem to run test
  const size_t required_smem_size = 70 << 10;
  if (properties->sharedMemPerBlockOptin < required_smem_size) {
    GTEST_SKIP() << "not enough shared memory space on device to run test";
  }

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

TEST_F(NVFuserTest, FusionGlobalIntermediate_CUDA) {
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
  tv2->setMemoryType(MemoryType::Global);
  // tv2[I0, R1oo, Ir1i{BIDx}] = tv0[I0, I1]
  // tv1[I0,        R1i{BIDx}] = tv2[I0, R1oo, Ir1i{BIDx}]

  tv0->computeAt(tv1, 1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  constexpr int numel_x = 65000, numel_y = 1024;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  auto lparams = LaunchParams(-1, -1, -1, runtime_threadIdx_dim, -1, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto cg_outputs = fe.runFusion({input}, lparams);

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion,
      cg_outputs,
      {input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionGlobalIntermediateDefaultSchedule_CUDA) {
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

  tv4->setMemoryType(MemoryType::Global);
  tv5->setMemoryType(MemoryType::Global);
  tv6->setMemoryType(MemoryType::Global);

  constexpr int M = 32, N = 810;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t1 = at::randn({M, N}, options);
  at::Tensor t2 = at::randn({M, N}, options);
  at::Tensor t3 = at::randn({M, N}, options);

  at::Tensor aten_output = (t1 + (t2 - t3)) - t0;

  std::vector<IValue> aten_inputs = {t0, t1, t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t2, t3});
  auto cg_outputs = fe.runFusion({t0, t1, t2, t3});

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionConstCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto one = IrBuilder::create<Int>(1);
  TORCH_CHECK(one->isConstScalar());

  auto one_x2 = mul(one, one);
  TORCH_CHECK(one_x2->isConstScalar());

  auto one_x3 = mul(one_x2, one);
  TORCH_CHECK(one_x3->isConstScalar());

  auto one_x4 = mul(one_x3, one);
  TORCH_CHECK(one_x4->isConstScalar());
}

TEST_F(NVFuserTest, FusionUnrollWithAlloc_CUDA) {
  const std::vector<int64_t> tensor_dims_in = {128, 128};
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(0));
  TensorView* tv2 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(tensor_dims_in, options);
  at::Tensor cg_output = at::empty({tensor_dims_in[0]}, options);

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto aten_output = (input + 0).to(at::kDouble).sum(1);

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// Test isZeroInt
TEST_F(NVFuserTest, FusionIsZeroInt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Int* x = IrBuilder::create<Int>(0);
  Int* y = IrBuilder::create<Int>(1);
  Val* z = mul(x, y);
  TORCH_CHECK(x->isZeroInt());
  TORCH_CHECK(!y->isZeroInt());
  TORCH_CHECK(!z->isZeroInt());
}

// Test isOneInt
TEST_F(NVFuserTest, FusionIsOneInt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Int* x = IrBuilder::create<Int>(1);
  Int* y = IrBuilder::create<Int>(1);
  Val* z = mul(x, y);
  TORCH_CHECK(x->isOneInt());
  TORCH_CHECK(y->isOneInt());
  TORCH_CHECK(!z->isOneInt());
}

// This is to verify no cycle of computeAt is created. A more complex
// variation of this pattern appears in one of the Python tests
// (test_random_topo).
TEST_F(NVFuserTest, FusionComputeAtNonterminatingOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Common intermediate tensor
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  // tv1 -> tv2
  auto tv2 = add(tv1, IrBuilder::create<Double>(2));
  // tv1 -> tv3 -> tv4
  auto tv3 = add(tv1, IrBuilder::create<Double>(3));
  auto tv4 = add(tv3, IrBuilder::create<Double>(4));

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

  TORCH_CHECK(tv3->hasComputeAt());
  TORCH_CHECK(!tv4->hasComputeAt());

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(100, options);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = t1 + 3;
  auto t4 = t3 + 4;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  std::vector<at::Tensor> aten_outputs = {t2, t4, t3};
  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3));
  TensorView* tv4 = add(tv1, IrBuilder::create<Double>(4));

  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  tv1->computeAt(tv3, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 10}, options);

  auto t1 = aten_input + 1;
  auto t2 = aten_input + 2;
  auto t3 = t1 + 3;
  auto t4 = t1 + 4;

  std::vector<at::Tensor> aten_outputs = {t2, t3, t4};

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

TEST_F(NVFuserTest, FusionTraversalOrder2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));

  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(3));
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));

  TensorView* tv5 = add(tv1, tv3);

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv1->computeAt(tv5, -1);
  tv3->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 10}, options);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = aten_input + 3;
  auto t4 = t3 + 4;
  auto t5 = t1 + t3;

  std::vector<at::Tensor> aten_outputs = {t2, t4, t5};

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

TEST_F(NVFuserTest, FusionTraversalOrder3_CUDA) {
  for (const auto i : c10::irange(2)) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeSymbolicTensor(1);
    fusion.addInput(tv0);

    TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
    TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));

    TensorView* tv3 = add(tv0, IrBuilder::create<Double>(3));
    TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));

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

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor aten_input = at::randn({100}, options);
    auto t1 = aten_input + 1;
    auto t2 = t1 + 2;
    auto t3 = aten_input + 3;
    auto t4 = t3 + 4;
    auto t5 = t1 + t3;

    std::vector<at::Tensor> aten_outputs = {t2, t4, t5};

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
}

TEST_F(NVFuserTest, FusionTraversalOrder4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // First tree
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv1, IrBuilder::create<Double>(3));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  // Second tree
  TensorView* tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);
  TensorView* tv5 = add(tv4, IrBuilder::create<Double>(5));
  TensorView* tv6 = add(tv5, IrBuilder::create<Double>(6));
  TensorView* tv7 = add(tv5, IrBuilder::create<Double>(7));
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  tv1->computeAt(tv2, -1);
  tv5->computeAt(tv6, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  at::Tensor t4 = at::rand_like(t0, options);

  auto t1 = t0 + 1;
  auto t2 = t1 + 2;
  auto t3 = t1 + 3;
  auto t5 = t4 + 5;
  auto t6 = t5 + 6;
  auto t7 = t5 + 7;

  std::vector<at::Tensor> aten_outputs = {t2, t3, t6, t7};
  std::vector<IValue> aten_inputs = {t0, t4};
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(t0, options),
      at::empty_like(t0, options),
      at::empty_like(t0, options),
      at::empty_like(t0, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  fe.runFusion(aten_inputs, cg_outputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(3));
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv1);
  fusion.addOutput(tv3);
  fusion.addOutput(tv5);

  tv2->computeAt(tv5, -1);
  tv4->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);
  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = aten_input + 3;
  auto t4 = t3 + 4;
  auto t5 = t2 + t4;

  std::vector<at::Tensor> aten_outputs = {t1, t3, t5};

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv1, tv2);
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));

  fusion.addOutput(tv4);

  tv1->split(0, 32);
  tv2->split(0, 32);
  tv3->split(0, 32);
  tv4->split(0, 32);

  tv3->computeAt(tv4, -2);
  tv1->computeAt(tv3, -1);
  tv2->computeAt(tv3, -2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);

  auto t1 = aten_input + 1;
  auto t2 = aten_input + 2;
  auto t3 = t1 + t2;
  auto aten_output = t3 + 4;

  at::Tensor cg_output = at::empty_like(aten_input, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2));
  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(3));
  TensorView* tv4 = add(tv3, IrBuilder::create<Double>(4));
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = aten_input + 3;
  auto t4 = t3 + 4;
  auto aten_output = t2 + t4;

  at::Tensor cg_output = at::empty_like(aten_input, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

// Test predication of grid reduction
TEST_F(NVFuserTest, FusionThreadPredicate_CUDA) {
  const int gdimx = 4;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv0);
  TensorView* tv2 = unaryOp(UnaryOpType::Neg, tv1);
  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(2));

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
  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);

  auto t2 = -aten_input.to(at::kDouble).sum({1});
  auto t3 = aten_input + 2.0;

  std::vector<at::Tensor> aten_outputs = {t3, t2};

  std::vector<at::Tensor> cg_outputs = {
      at::empty_like(aten_input, options), at::empty({numel_x}, options)};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLSTMCell_CUDA) {
  const int hidden_features = 512;
  const int batch_size = 64;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tvs[16];
  for (const auto i : c10::irange(16)) {
    tvs[i] = makeSymbolicTensor(2);
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

  std::vector<c10::IValue> aten_inputs;
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

  aten_inputs.insert(aten_inputs.end(), chunked0.begin(), chunked0.end());
  aten_inputs.insert(aten_inputs.end(), chunked1.begin(), chunked1.end());
  aten_inputs.insert(aten_inputs.end(), chunked2.begin(), chunked2.end());
  aten_inputs.insert(aten_inputs.end(), chunked3.begin(), chunked3.end());

  auto at_ingate =
      chunked0[0].add(chunked0[1]).add(chunked0[2]).add(chunked0[3]).sigmoid();
  auto at_forgetgate =
      chunked1[0].add(chunked1[1]).add(chunked1[2]).add(chunked1[3]).sigmoid();
  auto at_cellgate =
      chunked2[0].add(chunked2[1]).add(chunked2[2]).add(chunked2[3]).tanh();
  auto at_outgate =
      chunked3[0].add(chunked3[1]).add(chunked3[2]).add(chunked3[3]).sigmoid();

  auto at_cx = at::randn({batch_size, hidden_features}, options);
  aten_inputs.push_back(at_cx);
  auto at_cy = at_forgetgate.mul(at_cx).add(at_ingate.mul(at_cellgate));
  auto at_hy = at_outgate.mul(at_cy.tanh());

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {at_cy, at_hy}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeAtMultiBCast_CUDA) {
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    GTEST_SKIP() << "Somehow it fails on sm_80+ GPUs"
                 << " See https://github.com/pytorch/pytorch/issues/86717";
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(0.5));
  TensorView* tv2 = broadcast(tv1, {true, false});
  TensorView* tv3 = broadcast(tv1, {false, true});
  TensorView* tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Not possible to do computeAt at position -1 as recomputation
  // would be required. An exception should be thrown.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv1->computeAt(tv3, -1));
}

TEST_F(NVFuserTest, FusionReductionHalf_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = sum(tv2, {2});
  auto tv4 = castOp(DataType::Half, tv3);

  fusion.addOutput(tv4);

  const auto options =
      at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({8, 8, 16}, options);

  auto reduction_tv = tv3;

  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);

  auto aten_output = aten_input.add(1.0).to(at::kDouble).sum({2});

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

TEST_F(NVFuserTest, FusionReduceSingle_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({100, 1});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100, 1}, options);

  // Grab only tensor views, though there shouldn't be any other type
  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input});

  auto aten_output = aten_input.to(at::kDouble).sum({1});
  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, {red_dim, 2}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);
  auto aten_output = aten_input.to(at::kDouble).sum({red_dim, 2});

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

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast2_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {2}, IrBuilder::create<Double>(0), tv0);

  TensorView* tv2 = reductionOp(
      BinaryOpType::Add, {red_dim}, IrBuilder::create<Double>(0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);
  auto aten_output = aten_input.to(at::kDouble).sum({1, 2});

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

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast3_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, {red_dim}, IrBuilder::create<Double>(0), tv0);

  TensorView* tv2 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Double>(0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);

  // Apply reduction heuristic
  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);
  auto aten_output = aten_input.to(at::kDouble).sum({2, 1});

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

TEST_F(NVFuserTest, FusionTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({10, 20, 1});
  fusion.addInput(tv0);
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {2}, IrBuilder::create<Double>(0), tv0);
  fusion.addOutput(tv1);

  TORCH_CHECK(
      ir_utils::getReductionOps(&fusion, true /* ignore_trivial */).empty(),
      "Trivial reduction picked up by fusion");

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 20, 1}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});
  auto aten_output = aten_input.to(at::kDouble).sum({2});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 1, x = 1, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeConcreteTensor({w, x, y, z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = sum(tv1, {0});
  auto tv3 = sum(tv2, {0});
  auto tv4 = add(tv3, tv0);

  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);
  auto aten_output = t1.to(at::kDouble).sum({0}).sum({0}).add(t0);

  std::vector<IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialReduction3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int v = 1, w = 1, x = 1, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeConcreteTensor({v, w, x, y, z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = sum(tv1, {0, 1, 2});
  auto tv3 = add(tv2, tv0);

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);
  at::Tensor t1 = at::randn({v, w, x, y, z}, options);
  auto aten_output = t1.sum({0, 1, 2}).add(t0);

  std::vector<IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

// Make sure trivial reductions are correctly detected even with
// scheduling applied.
TEST_F(NVFuserTest, FusionDetectTrivialReduction1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  tv2->split(1, 4);
  tv2->split(1, 8);
  auto tv3 = tv2->rFactor({-1});
  auto tv4 = tv2->rFactor({-1});

  auto tv5 = broadcast(tv0, {true, false});
  auto tv6 = add(tv5, IrBuilder::create<Double>(1));
  auto tv7 = sub(tv6, IrBuilder::create<Double>(1));
  auto tv8 = sum(tv7, {0});
  fusion.addOutput(tv8);

  auto tv9 = broadcast(tv0, {false, true, true});
  auto tv10 = sum(tv9, {1});
  auto tv11 = sum(tv10, {1});
  fusion.addOutput(tv11);

  tv8->split(0, 3);
  tv10->split(1, 4);
  tv11->split(1, 5);

  tv0->computeAt(tv2, -1);
  tv0->computeAt(tv8, -1);
  tv0->computeAt(tv11, 1);

  // Test indexing to gmem-backed tensors
  tv3->setMemoryType(MemoryType::Global);
  tv8->setMemoryType(MemoryType::Global);

  GpuLower gpulw(&fusion);

  // No ReductionOp should be generated as all the reduction
  // exprs should be replaced with a unary set op.
  for (const auto expr : gpulw.kernel()->as<Fusion>()->exprs()) {
    TORCH_CHECK(!expr->isA<ReductionOp>());
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {t0, t0, t0}, __LINE__, __FILE__);
}

// Test detection of partially trivial reduction
TEST_F(NVFuserTest, FusionDetectTrivialReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->split(1, 1);
  // tv1->axis(1): non-trivial
  // tv1->axis(2): trivial

  auto tv3 = tv1->rFactor({-1});

  // Just to suppress register-allocation warning
  tv0->computeAt(tv2, 1);
  tv3->computeAt(tv1, -1);

  GpuLower gpulw(&fusion);

  // tv3's reduction axis is a trivial reduction. The only
  // ReductionOp should be for tv1.
  for (const auto expr : gpulw.kernel()->as<Fusion>()->exprs()) {
    if (expr->isA<ReductionOp>()) {
      auto reduction_out =
          expr->as<ReductionOp>()->outputs()[0]->as<TensorView>();
      TORCH_CHECK(reduction_out->name() == 1);
    }
  }
}

TEST_F(NVFuserTest, FusionInputsIdLookup_CUDA) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 8, 8}, options);
  at::Tensor t1 = at::randn({8, 8}, options);
  at::Tensor t2 = at::randn({6, 4}, options);

  // create a cache with max size 2;
  torch::jit::fuser::cuda::InputsIdLookup inputs_id_lookup(2);

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

TEST_F(NVFuserTest, FusionGroupGuardSimpleTensor_CUDA) {
  std::vector<int64_t> sizes_vec({16, 8, 8});
  std::vector<int64_t> strides_vec({64, 8, 1});
  auto tensor_type = TensorType::create(
      at::kFloat, c10::nullopt, sizes_vec, strides_vec, c10::nullopt);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // pass with identical shape
  auto t0 = at::randn({16, 8, 8}, options);
  TORCH_CHECK(complyWith(t0, tensor_type));

  // pass with dynamic shape
  auto t1 = at::randn({16, 16, 8}, options);
  TORCH_CHECK(complyWith(t1, tensor_type));

  // broadcasting semantic change failure
  auto t2 = at::randn({16, 1, 8}, options);
  TORCH_CHECK(!complyWith(t2, tensor_type));

  // contiguity failure via slicing
  auto t3 = t0.slice(1, 0, 8, 2);
  TORCH_CHECK(!complyWith(t3, tensor_type));

  // contiguity failure via slicing
  auto t4 = t0.slice(2, 0, 8, 2);
  TORCH_CHECK(!complyWith(t4, tensor_type));

  // rank failure
  auto t5 = at::randn({16, 8, 8, 8}, options);
  TORCH_CHECK(!complyWith(t5, tensor_type));

  // contiguity on stride 1 dimension with implicit broadcasting
  auto t = at::randn({4}, options);
  auto t6 = t.unsqueeze(1).expand({4, 8});
  TORCH_CHECK(complyWith(t6, TensorType::create(t6)));
}

TEST_F(NVFuserTest, FusionGroupGuardBroadcastTensor_CUDA) {
  std::vector<int64_t> sizes_vec({16, 1, 8});
  std::vector<int64_t> strides_vec({8, 8, 1});
  auto tensor_type = TensorType::create(
      at::kFloat, c10::nullopt, sizes_vec, strides_vec, c10::nullopt);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // broadcasting semantic change
  auto t0 = at::randn({16, 8, 8}, options);
  TORCH_CHECK(!complyWith(t0, tensor_type));

  // dtype failure
  auto t1 = at::randn({16, 1, 8}, options.dtype(at::kHalf));
  TORCH_CHECK(!complyWith(t1, tensor_type));

  // dtype failure
  auto t2 = at::randn({16, 1, 8}, options);
  TORCH_CHECK(complyWith(t2, tensor_type));

  // device inconsistency shouldn't fail
  auto t3 = at::randn({16, 1, 8}, options.device(at::kCPU, 0));
  TORCH_CHECK(complyWith(t3, tensor_type));
}

TEST_F(NVFuserTest, FusionGroupGuardPermutedTensor_CUDA) {
  std::vector<int64_t> sizes_vec({16, 8, 8});
  std::vector<int64_t> strides_vec({64, 1, 8});
  auto tensor_type = TensorType::create(
      at::kFloat, c10::nullopt, sizes_vec, strides_vec, c10::nullopt);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // failing permutation
  auto t0 = at::randn({16, 8, 8}, options);
  TORCH_CHECK(!complyWith(t0, tensor_type));

  // passing with dynamic shape
  auto t1 = t0.permute({0, 2, 1});
  TORCH_CHECK(complyWith(t1, tensor_type));
}

TEST_F(NVFuserTest, FusionGroupGuardRelaxedCheck_CUDA) {
  std::vector<int64_t> sizes_vec({16, 8, 8});
  std::vector<int64_t> strides_vec({128, 16, 1});
  auto tensor_type = TensorType::create(
      at::kFloat, c10::nullopt, sizes_vec, strides_vec, c10::nullopt);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // contiguity check passes although it differs
  auto t0 = at::randn({16, 16, 8}, options);
  TORCH_CHECK(complyWith(t0, tensor_type));

  // passing with dynamic shape
  auto t1 = t0.slice(1, 0, 16, 2);
  TORCH_CHECK(complyWith(t1, tensor_type));
}

TEST_F(NVFuserTest, FusionDisjointSet_CUDA) {
  DisjointSets<int> set;

  const std::set<int> group_x({0, 1, 2});
  const std::set<int> group_y({3, 4, 5});
  const std::set<int> group_z({6, 7, 8});
  const std::vector<std::set<int>> groups({group_x, group_y, group_z});
  std::set<int> group_all;
  std::for_each(groups.begin(), groups.end(), [&](const auto& g) {
    group_all.insert(g.begin(), g.end());
  });

  // Initially, nothing should be considered equivalent
  for (auto i : group_all) {
    for (auto j : group_all) {
      TORCH_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_x are equivalent
  for (auto i : group_x) {
    for (auto j : group_x) {
      set.mapEntries(i, j);
      TORCH_CHECK(set.mappingExists(i));
      TORCH_CHECK(set.mappingExists(j));
    }
  }

  // All values in group_x shoudl be equivalent with each other
  for (auto i : group_x) {
    for (auto j : group_x) {
      TORCH_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // But nothing else should be equivalent
  for (auto i : group_all) {
    for (auto j : group_y) {
      TORCH_CHECK(!set.permissiveAreMapped(i, j));
    }
    for (auto j : group_z) {
      TORCH_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_y are equivalent
  for (auto i : group_y) {
    for (auto j : group_y) {
      set.mapEntries(i, j);
      TORCH_CHECK(set.mappingExists(i));
      TORCH_CHECK(set.mappingExists(j));
    }
  }

  // group_x should be still equivalent
  for (auto i : group_x) {
    for (auto j : group_x) {
      TORCH_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // group_y should be now equivalent
  for (auto i : group_y) {
    for (auto j : group_y) {
      TORCH_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // But group_z should not be equivalent with anything yet
  for (auto i : group_all) {
    for (auto j : group_z) {
      TORCH_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_z are equivalent
  for (auto i : group_z) {
    for (auto j : group_z) {
      set.mapEntries(i, j);
      TORCH_CHECK(set.mappingExists(i));
      TORCH_CHECK(set.mappingExists(j));
    }
  }

  // Now each of the three groups should be equivalent within each
  // group
  for (const auto gi : c10::irange(groups.size())) {
    for (const auto gj : c10::irange(groups.size())) {
      for (auto i : groups[gi]) {
        for (auto j : groups[gj]) {
          TORCH_CHECK(
              (gi == gj && set.permissiveAreMapped(i, j)) ||
              (gi != gj && !set.permissiveAreMapped(i, j)));
        }
      }
    }
  }

  std::vector<int> all_elements = set.getAllElements().vector();
  std::sort(all_elements.begin(), all_elements.end());
  std::vector<int> group_all_vec(group_all.begin(), group_all.end());
  std::sort(group_all_vec.begin(), group_all_vec.end());
  TORCH_CHECK(all_elements == group_all_vec);

  set.clear();
  TORCH_CHECK(set.getAllElements().vector().size() == 0);

  // All cleared. Nothing should be considered equivalent.
  for (auto i : group_all) {
    for (auto j : group_all) {
      TORCH_CHECK(!set.permissiveAreMapped(i, j));
    }
  }
}

TEST_F(NVFuserTest, FusionNonUniqueBroadcastSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv0, {false, true});
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv3, tv2);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  // In order to do this, tv1->axis(1) and tv2->axis(1) must have the
  // same size, but we can't prove it, so this should throw an error.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv3->computeAt(tv4, -1));
}

TEST_F(NVFuserTest, FusionBiasGeluFwd_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;

  // bias vector
  auto t0 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(t0);
  auto t1 = castOp(DataType::Float, t0);
  // input tensor
  auto t2 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t2);
  auto t3 = castOp(DataType::Float, t2);
  auto t4 = broadcast(t1, {true, true, false});
  auto t5 = add(t4, t3);
  auto t6 = mul(t5, IrBuilder::create<Double>(0.5));
  auto t7 = mul(t5, IrBuilder::create<Double>(k_079));
  auto t8 = mul(t5, IrBuilder::create<Double>(k_004));
  auto t9 = mul(t8, t5);
  auto t10 = add(t9, IrBuilder::create<Int>(1));
  auto t11 = mul(t7, t10);
  auto t12 = unaryOp(UnaryOpType::Tanh, t11);
  auto t13 = add(t12, IrBuilder::create<Double>(1));
  auto t14 = mul(t6, t13);
  auto t15 = castOp(DataType::Half, t14);
  fusion.addOutput(t15);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::manual_seed(0);
  std::vector<int64_t> input_shape{6, 512, 4096};
  std::vector<int64_t> bias_shape{4096};

  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);

  auto at_x =
      at_bias.to(c10::ScalarType::Float) + at_input.to(c10::ScalarType::Float);
  auto aten_output_float =
      at_x * 0.5 * (1.0 + (k_079 * at_x * (1 + k_004 * at_x * at_x)).tanh());
  auto aten_output = aten_output_float.to(c10::ScalarType::Half);

  std::vector<IValue> aten_inputs = {at_bias, at_input};
  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBiasGeluBwd_CUDA) {
  if (at::cuda::getDeviceProperties(0)->major < 6) {
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;
  const float k_010 = 0.1070322243;

  // gradient tensor
  auto t0 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t0);
  auto t1 = castOp(DataType::Float, t0);
  // bias tensor
  auto t2 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(t2);
  auto t3 = castOp(DataType::Float, t2);
  // input tensor
  auto t4 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t4);
  auto t5 = castOp(DataType::Float, t4);
  auto t6 = broadcast(t3, {true, true, false});
  auto t7 = add(t6, t5);
  auto t8 = mul(t7, IrBuilder::create<Double>(k_079));
  auto t9 = mul(t7, IrBuilder::create<Double>(k_004));
  auto t10 = mul(t9, t7);
  auto t11 = add(t10, IrBuilder::create<Int>(1));
  auto t12 = mul(t8, t11);
  auto t13 = unaryOp(UnaryOpType::Tanh, t12);
  auto t14 = mul(t7, IrBuilder::create<Double>(0.5));
  auto t15 = mul(t13, t13);
  auto t16 = unaryOp(UnaryOpType::Neg, t15);
  auto t17 = add(t16, IrBuilder::create<Int>(1));
  auto t18 = mul(t7, IrBuilder::create<Double>(k_010));
  auto t19 = mul(t18, t7);
  auto t20 = add(t19, IrBuilder::create<Double>(k_079));
  auto t21 = mul(t17, t20);
  auto t22 = mul(t14, t21);
  auto t23 = add(t13, IrBuilder::create<Int>(1));
  auto t24 = mul(t23, IrBuilder::create<Double>(0.5));
  auto t25 = add(t22, t24);
  auto t26 = mul(t25, t1);
  // Save float output for validation
  fusion.addOutput(t26);
  auto t27 = castOp(DataType::Half, t26);
  fusion.addOutput(t27);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::manual_seed(1);
  std::vector<int64_t> input_shape{6, 512, 4096};
  std::vector<int64_t> bias_shape{4096};
  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);
  auto at_grad = at::randn(input_shape, options);

  auto at_x =
      at_bias.to(c10::ScalarType::Float) + at_input.to(c10::ScalarType::Float);
  auto at_tanh_out = (k_079 * at_x * (1 + k_004 * at_x * at_x)).tanh();
  auto at_ff = 0.5 * at_x *
          ((1 - at_tanh_out * at_tanh_out) * (k_079 + k_010 * at_x * at_x)) +
      0.5 * (1 + at_tanh_out);
  auto at_out = at_ff * at_grad;
  auto at_out_half = at_out.to(c10::ScalarType::Half);

  std::vector<IValue> aten_inputs = {at_grad, at_bias, at_input};
  std::vector<at::Tensor> aten_outputs = {at_out, at_out_half};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

// Reproducer of issue #459
TEST_F(NVFuserTest, FusionIssue459_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv1, tv3);

  // Create two outputs from the final arithmetic result
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);
  auto tv6 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv6);

  // Scheduling
  for (auto output : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    output->merge(-2, -1);
  }
  for (auto output : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    output->split(0, 128);
  }

  tv0->computeAt(tv5, -1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  const int numel_x = 10;
  const int numel_y = 20;
  auto t0 = at::randn({numel_x}, options);
  auto t1 = at::randn({numel_y, numel_x}, options);
  auto aten_output = (t0 + 1).unsqueeze(0) + t1 + 1;

  std::vector<IValue> aten_inputs = {t0, t1};

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion,
      cg_outputs,
      aten_inputs,
      {aten_output, aten_output},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionSmemIndexingSimple_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto aten_input = at::randn({12, 34}, options);
  at::Tensor aten_output = aten_input + 1.0 + 1.0 + 1.0;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSmemIndexing_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Int* symbolic_m_tile_dim = IrBuilder::create<Int>();
  Int* symbolic_split_k_tile_dim = IrBuilder::create<Int>();
  Int* symbolic_block_k_tile_dim = IrBuilder::create<Int>();
  // Compile-time integer for tiling
  int n_smem_tile = 32;

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
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
  // [M, rK, N]
  tv5->split(2, n_smem_tile);
  // [M, rK, No, Ni{32}]
  tv5->split(1, symbolic_block_k_tile_dim);
  // [M, rKo, rKi{i2}, No, Ni{32}]
  tv5->split(1, symbolic_split_k_tile_dim);
  // [M, rKoo, rKoi{i1}, rKi{i2}, No, Ni{32}]
  tv5->split(0, symbolic_m_tile_dim);
  // [Mo, Mi{i0}, rKoo, rKoi{i1}, rKi{i2}, No, Ni{32}]

  // Reorder so all outer tiles are in the leftmost 3 positions
  // [Mo, Mi{i0}, rKoo, rKoi{i1}, rKi{i2},     No, Ni{32}]
  // [Mo,     No, rKoo, rKoi{i1}, rKi{i2}, Mi{i0}, Ni{32}]
  tv5->reorder({{1, 5}, {5, 1}});

  // Factor out the outer reduction IterDomain, then run the inter-cta
  // reduction, and intra-cta reduction
  // [Mo, No, rKoo,  Koi{i1},  Ki{i2}, Mi{i0}, Ni{32}]
  // [Mo, No,       rKoi{i1}, rKi{i2}, Mi{i0}, Ni{32}]
  auto tv6 = tv5->rFactor({2});

  // Scope computations
  tv6->computeAt(tv5, 2);

  // [Mo, No, rKoo, Koi{i1},  Ki{i2}, Mi{i0}, Ni{32}]
  // [Mo, No, Ki{i2}, Mi{i0}, Ni{32}, rKoo, Koi{i1}]
  tv6->reorder({
      {5, -2},
      {6, -1},
      {2, 2},
      {3, 3},
      {4, 4},
  });

  // Setup compute at schedule
  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);
  tv4->computeAt(tv6, -1);

  // Cache smem tiles
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  std::vector<TensorView*> tv_list = {tv2, tv3, tv4, tv5, tv6};
  for (auto tv : tv_list) {
    tv->axis(-2)->parallelize(ParallelType::TIDz);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }

  constexpr int M = 31, K = 65, N = 32;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  // A, B, m_tile_dim, split_k, intra_cta_tile
  std::vector<IValue> aten_inputs = {t0, t1, 3, 4, 5};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

// Reproducer of issue 408
TEST_F(NVFuserTest, FusionCacheBeforeReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  tv2->split(0, 4);

  auto tv3 = tv2->cacheBefore();

  tv0->computeAt(tv3, -1);
  tv3->computeAt(tv2, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  auto aten_output = (aten_input + 1).to(at::kDouble).sum({1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  fe.runFusion({aten_input}, {cg_output});

  testValidate(
      &fusion, {cg_output}, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheBeforeReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto tv4 = tv2->cacheBefore();

  tv4->computeAt(tv3, 1);
  tv0->computeAt(tv4, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  const int numel_x = 10;
  const int numel_y = 20;
  const int numel_z = 30;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({numel_x, numel_y, numel_z}, options);
  auto t2 = (aten_input + 1).to(at::kDouble).sum({1});
  auto t3 = t2 + 1;
  std::vector<at::Tensor> aten_outputs = {t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue367_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Int* symbolic_m_tile_dim = IrBuilder::create<Int>();
  Int* symbolic_split_k_tile_dim = IrBuilder::create<Int>();
  Int* symbolic_block_k_tile_dim = IrBuilder::create<Int>();
  // Compile-time integer for tiling
  int n_smem_tile = 32;

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
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
  tv5->reorder({{1, 5}, {5, 1}});
  // [Mo, No, Koo, Koi, Ki, Mi, Ni]

  auto tv6 = tv5->rFactor({2});
  auto tv7 = tv5->rFactor({2});
  // [Mo, No, rKoo,  Koi,  Ki, Mi, Ni]
  // [Mo, No,       rKoi, rKi, Mi, Ni]

  // Scope computations
  tv6->computeAt(tv5, 2);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);
  tv4->computeAt(tv6, -1);

  // Cache smem tiles
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Local);
  tv6->setMemoryType(MemoryType::Local);
  tv7->setMemoryType(MemoryType::Local);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  std::vector<TensorView*> tv_list = {tv2, tv3, tv4, tv5, tv6, tv7};
  for (auto tv : tv_list) {
    tv->axis(-2)->parallelize(ParallelType::TIDz);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }
  tv2->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::TIDx);
  tv6->axis(3)->parallelize(ParallelType::TIDx);
  tv7->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(4)->parallelize(ParallelType::BIDx);
  tv3->axis(4)->parallelize(ParallelType::BIDx);
  tv4->axis(4)->parallelize(ParallelType::BIDx);
  tv6->axis(4)->parallelize(ParallelType::BIDx);
  tv7->axis(3)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::BIDx);

  constexpr int M = 3, K = 6, N = 16;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  // A, B, m, split_k, block_k
  std::vector<IValue> aten_inputs = {t0, t1, 2, 2, 3};
  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue468_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 100}, options);
  at::Tensor aten_output = aten_input.to(at::kDouble).sum({1}).sum({0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue363_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
  TensorView* tv5 = sum(tv4, {1});

  // Register inputs and outputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv4->setMemoryType(MemoryType::Global);

  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  tv5->axis(2)->parallelize(ParallelType::BIDx);

  constexpr int M = 3, K = 6, N = 16;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  std::vector<IValue> aten_inputs = {t0, t1};

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue484_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Double>(0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  constexpr int M = 100;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({M, M}, options);
  at::Tensor aten_output = aten_input.to(at::kDouble).sum({1});

  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue329_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = sum(tv1, {1});
  fusion.addOutput(tv3);

  tv1->computeAt(tv2, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> t0_shape{17, 19};
  auto aten_input = at::randn(t0_shape, options);
  auto t2 = (aten_input + 1).to(at::kDouble).sum({1});
  auto t3 = (aten_input + 1).to(at::kDouble).sum({1});
  std::vector<at::Tensor> aten_outputs = {t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue382_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv2->merge(1);
  tv4->merge(1);

  tv1->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);

  const int numel_x = 12;
  const int numel_y = 34;
  const int numel_z = 56;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({numel_x, numel_y}, options);
  auto t3 = at::randn({numel_x, numel_y, numel_z}, options);

  std::vector<IValue> aten_inputs = {t0, t3};
  auto aten_output = (t0 + 1).unsqueeze(-1) + t3;

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue507_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> t0_shape{17, 19};
  auto aten_input = at::randn(t0_shape, options);
  auto t1 = (aten_input + 1);
  auto aten_output = (t1 + 1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue532_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  const int M_BLOCK = 64;
  const int M_THREAD = 4;

  tv2->split(0, M_BLOCK);
  // tv2: [M/M_BLOCK, M_BLOCK]
  tv1->computeAt(tv2, 1);
  // tv1: [M/M_BLOCK, M_BLOCK]

  tv1->split(-1, M_BLOCK / M_THREAD);
  // tv1: [M/M_BLOCK, M_THREAD, M_BLOCK / M_THREAD]

  tv2->split(-1, M_THREAD);
  // tv2: [M/M_BLOCK, M_BLOCK / M_THREAD, M_THREAD]

  constexpr int M = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  at::Tensor aten_output = t0 + 1 + 1;

  testValidate(
      &fusion, outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLoopUnswitch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = add(tv0, IrBuilder::create<Double>(1));
  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv1->computeAt(tv2, -1);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  constexpr int M = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  at::Tensor aten_output = t0 + 1 + 1;

  testValidate(
      &fusion, outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue549_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2); // M, K
  TensorView* tv1 = makeSymbolicTensor(2); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));

  TensorView* tv3 = broadcast(tv2, {false, false, true});
  // tv3[I0, I1, B] = tv0[I0, I1]

  TensorView* tv4 = broadcast(tv1, {true, false, false});
  // tv4[B, I1, I2] = tv1[I1, I2]

  // tv5[I0, I1, I2] = tv3[I0, I1, B] * tv4[B, I1, I2]
  TensorView* tv5 = mul(tv3, tv4);
  // tv6[I0, R1, I2] = tv5[I0, I1, I2]
  TensorView* tv6 = sum(tv5, {1});
  fusion.addOutput(tv6);

  tv6->split(1, 32);
  // tv6[I0, R1o, R1i{32}, I2]

  auto tv7 = tv6->rFactor({1});
  // tv7[I0, R1o, I1i{32}, I2] = tv5[I0, I1, I2]
  // tv6[I0,    , R1i{32}, I2] = tv7[I0, R1o, I1i{32}, I2]

  tv6->split(0, 4);
  tv6->split(-1, 4);
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]

  tv0->computeAt(tv6, -1);
  tv1->computeAt(tv6, -1);

  // tv7[I0o, I0i{4}, R1o, I1i{32}, I2o, I2i{4}]
  // tv6[I0o, I0i{4},    , R1i{32}, I2o, I2i{4}]
  //--> (line symbolizes compute at location)
  // tv5[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, I1o]
  // tv7[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, R1o]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv0->computeAt(tv7, -1);
  tv1->computeAt(tv7, -1);
  // tv5[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, I1o |]
  // tv7[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, R1o |]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv6->axis(0)->parallelize(ParallelType::BIDz);
  tv6->axis(1)->parallelize(ParallelType::TIDz);

  tv6->axis(-2)->parallelize(ParallelType::BIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDy);

  tv6->axis(2)->parallelize(ParallelType::TIDx);
  tv7->axis(2)->parallelize(ParallelType::TIDx);

  constexpr int M = 65, K = 33, N = 17;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  // Lets specify a few bounds in launch params to make sure it works
  LaunchParams lparams(1, -1, -1, 32, 4, 4);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, lparams);
  fe.runFusion({t0, t1}, lparams);

  // Make sure bad launch params throws
  // TODO: Re-enable once we have parallelization validation in.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // ASSERT_ANY_THROW(fe.runFusion({t0, t1}, LaunchParams(1, 2, 3, 4, 5, 6)));

  // Don't specify any launch params
  auto cg_outputs = fe.runFusion({t0, t1});

  auto aten_output = (t0 + 1).to(at::kDouble).matmul(t1.to(at::kDouble));

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleCompileRtc_CUDA) {
  FusionExecutor fe;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 1> T0, Tensor<float, 1> T1) {
  if(threadIdx.x==0){
    for(size_t ki28 = 0; ki28 < T0.size[0]; ++ki28) {
      T1[ki28*T1.stride[0]] = T0[ki28*T0.stride[0]]*2;
    }
  }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      256, // gdimx
      1, // gdimy
      1, // gdimz
      1, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {8};
  auto in0 = at::randn(tensor_dims, options);
  auto out0 = at::empty_like(in0);
  fe.runRtc(lp, {in0, out0});

  auto out_ref = in0 * 2;
  TORCH_CHECK(out_ref.allclose(out0));
}

TEST_F(NVFuserTest, FusionSerialWelford_CUDA) {
  FusionExecutor fe;
  int x = 128, y = 64, z = 64;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_var,
    Tensor<float,1> out_avg
){
    for(int i0=0;i0<inp.size[0];i0++){
        float tmp_M2=0;
        float tmp_avg=0;
        long tmp_N=0;
        for(int i1=0;i1<inp.size[1];i1++){
            for(int i2=0;i2<inp.size[2];i2++){
                welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    inp[i0*inp.stride[0]+
                        i1*inp.stride[1]+
                        i2*inp.stride[2]],
                    0.f,
                    (long)1
                );
            }
        }
        out_var[i0*out_var.stride[0]]=
            tmp_M2/(tmp_N);
        out_avg[i0*out_avg.stride[0]]=
            tmp_avg;
    }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      1, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  fe.runRtc(lp, {in0, out_var, out_avg});

  TORCH_CHECK(in0.var({1, 2}, false).allclose(out_var));
  TORCH_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionBlockWelford_CUDA) {
  FusionExecutor fe;
  int x = 7, y = 8, z = 9;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,2> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> init_avg,
    Tensor<float,1> init_var,
    Tensor<long,0> init_N
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.stride[0]+
                        threadIdx.y*inp.stride[1]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    blockWelford<false,true,false>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long)1,
        threadIdx,
        blockDim,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.size[0]),
        0.f);
    __syncthreads();
    if(threadIdx.x<out_var.size[0] && threadIdx.y==0){
        welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    init_avg[threadIdx.x*init_avg.stride[0]],
                    init_var[threadIdx.x*init_var.stride[0]]*init_N[0],
                    init_N[0]
                );
        out_avg[threadIdx.x*out_avg.stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y};
  const std::vector<int64_t> init_dims = {x, z};

  // generate initial values
  auto init_in = at::randn(init_dims, options);
  auto init_var = init_in.var({1}, false);
  auto init_avg = init_in.mean({1});
  auto init_N =
      at::tensor(z, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  auto in0 = at::randn(tensor_dims, options);

  // run kernel
  auto out_var = at::zeros({x}, options);
  auto out_avg = at::zeros({x}, options);
  fe.runRtc(lp, {in0, out_avg, out_var, init_avg, init_var, init_N});

  // compare with reference output
  auto cat_tensor = at::cat({init_in, in0}, 1);
  TORCH_CHECK(cat_tensor.var({1}, false).allclose(out_var));
  TORCH_CHECK(
      cat_tensor.mean({1}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionBlockWelfordNoInit_CUDA) {
  FusionExecutor fe;
  int x = 7, y = 8, z = 9;

  // need support IValue for integer input as initial count
  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.stride[0]+
                        threadIdx.y*inp.stride[1]+
                        threadIdx.z*inp.stride[2]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    block_sync::init();
    blockWelford<false,true,true>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        threadIdx,
        blockDim,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.size[0]),
        0.f);
    __syncthreads();
    if(threadIdx.x<out_var.size[0] && threadIdx.y==0 && threadIdx.z==0){
        out_avg[threadIdx.x*out_var.stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      z // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  fe.runRtc(lp, {in0, out_avg, out_var});

  TORCH_CHECK(in0.var({1, 2}, false).allclose(out_var));
  TORCH_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionGridWelfordNoInit_CUDA) {
  FusionExecutor fe;
  int x = 128, y = 64, z = 128;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> work_buf_avg,
    Tensor<float,1> work_buf_M2,
    Tensor<long,1> work_buf_N,
    Tensor<int64_t,1> sync_flag
){
    __shared__ float shared_buf_avg[512];
    __shared__ float shared_buf_M2[512];
    __shared__ long shared_buf_N[512];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    float in = inp[ blockIdx.x  * inp.stride[0]+
                    blockIdx.y  * inp.stride[1]+
                    threadIdx.x * inp.stride[2]];
    block_sync::init();
    welford::gridWelford<
        true,true,false,
        true,false,false,
        false
    >(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        &work_buf_avg[0],
        &work_buf_M2[0],
        &work_buf_N[0],
        sync_flag,
        (float*)shared_buf_avg,
        (float*)shared_buf_M2,
        (long*)shared_buf_N,
        threadIdx.x<out_var.size[0],
        threadIdx.x<out_var.size[0],
        0.f,
        0,
        1);
    if(blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1){
        out_avg[threadIdx.x*out_avg.stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.stride[0]]=tmp_M2/tmp_N;
    }
}
    )";
  fe.compileRtc(kernel, "CudaCodeGen::kernel1");
  LaunchParams lp(
      x, // gdimx
      y, // gdimy
      1, // gdimz
      z, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const auto options_int =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);

  auto out_avg = at::empty({z}, options);
  auto out_var = at::empty({z}, options);
  auto work_buf_avg = at::empty({x * y * z}, options);
  auto work_buf_var = at::empty({x * y * z}, options);
  auto work_buf_N = at::empty({x * y * z}, options_int);
  auto sync_flag = at::zeros({1}, options_int);
  fe.runRtc(
      lp,
      {in0,
       out_avg,
       out_var,
       work_buf_avg,
       work_buf_var,
       work_buf_N,
       sync_flag});
  std::vector<int64_t> dims{0, 1};

  TORCH_CHECK(in0.mean(dims).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
  TORCH_CHECK(in0.var(dims, false).allclose(out_var));
}

TEST_F(NVFuserTest, FusionWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 32);
  tv_avg->split(0, 32);
  tv_avg->split(0, 4);
  tv_avg->reorder({{-1, -3}, {-3, -1}});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] /= N;

  testValidate(
      fe.kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionBlockWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->computeAt(tv_avg, -1);

  //
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] /= N;

  testValidate(
      fe.kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionGridWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(0)->parallelize(ParallelType::TIDx);
  tv_avg->axis(-1)->parallelize(ParallelType::BIDx);

  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] /= N;

  testValidate(
      fe.kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 4);
  ir_utils::rfactorHelper(tvs.avg, {2});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] /= N;

  testValidate(
      fe.kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordSchedule_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);
  // TODO: Why do we use launch params from here, but not scheduling???
  auto reduction_params = getReductionHeuristics(&fusion, {t0});
  scheduleReduction(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, lparams);
  auto outputs = fe.runFusion({t0}, lparams);

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] /= N;

  auto at_avg = t0.mean({1});
  auto at_var = t0.var({1}, false);
  auto at_n = at::ones({M}, options_int) * N;

  testValidate(
      fe.kernel(),
      outputs,
      {t0},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      reduction_params->lparams);
}

namespace {
void testWelford(DataType dtype, int red_axis, int odim, int rdim) {
  const int axis = red_axis;
  at::ScalarType aten_dtype = data_type_to_aten(dtype);

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(2, dtype);
  bool is_fp16 = dtype == DataType::Half;
  bool is_bf16 = dtype == DataType::BFloat16;
  TensorView* tv0_cast = tv0;
  if (is_fp16 || is_bf16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }
  fusion.addInput(tv0);
  auto tv1 = mul(tv0_cast, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {axis});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;

  TensorView* avg_cast = tv_avg;
  TensorView* M2_cast = tv_M2;

  if (is_fp16) {
    avg_cast = castOp(DataType::Half, tv_avg);
    M2_cast = castOp(DataType::Half, tv_M2);
  }
  if (is_bf16) {
    avg_cast = castOp(DataType::BFloat16, tv_avg);
    M2_cast = castOp(DataType::BFloat16, tv_M2);
  }

  fusion.addOutput(avg_cast);
  fusion.addOutput(M2_cast);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  std::vector<TensorView*> outputs_of_red;
  at::Tensor aten_input =
      (axis ? at::randn({odim, rdim}, options)
            : at::randn({rdim, odim}, options));

  if (is_fp16 || is_bf16) {
    outputs_of_red.push_back(avg_cast);
    outputs_of_red.push_back(M2_cast);
  }

  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  scheduleReduction(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  auto outputs = fe.runFusion({aten_input}, lparams);

  // by default Welford outputs sum of square diff so need to divide to
  // get var

  outputs[1] /= rdim;

  auto at_avg = aten_input.mean({axis});
  auto at_var = aten_input.var({axis}, false);
  auto at_n =
      (axis ? at::ones({odim, rdim}, options)
            : at::ones({rdim, odim}, options));
  at_n = at_n.sum({axis});

  testValidate(
      fe.kernel(),
      outputs,
      {aten_input},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      reduction_params->lparams);
}
} // namespace

TEST_F(NVFuserTest, FusionWelfordShmoo_CUDA) {
  std::vector<DataType> dtypes = {
      DataType::Double, DataType::Float, DataType::Half};
  // TODO: enable this for complex. Currently, complex yields
  // silent wrong results:
  //   Detected abs error of: 3.8062
  //     absolute tolerance was set to 2.23704e-06
  //     and relative tolerance set to 2.23704e-08
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
    for (auto& axis : red_axis) {
      for (auto& odim : output_dims) {
        for (auto& rdim : red_dims) {
          // TODO: original welford algorithm actually keeps a running sum of
          // squares, i.e. M_{2n} in the
          //       cf:
          //       https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
          //       algorithm notation, and it can reach inf for large numbers
          //       with half precision. skipping too large volumes for half for
          //       nwo might need further numerical experiments to re-design
          //       this.
          if (rdim > 32768 &&
              (dtype == DataType::Half || dtype == DataType::BFloat16)) {
            continue;
          }
          testWelford(dtype, axis, odim, rdim);
        }
      }
    }
  }
}

namespace {
void testVarMean(at::ScalarType dtype, int correction, bool keepdim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tvs = variance_mean(tv0, {1}, correction, keepdim);
  auto tv_mean = tvs.mean;
  auto tv_var = tvs.var;
  fusion->addOutput(tv_var);
  fusion->addOutput(tv_mean);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto at_var_mean = at::var_mean(t0, {1}, correction, keepdim);
  std::vector<at::Tensor> aten_outputs = {
      std::get<0>(at_var_mean), std::get<1>(at_var_mean)};

  testValidate(
      executor_cache.fusion(), outputs, {t0}, aten_outputs, __LINE__, __FILE__);
}
} // namespace

TEST_F(NVFuserTest, FusionVarMean_CUDA) {
  std::vector<at::ScalarType> dtypes = {at::kFloat, at::kDouble};
  std::vector<int> corrections = {0, 1};
  std::vector<bool> keepdims = {false, true};
  for (auto correction : corrections) {
    for (auto keepdim : keepdims) {
      for (auto dtype : dtypes) {
        testVarMean(dtype, correction, keepdim);
      }
    }
  }
}

TEST_F(NVFuserTest, FusionSimpleGemmTransposed_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views

  TensorView* tv0 = makeSymbolicTensor(2); // K, M
  TensorView* tv1 = makeSymbolicTensor(2); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv0_t = transpose(tv0);
  TensorView* tv1_t = transpose(tv1);

  TensorView* tv2 = broadcast(tv0_t, {false, false, true});
  // tv2[I0, I1, B] = tv0[I0, I1]

  TensorView* tv3 = broadcast(tv1_t, {true, false, false});
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

  tv0_t->computeAt(tv5, -1);
  tv1_t->computeAt(tv5, -1);

  // tv6[I0o, I0i{4}, R1o, I1i{32}, I2o, I2i{4}]
  // tv5[I0o, I0i{4},    , R1i{32}, I2o, I2i{4}]
  //--> (line symbolizes compute at location)
  // tv4[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, I1o]
  // tv6[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, R1o]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv0_t->computeAt(tv6, -1);
  tv1_t->computeAt(tv6, -1);
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

  at::Tensor t0 = at::randn({K, M}, options);
  at::Tensor t1 = at::randn({N, K}, options);

  // Lets specify a few bounds in launch params to make sure it works
  LaunchParams lparams(1, -1, -1, 32, 4, 4);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, lparams);
  fe.runFusion({t0, t1}, lparams);

  // Don't specify any launch params
  auto cg_outputs = fe.runFusion({t0, t1});

  auto aten_output = t0.t().to(at::kDouble).matmul(t1.t().to(at::kDouble));

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSoftmax3DTransposed_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(3);
  fusion.addInput(input_tv0);

  TensorView* input_t = transpose(input_tv0, 1, 2);

  TensorView* exp_tv1 = unaryOp(UnaryOpType::Exp, input_t);
  TensorView* sum_exp_tv2 = sum(exp_tv1, {-1});
  TensorView* bcast_sum_tv3 = broadcast(sum_exp_tv2, {false, false, true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* input_t_copy = transpose(input_tv0, 1, 2);
  TensorView* exp_tv1_copy = unaryOp(UnaryOpType::Exp, input_t_copy);

  TensorView* output_tv4 = div(exp_tv1_copy, bcast_sum_tv3);

  fusion.addOutput(output_tv4);

  bcast_sum_tv3->split(-1, tidx);

  sum_exp_tv2->split(-1, tidx);
  TensorView* sum_exp_rf_tv5 = sum_exp_tv2->rFactor({-2});

  output_tv4->split(-1, tidx);

  input_t->computeAt(sum_exp_rf_tv5, -1);
  input_t_copy->computeAt(output_tv4, -1);

  TensorView* tensors_to_parallelize[] = {
      sum_exp_tv2, bcast_sum_tv3, output_tv4, sum_exp_rf_tv5};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({dimx, dimz, dimy}, options);

  at::Tensor cg_output = at::empty({dimx, dimy, dimz}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_input_t = at::transpose(input, 1, 2);
  auto aten_output = at::_softmax(aten_input_t.to(at::kDouble), -1, false);

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed1_CUDA) {
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

  tv0 = transpose(tv0);

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

  // The this-position of the last tensor should be zero.
  TORCH_CHECK(
      tv7->nDims() == 3 && tv7->getComputeAtPosition() == 0 &&
      tv7->getMaxProducerPosition() == 1);
  TORCH_CHECK(
      tv6->nDims() == 3 && tv6->getComputeAtPosition() == 0 &&
      tv6->getMaxProducerPosition() == 1);
  // The position of every other tensor should be 1.
  for (auto tv : {tv1, tv2, tv3, tv4, tv5}) {
    TORCH_CHECK(tv->nDims() == 3 && tv->getComputeAtPosition() == 1);
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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  at::Tensor aten_input_t = aten_input.t();

  auto t1 = aten_input_t.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t1.add({3.0});
  auto t4 = t1.mul({2.0});
  auto t5 = t3.add(t2);
  auto t6 = t5.add(t4);
  auto t7 = t1.add(t4);

  std::vector<at::Tensor> aten_outputs = {t6, t7};

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed2_CUDA) {
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

  tv0 = transpose(tv0);

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  auto input_t = input.t();
  auto t1 = input_t.mul({-1.0});
  auto t2 = input_t.add({3.0});
  auto t3 = input_t.mul({2.0});
  auto t4 = t2.add(t1);
  auto t5 = t4.add(t3);
  auto t6 = t5.add(t3);

  std::vector<at::Tensor> aten_outputs = {t5, t6};

  testValidate(&fusion, cg_outputs, {input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed3_CUDA) {
  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  tv0 = permute(tv0, {3, 0, 1, 2});

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  tv1 = permute(tv1, {3, 0, 1, 2});

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

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t0_t = t0.permute({3, 0, 1, 2});
  auto t1_t = t1.permute({3, 0, 1, 2});
  auto t2 = t1_t.mul({0.979361});
  auto aten_output = t2.mul(t0_t);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed4_CUDA) {
  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  tv0 = permute(tv0, {3, 0, 1, 2});

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  tv1 = permute(tv1, {3, 0, 1, 2});

  TensorView* tv2 = makeSymbolicTensor(4);
  fusion.addInput(tv2);

  tv2 = permute(tv2, {3, 0, 1, 2});

  TensorView* tv3 = makeSymbolicTensor(4);
  fusion.addInput(tv3);

  tv3 = permute(tv3, {3, 0, 1, 2});

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

  std::vector<IValue> aten_inputs = {t0, t1, t2, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t0_t = t0.permute({3, 0, 1, 2});
  auto t1_t = t1.permute({3, 0, 1, 2});
  auto t2_t = t2.permute({3, 0, 1, 2});
  auto t3_t = t3.permute({3, 0, 1, 2});
  auto t4 = t2_t.sub(t3_t);
  auto t5 = t1_t.add(t4);
  auto aten_output = t5.sub(t0_t);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed5_CUDA) {
  // Case 5
  // tv2 = tv0 + 2.0
  // tv3 = tv1 * tv2
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  tv0 = transpose(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  tv1 = transpose(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(-1, 8);
  tv3->split(-1, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t2 = t0.t().add(2.0);
  auto aten_output = t1.t().mul(t2);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  tv0 = transpose(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  tv1 = transpose(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(-1, 8);
  tv2->split(-1, 4);
  tv3->merge(0);
  tv3->split(-1, 8);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t2 = t0.t().add(2.0);
  auto aten_output = t1.t().mul(t2);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmentReducePointwise_CUDA) {
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
  at::Tensor t0 = at::randn({128, 65}, options);
  at::Tensor t1 = at::randn({65}, options);
  at::Tensor t2 = at::randn({128, 65}, options);

  auto t3 = t0.add(1.0);
  auto t4 = std::get<0>(at::max(t3, 0));
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

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
      executor_cache.fusion(), outputs, {t0, t1, t2}, {t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMultipleVectorize_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  TensorView* tv1 = makeContigTensor(1);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv3 = add(tv0, tv1);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({40960}, options);
  at::Tensor t1 = at::randn({40960}, options);
  auto t2 = t0 + t1;

  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.profile(true);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime1 = executor_cache.getMostRecentKernelRuntime();
  auto log1 = std::dynamic_pointer_cast<PointwiseParams>(
      executor_cache.getMostRecentExecutorInfo().params);
  TORCH_CHECK(log1 != nullptr);
  TORCH_CHECK(log1->vectorize);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  t0 = at::randn({40964}, options);
  t1 = at::randn({40964}, options);
  t2 = t0 + t1;

  outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime2 = executor_cache.getMostRecentKernelRuntime();
  auto log2 = std::dynamic_pointer_cast<PointwiseParams>(
      executor_cache.getMostRecentExecutorInfo().params);
  TORCH_CHECK(log2 != nullptr);
  TORCH_CHECK(log2->vectorize);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  t0 = at::randn({40962}, options);
  t1 = at::randn({40962}, options);
  t2 = t0 + t1;

  outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime3 = executor_cache.getMostRecentKernelRuntime();
  auto log3 = std::dynamic_pointer_cast<PointwiseParams>(
      executor_cache.getMostRecentExecutorInfo().params);
  TORCH_CHECK(log3 != nullptr);
  TORCH_CHECK(log3->vectorize);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  TORCH_CHECK(runtime1 == runtime2);
  TORCH_CHECK(runtime1 != runtime3);
}

TEST_F(NVFuserTest, FusionVectorizeSimple_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(3);

  fusion.addInput(tv0);

  auto tv1 = unaryOp(UnaryOpType::Sin, tv0);

  fusion.addOutput(tv1);

  auto tv0_cache = tv0->cacheAfter();

  auto tv1_cache = tv1->cacheBefore();

  tv1->merge(0);
  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv1, 2);

  tv0_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::empty({2, 6, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  at::Tensor aten_output = aten_input.sin();

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleVectorizeUnroll_CUDA) {
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

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();
  auto tv3_cache = tv3->cacheBefore();

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);

  // Split by n_threads
  tv3->split(1, 2);
  tv3->split(0, 3);
  tv3->split(0, 1);

  // [bidx, unswitch, unroll{2}, tidx, vectorize{2}]

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(3)->parallelize(ParallelType::TIDx);

  tv3->reorder({{4, 2}});
  // [bidx, unswitch, vectorize{2}, unroll{2}, tidx]

  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);

  tv0_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv3->axis(2)->parallelize(ParallelType::Vectorize);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);
  tv1->computeAt(tv3, -1, ComputeAtMode::MostInlined);

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

TEST_F(NVFuserTest, FusionSegmentReduceSoftmax_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_shape{32, 64, 8};
  const int kReductionAxis = 1;

  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();

  fusion->addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = sum(tv1, {2}); // Group 0

  auto output = softmax(tv2, kReductionAxis); // Group 1
  fusion->addOutput(output);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({at_x});

  auto t1 = at_x.add(1.0);
  auto t2 = t1.sum({2});
  auto t3 = at::_softmax(t2.to(at::kDouble), -1, false);

  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen");
  TORCH_CHECK(
      optimized_fusion->fusionSegments()->groups().size() == 2,
      "segmentation didn't happen as expected");

  testValidate(
      executor_cache.fusion(), outputs, {at_x}, {t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSwizzle1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  fusion.addOutput(tv2);

  tv2->split(0, 7);
  tv2->split(0, 9);

  tv0->computeAt(tv2, 1);

  tv2->axis(0)->parallelize(ParallelType::BIDx);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->swizzle(SwizzleType::Transpose, {1, 2});

  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);

  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);

  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = (t0 + 1) * 2;

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSwizzle2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->split(-2, 4);

  tv2->split(-1, 4);
  tv2->split(-2, 4);

  tv0->computeAt(tv2, 1);

  tv2->reorder({{-1, -2}});

  tv1->setMemoryType(MemoryType::Shared);
  tv1->swizzle(SwizzleType::Transpose, {-2, -1});

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({123}, options);

  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = (t0 + 1) * 2;

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridPersistence_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  std::vector<TensorView*> tvs = {tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
  }

  const int numel_x = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = input.sum({0}).unsqueeze(-1).add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridPersistence2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  std::vector<TensorView*> tvs = {tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }

  const int numel_x = 10;
  const int numel_y = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = input.sum({0}).unsqueeze(0).add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordPersistence_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};

  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
  }

  const int numel_x = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(-1)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordPersistence2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true, false});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};
  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 10;
  const int numel_y = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto out = fe.runFusion({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(0)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue633_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int dx = 10;
  const int dy = 11;
  const int dz = 12;

  auto tv0 = makeConcreteTensor({dx, dy, dz});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({dx, dy, 1});
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->merge(1);
  tv2->merge(0);
  tv2->split(-1, 128);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dx, dy, dz}, options);
  at::Tensor t1 = at::randn({dx, dy, 1}, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastAcrossComputeAt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{17, 19};

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->split(1, 128);
  tv0->computeAt(tv3, 2);

  for (auto tv : {tv2, tv3}) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({shape[0]}, options);
  at::Tensor t1 = at::randn(shape, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t3 = t0.unsqueeze(-1).expand(shape) + t1;

  testValidate(&fusion, cg_outputs, aten_inputs, {t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedPointwise_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  const int kTDX = 64;
  const int kVecSize = 4;
  const int kNumElems = kTDX * kVecSize;

  tv2->split(1, kNumElems);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  tv2->split(-1, kVecSize);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 457;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedPointwiseMergeContig_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  auto tv1 = makeContigTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->reorder({{0, 1}, {1, 0}});
  tv2->merge(-2);

  const int kTDX = 64;
  const int kVecSize = 2;
  const int kNumElems = kTDX * kVecSize;

  tv2->split(-1, kNumElems);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  tv2->split(0, 128);
  tv2->split(-1, kVecSize);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int n = 32;
  const int c = 127;
  const int h = 51;
  const int w = 23;
  at::Tensor t0 = at::randn({n, c, h, w}, options);
  at::Tensor t1 = at::randn({n, c, h, w}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedPointwiseMergeSymbolicPass_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int kNumDims = 4;
  constexpr int kTDX = 64;
  constexpr int kVecSize = 2;
  constexpr int kNumElems = kTDX * kVecSize;

  auto tv0 = makeSymbolicTensor(kNumDims);
  auto tv1 = makeSymbolicTensor(kNumDims);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  // Create caches for vectorization
  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  // Merge all dimensions together except inner-most dim
  for (const auto idx : c10::irange(kNumDims - 2)) {
    tv2->merge(0);
  }
  // Split inner-most dim
  tv2->split(-1, kNumElems);
  tv2->split(-1, kVecSize);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  // Parallelization Strategy
  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int n = 5;
  const int c = 3;
  const int h = 51;
  const int w = 257;
  at::Tensor t0 = at::randn({n, c, h, w}, options);
  at::Tensor t1 = at::randn({n, c, h, w}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedPointwiseMergeSymbolicFail_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int kNumDims = 4;
  constexpr int kTDX = 64;
  constexpr int kVecSize = 2;
  constexpr int kNumElems = kTDX * kVecSize;
  std::vector<int64_t> bcast_shape{1, 1, 1, -1};

  auto tv0 = makeContigTensor(kNumDims);
  auto tv1 = TensorViewBuilder().shape(bcast_shape).build();
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  // Create caches for vectorization
  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  // Merge all dimensions together
  // Backward merge order is necessary for vectorize validation
  for (int idx = kNumDims - 1; idx > 0; --idx) {
    tv2->merge(idx - 1);
  }
  tv2->split(-1, kNumElems);
  tv2->split(-1, kVecSize);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  // Parallelization Strategy
  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int n = 32;
  const int c = 128;
  const int h = 51;
  const int w = 23;
  at::Tensor t0 = at::randn({n, c, h, w}, options);
  at::Tensor t1 = at::randn({1, 1, 1, w}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  // TODO: throw assertion - cannot merge non-contiguous vectorization axes
  // Make sure compilation fails
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedRFactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);

  auto tv3 = sum(tv2, {-1});

  fusion.addOutput(tv3);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();

  tv3->split(-1, 128 * 4);
  tv3->split(-1, 4);
  // Reduce outer dim first
  auto tv4 = tv3->rFactor({-3, -1});
  // Tv3 will reduce threads

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  tv2->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2050;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0.add(t1).sum(1);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedWrongDimFail_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    // Vectorize the wrong dimension
    tv->axis(-2)->parallelize(ParallelType::MisalignedVectorize);
  }

  FusionExecutor fe;
  // Make sure compilation fails
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedStride_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  const int kTDX = 64;
  const int kVecSize = 4;
  const int kNumElems = kTDX * kVecSize;

  tv2->split(1, kNumElems);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();

  tv2->split(-1, kVecSize);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2049;
  at::Tensor t0 = at::randn({bx, by}, options).index({"...", Slice(3)});
  at::Tensor t1 = at::randn({bx, by}, options).index({"...", Slice(3)});
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeMisalignedStrideFail_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  const int kTDX = 64;
  const int kVecSize = 4;
  const int kNumElems = kTDX * kVecSize;

  tv2->split(1, kNumElems);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  tv2->split(-1, kVecSize);

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  c0->axis(-1)->parallelize(ParallelType::MisalignedVectorize);
  c1->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::MisalignedVectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2049;
  at::Tensor t0 = at::randn({bx, by}, options).index({"...", Slice(3)});
  at::Tensor t1 = at::randn({bx, by}, options).index({"...", Slice(3)});
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  // Failure because the input + output tensors do not have the same stride
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion(aten_inputs));
}

TEST_F(NVFuserTest, FusionVectorization1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2048;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorization2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    // Vectorize the wrong dimension
    tv->axis(-2)->parallelize(ParallelType::Vectorize);
  }

  FusionExecutor fe;
  // Make sure compilation fails
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionVectorization3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  auto c2 = tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2049;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion(aten_inputs));

  aten_inputs[0] = t0.index({"...", Slice(1)});
  aten_inputs[1] = t1.index({"...", Slice(1)});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion(aten_inputs));

  t0 = at::randn({bx, 2048}, options).index({"...", Slice(4)});
  t1 = at::randn({bx, 2048}, options).index({"...", Slice(4)});
  aten_inputs = {t0, t1};
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0 + t1;
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizationRFactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);

  auto tv3 = sum(tv2, {-1});

  fusion.addOutput(tv3);

  tv3->split(-1, 128 * 4);
  tv3->split(-1, 4);
  // Reduce outer dim first
  auto tv4 = tv3->rFactor({-3, -1});
  // Tv3 will reduce threads

  auto tv6 = tv0->cacheAfter();
  auto tv7 = tv1->cacheAfter();

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  tv6->axis(-1)->parallelize(ParallelType::Vectorize);
  tv7->axis(-1)->parallelize(ParallelType::Vectorize);

  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2048;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto aten_output = t0.add(t1).sum(1);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);

  auto t3 = t0.add(t1).sum(1);

  testValidate(&fusion, cg_outputs, aten_inputs, {t3}, __LINE__, __FILE__);
}

// Unswitched loops with extent one may omit else clause.
TEST_F(NVFuserTest, FusionSizeOneLoop1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Progressively broadcast tensors
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = broadcast(tv0, {false, true});
  TensorView* tv4 = add(tv3, tv1);
  TensorView* tv5 = add(tv4, tv2);

  fusion.addOutput(tv5);

  // Split inner dimension
  tv5->split(1, 8);
  // Merge middle dims with outer dimensions
  tv5->merge(2);
  tv5->merge(0);

  // tv5[I0*I1o, I1i*I2]
  // Get a dim of size 1 to unswitch
  tv5->split(0, 1, false);

  // Compute everything inline
  tv0->computeAt(tv5, -1);

  tv5->axis(0)->parallelize(ParallelType::Unswitch);
  tv5->axis(1)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  // Make sure the unswitched loop does not have an else clause.
  GpuLower gpulw(&fusion);
  TORCH_CHECK(!UnswitchInElseChecker::check(gpulw));

  const int x = 11;
  const int y = 12;
  const int z = 13;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t1 = at::randn({x, y}, options);
  at::Tensor t2 = at::randn({z, x, y}, options);
  std::vector<IValue> aten_inputs = {t0, t1, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);
  auto t6 = (t0.unsqueeze(-1) + t1).unsqueeze(0) + t2;

  testValidate(&fusion, cg_outputs, aten_inputs, {t6}, __LINE__, __FILE__);
}

// The unswitched loop has extent one but inner loops don't. The else
// part should not be omitted.
TEST_F(NVFuserTest, FusionSizeOneLoop2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int x = 15;
  auto tv0 = makeConcreteTensor({x});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  fusion.addOutput(tv1);

  tv1->split(-1, 4);
  tv1->split(-2, 1);

  tv1->axis(-2)->parallelize(ParallelType::Unswitch);

  // Make sure the size-one unswitched loop does not omit the else clause.
  GpuLower gpulw(&fusion);
  TORCH_CHECK(UnswitchInElseChecker::check(gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);
  auto t1 = t0 + 1;

  testValidate(&fusion, cg_outputs, aten_inputs, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionValidateParallelize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);

  // Invalid as tv1 and tv2 do have the same ParallelType
  FusionExecutor fe;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(NVFuserTest, FusionValidateParallelize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv1->setMemoryType(MemoryType::Shared);

  // tv1 and tv2 do have the same ParallelType, but tv1 is on shared
  // memory, so it is valid
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->split(-1, 4);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Global);

  // tv1 and tv2 have the same shape and ParallelType
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->split(-1, 8);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Global);

  // tv1 and tv2 do not have the same shape but global memory comm is supported.
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 8);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // tv1 and tv2 do not have the same shape, but tv1 is on shared
  // memory, so it is valid
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

// See issue #995
TEST_F(NVFuserTest, FusionValidateParallelize6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t W = 5, X = 6, Y = 7, Z = 8;

  auto tv0 = makeConcreteTensor({X, Y, Z});
  auto tv1 = makeConcreteTensor({W, X, Y, Z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);
  tv4->split(0, 4);
  tv4->split(0, 3);
  tv4->split(0, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv3->computeAt(tv4, 2);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // Validation should throw an exception saying the first axes of tv2
  // and tv3 have incompatible parallelization. See also issue #995.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST_F(NVFuserTest, FusionDAGMerging_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(5);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Branch 0
  auto tv2 = sum(tv0, {0}); // 0
  auto tv3 = sum(tv2, {0}); // 1
  auto tv4 = sum(tv3, {0}); // 2
  auto tv5 = sum(tv4, {0}); // 3

  // Branch 1
  auto tv6 = add(tv1, IrBuilder::create<Double>(1)); // 4

  // Merge
  auto tv7 = add(tv6, tv5); // 5

  // Maximum expected output groups (can improve overtime):
  //  {0}, {1}, {2}, {3,4,5}
  //  without final merge would have been {0}, {1}, {2}, {3,4}, {5}

  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2, 2, 2}, options);
  at::Tensor t1 = at::randn({2}, options);

  std::vector<at::Tensor> aten_inputs = {t0, t1};

  KernelArgumentHolder args(KernelIndexMode::INT32);
  args.setDeviceIndex(0);
  args.push(aten_inputs);

  auto fusion_segments = fusion.segment(args);
  TORCH_CHECK(fusion_segments->groups().size() <= 4);
}

TEST_F(NVFuserTest, FusionDAGScalarMerging_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto i0 = IrBuilder::create<Double>();

  fusion->addInput(tv0);
  fusion->addInput(i0);

  auto i1 = add(i0, IrBuilder::create<Double>(1.0));
  auto i2 = mul(i1, i1);
  auto i3 = add(i2, i1);

  // Branch 0
  auto tv1 = sum(tv0, {0}); // 0
  auto tv2 = add(tv1, i2);
  // Branch 1
  auto tv3 = sum(tv2, {0}); // 1
  auto tv4 = add(tv3, i3);

  auto tv5 = add(tv4, i0);

  fusion->addOutput(tv5);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16, 16}, options);
  double s0 = 0.5;

  auto s1 = s0 + 1.0;
  auto s2 = s1 * s1;
  auto s3 = s2 + s1;
  auto t1 = t0.sum({0});
  auto t2 = t1 + s2;
  auto t3 = sum(t2, {0});
  auto t4 = t3 + s3;
  auto t5 = t4 + s0;

  auto outputs = executor_cache.runFusionWithInputs({t0, s0});

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
      executor_cache.fusion(), outputs, {t0, s0}, {t5}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBlockReduceInSerialLoop_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;
  constexpr int K = 20;

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = sum(tv0, {{1, 2}});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N, K}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);
  at::Tensor aten_output = t0.sum({1, 2});
  testValidate(
      &fusion, outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBlockWelfordInSerialLoop_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;
  constexpr int K = 20;

  auto tv0 = makeSymbolicTensor(3);
  auto tvs = Welford(tv0, {{1, 2}});
  fusion.addInput(tv0);
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);
  tv_avg->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, N, K}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);
  at::Tensor aten_avg = t0.mean({1, 2});
  at::Tensor aten_M2 = t0.var({1, 2}, false) * N * K;
  testValidate(
      &fusion, outputs, aten_inputs, {aten_avg, aten_M2}, __LINE__, __FILE__);
}

// See Issue #716
TEST_F(NVFuserTest, FusionIOTensorTrivialReductionRepro_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 11;

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  std::vector<int> reduction_axes = {1};
  std::vector<bool> broadcast_mask = {false, true};

  auto tv0_bcast = broadcast(tv0, broadcast_mask);
  auto path1_bcast = add(tv0_bcast, IrBuilder::create<Double>(1.0));
  auto path1 = sum(path1_bcast, reduction_axes);
  fusion.addOutput(path1);

  auto p = path1->split(1, 1);
  path1->rFactor({1});
  path1->axis(0)->parallelize(ParallelType::BIDx);
  tv0->computeAt(path1, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M}, options);
  at::Tensor t0_ref = t0.clone();
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  // inplace op, we are adding t0 to itself
  auto outputs = fe.runFusion(aten_inputs, {t0});

  TORCH_CHECK(outputs[0].allclose(t0_ref.add(1)));
}

TEST_F(NVFuserTest, FusionReductionPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto tv2 = tv0->cacheAfter();

  const int bdimx = 128;
  tv1->split(1, bdimx);
  tv1->split(1, 4);
  tv1->split(1, 1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Unroll);
  tv1->split(0, 10);
  tv0->computeAt(tv1, 4);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 650;
  int numel_y = 102;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  fe.runFusion({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({0});

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue728_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addOutput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addOutput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Double>(1));
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  auto tv6 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  // tv0 -> tv3 -+
  // tv1 --------+-> tv4 -> tv5
  //
  // tv2 -> tv6

  auto all_vals_under_tv3 =
      DependencyCheck::getAllValsBetween({tv3}, fusion.outputs());
  std::unordered_set<Val*> included_tensors({tv3, tv4, tv5});
  for (auto tv : included_tensors) {
    TORCH_CHECK(
        std::find(all_vals_under_tv3.begin(), all_vals_under_tv3.end(), tv) !=
            all_vals_under_tv3.end(),
        "TV",
        tv->name(),
        " not found");
  }
  for (auto tv : ir_utils::filterByType<TensorView>(fusion.vals())) {
    if (included_tensors.find(tv) == included_tensors.end()) {
      TORCH_CHECK(
          std::find(all_vals_under_tv3.begin(), all_vals_under_tv3.end(), tv) ==
              all_vals_under_tv3.end(),
          "TV",
          tv->name(),
          " should not be found");
    }
  }

  auto no_dependency = DependencyCheck::getAllValsBetween({}, fusion.outputs());
  TORCH_CHECK(no_dependency.empty(), "No val should be returned");

  auto no_dep_path = DependencyCheck::getAllValsBetween({tv0, tv1}, {tv6});
  TORCH_CHECK(no_dep_path.empty(), "No val should be returned");

  auto no_dep_path2 = DependencyCheck::getAllValsBetween({tv2}, {tv5});
  TORCH_CHECK(no_dep_path2.empty(), "No val should be returned");

  auto just_tv3 = DependencyCheck::getAllValsBetween({tv3}, {tv3});
  TORCH_CHECK(
      just_tv3.size() == 1 && *(just_tv3.begin()) == tv3,
      "Only tv3 should be included");
}

TEST_F(NVFuserTest, FusionIssue757_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv1->computeAt(tv4, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 650;
  int numel_y = 102;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t3 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0.sum({1});
  auto t2 = t1.unsqueeze(-1).expand({numel_x, numel_y});
  auto t4 = t2 + t3;

  testValidate(&fusion, outputs, inputs, {t4}, __LINE__, __FILE__);
}

// See issue #759
TEST_F(NVFuserTest, FusionPredicatedBlockBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 4);
  tv1->computeAt(tv4, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 100;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t3 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  auto t1 = t0.sum({1});
  auto t2 = t1.unsqueeze(-1).expand({numel_x, numel_y});
  auto t4 = t2 + t3;

  testValidate(&fusion, outputs, inputs, {t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmentVerticalMerge_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  // {first kernel}
  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, tv0);
  auto tv3 = sum(tv2, {0});
  auto tv4 = add(tv3, tv0);
  auto tv5 = sum(tv4, {0});
  auto tv6 = sum(tv5, {0});
  // {second kernel}
  auto tv7 = add(tv6, tv5);
  auto tv8 = add(tv7, tv5);
  auto tv9 = sum(tv8, {0});

  fusion->addOutput(tv9);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  KernelArgumentHolder args(KernelIndexMode::INT32);
  args.setDeviceIndex(0);
  args.push(t0);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), args, segment_options);

  TORCH_CHECK(segmented_fusion->groups().size() == 2);
}

TEST_F(NVFuserTest, FusionSegmentHorizontalMerge_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto i0 = IrBuilder::create<Double>();

  fusion->addInput(tv0);
  fusion->addInput(i0);

  // Branch 0 {first kernel}
  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv0, i0);
  auto tv3 = unaryOp(UnaryOpType::Rsqrt, tv2);
  auto tv4 = sum(tv3, {0});

  // Branch 1 {first kernel}
  auto tv5 = unaryOp(UnaryOpType::Rsqrt, tv3);
  auto tv6 = sum(tv5, {0});

  // Incompatible {second kernel}
  auto tv7 = sum(tv6, {0});

  fusion->addOutput(tv1);
  fusion->addOutput(tv4);
  fusion->addOutput(tv7);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  KernelArgumentHolder args(KernelIndexMode::INT32);
  args.setDeviceIndex(0);
  args.push(t0);
  c10::IValue scalar = 1.0;
  args.push(scalar);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), args, segment_options);

  TORCH_CHECK(segmented_fusion->groups().size() == 2);
}

TEST_F(NVFuserTest, FusionSegmentMixReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  // def of tv1 in kernel 1 through horizontal
  auto tv1 = sum(tv0, {0, 1});
  // kernel 2
  auto tv2 = sum(tv0, {2});
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = sum(tv4, {2});
  // end of kernel 2
  // kernel 1
  auto tv6 = unaryOp(UnaryOpType::Rsqrt, tv0);
  auto tv7 = sum(tv6, {0, 1});
  auto tv8 = sum(tv6, {0, 1});

  fusion->addOutput(tv1);
  fusion->addOutput(tv5);
  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  KernelArgumentHolder args(KernelIndexMode::INT32);
  args.setDeviceIndex(0);
  args.push(t0);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), args, segment_options);

  TORCH_CHECK(segmented_fusion->groups().size() <= 2);
}

TEST_F(NVFuserTest, FusionSBAR_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // N, H, W, C format
  std::vector<int64_t> input_shape{656, 7, 7, 64};

  auto x = makeContigTensor(4);
  auto y = makeContigTensor(4);
  auto weight = makeContigTensor(1);
  auto bias = makeContigTensor(1);

  fusion.addInput(x);
  fusion.addInput(y);
  fusion.addInput(weight);
  fusion.addInput(bias);

  const size_t kNumberOfDims = x->nDims();
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  for (const auto axis : c10::irange(kNumberOfDims - 1)) {
    broadcast_mask[axis] = true;
  }

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto scale = mul(x, weight_bcast);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto scale_bias = add(scale, bias_bcast);
  auto scale_bias_add = add(scale_bias, y);
  auto scale_bias_add_relu = unaryOp(UnaryOpType::Relu, scale_bias_add);

  fusion.addOutput(scale_bias_add_relu);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[3]}, options);
  at::Tensor at_bias = at::zeros({input_shape[3]}, options);

  // inputs
  std::vector<c10::IValue> inputs = {at_x, at_y, at_weight, at_bias};

  // outputs
  std::vector<at::Tensor> outputs;

  auto lparams = schedulePointwise(&fusion, inputs);

  FusionExecutor executor;
  executor.compileFusion(&fusion, inputs, lparams);
  outputs = executor.runFusion(inputs, lparams);

  auto at_scale = at::mul(at_x, at_weight);
  auto at_scale_bias = at::add(at_scale, at_bias);
  auto pwise_add = at::add(at_scale_bias, at_y);
  auto output = at::relu(pwise_add);

  testValidate(&fusion, outputs, inputs, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSingleElement_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(2.5));

  auto tv2 = add(tv1, IrBuilder::create<Double>(3.5));
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({}, options);

  at::Tensor cg_output = at::empty({}, options);

  auto lparams = schedulePointwise(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  fe.runFusion({input}, {cg_output}, lparams);

  auto aten_output = input.add(2.5).add(3.5);

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBNBackwardRepro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 4;
  int c = 4;
  int h = 4;
  int w = 4;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);
  auto save_mean = makeSymbolicTensor(1);
  fusion.addInput(save_mean);
  auto save_invstd = makeSymbolicTensor(1);
  fusion.addInput(save_invstd);

  auto grad_out_prev = makeSymbolicTensor(numDims);
  fusion.addInput(grad_out_prev);
  auto gt_0 =
      makeSymbolicTensor(numDims); // single tensor broadcasted is dangerous.
  fusion.addInput(gt_0);

  auto gt_bool = binaryOp(BinaryOpType::GT, gt_0, IrBuilder::create<Int>(1));
  auto gt_float = castOp(DataType::Float, gt_bool);

  auto grad_out = mul(grad_out_prev, gt_float);

  Val* eps_ptr = IrBuilder::create<Double>(1e-5);

  auto grads = batch_norm_backward(
      input,
      grad_out,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      true,
      eps_ptr,
      {true, true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);
  fusion.addOutput(grads.grad_bias);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({batch, c, h, w}, options);
  at::Tensor input1 = at::randn({c}, options);
  at::Tensor input2 = at::randn_like(input1);
  at::Tensor input3 = at::randn_like(input1);
  at::Tensor input4 = at::randn_like(input1);
  at::Tensor input5 = at::randn_like(input1);
  at::Tensor input6 = at::randn_like(input0);
  at::Tensor input7 = at::randn_like(input0);

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> inputs = {
      input0, input1, input2, input3, input4, input5, input6, input7};
  auto outputs = fec.runFusionWithInputs(inputs);
}

// TODO: We only changed inputs, merge this with the test above.
TEST_F(NVFuserTest, FusionBNBackwardRepro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 2;
  int c = 81;
  int h = 1;
  int w = 1;
  int numDims = 4;

  // auto input = makeSymbolicTensor(numDims);
  auto input = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);
  auto save_mean = makeSymbolicTensor(1);
  fusion.addInput(save_mean);
  auto save_invstd = makeSymbolicTensor(1);
  fusion.addInput(save_invstd);

  // auto grad_out_prev = makeSymbolicTensor(numDims);
  auto grad_out_prev = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(grad_out_prev);
  // auto gt_0 =
  //     makeSymbolicTensor(numDims); // single tensor broadcasted is dangerous.
  auto gt_0 = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(gt_0);

  auto gt_bool = binaryOp(BinaryOpType::GT, gt_0, IrBuilder::create<Int>(1));
  auto gt_float = castOp(DataType::Float, gt_bool);

  auto grad_out = mul(grad_out_prev, gt_float);

  Val* eps_ptr = IrBuilder::create<Double>(1e-5);

  auto grads = batch_norm_backward(
      input,
      grad_out,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      true,
      eps_ptr,
      {true, true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);
  fusion.addOutput(grads.grad_bias);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({batch, c, h, w}, options);
  at::Tensor input1 = at::randn({c}, options);
  at::Tensor input2 = at::randn_like(input1);
  at::Tensor input3 = at::randn_like(input1);
  at::Tensor input4 = at::randn_like(input1);
  at::Tensor input5 = at::randn_like(input1);
  at::Tensor input6 = at::randn_like(input0);
  at::Tensor input7 = at::randn_like(input0);

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> inputs = {
      input0, input1, input2, input3, input4, input5, input6, input7};
  auto outputs = fec.runFusionWithInputs(inputs);
}

TEST_F(NVFuserTest, FusionBNRepro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  int batch = 14;
  int c = 65;
  int h = 7;
  int w = 7;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto bias = makeSymbolicTensor(1);
  fusion.addInput(bias);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);

  auto momentum_ptr = IrBuilder::create<Double>(kMomentum);
  auto eps_ptr = IrBuilder::create<Double>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({batch, c, h, w}, options);
  at::Tensor input2 = at::randn({c}, options);
  at::Tensor input3 = at::randn_like(input2);
  at::Tensor input4 = at::randn_like(input2);
  at::Tensor input5 = at::randn_like(input2);

  auto input1_ref = input1.clone();
  auto input2_ref = input2.clone();
  auto input3_ref = input3.clone();
  auto input4_ref = input4.clone();
  auto input5_ref = input5.clone();

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> aten_inputs = {input1, input2, input3, input4, input5};
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto at_results = at::native_batch_norm(
      input1_ref,
      input2_ref,
      input3_ref,
      input4_ref,
      input5_ref,
      kTraining,
      kMomentum,
      kEps);

  auto at_output = std::get<0>(at_results);
  auto at_mean = std::get<1>(at_results);
  auto at_invstd = std::get<2>(at_results);

  std::vector<at::Tensor> aten_outputs = {at_output, at_mean, at_invstd};

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBNRepro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  int batch = 2;
  int c = 4;
  int h = 17;
  int w = 17;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);

  Val* momentum_ptr = IrBuilder::create<Double>(kMomentum);
  Val* eps_ptr = IrBuilder::create<Double>(kEps);

  auto result = batch_norm(
      input,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      kTraining,
      momentum_ptr,
      eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({batch, c, h, w}, options);

  auto input1_ref = input1.clone();
  at::Tensor r_m;
  at::Tensor r_v;
  at::Tensor weight;
  at::Tensor bias;

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> aten_inputs = {input1};
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto at_results = at::native_batch_norm(
      input1_ref, r_m, r_v, weight, bias, kTraining, kMomentum, kEps);

  auto at_output = std::get<0>(at_results);
  auto at_mean = std::get<1>(at_results);
  auto at_invstd = std::get<2>(at_results);

  std::vector<at::Tensor> aten_outputs = {at_output, at_mean, at_invstd};

  testValidate(
      &fusion, cg_outputs, aten_inputs, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionZeroSizeTensorPW_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(2.5));
  fusion.addOutput(tv2);

  // This test used to just have:
  // auto tv3 = makeConcreteTensor({0});
  // and somehow that was running through our system fine, but size-0 tensors
  // are not supported, so making sure this fails.
  auto tv3 = set(tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn({2}, options);
  at::Tensor input1 = at::randn({0}, options);
  at::Tensor cg_output2 = at::empty({2}, options);
  at::Tensor cg_output3 = at::empty({0}, options);

  // Fails at schedule pointwise because our (maybe only) size-0 check is in
  // binding input sizes which the scheduler ends up calling.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(schedulePointwise(&fusion, {input0, input1}));
}

TEST_F(NVFuserTest, FusionZeroSizeTensorReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({0});
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn({2, 4}, options);
  at::Tensor input1 = at::randn({0}, options);
  at::Tensor cg_output2 = at::empty({2}, options);
  at::Tensor cg_output3 = at::empty({0}, options);

  auto reduction_params = getReductionHeuristics(&fusion, {input0, input1});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  auto lparams = reduction_params->lparams;
  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto cg_outputs = fe.runFusion({input0, input1}, lparams);
  auto aten_output2 = input0.sum({1});
  at::Tensor aten_output3 = at::empty({0}, options);

  testValidate(
      &fusion,
      cg_outputs,
      {input0, input1},
      {aten_output2, aten_output3},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionZeroSizeTensorNormalization_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  auto tv5 = makeConcreteTensor({0});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn({2, 4}, options);
  at::Tensor input1 = at::randn({0}, options);
  at::Tensor cg_output2 = at::empty({2, 4}, options);
  at::Tensor cg_output3 = at::empty({0}, options);

  auto reduction_params = getPersistentHeuristics(&fusion, {input0, input1});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  schedulePersistentKernel(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;
  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto cg_outputs = fe.runFusion({input0, input1}, lparams);
  auto aten_output2 = input0.sum({0}).add(input0);
  at::Tensor aten_output3 = at::empty({0}, options);

  testValidate(
      &fusion,
      cg_outputs,
      {input0, input1},
      {aten_output2, aten_output3},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionSegmentIoAlias_CUDA) {
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

  // Note: test alias;
  fusion->aliasOutputToInput(tv6, tv0);
  // TODO: support output on aliased fusion #1488
  // remove tv7 after #1488
  // fusion->addOutput(tv6);
  TensorView* tv7 = add(tv6, IrBuilder::create<Double>(1)); // Group 0
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  at::Tensor t1 = at::randn({65}, options);
  at::Tensor t2 = at::randn({128, 65}, options);

  auto t3 = t0.add(1.0);
  auto t4 = std::get<0>(at::max(t3, 0));
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);
  auto t7 = t6.add(1.0);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  // TODO: support output on aliased fusion #1488
  // validating aliasing
  // TORCH_INTERNAL_ASSERT(outputs[0].data_ptr() == t0.data_ptr());

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
      executor_cache.fusion(), outputs, {t0, t1, t2}, {t7}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWelford1Output_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion->addOutput(tvs.var_sum);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.var({1}, false) * 65;
  testValidate(fusion, outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTranslate1Welford_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  auto tv_out = add(tv0, broadcast(tvs.avg, {false, true}));
  fusion->addOutput(tv_out);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});
    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    testValidate(
        fusion,
        outputs,
        {t0},
        {t0.add(t0.mean({1}).unsqueeze(1))},
        __LINE__,
        __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  TORCH_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 2);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);

  bool found_welford = false;
  for (auto group : runtime2->fusionSegments()->groups()) {
    for (auto expr : group->exprs()) {
      if (expr->isA<WelfordOp>()) {
        found_welford = true;
      }
    }
  }
  TORCH_CHECK(found_welford);
}

TEST_F(NVFuserTest, FusionTranslate2Welford_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto tv_out1 = add(tv0, broadcast(tvs1.avg, {false, true}));
  fusion->addOutput(tv_out1);

  auto tvs2 = Welford(tv0, {1});
  auto tv_out2 = add(tv0, broadcast(tvs2.avg, {false, true}));
  fusion->addOutput(tv_out2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    auto out = t0.add(t0.mean({1}).unsqueeze(1));
    testValidate(fusion, outputs, {t0}, {out, out}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  TORCH_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 4);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);
  // // Check it was not translated
  bool found_welford = false;
  for (auto group : runtime2->fusionSegments()->groups()) {
    for (auto expr : group->exprs()) {
      if (expr->isA<WelfordOp>()) {
        found_welford = true;
      }
    }
  }
  TORCH_CHECK(found_welford);
}

TEST_F(NVFuserTest, FusionLargeWelfordNormalization_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});

  fusion->addOutput(tvs1.var_sum);
  fusion->addOutput(sum_of_tv0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.var({1}, false) * inner_size;
    auto t2 = t0.sum({1});
    testValidate(fusion, outputs, {t0}, {t1, t2}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  auto runtime = run_test(65536);
  TORCH_CHECK(!runtime->isSegmented());
}

TEST_F(NVFuserTest, FusionWelfordOuterPersistence_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});
  auto sum_bcasted = broadcast(sum_of_tv0, {false, true});
  auto avg_bcasted = broadcast(tvs1.avg, {false, true});
  auto tv0_plus_sum = add(tv0, sum_bcasted);
  auto tv0_plus_avg = add(tv0, avg_bcasted);

  fusion->addOutput(tv0_plus_sum);
  fusion->addOutput(tv0_plus_avg);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.to(c10::kDouble).mean({1}).unsqueeze(1) + t0;
    auto t2 = t0.to(c10::kDouble).sum({1}).unsqueeze(1) + t0;
    testValidate(fusion, outputs, {t0}, {t2, t1}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  for (auto inner_size : {4096, 8192, 32768}) {
    auto runtime = run_test(inner_size);
    TORCH_CHECK(!runtime->isSegmented());
  }
}

TEST_F(NVFuserTest, FusionSegmentIslands_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = sum(tv0, {0});
  auto tv3 = sum(tv1, {1});
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16}, options);
  at::Tensor t1 = at::randn({16, 16}, options);

  FusionExecutorCache fusion_executor_cache(std::move(fusion));
  fusion_executor_cache.runFusionWithInputs({t0, t1});
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv3 = broadcast(tv0, {false, true, true, true});
  auto tv4 = broadcast(tv1, {false, false, true, true});
  auto tv5 = unaryOp(UnaryOpType::Rsqrt, tv2);

  auto tv6 = add(tv3, tv5);
  auto tv7 = add(tv4, tv5);
  auto tv8 = add(tv3, tv4);

  auto tv9 = add(tv6, tv7);
  auto tv10 = add(tv9, tv8);

  fusion->addOutput(tv10);

  tv0->computeAt(tv10, -2);
  tv1->computeAt(tv10, -2);
  tv2->computeAt(tv10, -2);

  TORCH_CHECK(tv3->getComputeAtPosition() == 1);
  TORCH_CHECK(tv4->getComputeAtPosition() == 2);
  TORCH_CHECK(tv5->getComputeAtPosition() == 3);

  TORCH_CHECK(tv6->getMaxProducerPosition() == 3);
  TORCH_CHECK(tv7->getMaxProducerPosition() == 3);
  TORCH_CHECK(tv8->getMaxProducerPosition() == 2);
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(3);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, false, true});
  auto tv3 = add(tv2, tv1);

  fusion->addOutput(tv3);
  tv3->split(-2, 4);
  tv3->reorder({{-1, -2}});
  tv0->computeAt(tv3, -2);
  tv1->computeAt(tv3, -2);
  TORCH_CHECK(tv2->getComputeAtPosition() == 2);
  TORCH_CHECK(tv3->getMaxProducerPosition() == 2);
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, false, true});
  auto tv3 = broadcast(tv2, {false, true, false, false});
  auto tv4 = add(tv3, tv1);

  fusion->addOutput(tv4);
  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);
  TORCH_CHECK(tv2->getComputeAtPosition() == 2);
  TORCH_CHECK(tv3->getMaxProducerPosition() == 3);
}

TEST_F(NVFuserTest, FusionSimpleWarp_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  tv1->split(1, 32);
  auto tv1_rf = tv1->rFactor({1});
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 128}, options);

  auto at_output = input1.sum({1}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});

  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleWarpPad_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->padToMultipleOfWarp(32);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp(32);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(-1)->padToMultipleOfWarp(32);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->padToMultipleOfWarp(32);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->padToMultipleOfWarp(32);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->padToMultipleOfWarp(32);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 127}, options);

  auto at_output = input1.sum({1}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpPadMergeSplit_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  auto tv2 = broadcast(tv1, {false, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 17, 128}, options);

  auto at_output = input1.sum({1, 2}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSerialWarpReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  auto tv2 = broadcast(tv1, {false, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 17, 128}, options);

  auto at_output = input1.sum({1, 2}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialWarpReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({17, 18, 128, 1});

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2, 3});
  auto tv2 = broadcast(tv1, {false, true, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-2)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({17, 18, 128, 1}, options);

  auto at_output = input1.sum({1, 2, 3}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMultipleDimBinding_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv_add = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv_add);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  auto tv4 = add(tv0, tv_add);

  fusion->addOutput(tv3);
  fusion->addOutput(tv4);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->padToMultipleOfWarp(32);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp(32);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(-1)->padToMultipleOfWarp(32);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->padToMultipleOfWarp(32);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->padToMultipleOfWarp(32);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->padToMultipleOfWarp(32);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->padToMultipleOfWarp(64);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 128}, options);
  at::Tensor input2 = at::randn({16, 128}, options);

  auto at_output = input1.sum({1}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1, input2});
  auto outputs = fe.runFusion({input1, input2});
  testValidate(
      fusion.get(),
      outputs,
      {input1, input2},
      {at_output, input1 + input2},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionPadNoWarpReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(0)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 31}, options);

  auto at_output = input1.sum({1}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpMutipleThreadDim_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  tv2->split(1, 8);
  auto tv2_rf = tv2->rFactor({-1});
  tv2_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv2_rf->axis(-1)->padToMultipleOfWarp();

  TransformPropagatorWithCheck propagator(tv2_rf);
  MaxRootDomainInfoSpanningTree(tv2_rf).traverse(&propagator);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv0->computeAt(tv2, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 31}, options);

  auto at_output = (input1 + 1).sum({1});

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpReduceUnrollOuterLoop_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  tv1->split(0, 4);
  auto tv1_rf = tv1->rFactor({2});

  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(1)->parallelize(ParallelType::Unroll);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  tv1->axis(1)->parallelize(ParallelType::Unroll);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxRootDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(1)->parallelize(ParallelType::Unroll);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unroll);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({16, 128}, options);

  auto at_output = input1.sum({1}, true).add(input1);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_output}, __LINE__, __FILE__);
}

// Repro of issue #1579
TEST_F(NVFuserTest, FusionWarpReducePredication_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1 = {1024};
  std::vector<int64_t> shape2 = {50};

  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto tv2 = makeConcreteTensor(shape2);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  auto tv4 = sum(tv3, {0});
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  // Just to fill the smem buffer by a thread block of 1024 threads
  // with some values
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Make the tv4_rf reduction a warp reduction to trigger the
  // bug. Since the smem buffer is filled with some values due to the
  // reduction of tv1, those values would be used by predicated-out
  // threads.
  tv4->split(-1, 10);
  auto tv4_rf = tv4->rFactor({-1});
  tv4_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv4_rf->axis(-1)->padToMultipleOfWarp();

  tv4_rf->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t2});
  auto cg_outputs = fe.runFusion({t0, t2});

  auto t1 = t0.sum({0});
  auto t4 = (t2 + 1).sum({0}) + 1;

  testValidate(&fusion, cg_outputs, {t0, t2}, {t1, t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegfaultReduction_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 2;
  int c = 1;
  int h = 1;
  int w = 1;
  int numDims = 4;

  auto input = makeConcreteTensor({-1, 1, 1, 1});
  fusion.addInput(input);
  auto bcast_bias = makeConcreteTensor({-1, 1, 1, 1});
  fusion.addInput(bcast_bias);

  std::vector<int64_t> at_sum_axes;
  std::vector<int> outer_reduction_axes;
  std::vector<bool> outer_broadcast_mask(numDims, false);
  Val* N = IrBuilder::create<Double>(1);
  for (const auto axis : c10::irange(numDims)) {
    if (axis != 1) {
      outer_reduction_axes.push_back(axis);
      at_sum_axes.push_back(axis);
      outer_broadcast_mask[axis] = true;
      N = mul(N, input->domain()->domain()[axis]->extent());
    }
  }

  auto output0 = mul(input, bcast_bias);
  fusion.addOutput(output0);
  auto output1 = sum(output0, outer_reduction_axes);
  fusion.addOutput(output1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({batch, c, h, w}, options);
  at::Tensor input1 = at::randn({batch, c, h, w}, options);

  auto at_output0 = input0.mul(input1);
  auto at_output1 = at_output0.sum(at_sum_axes);

  FusionExecutorCache fec(std::move(fusion_ptr));
  std::vector<IValue> inputs = {input0, input1};
  auto outputs = fec.runFusionWithInputs(inputs);

  testValidate(
      &fusion, outputs, inputs, {at_output0, at_output1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPredicateElimination1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2));
  auto tv3 = add(tv2, IrBuilder::create<Double>(3));

  fusion.addOutput(tv3);

  tv3->split(0, 32);
  tv0->computeAt(tv3, 1);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  {
    GpuLower gpulw(&fusion);
    TORCH_CHECK(!PredicatedChecker::isPredicated(tv2, gpulw));
  }

  tv2->axis(1)->parallelize(ParallelType::Serial);
  tv2->split(1, 5);

  {
    GpuLower gpulw(&fusion);
    TORCH_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));
  }
}

// Repro of issue #1571
TEST_F(NVFuserTest, FusionPredicateElimination2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  tv1->split(1, 4);
  tv1->split(0, 4);
  tv2->split(1, 4);
  tv2->split(0, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum({1}) + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPredicateElimination3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  auto tv3 = tv0->cacheAfter();

  tv1->split(0, 10);
  tv1->split(0, 33);
  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  auto tv4 = tv1->rFactor({-1});
  auto tv5 = tv1->rFactor({-1});

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  GpuLower gpulw(&fusion);

  // The fusion has three reductions: one within each thread, one
  // within each block, and another with the whole grid. All of them
  // should not need to be predicated as they use the same init value
  // and same reduction op.
  TORCH_CHECK(!PredicatedChecker::isPredicated(tv4, gpulw));
  TORCH_CHECK(!PredicatedChecker::isPredicated(tv5, gpulw));
  TORCH_CHECK(!PredicatedChecker::isPredicated(tv1, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  for (auto size : {1, 2, 999, 1001, 1234, 10000}) {
    auto t0 = at::randn({size}, options);

    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});

    auto ref = sum(t0) + 1;
    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionPredicateElimination4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});

  auto tv2 = sum(tv1, {0});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  auto tv4 = max(tv1, {0});
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv1->split(1, 7);
  tv1->split(0, 11);
  tv1->reorder({{1, 2}, {2, 1}});
  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  GpuLower gpulw(&fusion);

  // tv2 uses the same op and init with tv1, so tv2 should be fine
  // without a predicate. However, tv4, while it uses the tv1 as its
  // input, the reduction op and init value is different from those of
  // tv1, so tv4 needs to be predicated.
  TORCH_CHECK(!PredicatedChecker::isPredicated(tv2, gpulw));
  TORCH_CHECK(PredicatedChecker::isPredicated(tv4, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> sizes = {1, 2, 33, 34, 64, 99};
  for (auto s0 : sizes) {
    for (auto s1 : sizes) {
      auto t0 = at::randn({s0, s1}, options);

      FusionExecutor fe;
      fe.compileFusion(&fusion, {t0});
      auto cg_outputs = fe.runFusion({t0});

      auto t1 = t0.sum({1});
      auto t3 = t1.sum({0}) + 1;
      auto t5 = std::get<0>(t1.max(0)) + 1;

      testValidate(&fusion, cg_outputs, {t0}, {t3, t5}, __LINE__, __FILE__);
    }
  }
}

TEST_F(NVFuserTest, FusionPredicateElimination5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tvs2 = Welford(tv1, {0});
  auto tv3 = set(tvs2.avg);
  fusion.addOutput(tv3);

  tvs2.avg->split(0, 4);
  TransformPropagatorWithCheck propagator(tvs2.avg);
  MaxRootDomainInfoSpanningTree(tvs2.avg).traverse(&propagator);
  auto avg_rf = ir_utils::rfactorHelper(tvs2.avg, {1});

  avg_rf->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(avg_rf);

  GpuLower gpulw(&fusion);

  // The first per-thread welford needs to be predicated as the N
  // input is different from its init value. The second welford op
  // does not need a predicate.
  TORCH_CHECK(PredicatedChecker::isPredicated(avg_rf, gpulw));
  TORCH_CHECK(!PredicatedChecker::isPredicated(tvs2.avg, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> sizes = {1, 2, 33, 34, 64, 99};
  for (auto s0 : sizes) {
    auto t0 = at::randn({s0}, options);

    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});
    auto cg_outputs = fe.runFusion({t0});

    auto ref = t0.mean({0});

    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionPredicateElimination6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  fusion.addOutput(tv4);

  tv4->split(1, 5);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->reorder({{0, 1}, {1, 0}});
  tv3->computeAt(tv4, 1);

  GpuLower gpulw(&fusion);

  // The expression for tv2 is a local-to-local expression. It
  // satisfies all the requirements of predicate elimination, except
  // for the on on split root domains. As the second root axis of tv2
  // is split, its index exceeds its extent (i.e., 3 in this case)
  // without its predicate.
  TORCH_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));

  // Unlike tv2, tv3 is computed at tv4, so the second root axis does
  // have a zero domain. Its index should look like "i * 5 + j", where
  // i comes from the first root domain and j comes from the split
  // inner domain.
  TORCH_CHECK(!PredicatedChecker::isPredicated(tv3, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 4;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPredicateElimination7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  tv3->split(-1, 5);
  tv3->split(-1, 4);
  tv3->split(-1, 3);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // The last split of tv2 is a non-divisible split, and omitting it
  // is invalid.
  GpuLower gpulw(&fusion);
  TORCH_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 3;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionForceFp16Simple_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});

  // Group 2
  auto tv4 = add(tv3, tv1); // Edge: tv3: expect cast
  auto tv5 = castOp(DataType::Half, tv4);

  fusion->addOutput(tv5);

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  // Check the segmented edge is fp16
  auto segmented_fusion = fec.getMostRecentKernelRuntime()->fusionSegments();
  for (auto edge : segmented_fusion->edges()) {
    auto edge_tv = edge->val->as<TensorView>();
    TORCH_CHECK(edge_tv->getDataType() == DataType::Half);
  }
}

TEST_F(NVFuserTest, FusionForceBf16Simple_CUDA) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
    return;
  }

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});

  // Group 2
  auto tv4 = add(tv3, tv1); // Edge: tv3: expect cast
  auto tv5 = castOp(DataType::BFloat16, tv4);

  fusion->addOutput(tv5);

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  // Check the segmented edge is bf16
  auto segmented_fusion = fec.getMostRecentKernelRuntime()->fusionSegments();
  for (auto edge : segmented_fusion->edges()) {
    auto edge_tv = edge->val->as<TensorView>();
    TORCH_CHECK(edge_tv->getDataType() == DataType::BFloat16);
  }
#else
  GTEST_SKIP() << "requires cuda 11.0 or newer toolkit";
#endif
}

TEST_F(NVFuserTest, FusionForceFp16NotAllCast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv3 = sum(tv0, {1});
  auto tv4 = broadcast(tv3, {false, true, false});
  auto tv5 = sum(tv0, {1});

  // Group 2
  auto tv6 = add(tv4, tv1); // edge tv4, expect cast
  auto tv7 = castOp(DataType::Half, tv6);

  // Group 3
  auto tv8 = sum(tv5, {1}); // edge tv5, don't expect cast

  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  auto segmented_fusion = fec.getMostRecentKernelRuntime()->fusionSegments();
  auto complete_fusion = segmented_fusion->completeFusion();

  // Check that the edge that wasn't fp16 is the producer of the
  //  reduction op, i.e. tv8 = sum(tv5,{1});.
  for (auto edge : segmented_fusion->edges()) {
    auto edge_tv = edge->val->as<TensorView>();
    if (edge_tv->getDataType() == DataType::Float) {
      auto consumer = *(complete_fusion->unordered_uses(edge_tv).begin());
      TORCH_CHECK(consumer->isA<ReductionOp>());
    }
  }
}

TEST_F(NVFuserTest, FusionForceBf16NotAllCast_CUDA) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
    return;
  }

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv3 = sum(tv0, {1});
  auto tv4 = broadcast(tv3, {false, true, false});
  auto tv5 = sum(tv0, {1});

  // Group 2
  auto tv6 = add(tv4, tv1); // edge tv4, expect cast
  auto tv7 = castOp(DataType::BFloat16, tv6);

  // Group 3
  auto tv8 = sum(tv5, {1}); // edge tv5, don't expect cast

  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  auto segmented_fusion = fec.getMostRecentKernelRuntime()->fusionSegments();
  auto complete_fusion = segmented_fusion->completeFusion();

  // Check that the edge that wasn't fp16 is the producer of the
  //  reduction op, i.e. tv8 = sum(tv5,{1});.
  for (auto edge : segmented_fusion->edges()) {
    auto edge_tv = edge->val->as<TensorView>();
    if (edge_tv->getDataType() == DataType::Float) {
      auto consumer = *(complete_fusion->unordered_uses(edge_tv).begin());
      TORCH_CHECK(consumer->isA<ReductionOp>());
    }
  }
#else
  GTEST_SKIP() << "requires cuda 11.0 or newer toolkit";
#endif
}

TEST_F(NVFuserTest, FusionBufferReuseBroadCastMultiVisit_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Double>(2));
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv3, tv1);
  auto tv5 = mul(tv4, IrBuilder::create<Double>(3));
  fusion->addOutput(tv5);

  // t4 cannot inner re-use t2, because there's a broadcast
  //  between them.
  tv0->computeAt(tv5, 1, ComputeAtMode::BestEffort);
  tv3->computeAt(tv5, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);

  auto at_output = ((in0 * 2).unsqueeze(2) + in1) * 3;
  FusionExecutor fe;
  fe.compileFusion(fusion, {in0, in1});
  auto outputs = fe.runFusion({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseStressTest_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Double>(2));
  auto tv3 = mul(tv0, IrBuilder::create<Double>(3));
  auto tv4 = mul(tv2, tv3);
  // Broadcast buffer can be reused through outer sharing
  auto tv5 = broadcast(tv4, {true, false, false});
  auto tv6 = mul(tv5, IrBuilder::create<Double>(5));
  auto tv7 = mul(tv6, tv1);
  auto tv8 = mul(tv7, IrBuilder::create<Double>(7));
  // tv9 shouldn't alias to avoid buffer over-subscription
  auto tv9 = broadcast(tv4, {true, false, false});
  auto tv10 = mul(tv9, IrBuilder::create<Double>(9));
  auto tv11 = add(tv5, tv9);
  fusion->addOutput(tv7);
  fusion->addOutput(tv11);

  tv0->computeAt(tv5, 1, ComputeAtMode::BestEffort);
  tv0->computeAt(tv9, 1, ComputeAtMode::BestEffort);

  tv5->computeAt(tv7, 1, ComputeAtMode::BestEffort);
  tv5->computeAt(tv11, 1, ComputeAtMode::BestEffort);
  tv9->computeAt(tv11, 1, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);
  auto t2 = in0 * 2;
  auto t3 = in0 * 3;
  auto t4 = t2 * t3;
  auto t5 = t4.unsqueeze(0);
  auto t6 = t5 * 5;
  auto t7 = t6 * in1;
  auto t8 = t7 * 7;
  auto t9 = t4.unsqueeze(0);
  auto t10 = t9 * 9;
  auto t11 = t5 + t9;
  FusionExecutor fe;
  fe.compileFusion(fusion, {in0, in1});

  auto at_output = ((in0 * 2).unsqueeze(2) + in1) * 3;
  auto outputs = fe.runFusion({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, {t7, t11}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseLargeBuffer_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({256, 512});

  fusion->addInput(tv0);

  auto tv1 = mul(tv0, IrBuilder::create<Double>(2));
  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  auto tv3 = mul(tv2, IrBuilder::create<Double>(2));
  auto tv4 = mul(tv3, IrBuilder::create<Double>(2));
  auto tv5 = mul(tv4, IrBuilder::create<Double>(2));
  auto tv6 = mul(tv5, IrBuilder::create<Double>(2));

  fusion->addOutput(tv6);

  tv0->computeAt(tv6, 1, ComputeAtMode::BestEffort);
  tv6->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({256, 512}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {in0});
  auto outputs = fe.runFusion({in0});

  auto at_out = in0.mul(2).mul(2).mul(2).mul(2).mul(2).mul(2);

  testValidate(fusion, outputs, {in0}, {at_out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseNo2hop_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Double>(2));
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv3, tv1); // T4 to be inner aliased first, and
                            //  shouldn't outer alias on top
  auto tv5 = mul(tv4, IrBuilder::create<Double>(3));
  auto tv6 = mul(tv5, IrBuilder::create<Double>(3));
  fusion->addOutput(tv6);

  tv0->computeAt(tv6, 1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv6, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);
  FusionExecutor fe;
  fe.compileFusion(fusion, {in0, in1});
  auto outputs = fe.runFusion({in0, in1});

  auto at_out = (in0.mul(2.0).unsqueeze(2) + in1).mul(3.0).mul(3.0);

  testValidate(fusion, outputs, {in0, in1}, {at_out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseAllocationOrder_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({3, 3, 3});

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  auto tv3 = mul(tv2, IrBuilder::create<Double>(2));

  fusion->addOutput(tv3);

  // In this case tv1 "reuses" allocation of tv2
  //  due to the switched allocation order
  tv1->computeAt(tv2, 1, ComputeAtMode::BestEffort);

  tv0->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({3, 3, 3}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {in0});
  auto outputs = fe.runFusion({in0});

  auto at_out = in0.sum(1).mul(2).mul(2);

  testValidate(fusion, outputs, {in0}, {at_out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseLiveInterval_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({16, 16});

  fusion->addInput(tv0);

  auto tv1 = mul(tv0, IrBuilder::create<Double>(3));
  auto tv2 = mul(tv1, IrBuilder::create<Double>(2));
  auto tv3 = mul(tv2, IrBuilder::create<Double>(2));
  // tv1 used till here, cannot be reused by tv2 or tv3
  auto tv4 = mul(tv3, tv1);

  fusion->addOutput(tv4);

  tv0->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({16, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {in0});
  auto cg_outputs = fe.runFusion({in0});

  auto at_t0 = in0 * 3.0;
  auto at_out = at_t0 * 2.0 * 2.0 * at_t0;

  testValidate(fusion, cg_outputs, {in0}, {at_out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseNoAcrossBroadcast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Double>(2));
  auto tv3 = mul(tv0, IrBuilder::create<Double>(3));
  auto tv4 = mul(tv2, tv3);
  auto tv5 = broadcast(tv4, {false, false, true});
  auto tv6 = mul(tv5, tv1);
  auto tv7 = mul(tv6, IrBuilder::create<Double>(7));
  fusion->addOutput(tv7);

  // tv6 shouldn't re-use t2 or t3 because of
  //  the broadcast in between
  tv0->computeAt(tv4, 1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv7, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);
  FusionExecutor fe;
  fe.compileFusion(fusion, {in0, in1});
  auto outputs = fe.runFusion({in0, in1});

  auto t2 = in0 * 2;
  auto t3 = in0 * 3;
  auto t4 = t2 * t3;
  auto t5 = t4.unsqueeze(2);
  auto t6 = t5 * in1;
  auto t7 = t6 * 7;
  testValidate(fusion, outputs, {in0, in1}, {t7}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue970_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int nelm = 10;

  // tv3 = tv0 + sum(tv0)
  auto tv0 = makeConcreteTensor({nelm, nelm});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  fusion.addOutput(tv3);

  tv1->split(1, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({nelm, nelm}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = sum(t0, {1}).unsqueeze(-1).expand({nelm, nelm}) + t0;

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Reproducer of #1016
TEST_F(NVFuserTest, FusionIssue1016_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2));

  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 8);

  int numel_x = 10;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  auto ref = t0 + 1 + 2;

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Reproducer of #1021
TEST_F(NVFuserTest, FusionIssue1021_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  auto tv3 = tv2->cacheBefore();

  tv2->split(0, 2);

  tv1->computeAt(tv2, 1);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);
  std::vector<IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  auto ref = (t0 + 1).unsqueeze(-1);

  testValidate(&fusion, outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Reproducer of issue #1053
TEST_F(NVFuserTest, FusionNonUniqueThreadDim_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion->addOutput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv2);

  tv1->split(0, 8);
  auto tv1_rf = tv1->rFactor({-1});

  tv1_rf->computeAt(tv1, 1);

  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({32}, options);

  auto at_tv1 = (input1).sum({0});
  auto at_tv2 = input1 + 1;

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});
  testValidate(
      fusion.get(), outputs, {input1}, {at_tv1, at_tv2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap1_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  tv1->split(0, 8, false);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->split(0, 8, false);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  // The extents of tv1 and tv2 axes are equal even though their
  // actual values are not statically known
  GpuLower gpulw(fusion.get());
  const auto& pdmap = gpulw.parallelDimensionMap();
  for (const auto i : c10::irange(tv1->domain()->domain().size())) {
    auto dom1 = tv1->domain()->domain()[i];
    auto dom2 = tv2->domain()->domain()[i];
    TORCH_INTERNAL_ASSERT(pdmap.equalDim(dom1->extent(), dom2->extent()));
  }

  TORCH_CHECK(pdmap.isExact(ParallelType::TIDx));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({32}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});

  testValidate(
      fusion.get(),
      outputs,
      {input1},
      {input1 + 1, input1 + 1},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion->addOutput(tv3);

  tv3->split(-1, 8, false);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  GpuLower gpulw(fusion.get());
  const auto& pdmap = gpulw.parallelDimensionMap();
  TORCH_CHECK(pdmap.isExact(ParallelType::TIDx));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({11}, options);
  at::Tensor input2 = at::randn({11, 13}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  auto ref = input1.unsqueeze(-1) + input2;

  testValidate(
      fusion.get(), outputs, {input1, input2}, {ref}, __LINE__, __FILE__);
}

// Mix symbolic and concrete tensors
TEST_F(NVFuserTest, FusionParallelDimensionMap3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv2);
  auto tv3 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv3);

  tv2->split(0, 10);
  tv3->split(0, 20);

  auto tv4 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv4);
  auto tv5 = add(tv0, IrBuilder::create<Double>(1));
  fusion->addOutput(tv5);

  // Not mapped but equal extent
  tv4->split(0, 10);
  tv5->split(0, 10);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv4->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);

  GpuLower gpulw(fusion.get());
  const auto& pdmap = gpulw.parallelDimensionMap();
  TORCH_CHECK(!pdmap.isExact(ParallelType::TIDx));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");
  TORCH_CHECK(pdmap.isExact(ParallelType::TIDy));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDy)->isConst() &&
      pdmap.get(ParallelType::TIDy)->as<Int>()->value().value() == 10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({13}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {input1});
  auto outputs = fe.runFusion({input1});

  testValidate(
      fusion.get(),
      outputs,
      {input1},
      {input1 + 1, input1 + 1, input1 + 1, input1 + 1},
      __LINE__,
      __FILE__);
}

// Parallelizing merged broadcast domains
TEST_F(NVFuserTest, FusionParallelDimensionMap4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  tv4->reorder({{1, 2}, {2, 1}});
  tv4->merge(0);
  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  // TIDx is mapped to tv4.axis(0) as well as tv2.axis(0), so it's not
  // exact.
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  GpuLower gpulw(&fusion);
  const auto& pdmap = gpulw.parallelDimensionMap();
  TORCH_CHECK(!pdmap.isExact(ParallelType::TIDx));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({13}, options);
  at::Tensor input2 = at::randn({15, 13}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  auto ref = (input1 + 1).unsqueeze(0) + input2;

  testValidate(&fusion, outputs, {input1, input2}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv3 = broadcast(tv0, {false, true});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);

  GpuLower gpulw(&fusion);
  const auto& pdmap = gpulw.parallelDimensionMap();
  TORCH_CHECK(pdmap.isExact(ParallelType::TIDx));
  TORCH_CHECK(pdmap.isExact(ParallelType::TIDy));
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDx)->isConst() &&
      pdmap.get(ParallelType::TIDx)->as<Int>()->value().value() == 4);
  TORCH_CHECK(
      pdmap.get(ParallelType::TIDy)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDy)->as<NamedScalar>()->name() == "blockDim.y");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({13}, options);
  at::Tensor input2 = at::randn({13, 15}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  auto outputs = fe.runFusion({input1, input2});

  auto ref = (input1).unsqueeze(-1) + input2;

  testValidate(&fusion, outputs, {input1, input2}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmenterCombineReductionsCycleRepro_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto t0 = makeSymbolicTensor(3, DataType::Float);
  auto t1 = makeSymbolicTensor(3, DataType::Half);
  auto t3 = makeSymbolicTensor(3, DataType::Half);
  auto t5 = makeSymbolicTensor(3, DataType::Half);
  auto t7 = makeSymbolicTensor(1, DataType::Half);
  auto t11 = makeSymbolicTensor(3, DataType::Half);
  auto t13 = makeSymbolicTensor(3, DataType::Half);
  auto t15 = makeSymbolicTensor(3, DataType::Half);
  auto t17 = makeSymbolicTensor(3, DataType::Half);
  auto d56 = IrBuilder::create<Double>();

  fusion.addInput(t0);
  fusion.addInput(t1);
  fusion.addInput(t3);
  fusion.addInput(t5);
  fusion.addInput(t7);
  fusion.addInput(t11);
  fusion.addInput(t13);
  fusion.addInput(t15);
  fusion.addInput(t17);
  fusion.addInput(d56);

  auto t2 = castOp(DataType::Float, t1);
  auto t4 = castOp(DataType::Float, t3);
  auto t22 = sub(t2, t4);
  auto t6 = castOp(DataType::Float, t5);
  auto t23 = mul(t22, t6);
  auto t16 = castOp(DataType::Float, t15);
  auto t18 = castOp(DataType::Float, t17);
  auto t19 = add(t16, t18);
  auto t14 = castOp(DataType::Float, t13);
  auto t20 = add(t19, t14);
  auto t12 = castOp(DataType::Float, t11);
  auto t21 = add(t20, t12);
  auto t8 = castOp(DataType::Float, t7);
  auto t24 = broadcast(t8, {true, true, false});
  auto t25 = mul(t21, t24);
  auto t27 = sum(t25, {2});
  auto t28 = broadcast(t27, {false, false, true});
  auto t29 = mul(t25, t23);
  auto t30 = sum(t29, {2});
  auto t31 = broadcast(t30, {false, false, true});
  auto d59 =
      mul(t1->getRootDomain()[2]->extent(), IrBuilder::create<Double>(1));
  auto t26 = mul(d59, t25);
  auto txx = mul(t26, IrBuilder::create<Double>(1));
  auto t33 = sub(txx, t28);
  auto d70 = unaryOp(UnaryOpType::Reciprocal, d59);
  auto t35 = mul(d70, t6);
  auto t39 = sum(t21, {0, 1});
  auto t47 = castOp(DataType::Half, t39);
  auto t37 = mul(t21, t23);
  auto t38 = sum(t37, {0, 1});
  auto t46 = castOp(DataType::Half, t38);
  auto t32 = mul(t23, t31);
  auto t34 = sub(t33, t32);
  auto t36 = mul(t35, t34);
  auto t45 = castOp(DataType::Half, t36);
  auto t40 = mul(t36, t0);
  auto t41 = mul(t40, d56);
  auto t44 = castOp(DataType::Half, t41);
  auto t42 = sum(t41, {0, 1});
  auto t43 = castOp(DataType::Half, t42);

  fusion.addOutput(t43);
  fusion.addOutput(t44);
  fusion.addOutput(t45);
  fusion.addOutput(t46);
  fusion.addOutput(t47);

  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::randn({128, 64, 1024}, options_float);
  at::Tensor at_t1 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t3 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t5 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t7 = at::randn({1024}, options_half);
  at::Tensor at_t11 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t13 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t15 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t17 = at::randn({128, 64, 1024}, options_half);
  double at_d56 = 1.1111;

  std::vector<at::Tensor> aten_inputs = {
      at_t0, at_t1, at_t3, at_t5, at_t7, at_t11, at_t13, at_t15, at_t17};

  c10::IValue val = at_d56;

  KernelArgumentHolder args(KernelIndexMode::INT32);
  args.setDeviceIndex(0);
  args.push(aten_inputs);
  args.push(val);

  for (auto _ : c10::irange(5)) {
    auto segmented_fusion =
        SegmentCandidateFinder::segment(fusion_ptr.get(), args);
  }
}

TEST_F(NVFuserTest, FusionSerialAndParallelIndexing_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Double>(1));
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  fusion.addOutput(tv4);

  auto tv5 = add(tv0, IrBuilder::create<Double>(1));
  auto tv6 = add(tv5, IrBuilder::create<Double>(1));
  fusion.addOutput(tv6);

  // Case 1: local memory tensor computed serially and used by
  // parallel threads
  tv2->split(-1, 4);
  tv1->computeAt(tv2, -2);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // Case 2: shared memory tensor computed serially and used by BID
  tv4->split(-1, 4);
  tv3->computeAt(tv4, -2);
  tv4->axis(-1)->parallelize(ParallelType::BIDx);
  tv3->setMemoryType(MemoryType::Shared);

  // Case 3: shared memory tensor computed by TID and used by BID
  tv6->split(-1, 4);
  tv5->computeAt(tv6, -2);
  tv6->axis(-1)->parallelize(ParallelType::BIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->setMemoryType(MemoryType::Shared);

  const int nx = 11;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nx}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = t0 + 2;

  testValidate(
      &fusion, outputs, aten_inputs, {ref, ref, ref}, __LINE__, __FILE__);
}

// Repro of issue #1105
TEST_F(NVFuserTest, FusionWARSyncAliasedSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv3->split(0, 4);
  tv0->computeAt(tv3, 1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);

  // Make sure a WAR sync is inserted at the end of the outer loop
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.kernel()->topLevelExprs()) {
    if (auto loop = dynamic_cast<kir::ForLoop*>(kir_node)) {
      const auto& body = loop->body().exprs();
      TORCH_CHECK(!body.empty());
      auto last_expr = dynamic_cast<kir::BlockSync*>(body.back());
      TORCH_CHECK(last_expr != nullptr, "Invalid expr found");
      TORCH_CHECK(last_expr->isWarHazardSync(), "Not a sync for WAR hazard");
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 3;

  testValidate(&fusion, outputs, aten_inputs, {ref1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1099_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(1);
  fusion.addInput(tv3);

  // Just to make TIDx/y/z non-exact
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  auto tv6 = add(tv5, IrBuilder::create<Double>(1));
  fusion.addOutput(tv6);

  tv2->split(0, 4);
  tv0->computeAt(tv2, 1);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDz);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  tv1->setMemoryType(MemoryType::Shared);

  tv4->split(0, 5);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->setMemoryType(MemoryType::Shared);
  tv5->split(0, 6);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->setMemoryType(MemoryType::Shared);
  tv6->split(0, 7);
  tv6->axis(-1)->parallelize(ParallelType::TIDz);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t3 = at::randn({19}, options);
  std::vector<IValue> aten_inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref_t2 = t0 + 2;
  auto ref_t3 = t3 + 3;

  testValidate(
      &fusion, outputs, aten_inputs, {ref_t2, ref_t3}, __LINE__, __FILE__);
}

// Repro of issue #1080
TEST_F(NVFuserTest, FusionUnswitchPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv0->computeAt(tv2, 2);

  tv2->split(-1, 8);
  tv1->split(-1, 8);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);

  // swap TIDx and TIDy
  tv1->axis(-1)->parallelize(ParallelType::TIDy);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  const int nx = 4;
  const int ny = 10;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nx, ny}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = t0 + 2;

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1189_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({16, 16});
  auto tv1 = makeConcreteTensor({16, 16});

  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, false, true});

  fusion.addInput(tv0b);
  fusion.addInput(tv1b);

  auto tv2 = add(tv0b, tv1b);
  auto tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  auto parallelize = [](auto tv) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
    tv->axis(1)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
  };

  parallelize(tv0b);
  parallelize(tv1b);
  parallelize(tv2);
  parallelize(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16, 1}, options);
  at::Tensor t1 = at::randn({16, 16, 1}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto outputs = fe.runFusion({t0, t1});

  auto ref = (t0 + t1).sum({1});

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1052_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  auto tv3 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv2, {tv0});
  scheduler_utils::parallelizeAllLike(tv3, {tv1});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);
  at::Tensor t1 = at::randn({100}, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref_t2 = t0 + 1;
  auto ref_t3 = t1 + 1;

  testValidate(
      &fusion, outputs, aten_inputs, {ref_t2, ref_t3}, __LINE__, __FILE__);
}

// Repro of issue #1115
TEST_F(NVFuserTest, FusionPointwiseBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{3, 17, 80};
  std::vector<int64_t> output_shape{3, 17, 1, 80};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_bcast = broadcast(x_add_bias, {false, false, true, false});
  auto y = gelu(x_bcast);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);
  std::vector<IValue> aten_inputs = {at_x, at_bias};

  schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_view = at::native::view(at_x_add_bias, output_shape);
  auto aten_y = at::gelu(at_x_view);

  testValidate(&fusion, outputs, aten_inputs, {aten_y}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPointwiseVectorize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int size = 1024 * 64;

  TensorView* x = makeContigTensor(1);
  fusion.addInput(x);
  auto y = sin(x);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // PyTorch's CUDA caching allocator should always return aligned pointer for
  // freshly allocated tensor
  at::Tensor at_x = at::randn({size}, options);

  schedulePointwise(&fusion, {at_x});

  for (auto x_consumer : ir_utils::consumerTvsOf(x)) {
    bool found_vec_in_input = false;
    for (auto id : x_consumer->domain()->domain()) {
      if (isParallelTypeVectorize(id->getParallelType())) {
        found_vec_in_input = true;
        break;
      }
    }
    TORCH_CHECK(found_vec_in_input, "Expect input to be vectorized");
  }

  for (auto id : y->domain()->domain()) {
    if (isParallelTypeVectorize(id->getParallelType())) {
      return;
    }
  }
  TORCH_CHECK(false, "Expect output to be vectorized");
}

TEST_F(NVFuserTest, FusionSmemAliasSerial_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  // Just set the dimension of TIDx
  auto tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // tv1 and tv2 are on shared memory and are not parallelized with
  // TIDx. They should be predicated as they are redundant and can
  // interfere with smem aliasing (issue #1100).

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);
  at::Tensor t4 = at::randn({1024}, options);
  std::vector<IValue> aten_inputs = {t0, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 3;
  auto ref2 = t4 + 1;

  testValidate(&fusion, outputs, aten_inputs, {ref1, ref2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReductionWithNonExactParallelDimensions_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t2 = at::randn({19}, options);
  std::vector<IValue> aten_inputs = {t0, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 1;
  auto ref2 = sum(t2);

  testValidate(&fusion, outputs, aten_inputs, {ref1, ref2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridWelfordWithNonExactParallelDimensions_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = Welford(tv2, {0}).avg;
  fusion.addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t2 = at::randn({19}, options);
  std::vector<IValue> aten_inputs = {t0, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 1;
  auto ref2 = mean(t2, {0});

  testValidate(&fusion, outputs, aten_inputs, {ref1, ref2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReductionWithNonExactParallelDimensions2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0, 1});
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(3);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDz);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(2)->parallelize(ParallelType::BIDz);

  // TODO: This needs a fix for issue #1102.
  // Also, need to allow predicated grid reductions.
#if 0
  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 3}, options);
  at::Tensor t2 = at::randn({5, 6, 7}, options);
  at::Tensor t4 = at::randn({8, 9, 10}, options);
  std::vector<IValue> aten_inputs = {t0, t2, t4};
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0.sum(at::IntArrayRef{0, 1});
  auto ref2 = t2 + 1;
  auto ref3 = t4 + 1;

  testValidate(
      &fusion, outputs, aten_inputs, {ref1, ref2, ref3}, __LINE__, __FILE__);
#endif
}

TEST_F(NVFuserTest, FusionGridWelfordWithNonExactParallelDimensions2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0, 1});
  fusion.addOutput(tvs.avg);

  auto tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(3);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tvs.avg->axis(0)->parallelize(ParallelType::BIDx);
  tvs.avg->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDz);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(2)->parallelize(ParallelType::BIDz);

  // TODO: needs a fix for issue #1102
  // Also, need to allow predicated grid reductions.
#if 0
  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 3}, options);
  at::Tensor t2 = at::randn({5, 6, 7}, options);
  at::Tensor t4 = at::randn({8, 9, 10}, options);
  std::vector<IValue> aten_inputs = {t0, t2, t4};
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0.mean(at::IntArrayRef{0, 1});
  auto ref2 = t2 + 1;
  auto ref3 = t4 + 1;

  testValidate(
      &fusion, outputs, aten_inputs, {ref1, ref2, ref3}, __LINE__, __FILE__);
#endif
}

// Repro of issue #1102
TEST_F(NVFuserTest, FusionPredicateParallelizedDomains_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Just to make TIDx/y/z non-exact
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);

  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  auto tv6 = add(tv5, IrBuilder::create<Double>(1));
  auto tv7 = add(tv6, IrBuilder::create<Double>(1));
  auto tv8 = add(tv7, IrBuilder::create<Double>(1));
  auto tv9 = sum(tv8, {0});
  fusion.addOutput(tv9);

  tv1->split(0, 5);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->setMemoryType(MemoryType::Shared);
  tv2->split(0, 6);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);
  tv3->split(0, 7);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);

  tv9->split(0, 4);
  tv4->computeAt(tv9, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDz);
  tv7->axis(-1)->parallelize(ParallelType::TIDz);
  tv8->axis(-1)->parallelize(ParallelType::TIDz);
  tv9->axis(-1)->parallelize(ParallelType::TIDz);
  tv9->axis(0)->parallelize(ParallelType::BIDx);

  tv5->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t4 = at::randn({19}, options);
  std::vector<IValue> aten_inputs = {t0, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 3;
  auto ref2 = sum(t4 + 4);

  testValidate(&fusion, outputs, aten_inputs, {ref1, ref2}, __LINE__, __FILE__);
}

// Repro of #1102 and #1129
TEST_F(NVFuserTest, FusionSmemPredicateUnswitch_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  // Just to make TIDx/y/z non-exact
  auto tvx = add(tv1, IrBuilder::create<Double>(1));
  auto tvy = add(tvx, IrBuilder::create<Double>(1));
  auto tvz = add(tvy, IrBuilder::create<Double>(1));
  fusion.addOutput(tvz);

  tv5->split(0, 4);
  tv0->computeAt(tv5, 1);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->axis(0)->parallelize(ParallelType::Unswitch);

  tvx->split(0, 5);
  tvx->axis(-1)->parallelize(ParallelType::TIDx);
  tvy->split(0, 6);
  tvy->axis(-1)->parallelize(ParallelType::TIDy);
  tvz->split(0, 7);
  tvz->axis(-1)->parallelize(ParallelType::TIDz);

  for (auto tv : {tv2, tv3, tv4, tvx, tvy}) {
    tv->setMemoryType(MemoryType::Shared);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t1 = at::randn({19}, options);
  std::vector<IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref1 = t0 + 4;
  auto ref2 = t1 + 3;

  testValidate(&fusion, outputs, aten_inputs, {ref1, ref2}, __LINE__, __FILE__);
}

// Repro of issue #1136
TEST_F(NVFuserTest, FusionFloatPow_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Int>(4));
  // To check if pow(tv0, 2) is replaced with tv0 * tv0
  auto tv2 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Int>(2));
  // To check if pow(tv0, 2.0) is replaced with tv0 * tv0
  auto tv3 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Double>(2));
  auto tv4 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Int>(3));
  auto tv5 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Double>(3));
  auto s = binaryOp(
      BinaryOpType::Pow,
      IrBuilder::create<Double>(3),
      IrBuilder::create<Double>(3));
  auto tv6 = add(tv0, s);

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv1->split(0, 32);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv1, {tv2, tv3, tv4, tv5, tv6});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  // Negative inputs cause nan in Fuesr as use_fast_math is enabled
  t0 = abs(t0);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto p4 = at::pow(t0, 4);
  auto p2 = at::pow(t0, 2);
  auto p3 = at::pow(t0, 3);
  auto t6 = t0 + std::pow(3, 3);

  testValidate(
      &fusion,
      outputs,
      aten_inputs,
      {p4, p2, p2, p3, p3, t6},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1127_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int numel = 4;

  auto tv0 = makeConcreteTensor({numel});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});

  auto tv3 = makeConcreteTensor({numel, numel});
  fusion.addInput(tv3);

  auto tv4 = sum(tv3, {1});

  auto tv5 = add(tv2, tv4);
  fusion.addOutput(tv5);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // Lowering should fail since tv5 is predicated and paralellized with TIDx.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST_F(NVFuserTest, FusionChannelsLastParser_CUDA) {
  // This test may not pass if using a custom block sync as there may
  // be additional calls. Skip the test as it's not specifically
  // relevant with block synchronizatin.
  if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
    return;
  }
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
  graph(%0 : Half(8, 4, 10, 16, strides=[640, 1, 64, 4]),
        %1 : Half(8, 4, 10, 16, strides=[640, 160, 16, 1])):
    %o.1 : Half(8, 4, 10, 16, strides=[640, 1, 64, 4]) = aten::mul(%0, %1) # sum_dyn.py:5:6
    %3 : Half(8, 4, 10, 16, strides=[640, 1, 64, 4]) = aten::relu(%o.1) # sum_dyn.py:6:9
    return (%3))IR";
  parseIR(graph0_string, g.get());

  // strides are not yet supported in the irparser.
  {
    auto val = g->block()->inputs()[0];
    val->setType(val->type()->castRaw<TensorType>()->withSizesStrides(
        {8, 4, 10, 16}, {640, 1, 64, 4}));
  }

  {
    auto val = g->block()->inputs()[1];
    val->setType(val->type()->castRaw<TensorType>()->withSizesStrides(
        {8, 4, 10, 16}, {640, 160, 16, 1}));
  }

  for (auto node : g->block()->nodes()) {
    for (auto val : node->outputs()) {
      if (val->isCompleteTensor())
        val->setType(val->type()->castRaw<TensorType>()->withSizesStrides(
            {8, 4, 10, 16}, {640, 1, 64, 4}));
    }
  }

  auto fusion = parseJitIR(g);
  FusionGuard fg(fusion.get());
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor input0 =
      at::randn({2, 2, 2, 16}, options).clone(c10::MemoryFormat::ChannelsLast);
  at::Tensor input1 = at::randn({2, 2, 2, 16}, options);
  auto lparams = schedulePointwise(fusion.get(), {input0, input1});

  // CONSIDER:
  // 1. this can be moved to a dedicated "golden" file
  // 2. use a fuzzy compare (ignore non-significant whitespaces for example)
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<__half, 4> T0, Tensor<__half, 4> T2, Tensor<__half, 4> T7) {
  int64_t i171;
  i171 = (((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x);
  if ((i171 < (T0.size[0] * (T0.size[1] * (T0.size[2] * T0.size[3]))))) {
    __half T9[1];
    T9[0] = 0;
    T9[0]
       = T2[((((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) / (T0.size[1] * (T0.size[2] * T0.size[3]))) * ((T0.size[2] * T0.size[1]) * T0.size[3])) + ((((((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) % (T0.size[1] * (T0.size[2] * T0.size[3]))) % (T0.size[2] * T0.size[3])) % T0.size[3]) * (T0.size[2] * T0.size[1])) + (((((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) % (T0.size[1] * (T0.size[2] * T0.size[3]))) / (T0.size[2] * T0.size[3])) * T0.size[2]) + (((((((nvfuser_index_t)blockIdx.x) * 128) + ((nvfuser_index_t)threadIdx.x)) % (T0.size[1] * (T0.size[2] * T0.size[3]))) % (T0.size[2] * T0.size[3])) / T0.size[3])];
    __half T8[1];
    T8[0] = 0;
    T8[0]
       = T0[i171];
    float T3[1];
    T3[0]
       = __half2float(T9[0]);
    float T4[1];
    T4[0]
       = T3[0];
    float T1[1];
    T1[0]
       = __half2float(T8[0]);
    float T5[1];
    T5[0]
      = T1[0]
      * T4[0];
    float T6[1];
    T6[0]
       = relu(T5[0]);
    __half T10[1];
    T10[0]
       = __float2half(T6[0]);
    T7[i171]
       = T10[0];
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

  // TODO: runFusion hits assertion. I'm probably doing something wrong here.
  // FusionExecutor fe;
  // fe.compileFusion(fusion.get());
  // auto outputs = fe.runFusion({input0, input1}, lparams);
  // at::Tensor output_ref = (input0 * input1).relu();
  // TORCH_CHECK(output_ref.equal(outputs[0]));
}

TEST_F(NVFuserTest, FusionThreadPredicateUnswitch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10, 1024});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->computeAt(tv3, -1);
  tv3->axis(0)->parallelize(ParallelType::Unswitch);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 1024}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = sum(t0, {1}) + 2;

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionNonContigOutputs_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  fusion.addOutput(tv1);

  tv1->setContiguity(false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_input = at::randn({10}, options);
  at::Tensor at_output = at::empty_strided({10}, {2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_input});
  auto returned_outputs = fe.runFusion({at_input}, {at_output});

  // Returned outputs should only contain one tensor that is the same
  // as the output tensor given to runFusion
  TORCH_CHECK(returned_outputs.size() == 1);
  TORCH_CHECK(returned_outputs[0].is_same(at_output));
  TORCH_CHECK(!returned_outputs[0].is_contiguous());

  auto at_ref = at_input + 1;

  testValidate(&fusion, {at_output}, {at_input}, {at_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTestWarpSoftMax_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Setup softmax fusion
  auto input = makeContigTensor(2);
  fusion.addInput(input);
  auto output = softmax(input, 1);
  fusion.addOutput(output);

  // Setup runtime input
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({8, 16 * 197}, options);
  std::vector<c10::IValue> aten_inputs({aten_input});

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, aten_inputs, true);
  TORCH_CHECK(SchedulerEntry::canSchedule(
      ScheduleHeuristic::Persistent, &fusion, runtime_info));
  auto scheduler = SchedulerEntry::makeEntry(
      ScheduleHeuristic::Persistent, &fusion, runtime_info);
  scheduler->schedule(&fusion);

  // Modify the schedule to use warp reduction
  auto used_vals = fusion.usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (IterDomain* id : tv->domain()->domain()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        id->padToMultipleOfWarp();
      }
    }
  }

  // Test result
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);
  auto ref_output = at::_softmax(aten_input, 1, false);
  testValidate(&fusion, outputs, aten_inputs, {ref_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1133_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));

  fusion.addOutput(tv3);

  tv0->computeAt(tv3, 1);

  const int split_factor = 32;

  tv2->split(-1, split_factor);
  tv1->computeAt(tv2, -2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::Unswitch);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  // Both tv1 and tv2 should be allocated at the top-level scope
  GpuLower gpulw(&fusion);
  bool tv1_validated = false;
  bool tv2_validated = false;
  for (const auto& kir_node : gpulw.kernel()->topLevelExprs()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node)) {
      auto size = alloc->size();
      if (!(alloc->buffer()->name() == 1 || alloc->buffer()->name() == 2)) {
        // There should be no allocation other than those for tv1 and tv2
        TORCH_CHECK(false, "Invalid allocation detected");
      }
      TORCH_CHECK(size->isA<Int>(), "Invalid allocation size");
      TORCH_CHECK(size->as<Int>()->isConst(), "Allocation not constant");
      auto size_int = size->as<Int>()->value().value();
      if (alloc->buffer()->name() == 1) {
        TORCH_CHECK(
            size_int == split_factor,
            "Invalid allocation size: ",
            size->as<Int>()->value().value());
        tv1_validated = true;
      } else {
        TORCH_CHECK(
            size_int == 1,
            "Invalid allocation size: ",
            size->as<Int>()->value().value());
        tv2_validated = true;
      }
    }
  }

  TORCH_CHECK(tv1_validated, "Failed to validate tv1 allocation");
  TORCH_CHECK(tv2_validated, "Failed to validate tv2 allocation");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99, 101}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = (t0 + 1).sum({1}) + 1;

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorContigIDs_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  tv1->split(1, 32);

  auto tv2 = tv1->rFactor({1});

  // This merged domain is not contiguous.
  tv2->merge(0, 2);

  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99, 101}, options);
  std::vector<IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = t0.sum({1});

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPersistentBufferCalculation1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  TORCH_INTERNAL_ASSERT(buffers.size() == 1);
  TORCH_INTERNAL_ASSERT(resolution.size() == 1 && resolution[0].size() == 1);
  TORCH_INTERNAL_ASSERT(projectable.size() == 1);
  TORCH_INTERNAL_ASSERT(projectable_inputs.size() == 1);

  TORCH_INTERNAL_ASSERT(isTvWithinVec(buffers, tv1));
  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable, tv1));
  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  TORCH_INTERNAL_ASSERT(tv1_resolution_it != resolution.end())

  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0}, true);
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
}

TEST_F(NVFuserTest, FusionPersistentBufferCalculation2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = castOp(DataType::Half, tv5);
  fusion.addOutput(tv6);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  TORCH_INTERNAL_ASSERT(buffers.size() == 1);
  TORCH_INTERNAL_ASSERT(resolution.size() == 1 && resolution[0].size() == 1);
  TORCH_INTERNAL_ASSERT(projectable.size() == 1);
  TORCH_INTERNAL_ASSERT(projectable_inputs.size() == 1);

  TORCH_INTERNAL_ASSERT(isTvWithinVec(buffers, tv1));
  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable, tv1));
  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  TORCH_INTERNAL_ASSERT(tv1_resolution_it != resolution.end())

  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0}, true);
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Half)));
}

TEST_F(NVFuserTest, FusionPersistentBufferCalculation3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});

  auto tv5 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv5);

  auto tv6 = castOp(DataType::Float, tv5);

  auto tv7 = add(tv6, tv4);
  auto tv8 = set(tv1);
  auto tv9 = add(tv7, tv8);
  auto tv10 = sum(tv9, {1});
  auto tv11 = broadcast(tv10, {false, true});
  auto tv12 = set(tv7);
  auto tv13 = add(tv12, tv11);

  fusion.addOutput(tv13);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  TORCH_INTERNAL_ASSERT(buffers.size() == 2);
  TORCH_INTERNAL_ASSERT(
      resolution.size() == 2 && resolution[0].size() == 1 &&
      resolution[1].size() == 1);
  TORCH_INTERNAL_ASSERT(projectable.size() == 1);
  TORCH_INTERNAL_ASSERT(projectable_inputs.size() == 1);

  TORCH_INTERNAL_ASSERT(
      isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv7));
  TORCH_INTERNAL_ASSERT(
      isTvWithinVec(projectable, tv1) && !isTvWithinVec(projectable, tv7));

  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  TORCH_INTERNAL_ASSERT(tv1_resolution_it != resolution.end())
  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv1_resolution_it, tv9));

  auto tv7_resolution_it = tvEntryInVecVec(resolution, buffers, tv7);
  TORCH_INTERNAL_ASSERT(tv7_resolution_it != resolution.end())
  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv7_resolution_it, tv13));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);
  at::Tensor aten_t5 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0, aten_t5}, true);
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(
          aten_t0.size(1) * dataTypeSize(DataType::Float) * 2));
  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(
          aten_t0.size(1) *
          (dataTypeSize(DataType::Half) + dataTypeSize(DataType::Float))));
}

TEST_F(NVFuserTest, FusionPersistentBufferCalculation4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  TORCH_INTERNAL_ASSERT(buffers.size() == 2);
  TORCH_INTERNAL_ASSERT(
      resolution.size() == 2 && resolution[0].size() == 1 &&
      resolution[1].size() == 1);

  TORCH_INTERNAL_ASSERT(projectable.size() == 2);
  TORCH_INTERNAL_ASSERT(projectable_inputs.size() == 1);

  TORCH_INTERNAL_ASSERT(
      isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv2));
  TORCH_INTERNAL_ASSERT(
      isTvWithinVec(projectable, tv1) && isTvWithinVec(projectable, tv2));

  TORCH_INTERNAL_ASSERT(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  TORCH_INTERNAL_ASSERT(tv1_resolution_it != resolution.end())
  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv1_resolution_it, tv6));

  auto tv2_resolution_it = tvEntryInVecVec(resolution, buffers, tv2);
  TORCH_INTERNAL_ASSERT(tv2_resolution_it != resolution.end())
  TORCH_INTERNAL_ASSERT(isTvWithinVec(*tv2_resolution_it, tv8));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0}, true);
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(
          aten_t0.size(1) * dataTypeSize(DataType::Float) * 2));

  TORCH_INTERNAL_ASSERT(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Half)));
}

TEST_F(NVFuserTest, FusionPersistentBufferProjection_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  reduction_scheduler_utils::projectPersistentBuffers(&fusion);

  auto tv5_producers = ir_utils::producerTvsOf(tv5);
  auto tv7_producers = ir_utils::producerTvsOf(tv7);

  // Projection should have broken these dependencies

  TORCH_INTERNAL_ASSERT(
      std::find(tv5_producers.begin(), tv5_producers.end(), tv1) ==
      tv5_producers.end());
  TORCH_INTERNAL_ASSERT(
      std::find(tv7_producers.begin(), tv7_producers.end(), tv2) ==
      tv7_producers.end());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({aten_t0});

  auto aten_t1 = aten_t0.to(c10::kDouble);
  auto aten_t3 = aten_t1.sum({1});
  auto aten_t4 = aten_t3.unsqueeze(1);
  auto aten_t7 = aten_t4.add(aten_t1).add(aten_t1);

  testValidate(&fusion, cg_outputs, {aten_t0}, {aten_t7}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1223_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0, 1});
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Double>(0));
  fusion.addOutput(tv3);

  tv2->split(0, 4);
  tv2->split(1, 1, false);
  tv2->split(-1, 4);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(-3)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);

  tv1->computeAt(tv2, -1);

  // Make TIDx and TIDy non-exact
  tv3->split(0, 32);
  tv3->split(-1, 32);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDy);

  // The second axis of both tv1 and tv2 are fully unswitched, so they
  // don't need to predicate the parallel type usage of TIDy, whereas
  // the first axis is only partially unswitched, i.e., part of its
  // split output domains is outside the unswitched axis, so the first
  // axis, which uses TIDx, needs to predicate the parallel
  // dimension. Previously, as reported in issue #1223, unswitched
  // expressions didn't predicate parallel dimensions. It should be
  // fixed by PR #1222.

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::ones({11, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0});
  auto cg_outputs = fe.runFusion({at_t0});

  auto at_t1 = (at_t0 + 1).sum();

  testValidate(
      &fusion, cg_outputs, {at_t0}, {at_t1, at_t0}, __LINE__, __FILE__);
}

// See #1247 and #1250
TEST_F(NVFuserTest, FusionRfactorPredication1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = min(tv1, {0});

  fusion.addOutput(tv2);

  // Make TIDx non-exact
  auto tv3 = makeContigTensor(1);
  fusion.addInput(tv3);

  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  fusion.addOutput(tv4);

  tv2->split(0, 4);
  auto tv5 = tv2->rFactor({1});

  tv0->computeAt(tv2, 1);

  tv2->axis(0)->parallelize(ParallelType::TIDx);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::randn({9}, options);
  at_t0 = at::abs(at_t0);
  at::Tensor at_t3 = at::randn({128}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t3});
  auto cg_outputs = fe.runFusion({at_t0, at_t3});

  auto at_t2 = (at_t0 + 1).min();
  auto at_t4 = at_t3 + 1;

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t3}, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorPredication2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = min(tv0, {0});
  fusion.addOutput(tv1);

  // Make TIDx non-exact
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv3);

  tv1->split(0, 4);
  auto tv4 = tv1->rFactor({0});

  tv1->split(0, 3);

  // tv0->computeAt(tv1, 3);
  tv4->reorder({{0, 1}});
  tv4->split(0, 3);
  tv4->setMemoryType(MemoryType::Shared);

  // tv0: [I]
  // tv4: [4/3, 3, I/4]
  // tv1: [4/3, 3]

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1, {tv4});

  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor at_t0 = at::randn({9}, options);
  at_t0 = at::abs(at_t0);
  at::Tensor at_t3 = at::randn({128}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t3});
  auto cg_outputs = fe.runFusion({at_t0, at_t3});

  auto at_t2 = std::get<0>(at_t0.min(0));
  auto at_t4 = at_t3 + 1;

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t3}, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorIndirectRoot_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1692
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  fusion.addOutput(tv1);

  tv1->split(2, 4);
  tv1->split(1, 3);
  tv1->merge(2, 3);
  auto rf = tv1->rFactor({-1});

  tv1->split(0, 256);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  rf->computeAt(tv1, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto at_in = at::randn({6, 6, 6}, options);
  auto at_out = at_in.sum({1, 2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_in});
  auto cg_outputs = fe.runFusion({at_in});

  testValidate(&fusion, cg_outputs, {at_in}, {at_out}, __LINE__, __FILE__);
}

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
  TORCH_CHECK(!gpulw.concretizedBroadcastDomains().isConcretized(
      loweredTv(tv4, gpulw)->axis(1)));
  TORCH_CHECK(gpulw.concretizedBroadcastDomains().isConcretized(
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

  ConcretizedBroadcastDomains bcast_concretization_info;
  bcast_concretization_info.build(&fusion);

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
              pred->value()->toInlineString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 0);
          // hoisted_index * T0.stride[1]
          auto t0_index = in0->index(1);
          TORCH_CHECK(
              is_index_times_ns(t0_index, hoisted_index, "T0.stride[1]"),
              "Invalid index: ",
              t0_index->toInlineString());
        } else if (out_ti->view()->name() == 2) {
          // Ref: T3[*, hoisted_index] = T2[*, hoisted_index];
          auto out_index = out_ti->index(1);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 1);
          auto in0_index = in0->index(1);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString());
        } else if (out_ti->view()->name() == 3) {
          // Ref: T3[hoisted_index] = T2[hoisted_index];
          auto out_index = out_ti->index(0);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 2);
          auto in0_index = in0->index(0);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString());
        } else if (out_ti->view()->name() == 4) {
          // Ref: T4[0] = T3[hoisted_index];
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 3);
          auto in0_index = in0->index(0);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString());
        } else if (out_ti->view()->name() == 5) {
          // Ref: T5[hoisted_index] = T4[0]
          auto out_index = out_ti->index(0);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString());
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

  // Similar pattern as FusionTrivialReductionForwarding2 but no
  // trivial reduciton at the root domain

  // Create a trivial reduction by splitting with a factor of 1
  tv1->split(1, 1, false);
  // Merging with a trivial reduction
  tv1->merge(0, 1);
  tv1->split(0, 5);

  tv2->split(0, 5);

  // While the merge of tv1 is done with a trivial reduciton, it's not
  // a root domain, so forwarding is not enabled. BestEffortReplay
  // should only map the first axis of each tensor.

  PairwiseRootDomainMap root_map(tv1, tv2);
  auto p2c = BestEffortReplay::replayCasP(tv2, tv1, 2, root_map).getReplay();
  TORCH_CHECK(p2c.size() == 1, "Expected only one mapping found");
  TORCH_CHECK(p2c.begin()->first == tv1->getRootDomain().at(0));
  TORCH_CHECK(p2c.begin()->second == tv2->getRootDomain().at(0));
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
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
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

  // Make tv2 swizzled and half-inlined (unsupported).
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
      tv2->nDims() == 2, "Unpected unsqueeze result: ", tv2->toString());
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
  for (int64_t i = 0; i < replay_dom.size(); i++) {
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

  // Check that the splits are replayed on tv1, even though tv2
  //  is part of the boundary.
  TORCH_INTERNAL_ASSERT(
      tv2->nDims() == 4, "Propagator didn't propagate to tv2");
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
  const auto mask_options =
      at::TensorOptions().dtype(at::kBool).device(at::kCUDA, 0);
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

TEST_F(NVFuserTest, FusionInlinePropagatorMismatchedDims1_CUDA) {
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

  InlinePropagator inline_propagator(tv5, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&inline_propagator);

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

TEST_F(NVFuserTest, FusionInlinePropagatorMismatchedDims2_CUDA) {
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

  InlinePropagator inline_propagator(tv5, -1, ComputeAtMode::BestEffort);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&inline_propagator);

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

TEST_F(NVFuserTest, FusionInlinePropagatorMismatchedDims3_CUDA) {
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

  InlinePropagator inline_propagator(tv8, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv8).traverse(&inline_propagator);

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

TEST_F(NVFuserTest, FusionInlinePropagatorMismatchedDims4_CUDA) {
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
  InlinePropagator inline_propagator(tv0, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&inline_propagator);

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

TEST_F(NVFuserTest, FusionInlinePropagatorBroadcast_CUDA) {
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

  InlinePropagator inline_propagator(tv0, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&inline_propagator);

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

TEST_F(NVFuserTest, FusionInlinePropagatorBroadcastTrivialReduction_CUDA) {
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

  InlinePropagator inline_propagator(tv6, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv6).traverse(&inline_propagator);

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

  InlinePropagator inline_propagator(tv3, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&inline_propagator);

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

  InlinePropagator propagator(tv2, -1, ComputeAtMode::MostInlined);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

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

  InlinePropagator ip(tv0, -1, ComputeAtMode::MostInlined);
  tree.traverse(&ip);

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

  InlinePropagator ip(tv0, -1, ComputeAtMode::MostInlined);
  tree.traverse(&ip);

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

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
