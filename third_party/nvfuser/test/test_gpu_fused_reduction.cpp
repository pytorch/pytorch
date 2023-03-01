#if defined(USE_CUDA)
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
#include <lower2device.h>
#include <mutator.h>
#include <register_interface.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

namespace {

class KernelExprVisitor : private kir::IrVisitor {
 public:
  static std::vector<Expr*> getAllExprs(const kir::Kernel* kernel) {
    KernelExprVisitor visitor(kernel);
    return visitor.all_exprs_;
  }

 private:
  KernelExprVisitor(const kir::Kernel* kernel) {
    handle(kernel->topLevelExprs());
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    all_exprs_.push_back(expr);
    kir::IrVisitor::handle(expr);
  }

 private:
  std::vector<Expr*> all_exprs_;
};

void validateNoParallelBroadcastExist(kir::Kernel* kernel) {
  for (auto expr : KernelExprVisitor::getAllExprs(kernel)) {
    BroadcastOp* bc = dynamic_cast<BroadcastOp*>(expr);
    if (bc == nullptr) {
      auto grid_bc = dynamic_cast<kir::GridBroadcast*>(expr);
      if (grid_bc != nullptr) {
        std::cerr << "Grid broadcast: " << grid_bc->toString();
        bc = grid_bc->broadcast_op();
      }
    }
    if (bc == nullptr) {
      continue;
    }
    TORCH_CHECK(
        kernel->summary().broadcast_parallel_types.at(bc).none(),
        "Parallel broadcast should not exist but was found: ",
        bc->toString());
  }
}

} // namespace

TEST_F(NVFuserTest, FusionGridAllreduce1_CUDA) {
  const int nx = 999;
  const int tidx = 128;
  const int bidx = 4;

  if (ceilDiv(nx, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, tidx);
  tv3->split(0, bidx);
  tv3->split(0, 1); // unswitch
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::BIDy);
  tv3->axis(2)->parallelize(ParallelType::BIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  // Just to make sure fused_reduction and work buffers are allocated
  // uniquely
  tv1->axis(1)->parallelize(ParallelType::Unswitch);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = sum(t0).unsqueeze(0) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridAllreduce2_CUDA) {
  const int nx = 99;
  const int tidx = 32;

  if (ceilDiv(nx, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, tidx);
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, {tv2});

  // Broadcast on TIDy instead of TIDx. This still uses the fused
  // reduction as it's broadcast on BIDx as well. Since TIDy is not
  // predicated, the broadcast becomes a set op.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = sum(t0).unsqueeze(0) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grid reduction with serial non-reduction axis. The global work
// buffer is double buffered.
TEST_F(NVFuserTest, FusionGridAllreduce3_CUDA) {
  const int nx = 100;
  const int ny = 5000;
  const int tidx = 128;

  if (ceilDiv(ny, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(1, tidx);
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  tv3->axis(1)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx, ny}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = sum(t0, {1}).unsqueeze(-1) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Indirect reduction and broadcast
TEST_F(NVFuserTest, FusionGridAllreduce4_CUDA) {
  const int nx = 999;
  const int tidx = 128;

  if (ceilDiv(nx, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true});
  auto tv4 = add(tv0, tv3);

  fusion.addOutput(tv4);

  tv4->split(0, tidx);
  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (sum(t0) + 1).unsqueeze(0) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Unused block dimension in the kernel
TEST_F(NVFuserTest, FusionGridAllreduce5_CUDA) {
  const int nx = 999;
  const int tidx = 128;
  const int iter = 2;
  const int bdimx = 9; // One more than required by the reduction
  const int bdimy = 3; // Want an unused dimension

  // Going to bump the bdimx count for this test, ignor
  if (bdimx * bdimy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Didn't setup this test with inlining for register usage, so just leave the
  // iter dimension concrete
  auto tv0 = makeConcreteTensor({iter, -1});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);

  fusion.addOutput(tv4);

  // Dummy op to mess with parallelization
  auto tv5 = makeSymbolicTensor(2);
  fusion.addInput(tv5);
  auto tv6 = set(tv5);
  fusion.addOutput(tv6);

  // Setup the reduction
  tv4->split(1, tidx);
  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->axis(1)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv6->axis(0)->parallelize(ParallelType::BIDy);
  tv6->axis(1)->parallelize(ParallelType::BIDx);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({iter, nx}, options);
  auto t5 = at::randn({bdimy, bdimx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t5});
  auto cg_outputs = fe.runFusion({t0, t5});

  auto ref = (sum(t0, {1}) + 1).unsqueeze(-1) + t0;

  testValidate(&fusion, cg_outputs, {t0, t5}, {ref, t5}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridAllreduce6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({99, 200});

  const int vec = 4;
  const int tidx = 32;
  const int tidy = 8;
  const int bdimx = ceilDiv(shape[1], vec * tidx);
  const int bdimy = ceilDiv(shape[0], tidy);

  if (bdimx * bdimy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  tv1->split(1, vec);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv1->axis(4)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.sum({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridAllreduceWelford1_CUDA) {
  const int nx = 999;
  const int tidx = 128;

  if (ceilDiv(nx, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv2 = broadcast(tvs.avg, {true});
  auto tv3 = broadcast(tvs.var_sum, {true});
  auto tv4 = add(tv0, tv2);
  auto tv5 = add(tv4, tv3);

  fusion.addOutput(tv5);

  tv5->split(0, tidx);
  TransformPropagator propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5);

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref =
      (t0.mean({0}).unsqueeze(0) + t0) + t0.var({0}, false).unsqueeze(0) * nx;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grid welford reduction with serial non-reduction axis. The global
// work buffer is double buffered.
TEST_F(NVFuserTest, FusionGridAllreduceWelford2_CUDA) {
  const int nx = 100;
  const int ny = 5000;
  const int tidx = 128;

  if (ceilDiv(ny, tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {1});
  auto tv2 = broadcast(tvs.avg, {false, true});
  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(1, tidx);
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  tv3->axis(1)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  // There must be no parallel broadcast
  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({nx, ny}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (sum(t0, {1}) / ny).unsqueeze(-1) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Persistent batchnorm. Uses the fused reduction for grid welford and
// broadcast.
TEST_F(NVFuserTest, FusionFusedReductionBatchnorm_CUDA) {
  const std::vector<int64_t> input_shape{256, 2048, 14, 14};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(tv2);
  auto tv3 = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(tv3);
  auto tv4 = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(tv4);

  auto d34 = IrBuilder::create<Double>(1);
  auto tv5 = castOp(DataType::Float, tv0);
  auto tv6 = castOp(DataType::Float, tv1);
  auto tv7 = castOp(DataType::Float, tv2);
  auto tvs = Welford(tv5, {0, 2, 3});
  auto tv8 = tvs.avg;
  auto tv9 = tvs.var_sum;
  auto tv10 = tvs.n;
  auto tv11 = mul(tv8, IrBuilder::create<Double>(0.1));
  auto tv12 = mul(tv3, d34);
  auto tv13 = add(tv12, tv11);
  auto d43 = IrBuilder::create<Double>(0.5);
  auto tv14 = mul(tv9, d43);
  auto tv15 = mul(tv14, IrBuilder::create<Double>(0.1));
  auto tv16 = mul(tv4, d34);
  auto tv17 = add(tv16, tv15);
  auto tv18 = broadcast(tv8, {true, false, true, true});
  auto tv19 = sub(tv5, tv18);
  auto tv20 = mul(tv9, d43);
  auto tv21 = add(tv20, IrBuilder::create<Double>(0.0001));
  auto tv22 = rsqrt(tv21);
  auto tv23 = broadcast(tv22, {true, false, true, true});
  auto tv24 = mul(tv19, tv23);
  auto tv25 = broadcast(tv6, {true, false, true, true});
  auto tv26 = mul(tv24, tv25);
  auto tv27 = broadcast(tv7, {true, false, true, true});
  auto tv28 = add(tv26, tv27);
  auto tv29 = castOp(DataType::Half, tv28);
  fusion.addOutput(tv13);
  fusion.addOutput(tv17);
  fusion.addOutput(tv29);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();
  auto tv2_cache = tv2->cacheAfter();
  auto tv3_cache = tv3->cacheAfter();
  auto tv4_cache = tv4->cacheAfter();

  auto tv13_cache = tv13->cacheBefore();
  auto tv17_cache = tv17->cacheBefore();
  auto tv29_cache = tv29->cacheBefore();

  tv0->split(1, NamedScalar::getParallelDim(ParallelType::BIDx), false);
  tv0->split(0, NamedScalar::getParallelDim(ParallelType::BIDy), false);
  tv0->split(1, 8, false);
  tv0->split(2, 8, false);
  tv0->merge(-2, -1);
  tv0->split(-1, 2);
  tv0->split(-2, 1, false);
  tv0->split(-2, 1, false);
  tv0->reorder(
      {{4, 0},
       {5, 1},
       {0, 2},
       {3, 3},
       {8, 4},
       {1, 5},
       {7, 6},
       {2, 7},
       {9, 8},
       {6, 9}});

  TransformPropagator propagator(tv0);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);

  ir_utils::rfactorHelper(tvs.avg, {-5, -4, -3, -2, -1});

  tv0->computeAt(tv29, 2);
  tv1->computeAt(tv29, 2);
  tv2->computeAt(tv29, 2);
  tv3->computeAt(tv13, 2);
  tv4->computeAt(tv17, 2);

  tv29->axis(0)->parallelize(ParallelType::BIDx);
  tv29->axis(2)->parallelize(ParallelType::BIDy);
  tv29->axis(3)->parallelize(ParallelType::TIDz);
  tv29->axis(4)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv29);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(input_shape, options_half);
  auto t1 = at::randn(input_shape[1], options_half);
  auto t2 = at::randn(input_shape[1], options_half);
  auto t3 = at::randn(input_shape[1], options);
  auto t4 = at::randn(input_shape[1], options);
  std::vector<IValue> aten_inputs = {t0, t1, t2, t3, t4};

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  FusionExecutor fe;
  LaunchParams launch_params(2, 2, -1, -1, -1, -1);
  fe.compileFusion(&fusion, aten_inputs, launch_params);
  auto cg_outputs = fe.runFusion(aten_inputs, launch_params);

  auto t5 = t0.to(at::kFloat);
  auto t6 = t1.to(at::kFloat);
  auto t7 = t2.to(at::kFloat);
  auto t8 = t5.mean({0, 2, 3});
  auto t9 = t5.var({0, 2, 3}, false) * input_shape[0] * input_shape[2] *
      input_shape[3];
  auto t11 = t8 * 0.1;
  auto t12 = t3 * 1;
  auto t13 = t12 + t11;
  auto t14 = t9 * 0.5;
  auto t15 = t14 * 0.1;
  auto t16 = t4 * 1;
  auto t17 = t16 + t15;
  auto t18 = t8.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
  auto t19 = t5 - t18;
  auto t20 = t9 * 0.5;
  auto t21 = t20 + 0.0001;
  auto t22 = rsqrt(t21);
  auto t23 = t22.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
  auto t24 = t19 * t23;
  auto t25 = t6.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
  auto t26 = t24 * t25;
  auto t27 = t7.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
  auto t28 = t26 + t27;
  auto t29 = t28.to(at::kHalf);

  testValidate(
      &fusion,
      cg_outputs,
      aten_inputs,
      {t13, t17, t29},
      __LINE__,
      __FILE__,
      "",
      launch_params);
}

// Simple grouped reduction
TEST_F(NVFuserTest, FusionGroupedReduction1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = sum(tv0, {1});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  groupReductions({tv1, tv2});

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  std::vector<int64_t> shape({99, 999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = t0.sum({1}) * 2;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grouping reductions with different ops
TEST_F(NVFuserTest, FusionGroupedReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});

  auto tv3 = add(tv0, IrBuilder::create<Double>(2));
  auto tv4 = max(tv3, {1});

  auto tv5 = add(tv2, tv4);
  fusion.addOutput(tv5);

  groupReductions({tv2, tv4});

  tv2->split(1, 128);
  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv4, -1, ComputeAtMode::MostInlined);

  // tv4 is automatically parallelized in the same way
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  std::vector<int64_t> shape({99, 999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum({1}) + std::get<0>((t0 + 2).max(1));

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grouped reduction with different types
TEST_F(NVFuserTest, FusionGroupedReduction3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});

  auto tv2 = castOp(DataType::Double, tv0);
  auto tv3 = sum(tv2, {1});
  auto tv4 = castOp(DataType::Float, tv3);

  auto tv5 = add(tv1, tv4);
  fusion.addOutput(tv5);

  groupReductions({tv1, tv3});
  tv1->split(1, 128);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->computeAt(tv5, -1, ComputeAtMode::MostInlined);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDx);

  std::vector<int64_t> shape({99, 999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = t0.sum({1}) + t0.to(c10::kDouble).sum({1}).to(c10::kFloat);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Testing validation
TEST_F(NVFuserTest, FusionGroupedReduction4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = sum(tv1, {1});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Invalid grouping as tv2 and tv3 are not guaranteed to have the
  // same shape
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(groupReductions({tv2, tv3}));
}

// Testing validation
TEST_F(NVFuserTest, FusionGroupedReduction5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = sum(tv0, {1});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv1->split(1, 128);
  tv2->split(1, 64);

  // Invalid grouping as tv1 and tv2 don't have the same
  // transformations
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(groupReductions({tv1, tv2}));
}

// Grouping 3 reductions
TEST_F(NVFuserTest, FusionGroupedReduction6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});

  auto tv3 = add(tv0, IrBuilder::create<Double>(2));
  auto tv4 = sum(tv3, {1});

  auto tv5 = add(tv0, IrBuilder::create<Double>(3));
  auto tv6 = sum(tv5, {1});

  auto tv7 = add(add(tv2, tv4), tv6);

  fusion.addOutput(tv7);

  groupReductions({tv2, tv4, tv6});

  // There's no runtime grid reduction function that can take more
  // than 2 inputs, yet.
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv2);

  std::vector<int64_t> shape({99, 999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum({1}) + (t0 + 2).sum({1}) + (t0 + 3).sum({1});

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGroupedReduction7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  // Invalid grouping as tv3 depends on tv1
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(groupReductions({tv1, tv4}));
}

// Grouping rfactor'ed reductions
TEST_F(NVFuserTest, FusionGroupedReductionRfactor1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = sum(tv0, {0});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  const size_t gdimx = 10;
  const size_t bdimx = 128;

  tv1->split(0, gdimx, false);
  tv1->split(1, bdimx);
  auto tv1_rf = tv1->rFactor({1});

  tv2->split(0, gdimx, false);
  tv2->split(1, bdimx);
  auto tv2_rf = tv2->rFactor({1});

  groupReductions({tv1_rf, tv2_rf});
  groupReductions({tv1, tv2});

  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(2)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1_rf);

  std::vector<int64_t> shape({12345});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = t0.sum({0}) * 2;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Rfactoring grouped reductions
TEST_F(NVFuserTest, FusionGroupedReductionRfactor2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = sum(tv0, {0});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  groupReductions({tv1, tv2});

  const size_t gdimx = 10;
  const size_t bdimx = 128;

  tv1->split(0, gdimx, false);
  tv1->split(1, bdimx);

  // This should rfactor tv2 as well
  auto rf_tvs = tv1->rFactor({1}, {tv1, tv2});
  auto tv1_rf = rf_tvs.at(0);

  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(2)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1_rf);

  std::vector<int64_t> shape({12345});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = t0.sum({0}) * 2;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Group reductions of tensors that have computeAt positions set
TEST_F(NVFuserTest, FusionGroupedReductionAfterComputeAt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1});
  auto tv3 = sum(tv1, {1});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  const size_t bdimx = 128;

  tv2->split(1, bdimx);
  auto tv2_rf = tv2->rFactor({1});
  tv2_rf->reorder({{1, 2}});

  tv3->split(1, bdimx);
  auto tv3_rf = tv3->rFactor({1});
  tv3_rf->reorder({{1, 2}});

  tv0->computeAt(tv4, -1, ComputeAtMode::MostInlined);

  groupReductions({tv2_rf, tv3_rf});
  groupReductions({tv2, tv3});

  tv2->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  std::vector<int64_t> shape({3, 1234});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum({1}) * 2;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGroupAllreduce1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = sum(tv0, {0});
  auto tv4 = broadcast(tv3, {true});
  auto tv5 = add(tv0, tv2);
  auto tv6 = add(tv5, tv4);
  fusion.addOutput(tv6);

  groupReductions({tv1, tv3});

  tv2->split(0, 128);
  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  std::vector<int64_t> shape({999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t3 = t0.sum({0}).unsqueeze(-1);
  auto ref = t0 + t3 + t3;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grid reductionso of different types
TEST_F(NVFuserTest, FusionGroupAllreduce2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});

  auto tv3 = castOp(DataType::Double, tv0);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = castOp(DataType::Float, tv5);

  auto tv7 = add(tv0, tv2);
  auto tv8 = add(tv7, tv6);
  fusion.addOutput(tv8);

  const int tidx = 512;
  groupReductions({tv1, tv4});
  tv1->split(1, tidx);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->computeAt(tv8, -1, ComputeAtMode::MostInlined);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  std::vector<int64_t> shape({10, 999});

  if (shape.at(0) * ceilDiv(shape.at(1), tidx) > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t2 = t0.sum({1}).unsqueeze(-1);
  auto t6 = t0.to(c10::kDouble).sum({1}).unsqueeze(-1).to(c10::kFloat);
  auto ref = t0 + t2 + t6;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Grouping 3 grid allreduces
TEST_F(NVFuserTest, FusionGroupAllreduce3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = div(tv0, tv2);
  auto tv4 = max(tv0, {0});
  auto tv5 = broadcast(tv4, {true});
  auto tv6 = div(tv0, tv5);
  auto tv7 = min(tv0, {0});
  auto tv8 = broadcast(tv7, {true});
  auto tv9 = sub(tv0, tv8);
  fusion.addOutput(tv3);
  fusion.addOutput(tv6);
  fusion.addOutput(tv9);

  groupReductions({tv1, tv4, tv7});

  tv1->split(0, 128);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  std::vector<int64_t> shape({999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t3 = t0 / t0.sum({0}).unsqueeze(0);
  auto t6 = t0 / std::get<0>(t0.max(0)).unsqueeze(0);
  auto t9 = t0 - std::get<0>(t0.min(0)).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {t3, t6, t9}, __LINE__, __FILE__);
}

// Grouping 8 grid allreduces
TEST_F(NVFuserTest, FusionGroupAllreduce4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int num_reductions = 8;

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv_sum = tv0;
  std::vector<TensorView*> reduction_tvs;

  for (int i = 0; i < num_reductions; ++i) {
    auto reduction = sum(add(tv0, IrBuilder::create<Double>(i)), {0});
    reduction_tvs.push_back(reduction);
    auto avg = div(reduction, tv0->axis(0)->extent());
    auto bc = broadcast(avg, {true});
    tv_sum = add(tv_sum, bc);
  }

  fusion.addOutput(tv_sum);

  groupReductions(reduction_tvs);

  auto reduction_tv = reduction_tvs.at(0);

  reduction_tv->split(0, 128);
  TransformPropagator propagator(reduction_tv);
  MaxRootDomainInfoSpanningTree(reduction_tv).traverse(&propagator);

  reduction_tv->axis(0)->parallelize(ParallelType::BIDx);
  reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(reduction_tv);

  std::vector<int64_t> shape({999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  at::Tensor ref = t0;
  for (int i = 0; i < num_reductions; ++i) {
    auto reduction = sum(add(t0, i), {0});
    auto avg = reduction / t0.sizes()[0];
    auto bc = avg.unsqueeze(0);
    ref = add(ref, bc);
  }

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Variation of FusionGroupAllreduce5_CUDA but with different
// types. Exercise grouped allreduces with different types.
TEST_F(NVFuserTest, FusionGroupAllreduce5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = div(tv0, tv2);

  auto tv4 = makeSymbolicTensor(1, DataType::Double);
  fusion.addInput(tv4);
  auto tv5 = sum(tv4, {0});
  auto tv6 = broadcast(tv5, {true});
  auto tv7 = div(tv4, tv6);

  auto tv8 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv8);
  auto tv9 = sum(tv8, {0});
  auto tv10 = broadcast(tv9, {true});
  auto tv11 = div(tv8, tv10);

  auto out = add(
      add(castOp(DataType::Double, tv3), tv7), castOp(DataType::Double, tv11));

  fusion.addOutput(out);

  groupReductions({tv1, tv5, tv9});

  tv1->split(0, 128);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  std::vector<int64_t> shape({999});

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_double =
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  auto options_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options_float);
  auto t4 = at::randn(shape, options_double);
  auto t8 = torch::randint(0, 1000, shape, options_long);
  std::vector<IValue> aten_inputs = {t0, t4, t8};

  std::vector<at::indexing::TensorIndex> indices({at::indexing::Slice(0, 10)});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t3 = t0 / t0.sum({0}).unsqueeze(0).to(at::kDouble);
  auto t7 = t4 / t4.sum({0}).unsqueeze(0);
  auto t11 = t8 / t8.sum({0}).unsqueeze(0).to(at::kDouble);
  auto ref = t3 + t7 + t11;

  testValidate(fe.kernel(), outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Persistent batchnorm backward with grouped allreduce
TEST_F(NVFuserTest, FusionPersistentBNBackwardAllreduce_CUDA) {
  const std::vector<int64_t> shape({64, 1024, 14, 14});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto input = makeContigTensor(4);
  fusion.addInput(input);
  auto grad_output = makeContigTensor(4);
  fusion.addInput(grad_output);
  auto weight = makeContigTensor(1);
  fusion.addInput(weight);
  auto save_mean = makeContigTensor(1);
  fusion.addInput(save_mean);
  auto save_invstd = makeContigTensor(1);
  fusion.addInput(save_invstd);

  const bool kTraining = true;
  const bool channels_last = false;

  const size_t kNumberOfDims =
      TensorDomain::noReductions(input->getMaybeRFactorDomain()).size();
  size_t c_axis = channels_last ? kNumberOfDims - 1 : 1;

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  Val* num_features = nullptr;
  for (const auto axis : c10::irange(kNumberOfDims)) {
    if (axis != c_axis) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      if (num_features == nullptr) {
        num_features =
            castOp(DataType::Double, input->domain()->domain()[axis]->extent());
      } else {
        num_features =
            mul(num_features, input->domain()->domain()[axis]->extent());
      }
    }
  }

  auto mean = save_mean;
  auto invstd = save_invstd;

  mean = broadcast(mean, broadcast_mask);

  auto norm = reciprocal(num_features);

  auto grad_output_sum = sum(grad_output, reduction_axes);
  auto dot_p = sum(mul(grad_output, sub(input, mean)), reduction_axes);

  auto grad_mean = broadcast(mul(grad_output_sum, norm), broadcast_mask);

  auto proj_scale =
      broadcast(mul(mul(dot_p, norm), mul(invstd, invstd)), broadcast_mask);

  TensorView* grad_scale = nullptr;

  if (weight == nullptr) {
    grad_scale =
        mul(broadcast(invstd, broadcast_mask),
            IrBuilder::create<Double>(input->container(), 1));
  } else {
    grad_scale = mul(
        broadcast(invstd, broadcast_mask), broadcast(weight, broadcast_mask));
  }

  TensorView* grad_input = nullptr;
  if (kTraining) {
    auto proj = mul(sub(input, mean), proj_scale);
    grad_input = mul(sub(sub(grad_output, proj), grad_mean), grad_scale);
  } else {
    grad_input = mul(grad_output, grad_scale);
  }

  fusion.addOutput(grad_input);

  // Scheduling strategy
  // 1. Cache inputs
  // 2. Group the reductions (automatically fused with broadcasts)
  // 3. Merge HW and vectorize with the outer parallelized by TIDx
  // 4. Split N by TIDy with the outer parallelized by BIDx and
  // inner by TIDy
  // 5. Split C by BIDy and let the outer be the serial outermost loop

  auto input_cache = input->cacheAfter();
  auto grad_output_cache = grad_output->cacheAfter();
  auto weight_cache = weight->cacheAfter();
  auto save_mean_cache = save_mean->cacheAfter();
  auto save_invstd_cache = save_invstd->cacheAfter();

  // Group the two reductions
  groupReductions({grad_output_sum, dot_p});

  // Transform grad_input to: [C/bidy, N/tidy, tidy, bidy, HW/vec_width,
  // vec_width]
  const int tidy = 8;
  const int bidy = 4;
  const int bidx = ceilDiv(shape[0], (int64_t)tidy);
  const int vec_width = 4;
  TORCH_CHECK(
      (shape[2] * shape[3]) % vec_width == 0,
      "Invalid vector width: ",
      vec_width);

  grad_input->merge(-2, -1);
  grad_input->split(-1, vec_width);

  grad_input->split(0, tidy);
  grad_input->split(2, bidy);
  TORCH_CHECK(
      grad_input->nDims() == 6,
      "Unexpected number of dimensions: ",
      grad_input->toString());

  grad_input->reorder({{2, 0}, {0, 1}, {1, 2}});

  grad_input->axis(1)->parallelize(ParallelType::BIDx);
  grad_input->axis(2)->parallelize(ParallelType::TIDy);
  grad_input->axis(3)->parallelize(ParallelType::BIDy);
  grad_input->axis(4)->parallelize(ParallelType::TIDx);

  TransformPropagator propagator(grad_input);
  MaxRootDomainInfoSpanningTree(grad_input).traverse(&propagator);

  auto rf_tensors = grad_output_sum->rFactor(
      {-1}, std::vector<TensorView*>({grad_output_sum, dot_p}));

  for (auto fusion_input :
       ir_utils::filterByType<TensorView>(fusion.inputs())) {
    fusion_input->computeAt(grad_input, 1);
  }

  // Parallelization
  scheduler_utils::parallelizeAllLike(grad_input);
  input_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  grad_output_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_input = at::randn(shape, options);
  auto at_grad_output = at::randn(shape, options);
  auto at_weight = at::randn({shape[c_axis]}, options);
  auto at_save_mean = at::randn({shape[c_axis]}, options);
  auto at_save_invstd = at::randn({shape[c_axis]}, options);
  std::vector<IValue> aten_inputs(
      {at_input, at_grad_output, at_weight, at_save_mean, at_save_invstd});

  GpuLower gpulw(&fusion);
  validateNoParallelBroadcastExist(gpulw.kernel());

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  if (bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test";
  }

  auto outputs = fe.runFusion(aten_inputs);

  std::vector<int64_t> at_reduction_axes;
  std::copy(
      reduction_axes.begin(),
      reduction_axes.end(),
      std::back_inserter(at_reduction_axes));

  // MSVC bug on lambda non-capture of const integral type
  // https://developercommunity.visualstudio.com/t/lambda-fails-to-implicitly-capture-constexpr-value/610504
  auto at_bcast = [=](const auto& tensor) {
    if (channels_last) {
      tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0);
    } else {
      return tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
    }
  };

  auto at_mean = at_save_mean;
  const auto& at_invstd = at_save_invstd;
  at_mean = at_bcast(at_mean);
  auto at_norm = 1.0f / static_cast<float>(shape[0] * shape[2] * shape[3]);

  auto at_grad_output_sum = sum(at_grad_output, at_reduction_axes);
  auto at_dot_p =
      sum(mul(at_grad_output, sub(at_input, at_mean)), at_reduction_axes);

  auto at_grad_mean = at_bcast(at_grad_output_sum * at_norm);

  auto at_proj_scale = at_bcast((at_dot_p * at_norm) * (at_invstd * at_invstd));

  at::Tensor at_grad_scale;

  if (weight == nullptr) {
    at_grad_scale = at_bcast(at_invstd);
  } else {
    at_grad_scale = at_bcast(at_invstd) * at_bcast(at_weight);
  }

  at::Tensor at_grad_input;
  if (kTraining) {
    auto at_proj = (at_input - at_mean) * at_proj_scale;
    at_grad_input = (at_grad_output - at_proj - at_grad_mean) * at_grad_scale;
  } else {
    at_grad_input = at_grad_output * at_grad_scale;
  }

  testValidate(
      fe.kernel(), outputs, aten_inputs, {at_grad_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGroupedReductionReEntrant1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});

  auto tv3 = add(tv0, IrBuilder::create<Double>(2));
  auto tv4 = sum(tv3, {0});

  auto tv5 = add(tv2, tv4);
  fusion.addOutput(tv5);

  groupReductions({tv2, tv4});

  auto tv0_cache = tv0->cacheAfter();

  const int vec = 2;
  const int tidx = 64;
  const int tidy = 8;

  tv2->split(1, vec);
  tv2->split(1, tidx);

  tv2->split(0, tidy);
  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  tv0->computeAt(tv4, -1, ComputeAtMode::MostInlined);

  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv2->axis(2)->parallelize(ParallelType::BIDx);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv2);

  std::vector<int64_t> shape({99, 999});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = (t0_double + 1).sum({0}) + (t0_double + 2).sum({0});

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Channels-last batch norm with vectorization. Relies on re-entrant
// GroupedGridReduction
TEST_F(NVFuserTest, FusionGroupedReductionChannelsLastBatchNormLike_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape({64, 14, 14, 32});

  auto tv0 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  std::vector<int> reduction_axes({0, 1, 2});
  std::vector<bool> broadcast_mask({true, true, true, false});

  auto tv3 = castOp(DataType::Float, tv0);
  auto tv4 = castOp(DataType::Float, tv1);

  auto tv5 = sum(tv3, reduction_axes);

  auto tv6 = broadcast(tv2, broadcast_mask);
  auto tv7 = sub(tv4, tv6);
  auto tv8 = mul(tv3, tv7);
  auto tv9 = sum(tv8, reduction_axes);

  auto tv10 = castOp(DataType::Half, tv5);
  auto tv11 = castOp(DataType::Half, tv9);

  fusion.addOutput(tv10);
  fusion.addOutput(tv11);

  groupReductions({tv5, tv9});

  // Applies the outer-reduction schedule
  const int64_t num_channels = shape.back();
  const int64_t vector = 2;
  TORCH_CHECK(num_channels % vector == 0);
  // Use at most 32 TIDx threads
  const int64_t tidx = std::min<int64_t>(32l, num_channels / vector);
  const auto bidx = ceilDiv(num_channels, tidx * vector);

  const int64_t tidy = 8;
  const auto bidy = ceilDiv(
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4, bidx);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  auto ref = tv5;

  // Move the reduction domains inner positions
  ref->reorder({{0, 1}, {1, 2}, {2, 3}, {3, 0}});

  // Parallelizing the reduction domains
  ref->merge(2, 3);
  ref->merge(1, 2);
  ref->split(1, tidy);
  ref->split(1, bidy, false);

  // Parallelizing the iteration domains
  ref->split(0, vector);
  ref->split(0, tidx);

  // Move the vector axis to the innermost position
  ref->reorder({{2, 5}, {3, 2}, {4, 3}, {5, 4}});
  // Move the serial reduction to the right of the vector axis
  ref->reorder({{3, 4}, {4, 3}});

  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  auto rf_tvs = tv5->rFactor({-2}, {tv5, tv9});
  auto tv5_rf = rf_tvs.at(0);
  auto tv9_rf = rf_tvs.at(1);

  tv0->computeAt(tv5_rf, -2, ComputeAtMode::BestEffort);
  tv1->computeAt(tv9_rf, -2, ComputeAtMode::BestEffort);
  tv3->computeAt(tv5_rf, -1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv9_rf, -1, ComputeAtMode::BestEffort);

  ref = tv5_rf;

  ref->axis(0)->parallelize(ParallelType::BIDx);
  ref->axis(1)->parallelize(ParallelType::TIDx);
  ref->axis(2)->parallelize(ParallelType::BIDy);
  ref->axis(3)->parallelize(ParallelType::TIDy);
  ref->axis(4)->parallelize(ParallelType::Serial);
  ref->axis(5)->parallelize(ParallelType::Serial);

  scheduler_utils::parallelizeAllLike(ref);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options_half);
  auto t1 = at::randn(shape, options_half);
  auto t2 = at::randn({shape.back()}, options_float);
  std::vector<IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t0_double = t0.to(at::kDouble);
  auto t1_double = t1.to(at::kDouble);
  auto t2_double = t2.to(at::kDouble);

  std::vector<int64_t> at_reduction_axes(
      {reduction_axes.begin(), reduction_axes.end()});
  auto t5 = t0_double.sum(at_reduction_axes);
  auto t8 = t0_double *
      (t1_double - t2_double.unsqueeze(0).unsqueeze(0).unsqueeze(0));
  auto t9 = t8.sum(at_reduction_axes);

  testValidate(fe.kernel(), outputs, aten_inputs, {t5, t9}, __LINE__, __FILE__);
}

// Test the grouped grid allreduce with BN-like outer reductions
TEST_F(
    NVFuserTest,
    FusionGroupedReductionPersistentChannelsLastBatchNormLike_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape({64, 14, 14, 32});

  auto tv0 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  std::vector<int> reduction_axes({0, 1, 2});
  std::vector<bool> broadcast_mask({true, true, true, false});

  auto tv3 = castOp(DataType::Float, tv0);
  auto tv4 = castOp(DataType::Float, tv1);

  auto tv5 = sum(tv3, reduction_axes);

  auto tv6 = broadcast(tv2, broadcast_mask);
  auto tv7 = sub(tv4, tv6);
  auto tv8 = mul(tv3, tv7);
  auto tv9 = sum(tv8, reduction_axes);

  auto tv10 = broadcast(tv5, broadcast_mask);
  auto tv11 = add(tv3, tv10);

  auto tv12 = broadcast(tv9, broadcast_mask);
  auto tv13 = add(tv4, tv12);

  auto tv14 = castOp(DataType::Half, tv11);
  auto tv15 = castOp(DataType::Half, tv13);

  fusion.addOutput(tv14);
  fusion.addOutput(tv15);

  groupReductions({tv5, tv9});

  // Applies the outer-reduction schedule
  const int64_t num_channels = shape.back();
  const int64_t vector = 2;
  TORCH_CHECK(num_channels % vector == 0);
  // Use at most 32 TIDx threads
  const int64_t tidx = std::min<int64_t>(32l, num_channels / vector);
  const auto bidx = ceilDiv(num_channels, tidx * vector);

  const int64_t tidy = 8;
  const int64_t reduction_work_per_thread = 8;

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  auto ref = tv5;

  // Move the reduction domains inner positions
  ref->reorder({{0, 1}, {1, 2}, {2, 3}, {3, 0}});

  // Parallelizing the reduction domains
  ref->merge(2, 3);
  ref->merge(1, 2);
  ref->split(1, tidy);
  ref->split(1, reduction_work_per_thread);

  // Parallelizing the iteration domains
  ref->split(0, vector);
  ref->split(0, tidx);

  // Move the vector axis to the innermost position
  ref->reorder({{2, 5}, {3, 2}, {4, 3}, {5, 4}});
  // Move the serial reduction to the right of the vector axis
  ref->reorder({{3, 4}, {4, 3}});

  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  auto rf_tvs = tv5->rFactor({-2}, {tv5, tv9});
  auto tv5_rf = rf_tvs.at(0);
  auto tv9_rf = rf_tvs.at(1);

  tv0->computeAt(tv5_rf, -2, ComputeAtMode::BestEffort);
  tv1->computeAt(tv9_rf, -2, ComputeAtMode::BestEffort);
  tv3->computeAt(tv5_rf, -1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv9_rf, -1, ComputeAtMode::BestEffort);

  ref = tv5_rf;

  ref->axis(0)->parallelize(ParallelType::BIDx);
  ref->axis(1)->parallelize(ParallelType::TIDx);
  ref->axis(2)->parallelize(ParallelType::BIDy);
  ref->axis(3)->parallelize(ParallelType::TIDy);
  ref->axis(4)->parallelize(ParallelType::Serial);
  ref->axis(5)->parallelize(ParallelType::Serial);

  scheduler_utils::parallelizeAllLike(ref);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  tv5->axis(-1)->parallelize(ParallelType::Group);

  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options_half);
  auto t1 = at::randn(shape, options_half);
  auto t2 = at::randn({shape.back()}, options_float);
  std::vector<IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t0_double = t0.to(at::kDouble);
  auto t1_double = t1.to(at::kDouble);
  auto t2_double = t2.to(at::kDouble);

  std::vector<int64_t> at_reduction_axes(
      {reduction_axes.begin(), reduction_axes.end()});
  auto t5 = t0_double.sum(at_reduction_axes);
  auto t8 = t0_double *
      (t1_double - t2_double.unsqueeze(0).unsqueeze(0).unsqueeze(0));
  auto t9 = t8.sum(at_reduction_axes);

  auto t10 = t5.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t11 = t0_double + t10;
  auto t12 = t9.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t13 = t1_double + t12;

  testValidate(
      fe.kernel(), outputs, aten_inputs, {t11, t13}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduce1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  const int vec = 2;
  const int tidx = 32;
  const int tidy = 8;

  tv1->split(1, vec);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv2->axis(4)->parallelize(ParallelType::Group);

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction =
        dynamic_cast<kir::GroupedGridReduction*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    auto out = ir_utils::getTvOutput(grouped_grid_reduction);
    for (auto out_axis : out->domain()->domain()) {
      auto out_axis_pt = out_axis->getParallelType();
      TORCH_CHECK(
          isParallelTypeThread(out_axis_pt) ||
              out_axis_pt == ParallelType::Group,
          "Invalid parallel type of the reduction tensor: ",
          out_axis_pt,
          ". Reduction output tensor: ",
          out->toString());
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridReduction found.");

  std::vector<int64_t> shape({99, 101});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.sum({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Test grouping of two domains
TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduce2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  const int vec1 = 2;
  const int vec2 = 3;
  const int tidx = 16;
  const int tidy = 8;

  tv1->split(1, vec1);
  tv1->split(1, vec2);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv2->axis(4)->parallelize(ParallelType::Group);
  tv2->axis(5)->parallelize(ParallelType::Group);

  std::vector<int64_t> shape({99, 129});

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction =
        dynamic_cast<kir::GroupedGridReduction*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    auto out = ir_utils::getTvOutput(grouped_grid_reduction);
    for (auto out_axis : out->domain()->domain()) {
      auto out_axis_pt = out_axis->getParallelType();
      TORCH_CHECK(
          isParallelTypeThread(out_axis_pt) ||
              out_axis_pt == ParallelType::Group,
          "Invalid parallel type of the reduction tensor: ",
          out_axis_pt,
          ". Reduction output tensor: ",
          out->toString());
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridReduction found.");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.sum({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Group both expressions and iterations
TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduce3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv1, tv3);

  auto tv5 = add(tv0, IrBuilder::create<Double>(2));
  auto tv6 = sum(tv5, {0});
  auto tv7 = broadcast(tv6, {true, false});
  auto tv8 = add(tv5, tv7);

  auto tv9 = add(tv4, tv8);
  fusion.addOutput(tv9);

  groupReductions({tv2, tv6});

  const int vec = 2;
  const int tidx = 32;
  const int tidy = 8;

  tv1->split(1, vec);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv2->axis(4)->parallelize(ParallelType::Group);

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction =
        dynamic_cast<kir::GroupedGridReduction*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    auto out = ir_utils::getTvOutput(grouped_grid_reduction);
    for (auto out_axis : out->domain()->domain()) {
      auto out_axis_pt = out_axis->getParallelType();
      TORCH_CHECK(
          isParallelTypeThread(out_axis_pt) ||
              out_axis_pt == ParallelType::Group,
          "Invalid parallel type of the reduction tensor: ",
          out_axis_pt,
          ". Reduction output tensor: ",
          out->toString());
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridReduction found.");

  std::vector<int64_t> shape({99, 101});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto t4 = t0_double + 1 + (t0_double + 1).sum({0}).unsqueeze(0);
  auto t8 = t0_double + 2 + (t0_double + 2).sum({0}).unsqueeze(0);
  auto ref = t4 + t8;

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// ParallelType::Group with computeAt
TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduce4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  const int vec = 2;
  const int tidx = 32;
  const int tidy = 8;

  tv2->reorder({{0, 1}});
  tv2->split(0, vec);
  tv2->split(0, tidx);
  tv2->split(-1, tidy);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(2)->parallelize(ParallelType::Group);

  // This should avoid inlining the grouped domain
  tv0->computeAt(tv4, -1, ComputeAtMode::MostInlined);

  TORCH_CHECK(
      tv1->getComputeAtPosition() == 2,
      "Invalid computeAt position: ",
      tv1->toString());
  TORCH_CHECK(
      tv2->getComputeAtPosition() == 2,
      "Invalid computeAt position: ",
      tv2->toString());

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    tv->axis(-2)->parallelize(ParallelType::BIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction =
        dynamic_cast<kir::GroupedGridReduction*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    auto out = ir_utils::getTvOutput(grouped_grid_reduction);
    for (auto out_axis : out->domain()->domain()) {
      auto out_axis_pt = out_axis->getParallelType();
      TORCH_CHECK(
          isParallelTypeThread(out_axis_pt) ||
              out_axis_pt == ParallelType::Group,
          "Invalid parallel type of the reduction tensor: ",
          out_axis_pt,
          ". Reduction output tensor: ",
          out->toString());
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridReduction found.");

  std::vector<int64_t> shape({99, 101});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.sum({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduceWelford1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = Welford(tv1, {0}).avg;
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  const int vec = 2;
  const int tidx = 32;
  const int tidy = 8;

  tv1->split(1, vec);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv2->axis(4)->parallelize(ParallelType::Group);

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction = dynamic_cast<kir::GroupedGridWelford*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridWelford found.");

  std::vector<int64_t> shape({99, 101});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.mean({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Test grouping of two domains
TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduceWelford2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = Welford(tv1, {0}).avg;
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  const int vec1 = 2;
  const int vec2 = 3;
  const int tidx = 16;
  const int tidy = 8;

  tv1->split(1, vec1);
  tv1->split(1, vec2);
  tv1->split(1, tidx);
  tv1->split(0, tidy);
  TransformPropagator propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::BIDx);
  tv1->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv1);

  tv2->axis(4)->parallelize(ParallelType::Group);
  tv2->axis(5)->parallelize(ParallelType::Group);

  std::vector<int64_t> shape({99, 129});

  // Make sure the reduction expr is converted to GroupedGridReduciton
  // and the non-reduction domains of the output TV are either
  // grouped or parallelized
  GpuLower gpulw(&fusion);
  bool validated = false;
  for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
    auto grouped_grid_reduction = dynamic_cast<kir::GroupedGridWelford*>(expr);
    if (grouped_grid_reduction == nullptr) {
      continue;
    }
    validated = true;
  }
  TORCH_CHECK(
      validated, "Invalid lowered kernel. No GroupedGridWelford found.");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  auto t0_double = t0.to(at::kDouble);
  auto ref = t0_double + t0_double.mean({0}).unsqueeze(0);

  testValidate(fe.kernel(), outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Follows the pattern of persistent outer grid welford in batchnorm
TEST_F(NVFuserTest, FusionCrossIterationGroupedGridAllreduceWelfordShmoo_CUDA) {
  struct Params {
    int N;
    int H;
    int W;
    int C;
    int tidx;
    int tidy;
    int vect;
    int persistent_buffer;
    int bidx;
  };

  auto test = [](const Params& params) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    std::vector<bool> bcast_pattern{true, true, true, false};
    std::vector<int> reduction_dims{2, 1, 0};

    auto tv0 = makeSymbolicTensor(4);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    auto tvs = Welford(tv1, reduction_dims);
    auto tv2 = tvs.avg;
    auto tv3 = tvs.var_sum;
    auto tv4 = tvs.n;
    auto tv5 = broadcast(tv2, bcast_pattern);
    auto tv6 = broadcast(tv3, bcast_pattern);
    auto tv7 = broadcast(tv4, bcast_pattern);
    auto tv8 = sub(tv1, tv5);
    auto tv9 = add(tv8, tv6);
    // auto tv10 = div(tv9, tv7);
    // fusion.addOutput(tv10);
    fusion.addOutput(tv9);

    // Schedule the fusion as it will be done by the persistent
    // scheduler

    auto input_cache = tv1;
    auto output_cache = tv9->cacheBefore();

    auto transform_ref = tv2;

    transform_ref->merge(0)->merge(0);

    int reduction_pos = 1;

    transform_ref->split(0, params.tidy);
    ++reduction_pos;
    transform_ref->axis(1)->parallelize(ParallelType::TIDy);

    // Persistent buffer
    transform_ref->split(0, params.persistent_buffer);
    ++reduction_pos;

    // Unswitch
    transform_ref->split(0, 1);
    ++reduction_pos;
    transform_ref->axis(1)->parallelize(ParallelType::Unswitch);

    transform_ref->axis(0)->parallelize(ParallelType::BIDy);

    transform_ref->split(reduction_pos, params.vect);
    transform_ref->axis(reduction_pos + 1)
        ->parallelize(ParallelType::Vectorize);

    transform_ref->split(reduction_pos, params.tidx);
    transform_ref->axis(reduction_pos + 1)->parallelize(ParallelType::TIDx);
    transform_ref->split(reduction_pos, params.bidx);
    transform_ref->axis(reduction_pos + 1)->parallelize(ParallelType::BIDx);

    auto transform_ref_rf =
        reduction_scheduler_utils::sortAndRFactor(transform_ref);

    TransformPropagator propagator(transform_ref_rf);
    MaxRootDomainInfoSpanningTree(transform_ref_rf).traverse(&propagator);

    int vec_id = std::distance(
        transform_ref_rf->domain()->domain().begin(),
        std::find_if(
            transform_ref_rf->domain()->domain().begin(),
            transform_ref_rf->domain()->domain().end(),
            [](auto id) {
              return id->getParallelType() == ParallelType::Vectorize;
            }));
    transform_ref_rf->axis(vec_id)->parallelize(ParallelType::Serial);

    int unswitch_id = std::distance(
        transform_ref_rf->domain()->domain().begin(),
        std::find_if(
            transform_ref_rf->domain()->domain().begin(),
            transform_ref_rf->domain()->domain().end(),
            [](auto id) {
              return id->getParallelType() == ParallelType::Unswitch;
            }));
    transform_ref_rf->axis(unswitch_id)->parallelize(ParallelType::Serial);

    scheduler_utils::parallelizeAllLike(
        transform_ref_rf, ir_utils::allTvs(&fusion));

    ParallelType vec_pt = ParallelType::Vectorize;
    tv1->axis(vec_id)->parallelize(vec_pt);
    tv9->axis(vec_id)->parallelize(vec_pt);

    transform_ref->axis(vec_id)->parallelize(ParallelType::Group);

    transform_ref_rf->axis(unswitch_id)->parallelize(ParallelType::Unswitch);

    inlineMost();

    // Make sure the reduction expr is converted to GroupedGridReduciton
    // and the non-reduction domains of the output TV are either
    // grouped or parallelized
    GpuLower gpulw(&fusion);
    bool validated = false;
    for (auto expr : KernelExprVisitor::getAllExprs(gpulw.kernel())) {
      auto grouped_grid_reduction =
          dynamic_cast<kir::GroupedGridWelford*>(expr);
      validated = true;
    }
    TORCH_CHECK(
        validated, "Invalid lowered kernel. No GroupedGridWelford found.");

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::manual_seed(0);

    const std::vector<int64_t> input_shape{
        params.N, params.H, params.W, params.C};
    auto t0 = at::randn(input_shape, options);

    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0});

    // Skip the rest of this test size if the required number of SMs
    // exceeds the available SM count
    const auto num_required_sms = params.bidx *
        ceilDiv(ceilDiv(params.N * params.H * params.W, params.tidy),
                params.persistent_buffer);
    if (num_required_sms > deviceSMCount()) {
      return;
    }

    auto cg_outputs = fe.runFusion({t0});

    auto t1 = t0.to(at::kDouble);
    auto t2 = t1.mean({0, 1, 2}).unsqueeze(0).unsqueeze(0).unsqueeze(0);
    auto t3 =
        at::var(t1, {0, 1, 2}, false).unsqueeze(0).unsqueeze(0).unsqueeze(0);
    auto t4 = params.N * params.H * params.W;
    auto ref = (t1 - t2 + (t3 * t4));

    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__, "");
  };

  std::vector<Params> base_params;
  base_params.push_back({256, 7, 7, 1, 8, 32, 2, 32, 4});
  base_params.push_back({256, 7, 7, 1, 16, 16, 4, 50, 4});
  base_params.push_back({128, 7, 7, 1, 16, 16, 4, 32, 4});
  base_params.push_back({128, 14, 14, 1, 16, 16, 4, 32, 1});
  base_params.push_back({128, 14, 14, 1, 16, 16, 2, 64, 2});
  base_params.push_back({128, 14, 14, 1, 8, 32, 4, 50, 4});
  base_params.push_back({128, 14, 14, 1, 8, 32, 2, 50, 4});

  std::vector<Params> param_vec;
  for (const auto base_p : base_params) {
    for (const auto c_dim : {512, 1024, 2048}) {
      auto tmp = base_p;
      tmp.C = c_dim;
      param_vec.push_back(tmp);
    }
  }

  for (const auto& params : param_vec) {
    test(params);
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
