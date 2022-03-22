#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
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
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

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

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

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

TEST_F(NVFuserTest, FusionReduceAndBroadcast1_CUDA) {
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
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, tidx);
  TransformPropagator::from(tv3);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

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

TEST_F(NVFuserTest, FusionReduceAndBroadcast2_CUDA) {
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
  TransformPropagator::from(tv3);

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
TEST_F(NVFuserTest, FusionReduceAndBroadcast3_CUDA) {
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
  TransformPropagator::from(tv3);

  tv0->computeAt(tv3, 1);

  tv3->axis(1)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

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
TEST_F(NVFuserTest, FusionReduceAndBroadcast4_CUDA) {
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
  TransformPropagator::from(tv4);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4, ir_utils::allTvs(&fusion));

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
TEST_F(NVFuserTest, FusionReduceAndBroadcast5_CUDA) {
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
  TransformPropagator::from(tv4);

  tv4->axis(1)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4, ir_utils::allTvs(&fusion));

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

TEST_F(NVFuserTest, FusionWelfordAndBroadcast1_CUDA) {
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
  TransformPropagator::from(tv5);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5, ir_utils::allTvs(&fusion));

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
TEST_F(NVFuserTest, FusionWelfordAndBroadcast2_CUDA) {
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
  TransformPropagator::from(tv3);

  tv0->computeAt(tv3, 1);

  tv3->axis(1)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

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

  auto tv0_cache = tv0->cache_after();
  auto tv1_cache = tv1->cache_after();
  auto tv2_cache = tv2->cache_after();
  auto tv3_cache = tv3->cache_after();
  auto tv4_cache = tv4->cache_after();

  auto tv13_cache = tv13->cache_before();
  auto tv17_cache = tv17->cache_before();
  auto tv29_cache = tv29->cache_before();

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

  TransformPropagator::from(tv0);

  auto tvs_rf = tvs.rFactor({-5, -4, -3, -2, -1});

  tv0->computeAt(tv29, 2);
  tv1->computeAt(tv29, 2);
  tv2->computeAt(tv29, 2);
  tv3->computeAt(tv13, 2);
  tv4->computeAt(tv17, 2);

  tv29->axis(0)->parallelize(ParallelType::BIDx);
  tv29->axis(2)->parallelize(ParallelType::BIDy);
  tv29->axis(3)->parallelize(ParallelType::TIDz);
  tv29->axis(4)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv29, ir_utils::allTvs(&fusion));

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

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
