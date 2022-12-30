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
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>
#include <lower_magic_zero.h>
#include <mutator.h>
#include <ops/all_ops.h>
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
#include <parser.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
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

// Check if a lowered TensorView that corresponds to a given fusion
// TensorView has the computeWith position at pos with consumers
// specified by target_tvs fusion TensorViews
void checkComputeWith(
    kir::Kernel* kernel,
    TensorView* fusion_tv,
    unsigned int pos,
    const std::vector<TensorView*>& target_tvs) {
  TensorView* kernel_tv = nullptr;
  for (auto val : kernel->usedMathVals()) {
    auto tv = dynamic_cast<TensorView*>(val);
    if (tv == nullptr) {
      continue;
    }
    if (tv->name() == fusion_tv->name()) {
      kernel_tv = tv;
      break;
    }
  }

  TORCH_CHECK(
      kernel_tv != nullptr,
      "No corresponding TensorView found in lowered kernel: ",
      fusion_tv->toString());

  TORCH_CHECK(
      kernel_tv->getComputeWithPosition() == pos,
      "Invalid computeWith positon: ",
      kernel_tv->toString(),
      ". Expected: ",
      pos);

  TORCH_CHECK(
      kernel_tv->getComputeWithConsumers().size() == target_tvs.size(),
      "Invalid number of computeWith consumers: ",
      kernel_tv->toString(),
      ". Expected: ",
      target_tvs.size());

  for (auto consumer : kernel_tv->getComputeWithConsumers()) {
    TORCH_CHECK(
        std::find_if(
            target_tvs.begin(),
            target_tvs.end(),
            [&](TensorView* fusion_target) {
              return fusion_target->name() == consumer->name();
            }) != target_tvs.end(),
        "Invalid computeWith consumer: ",
        consumer->toString(),
        ". Tensor: ",
        kernel_tv->toString());
  }
}

} // namespace

TEST_F(NVFuserTest, FusionComputeWith1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  // Set the global inlining only with the outer axis
  std::unordered_set<IterDomain*> uninlinable;
  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv->nDims() == 2) {
      uninlinable.insert(tv->axis(1));
    }
  }

  inlineMost(uninlinable);

  // Move the compute position of tv1 innermost without moving its
  // allocation point. At this point, it is not resolved yet which of
  // the consumers would be computed with
  tv1->computeWith(-1);

  // It is now illegal to modify the innermost ID of the consumers of
  // tv1.
  for (auto consumer_of_tv1 : ir_utils::consumerTvsOf(tv1)) {
    TORCH_CHECK(
        consumer_of_tv1->getMaybeMaxProducerPosition() == 2,
        "Invalid producer position: ",
        consumer_of_tv1->toString());
    EXPECT_THAT(
        [&]() { consumer_of_tv1->split(-1, 4); },
        ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
            "Cannot split axis within max producer position")));
  }

  // Lowering should resolve the computeWith
  GpuLower gpulw(&fusion);
  checkComputeWith(gpulw.kernel(), tv1, tv1->nDims(), {tv2});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.sum({1}).unsqueeze(-1) + t0;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// StoreAt with 1D softmax
TEST_F(NVFuserTest, FusionComputeWith2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int vec = 4;
  const int tidx = 128;
  const int dimx = 1000;

  auto input_tv0 = makeContigTensor(1);
  fusion.addInput(input_tv0);

  auto exp_tv1 = unaryOp(UnaryOpType::Exp, input_tv0);
  auto sum_exp_tv2 = sum(exp_tv1, {-1});
  auto bcast_sum_tv3 = broadcast(sum_exp_tv2, {true});

  auto exp_tv1_copy = unaryOp(UnaryOpType::Exp, input_tv0);

  auto output_tv4 = div(exp_tv1_copy, bcast_sum_tv3);

  fusion.addOutput(output_tv4);

  auto input_tv0_cache = input_tv0->cacheAfter();

  input_tv0->split(-1, vec);
  input_tv0->split(-2, tidx);
  MaxRootDomainInfoSpanningTree tree(input_tv0);
  TransformPropagatorWithCheck tp(input_tv0);
  tree.traverse(&tp);

  auto sum_exp_rf_tv5 = sum_exp_tv2->rFactor({-1});

  inlineMost();
  input_tv0_cache->computeWith(-1);

  input_tv0_cache->axis(0)->parallelize(ParallelType::BIDx);
  input_tv0_cache->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(input_tv0_cache);

  GpuLower gpulw(&fusion);
  // Lowering should automatcially pick the first consumer of the
  // computed-with tensor as its target
  checkComputeWith(
      gpulw.kernel(), input_tv0_cache, input_tv0_cache->nDims(), {exp_tv1});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dimx}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto aten_output = at::_softmax(t0.to(at::kDouble), -1, false);

  testValidate(&fusion, cg_outputs, {t0}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeWith3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {true, false});
  fusion.addOutput(tv2);

  tv2->split(-1, 4);
  tv2->split(-2, 3);
  MaxRootDomainInfoSpanningTree tree(tv2);
  TransformPropagatorWithCheck tp(tv2);
  tree.traverse(&tp);

  // tv1: [    i0//4//3, 3, 4]
  // tv2: [b1, i0//4//3, 3, 4]

  tv2->axis(-3)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  // ComputeAt inlining with no storeAt inlining
  tv1->computeWith(-2);

  GpuLower gpulw(&fusion);
  checkComputeWith(gpulw.kernel(), tv1, 2, {tv2});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.unsqueeze(0);

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Compute a tensor that has siblings with a consumer. All of the
// siblings are computed with the same consumer.
TEST_F(NVFuserTest, FusionComputeWith4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tvs = Welford(tv0, {1});
  auto tv2 = add(tvs.avg, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv2->split(0, 32);

  MaxRootDomainInfoSpanningTree tree(tv2);
  TransformPropagatorWithCheck tp(tv2);
  tree.traverse(&tp);

  tv0->computeAt(tv2, 2);

  // All of the Welford outputs should be configured to have the
  // same computeWith positon
  tvs.avg->computeWith(3);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  GpuLower gpulw(&fusion);
  for (auto welford_out_tv : {tvs.avg, tvs.var_sum, tvs.n}) {
    checkComputeWith(gpulw.kernel(), welford_out_tv, 3, {tv2});
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({345, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.mean({1}) + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Compute a tensor with a consumer that has siblings. The tensor is
// computed with all of the siblings.
TEST_F(NVFuserTest, FusionComputeWith5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tvs = Welford(tv1, {1});
  auto tv2 = add(tvs.avg, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);

  MaxRootDomainInfoSpanningTree tree(tv1);
  TransformPropagatorWithCheck tp(tv1);
  tree.traverse(&tp);

  tv1->computeWith(-1);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  GpuLower gpulw(&fusion);
  checkComputeWith(gpulw.kernel(), tv1, 3, {tvs.avg, tvs.var_sum, tvs.n});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({345, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).mean({1}) + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Testing inlining with a fusion with an outer persistent grid welford
TEST_F(NVFuserTest, FusionComputeWith6_CUDA) {
  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int> reduction_dims{2, 1, 0};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);

  // Manually project persistent buffer
  auto tv1 = set(tv0);
  // First use of persistent buffer, cast to float
  auto tv2 = castOp(DataType::Float, tv1);
  auto tv3 = Welford(tv2, reduction_dims).avg;
  auto tv4 = broadcast(tv3, bcast_pattern);
  // Second use of manually projected persistent buffer
  auto tv5 = castOp(DataType::Float, tv1);
  auto tv6 = sub(tv5, tv4);
  auto tv7 = castOp(DataType::Half, tv6);
  fusion.addOutput(tv7);

  // Set parallel dimensions to be not too small but not too
  // large. Also set input tensor dimensions accordingly.

  int tidx = 8;
  int bidx = 2;
  int vect = 2;
  int c_serial_ext = 8;
  int64_t C = tidx * vect * bidx * c_serial_ext;

  int tidy = 16;
  int bidy = 7;
  int persist_buffer = 16;
  int us_factor = 4;

  int64_t N = bidy * us_factor, H = tidy, W = persist_buffer;

  auto tv0_cache = tv1;
  auto tv7_cache = tv7->cacheBefore();

  tv3->merge(0)->merge(0);

  tv3->split(0, tidy);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->split(0, persist_buffer, false);
  tv3->split(0, us_factor);
  tv3->axis(2)->parallelize(ParallelType::BIDy);

  // R: [PB/US, US, BIDY, TIDY] -> [BIDY, TIY, PB/US, US]
  tv3->reorder({{0, 2}, {1, 3}, {2, 0}, {3, 1}});

  tv3->reorder({{-1, 0}});

  tv3->split(0, vect);
  tv3->split(0, tidx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv3->split(0, bidx);
  tv3->axis(1)->parallelize(ParallelType::BIDx);

  // [I/BIDX, BIDX, TIDX, VEC, BIDY, TIY, PB/US, US]

  tv3->reorder({{3, -1}});

  // [I/BIDX, BIDX, TIDX, BIDY, TIY, PB/US, US, VEC]

  auto tv3_rf = ir_utils::rfactorHelper(tv3, {-3, -2});

  TransformPropagator propagator(tv3_rf);
  MaxRootDomainInfoSpanningTree(tv3_rf).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3_rf, ir_utils::allTvs(&fusion));

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv7->axis(-1)->parallelize(ParallelType::Vectorize);

  tv3_rf->axis(-2)->parallelize(ParallelType::Unswitch);

  inlineMost();

  // Inline the initial load of input at the first use of the input
  // without inlining the allocation point. The inlining position
  // should be the innermost vectorized axis.
  tv1->computeWith(-1, true);

  GpuLower gpulw(&fusion);
  // The innermost ID is vectorized, so the computeWith position
  // should be -2. The compute-with tensor should be the first
  // consumer of tv1, i.e., tv2 not tv5.
  checkComputeWith(gpulw.kernel(), tv1, tv1->nDims() - 1, {tv2});

  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::manual_seed(0);

  const std::vector<int64_t> input_shape{N, H, W, C};
  auto t0 = at::randn(input_shape, options_half);

  CompileOptions co;
  co.index_mode = KernelIndexMode::INT32;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, LaunchParams());
  auto cg_outputs = fe.runFusion({t0});

  auto t1 = t0.to(at::kFloat);
  auto t2 = t1.mean({0, 1, 2});
  auto t3 = t2.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t4 = t1 - t3;

  testValidate(&fusion, cg_outputs, {t0}, {t4}, __LINE__, __FILE__, "");
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
