#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <inlining.h>
#include <ir_utils.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

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

// Shmoo testing of the optimized grouped grid welford
TEST_F(NVFuserTest, FusionGroupedGridWelfordOuterOpt_CUDA) {
  struct OuterReductionParams {
    int vec = 1;
    int tidx = 1;
    int tidy = 1;
    int bidx = 1;
    int pb = 8;
    bool should_use_opt = false;
    DataType dtype = DataType::Half;
  };

  auto run_test = [&](const OuterReductionParams& params) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(2, params.dtype);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 =
        params.dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
    auto tvs = Welford(tv2, {0});
    auto tv3 = tvs.avg;
    auto tv4 = broadcast(tv3, {true, false});
    auto tv5 =
        params.dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
    auto tv6 = sub(tv5, tv4);
    fusion.addOutput(tv6);

    int64_t bidy = deviceSMCount() / params.bidx;

    // Skip if the available SM count is too small for this problem size
    if (deviceSMCount() <= params.bidx || bidy <= 1) {
      return;
    }

    int64_t reduction_size = params.tidy * bidy * params.pb;
    int64_t iteration_size = params.vec * params.tidx * params.bidx * 8;

    auto ref = tv3;

    ref->reorder({{0, 1}});

    ref->split(1, params.tidy);
    ref->split(1, params.pb);

    ref->split(0, params.vec);
    ref->split(0, params.tidx);
    // ref->axis(1)->parallelize(ParallelType::TIDx);
    ref->split(0, params.bidx);
    // ref->axis(1)->parallelize(ParallelType::BIDx);

    // Move the vectorized ID to the innermost position
    ref->reorder({{3, -1}});

    auto ref_rf = ref->rFactor({-3}, {tvs.avg, tvs.var_sum, tvs.n}).at(0);

    TransformPropagator propagator(ref_rf);
    MaxRootDomainInfoSpanningTree(ref_rf).traverse(&propagator);

    ref_rf->axis(1)->parallelize(ParallelType::BIDx);
    ref_rf->axis(2)->parallelize(ParallelType::TIDx);
    ref_rf->axis(3)->parallelize(ParallelType::BIDy);
    ref_rf->axis(5)->parallelize(ParallelType::TIDy);

    scheduler_utils::parallelizeAllLike(ref_rf, ir_utils::allTvs(&fusion));

    tv1->axis(-1)->parallelize(ParallelType::Vectorize);
    tv3->axis(-1)->parallelize(ParallelType::Group);

    inlineMost();

    auto at_dtype = params.dtype == DataType::Half ? at::kHalf : at::kFloat;
    auto options = at::TensorOptions().dtype(at_dtype).device(at::kCUDA, 0);
    at::manual_seed(0);

    const std::vector<int64_t> input_shape{reduction_size, iteration_size};
    auto t0 = at::randn(input_shape, options);
    std::vector<IValue> aten_inputs = {t0};

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);

    TORCH_CHECK(
        fe.kernel()->summary().has_outer_grouped_grid_welford ==
            params.should_use_opt,
        (params.should_use_opt ? "Failed to use the optimized implementation"
                               : "Should not use the optimized implementation"),
        ": ",
        params.vec,
        ", ",
        params.tidx,
        ", ",
        params.tidy,
        ", ",
        params.bidx);

    auto cg_outputs = fe.runFusion(aten_inputs);

    auto t1 = t0;
    auto t2 = params.dtype == DataType::Half ? t1.to(at::kFloat) : t1;
    auto t3 = t2.mean({0});
    auto t4 = t3.unsqueeze(0);
    auto t5 = params.dtype == DataType::Half ? t1.to(at::kFloat) : t1;
    auto t6 = t5 - t4;

    testValidate(
        &fusion, cg_outputs, aten_inputs, {t6}, __LINE__, __FILE__, "");
  };

  std::vector<OuterReductionParams> test_params;

  for (const auto& dtype : {DataType::Half, DataType::Float}) {
    for (int bidx = 1; bidx < 8; bidx *= 2) {
      if (dtype == DataType::Half) {
        test_params.push_back(
            {.vec = 8,
             .tidx = 16,
             .tidy = 16,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
        test_params.push_back(
            {.vec = 8,
             .tidx = 8,
             .tidy = 32,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
        test_params.push_back(
            {.vec = 8,
             .tidx = 4,
             .tidy = 64,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
      }
      test_params.push_back(
          {.vec = 4,
           .tidx = 16,
           .tidy = 16,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      test_params.push_back(
          {.vec = 4,
           .tidx = 8,
           .tidy = 32,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 4,
           .tidx = 4,
           .tidy = 64,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
      test_params.push_back(
          {.vec = 2,
           .tidx = 16,
           .tidy = 16,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 2,
           .tidx = 8,
           .tidy = 32,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 2,
           .tidx = 4,
           .tidy = 64,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
    }
  }

  for (const auto& params : test_params) {
    run_test(params);
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
