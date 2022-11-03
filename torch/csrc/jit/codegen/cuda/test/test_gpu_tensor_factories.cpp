#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionStandaloneARange_CUDA) {
  auto starts_ends = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto dtypes = {kFloat, kLong, kDouble};

  for (auto dtype : dtypes) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* start_int = IrBuilder::create<Int>();
    Val* end_int = IrBuilder::create<Int>();
    Val* step_int = IrBuilder::create<Int>();
    Val* start_double = IrBuilder::create<Double>();
    Val* end_double = IrBuilder::create<Double>();
    Val* step_double = IrBuilder::create<Double>();
    fusion->addInput(start_int);
    fusion->addInput(end_int);
    fusion->addInput(step_int);
    fusion->addInput(start_double);
    fusion->addInput(end_double);
    fusion->addInput(step_double);
    auto tv0 = arange(start_int, end_int, step_int, aten_to_data_type(dtype));
    auto tv1 =
        arange(start_double, end_double, step_double, aten_to_data_type(dtype));
    auto tv2 =
        arange(start_int, end_double, step_double, aten_to_data_type(dtype));
    auto tv3 =
        arange(start_double, end_double, step_int, aten_to_data_type(dtype));
    fusion->addOutput(tv0);
    fusion->addOutput(tv1);
    fusion->addOutput(tv2);
    fusion->addOutput(tv3);

    FusionExecutorCache executor_cache(std::move(fusion));

    const auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);

    for (auto start : starts_ends) {
      for (auto end : starts_ends) {
        for (auto step : steps) {
          if (std::signbit(end - start) != std::signbit(step)) {
            continue;
          }

          at::Tensor a =
              at::arange((int64_t)start, (int64_t)end, (int64_t)step, options);
          at::Tensor b =
              at::arange((double)start, (double)end, (double)step, options);
          at::Tensor c =
              at::arange((int64_t)start, (double)end, (double)step, options);
          at::Tensor d =
              at::arange((double)start, (double)end, (int64_t)step, options);

          auto cg_outputs = executor_cache.runFusionWithInputs(
              {(int64_t)start,
               (int64_t)end,
               (int64_t)step,
               (double)start,
               (double)end,
               (double)step});

          testValidate(
              executor_cache.fusion(),
              cg_outputs,
              {(int64_t)start,
               (int64_t)end,
               (int64_t)step,
               (double)start,
               (double)end,
               (double)step},
              {a, b, c, d},
              __LINE__,
              __FILE__);
        }
      }
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
