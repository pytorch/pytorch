#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <executor.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_iostream.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionStandaloneFull_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      kBool,
      kFloat,
      kLong,
      kDouble,
      kHalf,
      kBFloat16,
      kInt,
      kComplexFloat,
      kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Int>();
  Val* fill_val1 = IrBuilder::create<Int>();
  Val* fill_val2 = IrBuilder::create<Int>();
  Val* fill_val3 = IrBuilder::create<Int>();
  fusion->addInput(size);
  fusion->addInput(fill_val1);
  fusion->addInput(fill_val2);
  fusion->addInput(fill_val3);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = full({size}, fill_val1, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = full({size, size}, fill_val2, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = full_like(out_tv, fill_val3);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    std::vector<at::Tensor> expect;
    expect.reserve(dtypes.size());
    for (auto dtype : dtypes) {
      if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
        continue;
      }
      const auto options =
          at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      expect.emplace_back(at::full({size}, 11, options));
      expect.emplace_back(at::full({size, size}, 12, options));
      expect.emplace_back(at::full({size, size}, 13, options));
    }
    auto cg_outputs = executor_cache.runFusionWithInputs({size, 11, 12, 13});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {size, 11, 12, 13},
        expect,
        __LINE__,
        __FILE__);
  }
}

TEST_F(NVFuserTest, FusionStandaloneZeros_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      kBool,
      kFloat,
      kLong,
      kDouble,
      kHalf,
      kBFloat16,
      kInt,
      kComplexFloat,
      kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Int>();
  fusion->addInput(size);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = zeros({size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = zeros({size, size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = zeros_like(out_tv);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    std::vector<at::Tensor> expect;
    expect.reserve(dtypes.size());
    for (auto dtype : dtypes) {
      if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
        continue;
      }
      const auto options =
          at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      expect.emplace_back(at::zeros({size}, options));
      expect.emplace_back(at::zeros({size, size}, options));
      expect.emplace_back(at::zeros({size, size}, options));
    }
    auto cg_outputs = executor_cache.runFusionWithInputs({size});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {size},
        expect,
        __LINE__,
        __FILE__);
  }
}

TEST_F(NVFuserTest, FusionStandaloneOnes_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      kBool,
      kFloat,
      kLong,
      kDouble,
      kHalf,
      kBFloat16,
      kInt,
      kComplexFloat,
      kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Int>();
  fusion->addInput(size);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv = ones({size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = ones({size, size}, aten_to_data_type(dtype));
    fusion->addOutput(out_tv);
    out_tv = ones_like(out_tv);
    fusion->addOutput(out_tv);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    std::vector<at::Tensor> expect;
    expect.reserve(dtypes.size());
    for (auto dtype : dtypes) {
      if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
        continue;
      }
      const auto options =
          at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      expect.emplace_back(at::ones({size}, options));
      expect.emplace_back(at::ones({size, size}, options));
      expect.emplace_back(at::ones({size, size}, options));
    }
    auto cg_outputs = executor_cache.runFusionWithInputs({size});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {size},
        expect,
        __LINE__,
        __FILE__);
  }
}

TEST_F(NVFuserTest, FusionStandaloneARange_CUDA) {
  auto starts_ends = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto dtypes = {kFloat, kLong, kDouble};

  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }

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

TEST_F(NVFuserTest, FusionStandaloneEye_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      kBool,
      kFloat,
      kLong,
      kDouble,
      kHalf,
      kBFloat16,
      kInt,
      kComplexFloat,
      kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Int>();
  Val* maybe_m = IrBuilder::create<Int>();
  fusion->addInput(size);
  fusion->addInput(maybe_m);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto out_tv1 = eye(size, aten_to_data_type(dtype));
    fusion->addOutput(out_tv1);
    auto out_tv2 = eye(size, maybe_m, aten_to_data_type(dtype));
    fusion->addOutput(out_tv2);
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    std::vector<at::Tensor> expect;
    expect.reserve(dtypes.size());
    for (auto dtype : dtypes) {
      if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
        continue;
      }
      const auto options =
          at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      expect.emplace_back(at::eye(size, options));
      expect.emplace_back(at::eye(size, 15, options));
    }
    auto cg_outputs = executor_cache.runFusionWithInputs({size, 15});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {size, 15},
        expect,
        __LINE__,
        __FILE__);
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
