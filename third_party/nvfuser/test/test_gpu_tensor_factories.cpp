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

namespace nvfuser {

TEST_F(NVFuserTest, FusionStandaloneFull_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

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
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

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
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

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

TEST_F(NVFuserTest, FusionStandaloneIota_CUDA) {
  auto starts = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto lengths = {0, 1, 2, 10, 1023, 1024, 1024 * 1024};
  auto dtypes = {at::kInt, at::kLong, at::kFloat, at::kDouble};

  for (auto dtype : dtypes) {
    auto data_type = aten_to_data_type(dtype);
    auto input_type =
        (data_type == DataType::Int32 || data_type == DataType::Int
             ? DataType::Int
             : DataType::Double);

    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* length = IrBuilder::create<Int>();

    Val* start = IrBuilder::newScalar(input_type);
    Val* step = IrBuilder::newScalar(input_type);
    fusion->addInput(length);
    fusion->addInput(start);
    fusion->addInput(step);
    auto tv0 = iota(length, start, step, data_type);
    fusion->addOutput(tv0);

    FusionExecutorCache executor_cache(std::move(fusion));

    const auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);

    switch (dtype) {
      case at::kInt:
      case at::kLong: {
        for (auto length : lengths) {
          for (auto start : starts) {
            for (auto step : steps) {
              int64_t start_ = (int64_t)start;
              int64_t step_ = (int64_t)step;
              int64_t end_ = start_ + step_ * length;
              auto a = at::arange(start_, end_, step_, options);

              auto cg_outputs =
                  executor_cache.runFusionWithInputs({length, start_, step_});

              testValidate(
                  executor_cache.fusion(),
                  cg_outputs,
                  {length, start_, step_},
                  {a},
                  __LINE__,
                  __FILE__);
            }
          }
        }
        break;
      }
      case at::kFloat:
      case at::kDouble: {
        for (auto length : lengths) {
          for (auto start : starts) {
            for (auto step : steps) {
              double start_ = (double)start;
              double step_ = (double)step;

              // Due to rounding error, it can be hard to guarantee the size of
              // the output of arange to be exactly length, so we generate a
              // larger tensor and truncate it to length.
              double end_ = start_ + step_ * (length + 1);
              auto a =
                  at::arange(start_, end_, step_, options).narrow(0, 0, length);

              auto cg_outputs =
                  executor_cache.runFusionWithInputs({length, start_, step_});

              testValidate(
                  executor_cache.fusion(),
                  cg_outputs,
                  {length, start_, step_},
                  {a},
                  __LINE__,
                  __FILE__);
            }
          }
        }
        break;
      }
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }
}

TEST_F(NVFuserTest, FusionStandaloneARange_CUDA) {
  auto starts_ends = {-1., 0., 10.3, 1024. * 256};
  auto steps = {-1.5, 1., 2.};
  auto dtypes = {at::kFloat, at::kLong, at::kDouble};

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

TEST_F(NVFuserTest, FusionStandaloneEye_CUDA) {
  auto sizes = {0, 1, 10, 17, 1024};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

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

TEST_F(NVFuserTest, FusionARangeScalarHoisting1_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* start_int = IrBuilder::create<Int>();
  Val* end_int = IrBuilder::create<Int>();
  Val* step_int = IrBuilder::create<Int>();
  fusion->addInput(start_int);
  fusion->addInput(end_int);
  fusion->addInput(step_int);
  auto output1 = arange(start_int, end_int, step_int, DataType::Int);
  auto output2 = full_like(output1, output1->axis(0)->extent(), DataType::Int);
  fusion->addOutput(output1);
  fusion->addOutput(output2);

  int64_t start = 0, end = 100, step = 1;

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {start, end, step});
  auto cg_outputs = fe.runFusion({start, end, step});

  const auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(start, end, step, options);
  at::Tensor t1 = at::full_like(t0, end - start, options);

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(int64_t i0, int64_t i1, int64_t i2, Tensor<int64_t, 1> T0, Tensor<int64_t, 1> T1) {
  int64_t i3;
  i3 = i1 - i0;
  int64_t i4;
  i4 = abs(i3);
  int64_t i5;
  i5 = abs(i2);
  int64_t i6;
  i6 = ceilDiv(i4, i5);
  #pragma unroll 1
  for(nvfuser_index_t i8 = 0; i8 < i6; ++i8) {
    T0[i8] = (i0 + (i2 * i8));
  }
  #pragma unroll 1
  for(nvfuser_index_t i9 = 0; i9 < i6; ++i9) {
    T1[i9] = i6;
  }
}
)";

  assertCUDAKernel(fusion.get(), expected_kernel);

  testValidate(
      fusion.get(),
      cg_outputs,
      {start, end, step},
      {t0, t1},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
