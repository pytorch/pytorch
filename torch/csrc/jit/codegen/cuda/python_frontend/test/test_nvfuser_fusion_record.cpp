#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace nvfuser;
using namespace torch::jit::fuser::cuda;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*RecordFunctorEquality*"
TEST_F(NVFuserTest, RecordFunctorEquality_CUDA) {
  // Getting the std::function matching correct is error prone so providing
  // checks for OpRecord, CastOp, and ReductionOp that employ std::function
  // matching.

  // OpRecord Equality Check
  {
    auto t0 = nvfuser::State(0, StateType::Tensor);
    auto s1 = nvfuser::State(1, StateType::Scalar);
    auto out = nvfuser::State(2, StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(
        new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(
                Nvf::mul)));
    std::unique_ptr<RecordFunctor> test_record2(
        new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(
                Nvf::mul)));
    std::unique_ptr<RecordFunctor> test_record3(
        new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(
                Nvf::mul)));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }

  // CastOpRecord Equality Check
  {
    auto t0 = nvfuser::State(0, StateType::Tensor);
    auto out = nvfuser::State(1, StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(
        new CastOpRecord<Nvf::TensorView*, Nvf::TensorView*>(
            {t0},
            {out},
            "ops.cast",
            static_cast<Nvf::TensorView* (*)(Nvf::DataType, Nvf::TensorView*)>(
                Nvf::castOp),
            Nvf::DataType::Half));
    std::unique_ptr<RecordFunctor> test_record2(
        new CastOpRecord<Nvf::TensorView*, Nvf::TensorView*>(
            {t0},
            {out},
            "ops.cast",
            static_cast<Nvf::TensorView* (*)(Nvf::DataType, Nvf::TensorView*)>(
                Nvf::castOp),
            Nvf::DataType::Half));
    std::unique_ptr<RecordFunctor> test_record3(
        new CastOpRecord<Nvf::TensorView*, Nvf::TensorView*>(
            {t0},
            {out},
            "ops.cast",
            static_cast<Nvf::TensorView* (*)(Nvf::DataType, Nvf::TensorView*)>(
                Nvf::castOp),
            Nvf::DataType::Half));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }

  // ReductionOpRecord Equality Check
  {
    auto t0 = nvfuser::State(0, StateType::Tensor);
    auto out = nvfuser::State(1, StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        static_cast<
            Nvf::
                TensorView* (*)(Nvf::TensorView*, const std::vector<int>&, bool, Nvf::DataType)>(
            Nvf::sum),
        {0},
        false,
        Nvf::DataType::Float));
    std::unique_ptr<RecordFunctor> test_record2(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        static_cast<
            Nvf::
                TensorView* (*)(Nvf::TensorView*, const std::vector<int>&, bool, Nvf::DataType)>(
            Nvf::sum),
        {0},
        false,
        Nvf::DataType::Float));
    std::unique_ptr<RecordFunctor> test_record3(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        static_cast<
            Nvf::
                TensorView* (*)(Nvf::TensorView*, const std::vector<int>&, bool, Nvf::DataType)>(
            Nvf::sum),
        {0},
        false,
        Nvf::DataType::Float));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
