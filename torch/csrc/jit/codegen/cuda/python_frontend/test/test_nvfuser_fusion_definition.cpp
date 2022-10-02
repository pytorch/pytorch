#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_interface.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace nvfuser;
using namespace torch::jit::fuser::cuda;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*FusionDefinition*"
TEST_F(NVFuserTest, FusionDefinition_CUDA) {
  // Test that the FusionDefinition asserts on max_length == 0
  {
    FusionDefinition fd(nullptr, 0);

    try {
      fd.enter();
      FAIL() << "You should trigger an assert with 0 Records allowed!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Test that the FusionDefinition asserts on a null FusionManager ptr
  {
    FusionDefinition fd(nullptr, 5);

    try {
      fd.enter();
      FAIL() << "You should trigger an assert with a null FusionInterface!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Create a new FusionDefinition that is not found in the cache
  {
    std::unique_ptr<FusionInterface> fusion =
        std::make_unique<FusionInterface>();
    FusionDefinition fd(fusion.get(), 4);

    try {
      fd.enter();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! "
             << e.what();
    }

    auto t0 = fd.defineTensor();
    try {
      fd.defineRecord(new TensorRecord(
          {fd.recordingState(t0())}, {3}, {true}, Nvf::DataType::Float));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(
          new ScalarRecord({fd.recordingState(s1())}, Nvf::DataType::Double));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor();
    try {
      fd.defineRecord(
          new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
              {fd.recordingState(t0()), fd.recordingState(s1())},
              {fd.recordingState(t2())},
              "ops.add",
              static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(
                  Nvf::add)));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }

    try {
      fd.defineRecord(
          new OutputRecord<Nvf::TensorView>({fd.recordingState(t2())}));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.defineRecord(new OutputRecord<Nvf::Val>({fd.recordingState(s1())}));
      FAIL() << "Expected an assert for too many records!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fd.exit();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! "
             << e.what();
    }
  }

  // Look up a FusionDefinition with a defined Fusion
  {
    std::unique_ptr<FusionInterface> fusion =
        std::make_unique<FusionInterface>(0);
    FusionDefinition fd(fusion.get(), 1);

    try {
      fd.enter();
      FAIL() << "You should trigger an assert with a defined FusionInterface!";
    } catch (const std::exception& e) {
      SUCCEED();
    }
  }

  // Look up a FusionDefinition completely in the cache
  {
    std::unique_ptr<FusionInterface> fusion =
        std::make_unique<FusionInterface>();
    FusionDefinition fd(fusion.get(), 4);

    try {
      fd.enter();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! "
             << e.what();
    }

    auto t0 = fd.defineTensor();
    try {
      fd.defineRecord(new TensorRecord(
          {fd.recordingState(t0())}, {3}, {true}, Nvf::DataType::Float));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(
          new ScalarRecord({fd.recordingState(s1())}, Nvf::DataType::Double));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor();
    try {
      fd.defineRecord(
          new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
              {fd.recordingState(t0()), fd.recordingState(s1())},
              {fd.recordingState(t2())},
              "ops.add",
              static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(
                  Nvf::add)));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }

    try {
      fd.defineRecord(
          new OutputRecord<Nvf::TensorView>({fd.recordingState(t2())}));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.exit();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! "
             << e.what();
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
