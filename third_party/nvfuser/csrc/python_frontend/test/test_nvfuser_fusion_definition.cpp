#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {
using namespace nvfuser::python_frontend;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*FusionDefinition*"
TEST_F(NVFuserTest, FusionDefinition_CUDA) {
  // Test that the FusionDefinition asserts on max_length == 0
  {
    FusionDefinition fd(c10::nullopt, 0);

    try {
      fd.setupDefinition();
      FAIL() << "You should trigger an assert with 0 Records allowed!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Create a new FusionDefinition that is not found in the cache
  {
    FusionDefinition fd(c10::nullopt, 4);

    try {
      fd.setupDefinition();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! "
             << e.what();
    }

    auto t0 = fd.defineTensor(2);
    try {
      fd.defineRecord(new TensorRecord(
          {fd.recordingState(t0())}, {3}, {true}, DataType::Float));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(
          new ScalarRecord({fd.recordingState(s1())}, DataType::Double));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor(2);
    try {
      fd.defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(
          {fd.recordingState(t0()), fd.recordingState(s1())},
          {fd.recordingState(t2())},
          "ops.add",
          static_cast<TensorView* (*)(TensorView*, Val*)>(add)));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }

    try {
      fd.defineRecord(new OutputRecord<TensorView>({fd.recordingState(t2())}));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.defineRecord(new OutputRecord<Val>({fd.recordingState(s1())}));
      FAIL() << "Expected an assert for too many records!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fd.finalizeDefinition();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! "
             << e.what();
    }
  }

  // Look up a FusionDefinition with a defined Fusion #id 1
  {
    FusionDefinition fd(1);

    try {
      fd.setupDefinition();
      FAIL() << "You should trigger an assert with a defined fusion!";
    } catch (const std::exception& e) {
      SUCCEED();
    }
  }

  // Look up a FusionDefinition completely in the cache
  {
    FusionDefinition fd(c10::nullopt, 4);

    try {
      fd.setupDefinition();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! "
             << e.what();
    }

    auto t0 = fd.defineTensor(2);
    try {
      fd.defineRecord(new TensorRecord(
          {fd.recordingState(t0())}, {3}, {true}, DataType::Float));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(
          new ScalarRecord({fd.recordingState(s1())}, DataType::Double));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor(2);
    try {
      fd.defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(
          {fd.recordingState(t0()), fd.recordingState(s1())},
          {fd.recordingState(t2())},
          "ops.add",
          static_cast<TensorView* (*)(TensorView*, Val*)>(add)));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }

    try {
      fd.defineRecord(new OutputRecord<TensorView>({fd.recordingState(t2())}));
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.finalizeDefinition();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! "
             << e.what();
    }
  }
}

} // namespace nvfuser
