#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>

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
  
  // Test that the FusionDefinition asserts a null FusionManager ptr
  {
    FusionDefinition fd(nullptr, 5);
 
    try {
      fd.enter();
      FAIL() << "You should trigger an assert with a null FusionManager!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Create a new FusionDefinition that is not found in the cache
  {
    FusionDefinition fd(FusionManager::get(1), 4);
 
    try {
      fd.enter();
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! " << e.what();
    }

    auto t0 = fd.defineTensor();
    try {
      fd.defineRecord(new TensorRecord(
          {*static_cast<State*>(t0)},
          {3},
          {true},
          Nvf::DataType::Float));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(new ScalarRecord(
          {*static_cast<State*>(s1)},
          Nvf::DataType::Double));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor();
    try {
      fd.defineRecord(new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
          {*static_cast<State*>(t0), *static_cast<State*>(s1)},
          {*static_cast<State*>(t2)},
          "ops.add",
          static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(Nvf::add)
          ));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }
 
    try {
      fd.defineRecord(new OutputRecord<Nvf::TensorView>(
          {*static_cast<State*>(t2)}));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.defineRecord(new OutputRecord<Nvf::Val>(
          {*static_cast<State*>(s1)}));
      FAIL() << "Expected an assert for too many records!";
    } catch(...) {
      SUCCEED();
    }

    try {
      fd.exit();
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! " << e.what();
    }
  }
 
  // Look up a FusionDefinition completely in the cache 
  {
    FusionDefinition fd(FusionManager::get(1), 4);
 
    try {
      fd.enter();
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert while entering FusionDefinition context! " << e.what();
    }

    auto t0 = fd.defineTensor();
    try {
      fd.defineRecord(new TensorRecord(
          {*static_cast<State*>(t0)},
          {3},
          {true},
          Nvf::DataType::Float));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Tensor Record creation! " << e.what();
    }

    auto s1 = fd.defineScalar();
    try {
      fd.defineRecord(new ScalarRecord(
          {*static_cast<State*>(s1)},
          Nvf::DataType::Double));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Scalar Record creation! " << e.what();
    }

    auto t2 = fd.defineTensor();
    try {
      fd.defineRecord(new OpRecord<Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*>(
          {*static_cast<State*>(t0), *static_cast<State*>(s1)},
          {*static_cast<State*>(t2)},
          "ops.add",
          static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>(Nvf::add)
          ));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Add Record creation! " << e.what();
    }
 
    try {
      fd.defineRecord(new OutputRecord<Nvf::TensorView>(
          {*static_cast<State*>(t2)}));
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during Output Record creation! " << e.what();
    }

    try {
      fd.exit();
      SUCCEED();
    } catch (const std::exception &e) {
      FAIL() << "Unexpected assert during creation of a new Fusion! " << e.what();
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
