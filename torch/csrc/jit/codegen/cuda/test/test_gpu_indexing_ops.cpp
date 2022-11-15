#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionSelectOpPointwise_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t1 = t0.select(0, idx);
  auto t2 = t0.select(1, idx);
  auto t3 = t0.select(2, idx);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t1, t2, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSelectOpReduction_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0});
  auto tv5 = sum(tv2, {1});
  auto tv6 = sum(tv3, {0, 1});

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t4 = t0.select(0, idx).sum(0);
  auto t5 = t0.select(1, idx).sum(1);
  auto t6 = t0.select(2, idx).sum(IntArrayRef{0, 1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t4, t5, t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSelectOpPersistent_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Int>();
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0}, true);
  auto tv5 = sum(tv2, {1}, true);
  auto tv6 = sum(tv3, {0, 1}, true);

  auto tv7 = add(tv1, tv4);
  auto tv8 = add(tv2, tv5);
  auto tv9 = add(tv3, tv6);

  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto t1 = t0.select(0, idx);
  auto t2 = t0.select(1, idx);
  auto t3 = t0.select(2, idx);
  auto t4 = t1.sum(0, true);
  auto t5 = t2.sum(1, true);
  auto t6 = t3.sum(IntArrayRef{0, 1}, true);
  auto t7 = t1 + t4;
  auto t8 = t2 + t5;
  auto t9 = t3 + t6;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(
      &fusion, cg_outputs, {t0, idx}, {t7, t8, t9}, __LINE__, __FILE__);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
