#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionSplitDims_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t* p = prime_numbers;
  auto tv = makeConcreteTensor(
      {p[0] * p[1] * p[2], p[3], p[4], p[5] * p[6], p[7], p[8], p[9] * p[10]});
  std::vector<size_t> dims{0, 1, 2, 3, 4, 5, 6};
  scheduler_utils::splitDims(
      tv, {{0, p[2]}, {0, p[1]}, {3, p[6]}, {6, p[10]}}, dims);
  TORCH_CHECK(tv->nDims() == 11);
  for (auto i : c10::irange(11)) {
    TORCH_CHECK(tv->axis(i)->extent()->evaluateInt() == p[i]);
  }
  std::vector<size_t> expect{0, 3, 4, 5, 7, 8, 9};
  TORCH_CHECK(dims == expect);
}

TEST_F(NVFuserTest, FusionMergeDims_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t* p = prime_numbers;
  auto tv = makeConcreteTensor(
      {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]});
  std::vector<size_t> dims{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto merged = scheduler_utils::mergeDims(tv, {2, 3, 7, 8, 9}, dims);
  TORCH_CHECK(merged == 2);
  std::vector<int64_t> expect_shape{
      p[0], p[1], p[2] * p[3] * p[7] * p[8] * p[9], p[4], p[5], p[6], p[10]};
  TORCH_CHECK(tv->nDims() == expect_shape.size());
  for (auto i : c10::irange(expect_shape.size())) {
    TORCH_CHECK(tv->axis(i)->extent()->evaluateInt() == expect_shape[i]);
  }
  std::vector<size_t> expect_dims{0, 1, 2, 2, 3, 4, 5, 2, 2, 2, 6};
  TORCH_CHECK(dims == expect_dims);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
