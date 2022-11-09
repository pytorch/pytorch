#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
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

TEST_F(NVFuserTest, FusionReorderAsRFactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int a = 1, b = 2, c = 3, d = 4;

  TensorView* tv0 = makeConcreteTensor({a, b, c, d});
  fusion.addInput(tv0);
  fusion.addOutput(tv0);

  // [a, b, c, d]
  tv0->merge(0, 2);
  // [a*c, b, d]
  tv0->split(1, 2);
  // [a*c, bo, bi, d]
  tv0->split(3, 3);
  // [a*c, bo, bi, do, di]
  tv0->reorder({{1, 4}, {2, 1}, {3, 3}, {4, 2}});
  // [a*c, bi, di, do, bo]
  tv0->merge(3);
  tv0->merge(1);
  // [a*c, bi*di, do*bo]
  tv0->reorder({{0, 2}});
  // [bi*di, do*bo, a*c]
  // Order we want is:
  // [a*c, do*bo, bi*di]
  auto old2new = scheduler_utils::domainReorderAsRfactorMap(tv0);
  TORCH_CHECK(old2new[0] == 2);
  TORCH_CHECK(old2new[1] == 1);
  TORCH_CHECK(old2new[2] == 0);
}

TEST_F(NVFuserTest, FusionDisjointViewSet_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion->addInput(tv0);

  auto tv1 = view(tv0, {2, 3, 4}, {2, 12});

  auto tv2 = makeConcreteTensor({2, 12});
  fusion->addInput(tv2);

  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto disjoint_exact = scheduler_utils::disjointViewSets(fusion.get());

  TORCH_INTERNAL_ASSERT(
      disjoint_exact.strictAreMapped(tv0->axis(1), tv0->axis(2)));
}

TEST_F(NVFuserTest, FusionBroadcastViewMultiples_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int a = 2, b = 3, c = 5, d = 7, e = 11, f = 13;

  auto tv0 = makeConcreteTensor({a, b, c, d, e, f});
  fusion.addInput(tv0);

  // tie e and f together (swapping values next to eachother enforces they'll be
  // merged then split by view)
  auto tv1 = view(tv0, {a, b, c, d, e, f}, {a, b, c, d, f, e});
  fusion.addOutput(tv1);

  // swap d and e
  auto tv2 = transpose(tv1, 3, 4);
  // tie c and e together
  auto tv3 = view(tv2, {a, b, c, e, d, f}, {a, b, e, c, d, f});

  fusion.addOutput(tv3);

  auto tv4 = set(tv0);
  // Use tv4 as the reference
  fusion.addOutput(tv4);

  // a, b, d aren't tied to anything so they are valid broadcasts from the
  // perspective of broadcast multiples analysis.
  auto tv5 = makeConcreteTensor({1, 1, c, 1, e, f});
  fusion.addInput(tv5);

  // c, e, and f are tied together so this shouldn't be counted as a broadcast
  // dim in the reference since it's a partial bcast
  auto tv6 = makeConcreteTensor({a, b, c, 1, 1, 1});
  fusion.addInput(tv6);

  // c, e, and f are tied together this should be counted as a broadcast dim in
  // the reference since it's a partial bcast
  auto tv7 = makeConcreteTensor({a, b, 1, 1, 1, 1});
  fusion.addInput(tv7);

  // plug the broadcasts into the fusion
  auto tv8 = add(tv5, tv4);
  auto tv9 = add(tv6, tv8);
  auto tv10 = add(tv7, tv9);
  fusion.addOutput(tv10);

  auto bcast_info =
      scheduler_utils::getBroadcastMultiples(tv4, DataType::Int32);

  // linked c, e, and f together so they should have the same id.
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[5] == 0);
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[4] == 0);
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[3] == 1);
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[2] == 0);
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[1] == 2);
  TORCH_CHECK(bcast_info.view_disjoint_set_ids[0] == 3);

  TORCH_CHECK(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 0));
  TORCH_CHECK(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 1));
  TORCH_CHECK(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 2));
  TORCH_CHECK(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 3));
  TORCH_CHECK(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 4));
  TORCH_CHECK(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 5));

  // tv0  [a, b, c, d, e, f]
  // tv1  [a, b, c, d, e, f]
  // tv3  [a, b, c, d, e, f]
  // tv4  [a, b, c, d, e, f]
  // tv5  [1, 1, c, 1, e, f] -> Left bcasts should show up in some multiples
  // tv6  [a, b, c, 1, 1, 1] -> view interferes with bcasts, non of these should
  //                            show up
  // tv7  [a, b, 1, 1, 1, 1] -> These broadcasts could be recognized
  // tv10 [a, b, c, d, e, f]

  TORCH_CHECK(
      bcast_info.broadcast_multiples[0].lhs_multiple == 0 &&
      bcast_info.broadcast_multiples[0].rhs_multiple == 8 * 4);

  TORCH_CHECK(
      bcast_info.broadcast_multiples[1].lhs_multiple == 7 * 4 &&
      bcast_info.broadcast_multiples[1].rhs_multiple == 8 * 4);

  TORCH_CHECK(
      bcast_info.broadcast_multiples[2].lhs_multiple == 7 * 4 &&
      bcast_info.broadcast_multiples[2].rhs_multiple == 7 * 4);

  TORCH_CHECK(
      bcast_info.broadcast_multiples[3].lhs_multiple == 8 * 4 &&
      bcast_info.broadcast_multiples[3].rhs_multiple == 7 * 4);

  TORCH_CHECK(
      bcast_info.broadcast_multiples[4].lhs_multiple == 8 * 4 &&
      bcast_info.broadcast_multiples[4].rhs_multiple == 7 * 4);

  TORCH_CHECK(
      bcast_info.broadcast_multiples[5].lhs_multiple == 8 * 4 &&
      bcast_info.broadcast_multiples[5].rhs_multiple == 7 * 4);
}

TEST_F(NVFuserTest, FusionTVDomainGuard_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<bool> all_true = {true, true};
  std::vector<bool> all_false = {false, false};
  std::vector<bool> false_true = {false, true};
  auto tv = TensorViewBuilder().ndims(2).contiguity(false_true).build();
  TORCH_CHECK(tv->domain()->contiguity() == false_true);
  {
    auto guard = ir_utils::overrideContiguityGuard(tv, true);
    TORCH_CHECK(tv->domain()->contiguity() == all_true);
  }
  TORCH_CHECK(tv->domain()->contiguity() == false_true);
  {
    auto guard = ir_utils::overrideContiguityGuard(tv, false);
    TORCH_CHECK(tv->domain()->contiguity() == all_false);
  }
  TORCH_CHECK(tv->domain()->contiguity() == false_true);
  {
    auto guard1 = ir_utils::overrideContiguityGuard(tv, true);
    auto guard2 = std::move(guard1);
    TORCH_CHECK(tv->domain()->contiguity() == all_true);
  }
  TORCH_CHECK(tv->domain()->contiguity() == false_true);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
