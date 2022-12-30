#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor_utils.h>
#include <fusion.h>
#include <lower_utils.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

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
  TORCH_CHECK(merged == (size_t)2);
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

namespace {

bool checkProjectedExtent(
    vectorize_helper::ProjectedExtent& pe,
    ExpressionEvaluator& expr_eval,
    int64_t expected_numer,
    int64_t expected_denom) {
  auto numerator_val = expr_eval.evaluate(pe.getNumerator());
  auto denominator_val = expr_eval.evaluate(pe.getDenominator());

  if (!numerator_val.has_value() || !denominator_val.has_value()) {
    return false;
  }

  if (expected_numer == numerator_val->as<int64_t>() &&
      expected_denom == denominator_val->as<int64_t>()) {
    return true;
  }

  if (numerator_val->as<int64_t>() % denominator_val->as<int64_t>() == 0) {
    if (expected_numer ==
            numerator_val->as<int64_t>() / denominator_val->as<int64_t>() &&
        expected_denom == 1) {
      return true;
    }
  }
  return false;
}

bool checkProjectedExtent(
    vectorize_helper::ProjectedExtent& pe,
    int64_t expected_numer,
    int64_t expected_denom) {
  ExpressionEvaluator expr_eval;
  return checkProjectedExtent(pe, expr_eval, expected_numer, expected_denom);
}

bool trivialOrOneProjectedExtent(vectorize_helper::ProjectedExtent& pe) {
  if (pe.isZero()) {
    return true;
  }

  auto numerator_val = pe.getNumerator();
  if (!numerator_val->isConstInt()) {
    return false;
  }

  if (numerator_val->evaluateInt() != 1) {
    return false;
  }

  return true;
}
} // namespace

// Test simple backward mapping through split
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3}, {2, 3});
  fusion.addOutput(tv1);

  {
    // No mappings
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv1, {});

    TORCH_CHECK(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
    TORCH_CHECK(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(1)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 3, 1),
        mapper.getMappedExtent(tv0->axis(0)).toString());
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0), tv1->axis(1)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 2, 1),
        mapper.getMappedExtent(tv1->axis(1)).toString());
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 3, 1),
        mapper.getMappedExtent(tv1->axis(1)).toString());

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 2 * 3, 1),
        mapper.getMappedExtent(tv0->axis(0)).toString());
  }
}

// Test backward mapping through multiple splits
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3 * 4}, {2 * 3, 4});
  auto tv2 = view(tv1, {2 * 3, 4}, {2, 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1), tv2->axis(2)});

  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv2->axis(0))),
      mapper.getMappedExtent(tv2->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 3, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(2)), 4, 1),
      mapper.getMappedExtent(tv2->axis(2)).toString());
  // Inner dim fully maps, outer dim of split partially maps
  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(2)));

  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 4, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());
  // Inner dim fully maps, outer dim of split partially maps
  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 4 * 3, 1),
      mapper.getMappedExtent(tv0->axis(0)).toString());
  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
}

// Test backward mapping through multiple splits
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3 * 4}, {2, 3 * 4});
  auto tv2 = view(tv1, {2, 3 * 4}, {2, 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(2)});

  // Partial map forwarding
  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 4, 1),
      mapper.getMappedExtent(tv0->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 4, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(2)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(2)), 4, 1),
      mapper.getMappedExtent(tv2->axis(2)).toString());
}

// Test simple backward mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper4_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3}, {2 * 3});
  fusion.addOutput(tv1);

  {
    // No mapping
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv1, {});

    TORCH_CHECK(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
    TORCH_CHECK(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
  }

  {
    // Full merge mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 2, 1),
        mapper.getMappedExtent(tv0->axis(0)).toString());
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3, 1),
        mapper.getMappedExtent(tv0->axis(1)).toString());

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 2 * 3, 1),
        mapper.getMappedExtent(tv1->axis(0)).toString());
  }
}

// Test symbolic partial mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper5_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp = at::randn({2 * 3, 4}, options);

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv0->axis(0)), expr_eval, 3, 1),
      mapper.getMappedExtent(tv0->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv0->axis(1)), expr_eval, 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv1->axis(0)), expr_eval, 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv2->axis(1)), expr_eval, 3 * 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
}

// Test concrete partial outer dim mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper6_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3, 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 3, 1),
      mapper.getMappedExtent(tv0->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
}

// Test concrete exact inner dim mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper7_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv0->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
}

// Test concrete partial inner dim mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper8_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2 * 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv0->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
}

// Test concrete partial inner dim mapping through merge
TEST_F(NVFuserTest, FusionVectorizeBackwardMapper9_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({3, 5, 7});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {3, 5, 7}, {7, 5 * 3});
  auto tv2 = view(tv1, {7, 5 * 3}, {3, 5, 7});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1), tv2->axis(2)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 3);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[2]->sameAs(tv0->axis(2)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv0->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 5, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(2)), 7, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 7, 3),
      mapper.getMappedExtent(tv1->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 15, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(2)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 5, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(2)), 7, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
}

// Similar to FusionVectorizeBackwardMapper1_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3}, {2 * 3});
  fusion.addOutput(tv1);
  {
    // No mappings
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, {});

    TORCH_CHECK(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
    TORCH_CHECK(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(1)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3, 1),
        mapper.getMappedExtent(tv1->axis(0)).toString());
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0), tv0->axis(1)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 2, 1),
        mapper.getMappedExtent(tv0->axis(0)).toString());
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3, 1),
        mapper.getMappedExtent(tv0->axis(1)).toString());

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 2 * 3, 1),
        mapper.getMappedExtent(tv1->axis(0)).toString());
  }
}

// Similar to FusionVectorizeBackwardMapper2_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3, 4}, {2 * 3, 4});
  auto tv2 = view(tv1, {2 * 3, 4}, {2 * 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1), tv0->axis(2)});

  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv0->axis(0))),
      mapper.getMappedExtent(tv0->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(2)), 4, 1),
      mapper.getMappedExtent(tv0->axis(2)).toString());
  // Inner dim fully maps, outer dim of split partially maps
  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(2)));

  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 4, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());
  // Inner dim fully maps, outer dim of split partially maps
  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(0)), 4 * 3, 1),
      mapper.getMappedExtent(tv2->axis(0)).toString());
  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
}

// Similar to FusionVectorizeBackwardMapper3_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3, 4}, {2, 3 * 4});
  auto tv2 = view(tv1, {2, 3 * 4}, {2 * 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(2)});

  // Partial map forwarding
  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(0)), 4, 1),
      mapper.getMappedExtent(tv2->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 4, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(2)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(2)), 4, 1),
      mapper.getMappedExtent(tv0->axis(2)).toString());
}

// Similar to FusionVectorizeBackwardMapper4_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper4_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3}, {2, 3});
  fusion.addOutput(tv1);

  {
    // No mapping
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, {});

    TORCH_CHECK(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
    TORCH_CHECK(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0)});

    TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 2);
    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 2, 1),
        mapper.getMappedExtent(tv1->axis(0)).toString());
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 3, 1),
        mapper.getMappedExtent(tv1->axis(1)).toString());

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(0)), 2 * 3, 1),
        mapper.getMappedExtent(tv0->axis(0)).toString());
  }
}

// Similar to FusionVectorizeBackwardMapper5_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper5_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2 * 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp = at::randn({2, 3 * 4}, options);

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv2->axis(0)), expr_eval, 3, 1),
      mapper.getMappedExtent(tv2->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv2->axis(1)), expr_eval, 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv1->axis(0)), expr_eval, 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(
          mapper.getMappedExtent(tv0->axis(1)), expr_eval, 3 * 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
}

// Similar to FusionVectorizeBackwardMapper6_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper6_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {{2 * 3, 4}});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(0)), 3, 1),
      mapper.getMappedExtent(tv2->axis(0)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
}

// Similar to FusionVectorizeBackwardMapper7_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper7_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv2->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 3 * 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 3 * 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
}

// Similar to FusionVectorizeBackwardMapper8_CUDA but in the reverse direction
TEST_F(NVFuserTest, FusionVectorizeForwardMapper8_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3, 4});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = view(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv2->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 4, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 4, 1),
      mapper.getMappedExtent(tv1->axis(0)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 4, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
}

// Make sure partial mappings maintain their total values.
TEST_F(NVFuserTest, FusionVectorizeForwardMapper9_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({3, 5, 7});
  fusion.addInput(tv0);
  auto tv1 = view(tv0, {3, 5, 7}, {7, 5 * 3});
  auto tv2 = view(tv1, {7, 5 * 3}, {3, 5, 7});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1), tv0->axis(2)});

  TORCH_CHECK(mapper.mappedRFactorIds(tv2).size() == 3);
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv2)[2]->sameAs(tv2->axis(2)));
  TORCH_CHECK(
      trivialOrOneProjectedExtent(mapper.getMappedExtent(tv2->axis(0))));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(1)), 5, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv2->axis(2)), 7, 1),
      mapper.getMappedExtent(tv2->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv1).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(0)), 7, 3),
      mapper.getMappedExtent(tv1->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv1->axis(1)), 15, 1),
      mapper.getMappedExtent(tv1->axis(1)).toString());

  TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 2);
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  TORCH_CHECK(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(2)));
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 5, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
  TORCH_CHECK(
      checkProjectedExtent(mapper.getMappedExtent(tv0->axis(2)), 7, 1),
      mapper.getMappedExtent(tv0->axis(1)).toString());
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(NVFuserTest, FusionVectorizeMapperAdvanced_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // For broadcast we can't back propogate mapped axes to the left of bcast
  // axis.
  // For reduction we can't forward propogate mapped axes to the left of the
  // reduce axis.

  auto tv0 = makeContigConcreteTensor({3, 4 * 6});
  fusion.addInput(tv0);

  auto tv1 = view(tv0, {3, 4 * 6}, {3, 4, 6});
  auto tv2 = broadcast(tv1, {false, false, true, false});

  auto tv3 = makeContigConcreteTensor({3, 4, 5, 6});
  fusion.addInput(tv3);
  auto tv4 = add(tv3, tv2);

  auto tv5 = view(tv4, {3, 4, 5, 6}, {3 * 4 * 5, 6});

  // Broadcast path from tv0->tv5
  fusion.addOutput(tv5);

  // Sum path from tv3->tv6
  auto tv6 = sum(tv3, {2});
  auto tv7 = view(tv6, {3, 4, 6}, {3, 4 * 6});
  fusion.addOutput(tv7);
  {
    // tv5[3*4*5, 6]
    // tv0[3, 4*6]
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv5, {tv5->axis(0), tv5->axis(1)});
    TORCH_CHECK(mapper.mappedRFactorIds(tv0).size() == 1);
    TORCH_CHECK(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv0->axis(1)), 6, 1),
        mapper.getMappedExtent(tv0->axis(1)).toString());
  }

  {
    // tv3[3, 4, 5, 6]
    // tv7[3, 4*6]
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv3, {tv3->axis(0), tv3->axis(1), tv3->axis(2), tv3->axis(3)});
    TORCH_CHECK(mapper.mappedRFactorIds(tv7).size() == 1);
    TORCH_CHECK(mapper.mappedRFactorIds(tv7)[0]->sameAs(tv7->axis(1)));

    TORCH_CHECK(
        checkProjectedExtent(mapper.getMappedExtent(tv7->axis(1)), 6, 1),
        mapper.getMappedExtent(tv7->axis(1)).toString());
  }
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(NVFuserTest, FusionVectorizeSpanningTree_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<TensorView*> inputs;
  std::vector<TensorView*> intermediates;
  std::vector<TensorView*> outputs;

  auto bcast_inp = makeContigConcreteTensor({2});
  inputs.push_back(bcast_inp);
  auto bcast = broadcast(bcast_inp, {false, true});

  for (auto i : c10::irange(10)) {
    auto resolution_inp = makeContigConcreteTensor({2, 2});
    inputs.push_back(resolution_inp);
    auto intermediate = add(bcast, resolution_inp);
    if (i > 0) {
      auto output = add(intermediates.back(), intermediate);
      outputs.push_back(output);
    }
    intermediates.push_back(intermediate);
  }

  for (auto rev_inp : {false, true}) {
    for (auto rev_out : {false, true}) {
      // Clear fusion inputs / outputs
      {
        auto fusion_outs = fusion.outputs();
        for (auto out : fusion_outs) {
          fusion.removeOutput(out);
        }
        auto fusion_inps = fusion.inputs();
        for (auto inp : fusion_inps) {
          fusion.removeOutput(inp);
        }
      }

      if (rev_inp) {
        std::reverse(inputs.begin(), inputs.end());
      }

      if (rev_out) {
        std::reverse(outputs.begin(), outputs.end());
      }

      {
        // Populate outputs and inputs
        for (auto out : outputs) {
          fusion.addOutput(out);
        }

        for (auto inp : inputs) {
          fusion.addInput(inp);
        }
      }

      for (auto out : outputs) {
        auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
            out, {out->axis(0), out->axis(1)});

        for (auto tv : ir_utils::allTvs(&fusion)) {
          if (tv->name() == 0 || tv->name() == 1) {
            continue;
          }
          for (auto axis : tv->getRootDomain()) {
            TORCH_INTERNAL_ASSERT(
                mapper.getMappedExtent(axis).getNumerator()->evaluateInt() ==
                2);
            TORCH_INTERNAL_ASSERT(
                mapper.getMappedExtent(axis).getDenominator()->evaluateInt() ==
                1);
          }
        }
      }
    }
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
