#include <torch/jit.h>
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/runtime/argument_spec.h"

namespace torch {
namespace jit {

int device(const autograd::Variable& v) {
  return v.device().is_cuda() ? v.get_device() : -1;
}

bool isEqual(at::IntArrayRef lhs, at::IntArrayRef rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const CompleteArgumentInfo& ti, const autograd::Variable& v) {
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && isEqual(ti.sizes(), v.sizes()) &&
      isEqual(ti.strides(), v.strides());
}

bool isEqual(const ArgumentInfo& ti, const autograd::Variable& v) {
  if (!ti.defined())
    return ti.defined() == v.defined();
  return ti.device() == device(v) && ti.requires_grad() == v.requires_grad() &&
      ti.type() == v.scalar_type() && ti.dim() == v.dim();
}

autograd::Variable var(at::TensorOptions t, at::IntArrayRef sizes, bool requires_grad) {
  return autograd::make_variable(at::rand(sizes, t), requires_grad);
}
autograd::Variable undef() {
  return autograd::Variable();
}

void testCompleteArgumentSpec() {
  auto const CF = at::CPU(at::kFloat);
  auto const CD = at::CPU(at::kDouble);
  auto const GF = at::CUDA(at::kFloat);
  auto const GD = at::CUDA(at::kDouble);

  auto list = createStack({var(CF, {1}, true),
                           var(CD, {1, 2}, false),
                           var(GF, {}, true),
                           var(GD, {4, 5, 6}, false),
                           undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack({var(CF, {1}, true),
                            var(CD, {1, 2}, false),
                            var(GF, {}, true),
                            var(GD, {4, 5, 6}, false),
                            undef()});
  list2[1].toTensor().transpose_(0, 1);

  CompleteArgumentSpec a(true, list);
  CompleteArgumentSpec b(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  CompleteArgumentSpec d(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    ASSERT_TRUE(isEqual(a.at(i), list[i].toTensor()));
  }
  CompleteArgumentSpec no_grad(/*with_grad=*/false, list);
  ASSERT_TRUE(no_grad != a);

  std::unordered_set<CompleteArgumentSpec> spec;
  spec.insert(a); // we use a below, so no move
  ASSERT_TRUE(spec.count(b) > 0);
  ASSERT_EQ(spec.count(no_grad), 0);
  spec.insert(std::move(no_grad));
  ASSERT_EQ(spec.count(CompleteArgumentSpec(true, list)), 1);

  list2[1].toTensor().transpose_(0, 1);
  CompleteArgumentSpec c(true, list2); // same as list, except for one stride
  ASSERT_FALSE(c == a);
  ASSERT_EQ(spec.count(c), 0);

  Stack stack = {var(CF, {1, 2}, true), 3, var(CF, {1, 2}, true)};
  CompleteArgumentSpec with_const(true, stack);
  ASSERT_EQ(with_const.at(2).sizes().size(), 2);
}

size_t hashCode(const TensorTypePtr& ptr) {
  return std::hash<TensorType>()(*ptr.get());
}

void testProfiledTensorTypeHashing() {
  c10::VaryingShape vs(c10::optional<size_t>{});
  auto ptt_empty1 = TensorType::create({}, {}, vs, vs, false);
  auto ptt_empty2 = TensorType::create({}, {}, vs, vs, false);
  ASSERT_EQ(hashCode(ptt_empty1), hashCode(ptt_empty2));

  c10::VaryingShape vs22(std::vector<int64_t>{2, 2});
  auto ptt_vs22_1 = TensorType::create({}, {}, vs22, vs, false);
  auto ptt_vs22_2 = TensorType::create({}, {}, vs22, vs, false);
  ASSERT_EQ(hashCode(ptt_vs22_1), hashCode(ptt_vs22_2));

  c10::VaryingShape vs23(std::vector<int64_t>{2, 3});
  auto ptt_vs23_1 = TensorType::create({}, {}, vs23, vs, false);
  ASSERT_NE(hashCode(ptt_vs22_1), hashCode(ptt_vs23_1));

  auto ptt_vs22_vs22_1 = TensorType::create({}, {}, vs22, vs22, false);
  auto ptt_vs22_vs22_2 = TensorType::create({}, {}, vs22, vs22, false);
  ASSERT_EQ(hashCode(ptt_vs22_vs22_1), hashCode(ptt_vs22_vs22_2));

  auto ptt_vs22_vs23_2 = TensorType::create({}, {}, vs22, vs23, false);
  ASSERT_NE(hashCode(ptt_vs22_vs22_1), hashCode(ptt_vs22_vs23_2));

  auto ptt_vs22_vs22_1_true = TensorType::create({}, {}, vs22, vs22, true);
  auto ptt_vs22_vs22_2_true = TensorType::create({}, {}, vs22, vs22, true);
  ASSERT_EQ(hashCode(ptt_vs22_vs22_1_true), hashCode(ptt_vs22_vs22_2_true));

  auto ptt_vs22_vs22_1_false = TensorType::create({}, {}, vs22, vs22, false);
  ASSERT_NE(hashCode(ptt_vs22_vs22_1_true), hashCode(ptt_vs22_vs22_1_false));
}

void testArgumentSpec() {
  auto& CF = at::CPU(at::kFloat);
  auto& CD = at::CPU(at::kDouble);
  auto& GF = at::CUDA(at::kFloat);
  auto& GD = at::CUDA(at::kDouble);

  auto graph = jit::compile(R"JIT(
   def fn(a, b, c, d, e):
      return a, b, c, d, e
   )JIT")
                   ->get_function("fn")
                   .graph();

  ArgumentSpecCreator arg_spec_creator(*graph);

  auto list = createStack({var(CF, {1}, true),
                           var(CD, {1, 2}, false),
                           var(GF, {}, true),
                           var(GD, {4, 5, 6}, false),
                           undef()});

  // make sure we have some non-standard strides
  list[1].toTensor().transpose_(0, 1);

  // same list but different backing values
  auto list2 = createStack({var(CF, {1}, true),
                            var(CD, {1, 2}, false),
                            var(GF, {}, true),
                            var(GD, {4, 5, 6}, false),
                            undef()});
  list2[1].toTensor().transpose_(0, 1);


  ArgumentSpec a = arg_spec_creator.create(true, list);
  ArgumentSpec b = arg_spec_creator.create(true, list);
  ASSERT_EQ(a.hashCode(), b.hashCode());

  ASSERT_EQ(a, b);
  ArgumentSpec d = arg_spec_creator.create(true, list2);
  ASSERT_EQ(d, a);
  ASSERT_EQ(d.hashCode(), a.hashCode());

  for (size_t i = 0; i < list.size(); ++i) {
    ASSERT_TRUE(isEqual(a.tensorAt(i), list[i].toTensor()));
  }
  ArgumentSpec no_grad = arg_spec_creator.create(/*with_grad=*/false, list);
  ASSERT_TRUE(no_grad != a);

  std::unordered_set<ArgumentSpec> spec;
  spec.insert(a); // we still need a for the test below
  ASSERT_TRUE(spec.count(b) > 0);
  ASSERT_EQ(spec.count(no_grad), 0);
  spec.insert(std::move(no_grad));
  ASSERT_EQ(spec.count(arg_spec_creator.create(true, list)), 1);

  list2[1].toTensor().transpose_(0, 1);
  ArgumentSpec c = arg_spec_creator.create(
      true, list2); // same as list, except for one stride, used to be
                    // different, now the same
  ASSERT_TRUE(c == a);
  ASSERT_EQ(spec.count(c), 1);
}

} // namespace jit
} // namespace torch
