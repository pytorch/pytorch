#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "ATen/core/ivalue.h"

namespace c10 {

TEST(IValueTest, Basic) {
  c10::List<int64_t> foo({3, 4, 5});
  ASSERT_EQ(foo.use_count(), 1);
  IValue bar{foo};
  ASSERT_EQ(foo.use_count(), 2);
  auto baz = bar;
  ASSERT_EQ(foo.use_count(), 3);
  auto foo2 = std::move(bar);
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(foo2.isIntList());
  ASSERT_TRUE(bar.isNone());
  foo2 = IValue(4.0);
  ASSERT_TRUE(foo2.isDouble());
  ASSERT_EQ(foo2.toDouble(), 4.0);
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.toIntVector() == std::vector<int64_t>({3, 4, 5}));

  auto move_it = std::move(baz).toIntList();
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.isNone());
  IValue i(4);
  ASSERT_TRUE(i.isInt());
  ASSERT_EQ(i.toInt(), 4);
  IValue dlist(c10::List<double>({3.5}));
  ASSERT_TRUE(dlist.isDoubleList());
  ASSERT_TRUE(dlist.toDoubleVector() == std::vector<double>({3.5}));
  std::move(dlist).toDoubleList();
  ASSERT_TRUE(dlist.isNone());
  dlist = IValue(c10::List<double>({3.4}));
  ASSERT_TRUE(dlist.toDoubleVector() == std::vector<double>({3.4}));
  IValue the_list(
      at::ivalue::Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(the_list.isTuple());
  auto first = the_list.toTuple()->elements()[1];
  ASSERT_EQ(first.toInt(), 4);
  at::Tensor tv = at::rand({3, 4});
  IValue ten(tv);
  ASSERT_EQ(tv.use_count(), 2);
  auto ten2 = ten;
  ASSERT_EQ(tv.use_count(), 3);
  ASSERT_TRUE(ten2.toTensor().equal(ten.toTensor()));
  std::move(ten2).toTensor();
  ASSERT_EQ(tv.use_count(), 2);
}

TEST(IValueTest, Tuple) {
  std::tuple<int64_t, at::Tensor> t = std::make_tuple(123, at::randn({1}));
  auto iv = IValue(t);
  auto t_ = iv.to<std::tuple<int64_t, at::Tensor>>();
  ASSERT_EQ(std::get<0>(t_), 123);
  ASSERT_EQ(
      std::get<1>(t_).item().to<float>(), std::get<1>(t).item().to<float>());
}

TEST(IValueTest, unsafeRemoveAttr) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addAttribute("attr1", TensorType::get());
  cls->addAttribute("attr2", TensorType::get());
  auto obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(cu, cls), cls->numAttributes());
  obj->unsafeRemoveAttr("attr1");
  // attr1 is not removed in the type
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(obj->slots().size() == 1);
}

TEST(IValueTest, TuplePrint) {
  {
    IValue tp = std::make_tuple(3);

    std::stringstream ss;
    ss << tp;
    ASSERT_EQ(ss.str(), "(3,)");
  }

  {
    IValue tp = std::make_tuple(3, 3);
    std::stringstream ss;
    ss << tp;
    ASSERT_EQ(ss.str(), "(3, 3)");
  }
}

TEST(IValueTest, BasicFuture) {
  auto f1 = c10::make_intrusive<ivalue::Future>(IntType::get());
  ASSERT_FALSE(f1->completed());

  f1->markCompleted(IValue(42));
  ASSERT_TRUE(f1->completed());
  ASSERT_EQ(42, f1->value().toInt());
  IValue iv(f1);
  ASSERT_EQ(42, iv.toFuture()->value().toInt());
}

TEST(IValueTest, FutureCallbacks) {
  auto f2 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimesA = 0;
  int calledTimesB = 0;
  f2->addCallback([f2, &calledTimesA]() {
    ASSERT_TRUE(f2->completed());
    ASSERT_EQ(f2->value().toInt(), 43);
    ++calledTimesA;
  });
  f2->markCompleted(IValue(43));
  ASSERT_EQ(calledTimesA, 1);
  ASSERT_EQ(calledTimesB, 0);
  // Post-markCompleted()
  f2->addCallback([f2, &calledTimesB]() {
    ASSERT_TRUE(f2->completed());
    ASSERT_EQ(f2->value().toInt(), 43);
    ++calledTimesB;
  });
  ASSERT_EQ(calledTimesA, 1);
  ASSERT_EQ(calledTimesB, 1);
}

TEST(IValueTest, FutureExceptions) {
  auto f3 = c10::make_intrusive<ivalue::Future>(IntType::get());
  int calledTimes = 0;
  f3->addCallback([f3, &calledTimes]() {
    ASSERT_TRUE(f3->completed());
    try {
      (void)f3->value();
    } catch (const std::exception& e) {
      if (std::string(e.what()) == "My Error") {
        ++calledTimes;
      }
    }
  });
  ivalue::Future::FutureError err("My Error");
  f3->markCompleted(std::move(err));
  ASSERT_EQ(calledTimes, 1);
}
} // namespace c10
