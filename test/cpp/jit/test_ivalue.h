#pragma once

#include <ATen/ATen.h>
#include "ATen/core/ivalue.h"
#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

namespace torch {
namespace jit {
namespace {

using Var = SymbolicVariable;

using namespace torch::autograd;

void testIValue() {
  Shared<IntList> foo = IntList::create({3, 4, 5});
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
  ASSERT_TRUE(ArrayRef<int64_t>(baz.toIntList()->elements()).equals({3, 4, 5}));

  auto move_it = std::move(baz).toIntList();
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.isNone());
  IValue i(4);
  ASSERT_TRUE(i.isInt());
  ASSERT_EQ(i.toInt(), 4);
  IValue dlist(DoubleList::create({3.5}));
  ASSERT_TRUE(dlist.isDoubleList());
  ASSERT_TRUE(ArrayRef<double>(std::move(dlist).toDoubleList()->elements())
                  .equals({3.5}));
  ASSERT_TRUE(dlist.isNone());
  dlist = IValue(DoubleList::create({3.4}));
  ASSERT_TRUE(ArrayRef<double>(dlist.toDoubleList()->elements()).equals({3.4}));
  IValue the_list(Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
  ASSERT_EQ(foo.use_count(), 3);
  ASSERT_TRUE(the_list.isTuple());
  auto first = std::move(the_list).toTuple()->elements().at(1);
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

} // namespace
} // namespace jit
} // namespace torch
