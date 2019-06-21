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
  c10::List<int64_t> foo = c10::make_list<int64_t>({3, 4, 5});
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
  ASSERT_TRUE(baz.toIntListRef().equals({3, 4, 5}));

  auto move_it = std::move(baz).toIntList();
  ASSERT_EQ(foo.use_count(), 2);
  ASSERT_TRUE(baz.isNone());
  IValue i(4);
  ASSERT_TRUE(i.isInt());
  ASSERT_EQ(i.toInt(), 4);
  IValue dlist(c10::make_list<double>({3.5}));
  ASSERT_TRUE(dlist.isDoubleList());
  ASSERT_TRUE(dlist.toDoubleListRef()
                  .equals({3.5}));
  std::move(dlist).toDoubleList();
  ASSERT_TRUE(dlist.isNone());
  dlist = IValue(c10::make_list<double>({3.4}));
  ASSERT_TRUE(dlist.toDoubleListRef().equals({3.4}));
  IValue the_list(ivalue::Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
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

} // namespace
} // namespace jit
} // namespace torch
