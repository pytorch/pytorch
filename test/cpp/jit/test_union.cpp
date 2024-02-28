#include <gtest/gtest.h>

#include <ATen/core/jit_type.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

class UnionTypeTest : public ::testing::Test {
 public:
  // None
  const TypePtr none = NoneType::get();

  // List[str]
  const TypePtr l1 = ListType::ofStrings();

  // Optional[int]
  const TypePtr opt1 = OptionalType::create(IntType::get());

  // Optional[float]
  const TypePtr opt2 = OptionalType::create(FloatType::get());

  // Optional[List[str]]
  const TypePtr opt3 = OptionalType::create(ListType::ofStrings());

  // Tuple[Optional[int], int]
  const TypePtr tup1 =
      TupleType::create({OptionalType::create(IntType::get()), IntType::get()});

  // Tuple[int, int]
  const TypePtr tup2 = TupleType::create({IntType::get(), IntType::get()});

  bool hasType(UnionTypePtr u, TypePtr t) {
    auto res = std::find(u->getTypes().begin(), u->getTypes().end(), t);
    return res != u->getTypes().end();
  }
};

TEST_F(UnionTypeTest, UnionOperatorEquals) {
  const UnionTypePtr u1 = UnionType::create({l1, tup2, StringType::get()});

  // Same thing, but using different TypePtrs
  const TypePtr l1_ = ListType::ofStrings();
  const TypePtr tup2_ = TupleType::create({IntType::get(), IntType::get()});
  const UnionTypePtr u2 = UnionType::create({l1_, tup2_, StringType::get()});

  ASSERT_TRUE(*u1 == *u2);
}

TEST_F(UnionTypeTest, UnionCreate_OptionalT1AndOptionalT2) {
  // Goal: Union[int, float, None]
  const UnionTypePtr u = UnionType::create({opt1, opt2});

  ASSERT_EQ(u->getTypes().size(), 3);
  ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, FloatType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));
}

TEST_F(UnionTypeTest, UnionCreate_OptionalTAndT) {
  // Goal: Union[int, None]
  const UnionTypePtr u = UnionType::create({opt1, IntType::get()});

  ASSERT_EQ(u->getTypes().size(), 2);
  ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));
}

TEST_F(UnionTypeTest, UnionCreate_TupleWithSubtypingRelationship) {
  // Goal: Union[Tuple[Optional[int], int], str]
  const UnionTypePtr u = UnionType::create({StringType::get(), tup1, tup2});

  ASSERT_EQ(u->getTypes().size(), 2);
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, tup1));
}

TEST_F(UnionTypeTest, UnionCreate_ContainerTAndT) {
  // Goal: Union[List[str], str]
  const UnionTypePtr u = UnionType::create({l1, StringType::get()});

  ASSERT_EQ(u->getTypes().size(), 2);
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));
}

TEST_F(UnionTypeTest, UnionCreate_OptionalContainerTAndContainerTAndT) {
  // Goal: Union[List[str], None, str]
  const UnionTypePtr u = UnionType::create({l1, opt3, StringType::get()});

  ASSERT_EQ(u->getTypes().size(), 3);
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
  ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));
}

TEST_F(UnionTypeTest, Subtyping_NumberType) {
  // Union[int, float, Complex]
  const UnionTypePtr union1 =
      UnionType::create({IntType::get(), FloatType::get(), ComplexType::get()});

  // Union[int, float, Complex, None]
  const UnionTypePtr union2 = UnionType::create(
      {IntType::get(), FloatType::get(), ComplexType::get(), NoneType::get()});

  const NumberTypePtr num = NumberType::get();

  ASSERT_TRUE(num->isSubtypeOf(*union1));
  ASSERT_TRUE(union1->isSubtypeOf(*num));
  ASSERT_TRUE(*num == *union1);

  ASSERT_TRUE(num->isSubtypeOf(*union2));
  ASSERT_FALSE(union2->isSubtypeOf(*num));
  ASSERT_FALSE(*num == *union2);
}

TEST_F(UnionTypeTest, Subtyping_OptionalType) {
  // Union[int, None]
  const UnionTypePtr union1 =
      UnionType::create({IntType::get(), NoneType::get()});

  // Union[int, str, None]
  const UnionTypePtr union2 =
      UnionType::create({IntType::get(), StringType::get(), NoneType::get()});

  // Union[int, str, List[str]]
  const UnionTypePtr union3 = UnionType::create(
      {IntType::get(), StringType::get(), ListType::ofStrings()});

  ASSERT_TRUE(none->isSubtypeOf(opt1));
  ASSERT_TRUE(none->isSubtypeOf(union1));
  ASSERT_TRUE(none->isSubtypeOf(union2));
  ASSERT_FALSE(none->isSubtypeOf(union3));

  ASSERT_FALSE(opt1->isSubtypeOf(none));
  ASSERT_TRUE(opt1->isSubtypeOf(union1));
  ASSERT_TRUE(opt1->isSubtypeOf(union2));
  ASSERT_FALSE(opt1->isSubtypeOf(union3));

  ASSERT_FALSE(union1->isSubtypeOf(none));
  ASSERT_TRUE(union1->isSubtypeOf(opt1));
  ASSERT_TRUE(union1->isSubtypeOf(union2));
  ASSERT_FALSE(union1->isSubtypeOf(union3));

  ASSERT_FALSE(union2->isSubtypeOf(union1));
}

} // namespace jit
} // namespace torch
