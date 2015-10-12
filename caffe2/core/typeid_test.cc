#include "caffe2/core/typeid.h"
#include "gtest/gtest.h"

namespace caffe2 {
namespace {

class Foo {};
class Bar {};

TEST(TypeMetaTest, TypeMetaStatic) {
  EXPECT_EQ(TypeMeta::ItemSize<int>(), sizeof(int));
  EXPECT_EQ(TypeMeta::ItemSize<float>(), sizeof(float));
  EXPECT_EQ(TypeMeta::ItemSize<Foo>(), sizeof(Foo));
  EXPECT_EQ(TypeMeta::ItemSize<Bar>(), sizeof(Bar));
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<float>());
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<Foo>());
  EXPECT_NE(TypeMeta::Id<Foo>(), TypeMeta::Id<Bar>());
  EXPECT_EQ(TypeMeta::Id<int>(), TypeMeta::Id<int>());
  EXPECT_EQ(TypeMeta::Id<Foo>(), TypeMeta::Id<Foo>());
}

TEST(TypeMetaTest, TypeMeta) {
  TypeMeta int_meta = TypeMeta::Make<int>();
  TypeMeta float_meta = TypeMeta::Make<float>();
  TypeMeta foo_meta = TypeMeta::Make<Foo>();
  TypeMeta bar_meta = TypeMeta::Make<Bar>();

  TypeMeta another_int_meta = TypeMeta::Make<int>();
  TypeMeta another_foo_meta = TypeMeta::Make<Foo>();

  EXPECT_EQ(int_meta, another_int_meta);
  EXPECT_EQ(foo_meta, another_foo_meta);
  EXPECT_NE(int_meta, float_meta);
  EXPECT_NE(int_meta, foo_meta);
  EXPECT_NE(foo_meta, bar_meta);
  EXPECT_TRUE(int_meta.Match<int>());
  EXPECT_TRUE(foo_meta.Match<Foo>());
  EXPECT_FALSE(int_meta.Match<float>());
  EXPECT_FALSE(int_meta.Match<Foo>());
  EXPECT_FALSE(foo_meta.Match<int>());
  EXPECT_FALSE(foo_meta.Match<Bar>());
  EXPECT_EQ(int_meta.id(), TypeMeta::Id<int>());
  EXPECT_EQ(float_meta.id(), TypeMeta::Id<float>());
  EXPECT_EQ(foo_meta.id(), TypeMeta::Id<Foo>());
  EXPECT_EQ(bar_meta.id(), TypeMeta::Id<Bar>());
  EXPECT_EQ(int_meta.itemsize(), TypeMeta::ItemSize<int>());
  EXPECT_EQ(float_meta.itemsize(), TypeMeta::ItemSize<float>());
  EXPECT_EQ(foo_meta.itemsize(), TypeMeta::ItemSize<Foo>());
  EXPECT_EQ(bar_meta.itemsize(), TypeMeta::ItemSize<Bar>());
  EXPECT_EQ(int_meta.name(), TypeMeta::Name<int>());
  EXPECT_EQ(float_meta.name(), TypeMeta::Name<float>());
  EXPECT_EQ(foo_meta.name(), TypeMeta::Name<Foo>());
  EXPECT_EQ(bar_meta.name(), TypeMeta::Name<Bar>());
}

}  // namespace
}  // namespace caffe2