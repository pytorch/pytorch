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

TEST(TypeMetaTest, Names) {
  TypeMeta null_meta;
  EXPECT_TRUE(string(null_meta.name()) == "Unknown Type");
#ifdef __GXX_RTTI
  TypeMeta int_meta = TypeMeta::Make<int>();
  TypeMeta string_meta = TypeMeta::Make<string>();
  EXPECT_TRUE(string(int_meta.name()) == "int");
  // For string, we should have a demangled name.
  EXPECT_TRUE(
      string(string_meta.name()) != typeid(string).name());
  EXPECT_TRUE(
      string(string_meta.name()) == Demangle(typeid(string).name()));
#endif  // __GXX_RTTI
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


class ClassAllowAssignment {
 public:
  ClassAllowAssignment() : x(42) {}
  ClassAllowAssignment(const ClassAllowAssignment& src) : x(src.x) {}
  ClassAllowAssignment& operator=(const ClassAllowAssignment& src) = default;
  int x;
};

class ClassNoAssignment {
 public:
  ClassNoAssignment() : x(42) {}
  ClassNoAssignment(const ClassNoAssignment& src) = delete;
  ClassNoAssignment& operator=(const ClassNoAssignment& src) = delete;
  int x;
};

TEST(TypeMetaTest, CtorDtorAndCopy) {
  TypeMeta fundamental_meta = TypeMeta::Make<int>();
  EXPECT_EQ(fundamental_meta.ctor(), nullptr);
  EXPECT_EQ(fundamental_meta.dtor(), nullptr);
  EXPECT_EQ(fundamental_meta.copy(), nullptr);

  TypeMeta meta_a = TypeMeta::Make<ClassAllowAssignment>();
  EXPECT_TRUE(meta_a.ctor() != nullptr);
  EXPECT_TRUE(meta_a.dtor() != nullptr);
  EXPECT_TRUE(meta_a.copy() != nullptr);
  ClassAllowAssignment src;
  src.x = 10;
  ClassAllowAssignment dst;
  EXPECT_EQ(dst.x, 42);
  meta_a.copy()(&src, &dst, 1);
  EXPECT_EQ(dst.x, 10);

  TypeMeta meta_b = TypeMeta::Make<ClassNoAssignment>();

  EXPECT_TRUE(meta_b.ctor() != nullptr);
  EXPECT_TRUE(meta_b.dtor() != nullptr);
  EXPECT_EQ(meta_b.copy(),
            (TypeMeta::_CopyNotAllowed<ClassNoAssignment>));
}

}  // namespace
}  // namespace caffe2
