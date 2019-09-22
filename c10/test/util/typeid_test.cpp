#include "c10/util/typeid.h"
#include <gtest/gtest.h>

using std::string;

namespace caffe2 {
namespace {

class TypeMetaTestFoo {};
class TypeMetaTestBar {};
}

CAFFE_KNOWN_TYPE(TypeMetaTestFoo);
CAFFE_KNOWN_TYPE(TypeMetaTestBar);

namespace {

TEST(TypeMetaTest, TypeMetaStatic) {
  EXPECT_EQ(TypeMeta::ItemSize<int>(), sizeof(int));
  EXPECT_EQ(TypeMeta::ItemSize<float>(), sizeof(float));
  EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestFoo>(), sizeof(TypeMetaTestFoo));
  EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestBar>(), sizeof(TypeMetaTestBar));
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<float>());
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<TypeMetaTestFoo>());
  EXPECT_NE(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestBar>());
  EXPECT_EQ(TypeMeta::Id<int>(), TypeMeta::Id<int>());
  EXPECT_EQ(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestFoo>());
}

TEST(TypeMetaTest, Names) {
  TypeMeta null_meta;
  EXPECT_EQ("nullptr (uninitialized)", null_meta.name());
  TypeMeta int_meta = TypeMeta::Make<int>();
  EXPECT_EQ("int", int_meta.name());
  TypeMeta string_meta = TypeMeta::Make<string>();
  EXPECT_TRUE(c10::string_view::npos != string_meta.name().find("string"));
}

TEST(TypeMetaTest, TypeMeta) {
  TypeMeta int_meta = TypeMeta::Make<int>();
  TypeMeta float_meta = TypeMeta::Make<float>();
  TypeMeta foo_meta = TypeMeta::Make<TypeMetaTestFoo>();
  TypeMeta bar_meta = TypeMeta::Make<TypeMetaTestBar>();

  TypeMeta another_int_meta = TypeMeta::Make<int>();
  TypeMeta another_foo_meta = TypeMeta::Make<TypeMetaTestFoo>();

  EXPECT_EQ(int_meta, another_int_meta);
  EXPECT_EQ(foo_meta, another_foo_meta);
  EXPECT_NE(int_meta, float_meta);
  EXPECT_NE(int_meta, foo_meta);
  EXPECT_NE(foo_meta, bar_meta);
  EXPECT_TRUE(int_meta.Match<int>());
  EXPECT_TRUE(foo_meta.Match<TypeMetaTestFoo>());
  EXPECT_FALSE(int_meta.Match<float>());
  EXPECT_FALSE(int_meta.Match<TypeMetaTestFoo>());
  EXPECT_FALSE(foo_meta.Match<int>());
  EXPECT_FALSE(foo_meta.Match<TypeMetaTestBar>());
  EXPECT_EQ(int_meta.id(), TypeMeta::Id<int>());
  EXPECT_EQ(float_meta.id(), TypeMeta::Id<float>());
  EXPECT_EQ(foo_meta.id(), TypeMeta::Id<TypeMetaTestFoo>());
  EXPECT_EQ(bar_meta.id(), TypeMeta::Id<TypeMetaTestBar>());
  EXPECT_EQ(int_meta.itemsize(), TypeMeta::ItemSize<int>());
  EXPECT_EQ(float_meta.itemsize(), TypeMeta::ItemSize<float>());
  EXPECT_EQ(foo_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestFoo>());
  EXPECT_EQ(bar_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestBar>());
  EXPECT_EQ(int_meta.name(), "int");
  EXPECT_EQ(float_meta.name(), "float");
  EXPECT_NE(foo_meta.name().find("TypeMetaTestFoo"), c10::string_view::npos);
  EXPECT_NE(bar_meta.name().find("TypeMetaTestBar"), c10::string_view::npos);
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
}

CAFFE_KNOWN_TYPE(ClassAllowAssignment);
CAFFE_KNOWN_TYPE(ClassNoAssignment);

namespace {

TEST(TypeMetaTest, CtorDtorAndCopy) {
  TypeMeta fundamental_meta = TypeMeta::Make<int>();
  EXPECT_EQ(fundamental_meta.placementNew(), nullptr);
  EXPECT_EQ(fundamental_meta.placementDelete(), nullptr);
  EXPECT_EQ(fundamental_meta.copy(), nullptr);

  TypeMeta meta_a = TypeMeta::Make<ClassAllowAssignment>();
  EXPECT_TRUE(meta_a.placementNew() != nullptr);
  EXPECT_TRUE(meta_a.placementDelete() != nullptr);
  EXPECT_TRUE(meta_a.copy() != nullptr);
  ClassAllowAssignment src;
  src.x = 10;
  ClassAllowAssignment dst;
  EXPECT_EQ(dst.x, 42);
  meta_a.copy()(&src, &dst, 1);
  EXPECT_EQ(dst.x, 10);

  TypeMeta meta_b = TypeMeta::Make<ClassNoAssignment>();

  EXPECT_TRUE(meta_b.placementNew() != nullptr);
  EXPECT_TRUE(meta_b.placementDelete() != nullptr);
#ifndef __clang__
  // gtest seems to have some problem with function pointers and
  // clang right now... Disabling it.
  // TODO: figure out the real cause.
  EXPECT_EQ(meta_b.copy(), &(detail::_CopyNotAllowed<ClassNoAssignment>));
#endif
}

TEST(TypeMetaTest, Float16IsNotUint16) {
  EXPECT_NE(TypeMeta::Id<uint16_t>(), TypeMeta::Id<at::Half>());
}

}  // namespace
}  // namespace caffe2
