#include "caffe2/core/typeid.h"
#include "caffe2/core/types.h"
#include <gtest/gtest.h>

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
  EXPECT_TRUE(string(null_meta.name()) == "nullptr (uninitialized)");
  TypeMeta int_meta = TypeMeta::Make<int>();
  EXPECT_TRUE(string(int_meta.name()) == "int");
#ifdef __GXX_RTTI
  TypeMeta string_meta = TypeMeta::Make<string>();
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
  EXPECT_STREQ(int_meta.name(), "int");
  EXPECT_STREQ(float_meta.name(), "float");
#ifdef __GXX_RTTI
  EXPECT_NE(
      std::string(foo_meta.name()).find("TypeMetaTestFoo"), std::string::npos);
  EXPECT_NE(
      std::string(bar_meta.name()).find("TypeMetaTestBar"), std::string::npos);
#endif
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
#ifndef __clang__
  // gtest seems to have some problem with function pointers and
  // clang right now... Disabling it.
  // TODO: figure out the real cause.
  EXPECT_EQ(meta_b.copy(),
            &(TypeMeta::_CopyNotAllowed<ClassNoAssignment>));
#endif
}

TEST(TypeMetaTest, Float16IsNotUint16) {
  EXPECT_NE(TypeMeta::Id<uint16_t>(), TypeMeta::Id<float16>());
}

}  // namespace
}  // namespace caffe2
