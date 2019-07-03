#include <c10/util/intrusive_ptr.h>

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

using c10::intrusive_ptr;
using c10::intrusive_ptr_target;
using c10::make_intrusive;

#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wself-move"
#endif

namespace {
class SomeClass0Parameters : public intrusive_ptr_target {};
class SomeClass1Parameter : public intrusive_ptr_target {
 public:
  SomeClass1Parameter(int param_) : param(param_) {}
  int param;
};
class SomeClass2Parameters : public intrusive_ptr_target {
 public:
  SomeClass2Parameters(int param1_, int param2_)
      : param1(param1_), param2(param2_) {}
  int param1;
  int param2;
};
using SomeClass = SomeClass0Parameters;
struct SomeBaseClass : public intrusive_ptr_target {
  SomeBaseClass(int v_) : v(v_) {}
  int v;
};
struct SomeChildClass : SomeBaseClass {
  SomeChildClass(int v) : SomeBaseClass(v) {}
};

class DestructableMock : public intrusive_ptr_target {
 public:
  DestructableMock(bool* resourcesReleased, bool* wasDestructed)
      : resourcesReleased_(resourcesReleased), wasDestructed_(wasDestructed) {}

  ~DestructableMock() override {
    *wasDestructed_ = true;
  }

  void release_resources() override {
    *resourcesReleased_ = true;
  }

 private:
  bool* resourcesReleased_;
  bool* wasDestructed_;
};

class ChildDestructableMock final : public DestructableMock {
 public:
  ChildDestructableMock(bool* resourcesReleased, bool* wasDestructed)
      : DestructableMock(resourcesReleased, wasDestructed) {}
};
class NullType1 final {
  static SomeClass singleton_;
public:
  static constexpr SomeClass* singleton() {
    return &singleton_;
  }
};
SomeClass NullType1::singleton_;
class NullType2 final {
  static SomeClass singleton_;
public:
  static constexpr SomeClass* singleton() {
    return &singleton_;
  }
};
SomeClass NullType2::singleton_;
static_assert(NullType1::singleton() != NullType2::singleton(), "");
} // namespace

static_assert(
    std::is_same<SomeClass, intrusive_ptr<SomeClass>::element_type>::value,
    "intrusive_ptr<T>::element_type is wrong");

TEST(MakeIntrusiveTest, ClassWith0Parameters) {
  intrusive_ptr<SomeClass0Parameters> var =
      make_intrusive<SomeClass0Parameters>();
  // Check that the type is correct
  EXPECT_EQ(var.get(), dynamic_cast<SomeClass0Parameters*>(var.get()));
}

TEST(MakeIntrusiveTest, ClassWith1Parameter) {
  intrusive_ptr<SomeClass1Parameter> var =
      make_intrusive<SomeClass1Parameter>(5);
  EXPECT_EQ(5, var->param);
}

TEST(MakeIntrusiveTest, ClassWith2Parameters) {
  intrusive_ptr<SomeClass2Parameters> var =
      make_intrusive<SomeClass2Parameters>(7, 2);
  EXPECT_EQ(7, var->param1);
  EXPECT_EQ(2, var->param2);
}

TEST(MakeIntrusiveTest, TypeIsAutoDeductible) {
  auto var2 = make_intrusive<SomeClass0Parameters>();
  auto var3 = make_intrusive<SomeClass1Parameter>(2);
  auto var4 = make_intrusive<SomeClass2Parameters>(2, 3);
}

TEST(MakeIntrusiveTest, CanAssignToBaseClassPtr) {
  intrusive_ptr<SomeBaseClass> var = make_intrusive<SomeChildClass>(3);
  EXPECT_EQ(3, var->v);
}

TEST(IntrusivePtrTargetTest, whenAllocatedOnStack_thenDoesntCrash) {
  SomeClass myClass;
}

TEST(IntrusivePtrTest, givenValidPtr_whenCallingGet_thenReturnsObject) {
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  EXPECT_EQ(5, obj.get()->param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenCallingConstGet_thenReturnsObject) {
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  EXPECT_EQ(5, obj.get()->param);
}

TEST(IntrusivePtrTest, givenInvalidPtr_whenCallingGet_thenReturnsNullptr) {
  intrusive_ptr<SomeClass1Parameter> obj;
  EXPECT_EQ(nullptr, obj.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenDereferencing_thenReturnsObject) {
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  EXPECT_EQ(5, (*obj).param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenConstDereferencing_thenReturnsObject) {
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  EXPECT_EQ(5, (*obj).param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenArrowDereferencing_thenReturnsObject) {
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(3);
  EXPECT_EQ(3, obj->param);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenConstArrowDereferencing_thenReturnsObject) {
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(3);
  EXPECT_EQ(3, obj->param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigning_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  obj1 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj1.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigningToSelf_thenStaysValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  obj1 = std::move(obj1);
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid) {
  intrusive_ptr<SomeClass> obj1;
  obj1 = std::move(obj1);
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  SomeClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  SomeClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj2.defined());
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(1);
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  SomeBaseClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj2.get());
  EXPECT_EQ(1, obj2->v);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(1);
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  intrusive_ptr<SomeBaseClass> obj2;
  SomeBaseClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  intrusive_ptr<SomeBaseClass> obj2;
  SomeBaseClass* obj1ptr = obj1.get();
  obj2 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj2.get());
  EXPECT_EQ(5, obj2->v);
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid) {
  intrusive_ptr<SomeChildClass> obj1;
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  EXPECT_TRUE(obj2.defined());
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr) {
  intrusive_ptr<SomeClass, NullType1> obj1;
  intrusive_ptr<SomeClass, NullType2> obj2;
  obj2 = std::move(obj1);
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigning_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  obj2 = obj1;
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigning_thenOldInstanceValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = obj1;
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  obj1 = obj1;
  EXPECT_EQ(obj1ptr, obj1.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigningToSelf_thenStaysValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  obj1 = obj1;
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid) {
  intrusive_ptr<SomeClass> obj1;
  obj1 = obj1;
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  SomeClass* obj1ptr = obj1.get();
  obj2 = obj1;
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  intrusive_ptr<SomeBaseClass> base = make_intrusive<SomeBaseClass>(10);
  base = child;
  EXPECT_EQ(3, base->v);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(3);
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(10);
  obj2 = obj1;
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  intrusive_ptr<SomeBaseClass> obj2;
  SomeBaseClass* obj1ptr = obj1.get();
  obj2 = obj1;
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  intrusive_ptr<SomeBaseClass> obj2;
  SomeBaseClass* obj1ptr = obj1.get();
  obj2 = obj1;
  EXPECT_EQ(obj1ptr, obj2.get());
  EXPECT_EQ(5, obj2->v);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid) {
  intrusive_ptr<SomeChildClass> obj1;
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  EXPECT_TRUE(obj2.defined());
  obj2 = obj1;
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr) {
  intrusive_ptr<SomeClass, NullType1> obj1;
  intrusive_ptr<SomeClass, NullType2> obj2;
  obj2 = obj1;
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenOldInstanceInvalid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  EXPECT_FALSE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenNewInstanceValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  SomeBaseClass* objptr = child.get();
  intrusive_ptr<SomeBaseClass> base = std::move(child);
  EXPECT_EQ(3, base->v);
  EXPECT_EQ(objptr, base.get());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  intrusive_ptr<SomeBaseClass> base = std::move(child);
  EXPECT_FALSE(child.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid) {
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(2);
  intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  intrusive_ptr<SomeChildClass> obj1;
  intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr) {
  intrusive_ptr<SomeClass, NullType1> obj1;
  intrusive_ptr<SomeClass, NullType2> obj2 = std::move(obj1);
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenPointsToSameObject) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  intrusive_ptr<SomeClass> obj2 = obj1;
  EXPECT_EQ(obj1ptr, obj2.get());
  EXPECT_TRUE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenOldInstanceValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj1;
  EXPECT_TRUE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenNewInstanceValid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj1;
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = obj1;
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  SomeBaseClass* objptr = child.get();
  intrusive_ptr<SomeBaseClass> base = child;
  EXPECT_EQ(3, base->v);
  EXPECT_EQ(objptr, base.get());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  intrusive_ptr<SomeBaseClass> base = child;
  EXPECT_TRUE(child.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid) {
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  intrusive_ptr<SomeBaseClass> base = child;
  EXPECT_TRUE(base.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  intrusive_ptr<SomeChildClass> obj1;
  intrusive_ptr<SomeBaseClass> obj2 = obj1;
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr) {
  intrusive_ptr<SomeClass, NullType1> obj1;
  intrusive_ptr<SomeClass, NullType2> obj2 = obj1;
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapFunction) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  SomeClass* obj2ptr = obj2.get();
  swap(obj1, obj2);
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapMethod) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj1ptr = obj1.get();
  SomeClass* obj2ptr = obj2.get();
  obj1.swap(obj2);
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapFunctionFromInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj2ptr = obj2.get();
  swap(obj1, obj2);
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_TRUE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapMethodFromInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  SomeClass* obj2ptr = obj2.get();
  obj1.swap(obj2);
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_TRUE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapFunctionWithInvalid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  SomeClass* obj1ptr = obj1.get();
  swap(obj1, obj2);
  EXPECT_FALSE(obj1.defined());
  EXPECT_TRUE(obj2.defined());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapMethodWithInvalid) {
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  SomeClass* obj1ptr = obj1.get();
  obj1.swap(obj2);
  EXPECT_FALSE(obj1.defined());
  EXPECT_TRUE(obj2.defined());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapFunctionInvalidWithInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2;
  swap(obj1, obj2);
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapMethodInvalidWithInvalid) {
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2;
  obj1.swap(obj2);
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, CanBePutInContainer) {
  std::vector<intrusive_ptr<SomeClass1Parameter>> vec;
  vec.push_back(make_intrusive<SomeClass1Parameter>(5));
  EXPECT_EQ(5, vec[0]->param);
}

TEST(IntrusivePtrTest, CanBePutInSet) {
  std::set<intrusive_ptr<SomeClass1Parameter>> set;
  set.insert(make_intrusive<SomeClass1Parameter>(5));
  EXPECT_EQ(5, (*set.begin())->param);
}

TEST(IntrusivePtrTest, CanBePutInUnorderedSet) {
  std::unordered_set<intrusive_ptr<SomeClass1Parameter>> set;
  set.insert(make_intrusive<SomeClass1Parameter>(5));
  EXPECT_EQ(5, (*set.begin())->param);
}

TEST(IntrusivePtrTest, CanBePutInMap) {
  std::map<
      intrusive_ptr<SomeClass1Parameter>,
      intrusive_ptr<SomeClass1Parameter>>
      map;
  map.insert(std::make_pair(
      make_intrusive<SomeClass1Parameter>(5),
      make_intrusive<SomeClass1Parameter>(3)));
  EXPECT_EQ(5, map.begin()->first->param);
  EXPECT_EQ(3, map.begin()->second->param);
}

TEST(IntrusivePtrTest, CanBePutInUnorderedMap) {
  std::unordered_map<
      intrusive_ptr<SomeClass1Parameter>,
      intrusive_ptr<SomeClass1Parameter>>
      map;
  map.insert(std::make_pair(
      make_intrusive<SomeClass1Parameter>(3),
      make_intrusive<SomeClass1Parameter>(5)));
  EXPECT_EQ(3, map.begin()->first->param);
  EXPECT_EQ(5, map.begin()->second->param);
}

TEST(IntrusivePtrTest, Equality_AfterCopyConstructor) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = var1;
  EXPECT_TRUE(var1 == var2);
  EXPECT_FALSE(var1 != var2);
}

TEST(IntrusivePtrTest, Equality_AfterCopyAssignment) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  var2 = var1;
  EXPECT_TRUE(var1 == var2);
  EXPECT_FALSE(var1 != var2);
}

TEST(IntrusivePtrTest, Equality_Nullptr) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  EXPECT_TRUE(var1 == var2);
  EXPECT_FALSE(var1 != var2);
}

TEST(IntrusivePtrTest, Nonequality) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(var1 != var2);
  EXPECT_FALSE(var1 == var2);
}

TEST(IntrusivePtrTest, Nonequality_NullptrLeft) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(var1 != var2);
  EXPECT_FALSE(var1 == var2);
}

TEST(IntrusivePtrTest, Nonequality_NullptrRight) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2;
  EXPECT_TRUE(var1 != var2);
  EXPECT_FALSE(var1 == var2);
}

TEST(IntrusivePtrTest, HashIsDifferent) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_NE(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, HashIsDifferent_ValidAndInvalid) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_NE(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, HashIsSame_AfterCopyConstructor) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = var1;
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, HashIsSame_AfterCopyAssignment) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  var2 = var1;
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, HashIsSame_BothNullptr) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, OneIsLess) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(
      std::less<intrusive_ptr<SomeClass>>()(var1, var2) !=
      std::less<intrusive_ptr<SomeClass>>()(var2, var1));
}

TEST(IntrusivePtrTest, NullptrIsLess1) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, NullptrIsLess2) {
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2;
  EXPECT_FALSE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, NullptrIsNotLessThanNullptr) {
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  EXPECT_FALSE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenIsInvalid) {
  auto obj = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj.defined());
  obj.reset();
  EXPECT_FALSE(obj.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenHoldsNullptr) {
  auto obj = make_intrusive<SomeClass>();
  EXPECT_NE(nullptr, obj.get());
  obj.reset();
  EXPECT_EQ(nullptr, obj.get());
}

TEST(IntrusivePtrTest, givenPtr_whenDestructed_thenDestructsObject) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto obj2 = std::move(obj);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    intrusive_ptr<DestructableMock> obj2 = std::move(obj);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(IntrusivePtrTest, givenPtr_whenMoveAssigned_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj2 = std::move(obj);
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj2 = std::move(obj);
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      auto copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = std::move(obj);
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 =
        make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = std::move(obj);
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = std::move(obj);
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto obj2 = make_intrusive<DestructableMock>(&dummy, &dummy);
    obj2 = std::move(obj);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto obj2 = make_intrusive<DestructableMock>(&dummy, &dummy);
    obj2 = std::move(obj);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    auto obj = make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    intrusive_ptr<DestructableMock> copy = obj;
    obj.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    auto obj = make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    intrusive_ptr<DestructableMock> copy = obj;
    obj.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy =
          make_intrusive<DestructableMock>(&dummy, &dummy);
      copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    auto obj = make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy =
          make_intrusive<DestructableMock>(&dummy, &dummy);
      copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    auto copy = make_intrusive<DestructableMock>(&dummy, &dummy);
    {
      auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
      copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool wasDestructed = false;
  bool resourcesReleased = false;
  bool dummy = false;
  {
    auto copy = make_intrusive<DestructableMock>(&dummy, &dummy);
    {
      auto obj =
          make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
      copy = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(IntrusivePtrTest, givenPtr_whenCopyAssigned_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj2 = obj;
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj2 = obj;
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      auto copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 =
        make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj2;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenDestructs) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  EXPECT_FALSE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  obj.reset();
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto copy = obj;
    obj.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    copy.reset();
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto copy = obj;
    copy.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj.reset();
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto moved = std::move(obj);
    obj.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    moved.reset();
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto moved = std::move(obj);
    moved.reset();
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(IntrusivePtrTest, AllowsMoveConstructingToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();
  intrusive_ptr<const SomeClass> b = std::move(a);
}

TEST(IntrusivePtrTest, AllowsCopyConstructingToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();
  intrusive_ptr<const SomeClass> b = a;
}

TEST(IntrusivePtrTest, AllowsMoveAssigningToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();
  intrusive_ptr<const SomeClass> b = make_intrusive<SomeClass>();
  b = std::move(a);
}

TEST(IntrusivePtrTest, AllowsCopyAssigningToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();
  intrusive_ptr<const SomeClass> b = make_intrusive<const SomeClass>();
  b = a;
}

TEST(IntrusivePtrTest, givenNewPtr_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  EXPECT_EQ(1, obj.use_count());
}

TEST(IntrusivePtrTest, givenNewPtr_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj.unique());
}

TEST(IntrusivePtrTest, givenEmptyPtr_thenHasUseCount0) {
  intrusive_ptr<SomeClass> obj;
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenEmptyPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj;
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenResetPtr_thenHasUseCount0) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  obj.reset();
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenResetPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  obj.reset();
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenOldHasUseCount0) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenOldIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = std::move(obj);
  EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = std::move(obj);
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenOldHasUseCount0) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = std::move(obj);
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenOldIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = std::move(obj);
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenHasUseCount2) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_EQ(2, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_FALSE(obj2.unique());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenOldHasUseCount2) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_EQ(2, obj.use_count());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenOldIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_FALSE(obj.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenDestructingCopy_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    intrusive_ptr<SomeClass> obj2 = obj;
    EXPECT_EQ(2, obj.use_count());
  }
  EXPECT_EQ(1, obj.use_count());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenDestructingCopy_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    intrusive_ptr<SomeClass> obj2 = obj;
    EXPECT_FALSE(obj.unique());
  }
  EXPECT_TRUE(obj.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenReassigningCopy_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_EQ(2, obj.use_count());
  obj2 = make_intrusive<SomeClass>();
  EXPECT_EQ(1, obj.use_count());
  EXPECT_EQ(1, obj2.use_count());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenReassigningCopy_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = obj;
  EXPECT_FALSE(obj.unique());
  obj2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj.unique());
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_thenHasUseCount2) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = obj;
  EXPECT_EQ(2, obj.use_count());
  EXPECT_EQ(2, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = obj;
  EXPECT_FALSE(obj.unique());
  EXPECT_FALSE(obj2.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyAssignedPtr_whenDestructingCopy_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
    obj2 = obj;
    EXPECT_EQ(2, obj.use_count());
  }
  EXPECT_EQ(1, obj.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_whenDestructingCopy_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
    obj2 = obj;
    EXPECT_FALSE(obj.unique());
  }
  EXPECT_TRUE(obj.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyAssignedPtr_whenReassigningCopy_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = obj;
  EXPECT_EQ(2, obj.use_count());
  obj2 = make_intrusive<SomeClass>();
  EXPECT_EQ(1, obj.use_count());
  EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_whenReassigningCopy_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  obj2 = obj;
  EXPECT_FALSE(obj.unique());
  obj2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj.unique());
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenPtr_whenReleasedAndReclaimed_thenDoesntCrash) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  SomeClass* ptr = obj.release();
  EXPECT_FALSE(obj.defined());
  intrusive_ptr<SomeClass> reclaimed = intrusive_ptr<SomeClass>::reclaim(ptr);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    intrusive_ptr<DestructableMock> outer;
    {
      intrusive_ptr<DestructableMock> inner =
          make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
      DestructableMock* ptr = inner.release();
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      outer = intrusive_ptr<DestructableMock>::reclaim(ptr);
    }
    // inner is destructed
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // outer is destructed
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(IntrusivePtrTest, givenStackObject_whenReclaimed_thenCrashes) {
  // This would cause very weird bugs on destruction.
  // Better to crash early on creation.
  SomeClass obj;
  intrusive_ptr<SomeClass> ptr;
  EXPECT_ANY_THROW(ptr = intrusive_ptr<SomeClass>::reclaim(&obj));
}

TEST(IntrusivePtrTest, givenPtr_whenNonOwningReclaimed_thenDoesntCrash) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  SomeClass* raw_ptr = obj.get();
  EXPECT_TRUE(obj.defined());
  intrusive_ptr<SomeClass> reclaimed =
      intrusive_ptr<SomeClass>::unsafe_reclaim_from_nonowning(raw_ptr);
  EXPECT_TRUE(reclaimed.defined());
  EXPECT_EQ(reclaimed.get(), obj.get());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenNonOwningReclaimed_thenIsDestructedAtEnd) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    intrusive_ptr<DestructableMock> outer;
    {
      intrusive_ptr<DestructableMock> inner =
          make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
      DestructableMock* raw_ptr = inner.get();
      outer = intrusive_ptr<DestructableMock>::unsafe_reclaim_from_nonowning(
          raw_ptr);
    }
    // inner is destructed
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // outer is destructed
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

