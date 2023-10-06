//===- llvm/unittest/ADT/SmallVectorTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SmallVector unit tests.
//
//===----------------------------------------------------------------------===//

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <gtest/gtest.h>
#include <cstdarg>
#include <list>

// NOLINTBEGIN(*arrays, bugprone-forwarding-reference-overload)
using c10::SmallVector;
using c10::SmallVectorImpl;

namespace {

/// A helper class that counts the total number of constructor and
/// destructor calls.
class Constructable {
 private:
  static int numConstructorCalls;
  static int numMoveConstructorCalls;
  static int numCopyConstructorCalls;
  static int numDestructorCalls;
  static int numAssignmentCalls;
  static int numMoveAssignmentCalls;
  static int numCopyAssignmentCalls;

  bool constructed;
  int value;

 public:
  Constructable() : constructed(true), value(0) {
    ++numConstructorCalls;
  }

  Constructable(int val) : constructed(true), value(val) {
    ++numConstructorCalls;
  }

  Constructable(const Constructable& src)
      : constructed(true), value(src.value) {
    ++numConstructorCalls;
    ++numCopyConstructorCalls;
  }

  Constructable(Constructable&& src) noexcept
      : constructed(true), value(src.value) {
    src.value = 0;
    ++numConstructorCalls;
    ++numMoveConstructorCalls;
  }

  ~Constructable() {
    EXPECT_TRUE(constructed);
    ++numDestructorCalls;
    constructed = false;
  }

  Constructable& operator=(const Constructable& src) {
    EXPECT_TRUE(constructed);
    value = src.value;
    ++numAssignmentCalls;
    ++numCopyAssignmentCalls;
    return *this;
  }

  Constructable& operator=(Constructable&& src) noexcept {
    EXPECT_TRUE(constructed);
    value = src.value;
    src.value = 0;
    ++numAssignmentCalls;
    ++numMoveAssignmentCalls;
    return *this;
  }

  int getValue() const {
    return abs(value);
  }

  static void reset() {
    numConstructorCalls = 0;
    numMoveConstructorCalls = 0;
    numCopyConstructorCalls = 0;
    numDestructorCalls = 0;
    numAssignmentCalls = 0;
    numMoveAssignmentCalls = 0;
    numCopyAssignmentCalls = 0;
  }

  static int getNumConstructorCalls() {
    return numConstructorCalls;
  }

  static int getNumMoveConstructorCalls() {
    return numMoveConstructorCalls;
  }

  static int getNumCopyConstructorCalls() {
    return numCopyConstructorCalls;
  }

  static int getNumDestructorCalls() {
    return numDestructorCalls;
  }

  static int getNumAssignmentCalls() {
    return numAssignmentCalls;
  }

  static int getNumMoveAssignmentCalls() {
    return numMoveAssignmentCalls;
  }

  static int getNumCopyAssignmentCalls() {
    return numCopyAssignmentCalls;
  }

  friend bool operator==(const Constructable& c0, const Constructable& c1) {
    return c0.getValue() == c1.getValue();
  }

  friend bool C10_UNUSED
  operator!=(const Constructable& c0, const Constructable& c1) {
    return c0.getValue() != c1.getValue();
  }
};

int Constructable::numConstructorCalls;
int Constructable::numCopyConstructorCalls;
int Constructable::numMoveConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;
int Constructable::numCopyAssignmentCalls;
int Constructable::numMoveAssignmentCalls;

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(NonCopyable&&) noexcept = default;
  NonCopyable& operator=(NonCopyable&&) noexcept = default;

  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

C10_USED void CompileTest() {
  SmallVector<NonCopyable, 0> V;
  V.resize(42);
}

class SmallVectorTestBase : public testing::Test {
 protected:
  void SetUp() override {
    Constructable::reset();
  }

  template <typename VectorT>
  void assertEmpty(VectorT& v) {
    // Size tests
    EXPECT_EQ(0u, v.size());
    EXPECT_TRUE(v.empty());

    // Iterator tests
    EXPECT_TRUE(v.begin() == v.end());
  }

  // Assert that v contains the specified values, in order.
  template <typename VectorT>
  void assertValuesInOrder(VectorT& v, size_t size, ...) {
    EXPECT_EQ(size, v.size());

    va_list ap;
    va_start(ap, size);
    for (size_t i = 0; i < size; ++i) {
      int value = va_arg(ap, int);
      EXPECT_EQ(value, v[i].getValue());
    }

    va_end(ap);
  }

  // Generate a sequence of values to initialize the vector.
  template <typename VectorT>
  void makeSequence(VectorT& v, int start, int end) {
    for (int i = start; i <= end; ++i) {
      v.push_back(Constructable(i));
    }
  }
};

// Test fixture class
template <typename VectorT>
class SmallVectorTest : public SmallVectorTestBase {
 protected:
  VectorT theVector;
  VectorT otherVector;
};

typedef ::testing::Types<
    SmallVector<Constructable, 0>,
    SmallVector<Constructable, 1>,
    SmallVector<Constructable, 2>,
    SmallVector<Constructable, 4>,
    SmallVector<Constructable, 5>>
    SmallVectorTestTypes;
TYPED_TEST_SUITE(SmallVectorTest, SmallVectorTestTypes, );

// Constructor test.
TYPED_TEST(SmallVectorTest, ConstructorNonIterTest) {
  SCOPED_TRACE("ConstructorTest");
  this->theVector = SmallVector<Constructable, 2>(2, 2);
  this->assertValuesInOrder(this->theVector, 2u, 2, 2);
}

// Constructor test.
TYPED_TEST(SmallVectorTest, ConstructorIterTest) {
  SCOPED_TRACE("ConstructorTest");
  int arr[] = {1, 2, 3};
  this->theVector =
      SmallVector<Constructable, 4>(std::begin(arr), std::end(arr));
  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// New vector test.
TYPED_TEST(SmallVectorTest, EmptyVectorTest) {
  SCOPED_TRACE("EmptyVectorTest");
  this->assertEmpty(this->theVector);
  EXPECT_TRUE(this->theVector.rbegin() == this->theVector.rend());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
}

// Simple insertions and deletions.
TYPED_TEST(SmallVectorTest, PushPopTest) {
  SCOPED_TRACE("PushPopTest");

  // Track whether the vector will potentially have to grow.
  bool RequiresGrowth = this->theVector.capacity() < 3;

  // Push an element
  this->theVector.push_back(Constructable(1));

  // Size tests
  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_FALSE(this->theVector.begin() == this->theVector.end());
  EXPECT_FALSE(this->theVector.empty());

  // Push another element
  this->theVector.push_back(Constructable(2));
  this->assertValuesInOrder(this->theVector, 2u, 1, 2);

  // Insert at beginning. Reserve space to avoid reference invalidation from
  // this->theVector[1].
  this->theVector.reserve(this->theVector.size() + 1);
  this->theVector.insert(this->theVector.begin(), this->theVector[1]);
  this->assertValuesInOrder(this->theVector, 3u, 2, 1, 2);

  // Pop one element
  this->theVector.pop_back();
  this->assertValuesInOrder(this->theVector, 2u, 2, 1);

  // Pop remaining elements
  this->theVector.pop_back_n(2);
  this->assertEmpty(this->theVector);

  // Check number of constructor calls. Should be 2 for each list element,
  // one for the argument to push_back, one for the argument to insert,
  // and one for the list element itself.
  if (!RequiresGrowth) {
    EXPECT_EQ(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(5, Constructable::getNumDestructorCalls());
  } else {
    // If we had to grow the vector, these only have a lower bound, but should
    // always be equal.
    EXPECT_LE(5, Constructable::getNumConstructorCalls());
    EXPECT_EQ(
        Constructable::getNumConstructorCalls(),
        Constructable::getNumDestructorCalls());
  }
}

// Clear test.
TYPED_TEST(SmallVectorTest, ClearTest) {
  SCOPED_TRACE("ClearTest");

  this->theVector.reserve(2);
  this->makeSequence(this->theVector, 1, 2);
  this->theVector.clear();

  this->assertEmpty(this->theVector);
  EXPECT_EQ(4, Constructable::getNumConstructorCalls());
  EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

// Resize smaller test.
TYPED_TEST(SmallVectorTest, ResizeShrinkTest) {
  SCOPED_TRACE("ResizeShrinkTest");

  this->theVector.reserve(3);
  this->makeSequence(this->theVector, 1, 3);
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
  EXPECT_EQ(6, Constructable::getNumConstructorCalls());
  EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// Resize bigger test.
TYPED_TEST(SmallVectorTest, ResizeGrowTest) {
  SCOPED_TRACE("ResizeGrowTest");

  this->theVector.resize(2);

  EXPECT_EQ(2, Constructable::getNumConstructorCalls());
  EXPECT_EQ(0, Constructable::getNumDestructorCalls());
  EXPECT_EQ(2u, this->theVector.size());
}

TYPED_TEST(SmallVectorTest, ResizeWithElementsTest) {
  this->theVector.resize(2);

  Constructable::reset();

  this->theVector.resize(4);

  size_t Ctors = Constructable::getNumConstructorCalls();
  EXPECT_TRUE(Ctors == 2 || Ctors == 4);
  size_t MoveCtors = Constructable::getNumMoveConstructorCalls();
  EXPECT_TRUE(MoveCtors == 0 || MoveCtors == 2);
  size_t Dtors = Constructable::getNumDestructorCalls();
  EXPECT_TRUE(Dtors == 0 || Dtors == 2);
}

// Resize with fill value.
TYPED_TEST(SmallVectorTest, ResizeFillTest) {
  SCOPED_TRACE("ResizeFillTest");

  this->theVector.resize(3, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 77, 77, 77);
}

TEST(SmallVectorTest, ResizeForOverwrite) {
  {
    // Heap allocated storage.
    SmallVector<unsigned, 0> V;
    V.push_back(5U);
    V.pop_back();
    V.resize_for_overwrite(V.size() + 1U);
    EXPECT_EQ(5U, V.back());
    V.pop_back();
    V.resize(V.size() + 1);
    EXPECT_EQ(0U, V.back());
  }
  {
    // Inline storage.
    SmallVector<unsigned, 2> V;
    V.push_back(5U);
    V.pop_back();
    V.resize_for_overwrite(V.size() + 1U);
    EXPECT_EQ(5U, V.back());
    V.pop_back();
    V.resize(V.size() + 1);
    EXPECT_EQ(0U, V.back());
  }
}

// Overflow past fixed size.
TYPED_TEST(SmallVectorTest, OverflowTest) {
  SCOPED_TRACE("OverflowTest");

  // Push more elements than the fixed size.
  this->makeSequence(this->theVector, 1, 10);

  // Test size and values.
  EXPECT_EQ(10u, this->theVector.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i + 1, this->theVector[i].getValue());
  }

  // Now resize back to fixed size.
  this->theVector.resize(1);

  this->assertValuesInOrder(this->theVector, 1u, 1);
}

// Iteration tests.
TYPED_TEST(SmallVectorTest, IterationTest) {
  this->makeSequence(this->theVector, 1, 2);

  // Forward Iteration
  typename TypeParam::iterator it = this->theVector.begin();
  EXPECT_TRUE(*it == this->theVector.front());
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());
  ++it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_TRUE(*it == this->theVector.back());
  EXPECT_EQ(2, it->getValue());
  ++it;
  EXPECT_TRUE(it == this->theVector.end());
  --it;
  EXPECT_TRUE(*it == this->theVector[1]);
  EXPECT_EQ(2, it->getValue());
  --it;
  EXPECT_TRUE(*it == this->theVector[0]);
  EXPECT_EQ(1, it->getValue());

  // Reverse Iteration
  typename TypeParam::reverse_iterator rit = this->theVector.rbegin();
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
  ++rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  ++rit;
  EXPECT_TRUE(rit == this->theVector.rend());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[0]);
  EXPECT_EQ(1, rit->getValue());
  --rit;
  EXPECT_TRUE(*rit == this->theVector[1]);
  EXPECT_EQ(2, rit->getValue());
}

// Swap test.
TYPED_TEST(SmallVectorTest, SwapTest) {
  SCOPED_TRACE("SwapTest");

  this->makeSequence(this->theVector, 1, 2);
  std::swap(this->theVector, this->otherVector);

  this->assertEmpty(this->theVector);
  this->assertValuesInOrder(this->otherVector, 2u, 1, 2);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");

  this->makeSequence(this->otherVector, 2, 3);

  this->theVector.push_back(Constructable(1));
  this->theVector.append(this->otherVector.begin(), this->otherVector.end());

  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Append repeated test
TYPED_TEST(SmallVectorTest, AppendRepeatedTest) {
  SCOPED_TRACE("AppendRepeatedTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.append(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 3u, 1, 77, 77);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendNonIterTest) {
  SCOPED_TRACE("AppendRepeatedTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.append(2, 7);
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

struct output_iterator {
  typedef std::output_iterator_tag iterator_category;
  typedef int value_type;
  typedef int difference_type;
  typedef value_type* pointer;
  typedef value_type& reference;
  operator int() {
    return 2;
  }
  operator Constructable() {
    return 7;
  }
};

TYPED_TEST(SmallVectorTest, AppendRepeatedNonForwardIterator) {
  SCOPED_TRACE("AppendRepeatedTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.append(output_iterator(), output_iterator());
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

TYPED_TEST(SmallVectorTest, AppendSmallVector) {
  SCOPED_TRACE("AppendSmallVector");

  SmallVector<Constructable, 3> otherVector = {7, 7};
  this->theVector.push_back(Constructable(1));
  this->theVector.append(otherVector);
  this->assertValuesInOrder(this->theVector, 3u, 1, 7, 7);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignTest) {
  SCOPED_TRACE("AssignTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.assign(2, Constructable(77));
  this->assertValuesInOrder(this->theVector, 2u, 77, 77);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignRangeTest) {
  SCOPED_TRACE("AssignTest");

  this->theVector.push_back(Constructable(1));
  int arr[] = {1, 2, 3};
  this->theVector.assign(std::begin(arr), std::end(arr));
  this->assertValuesInOrder(this->theVector, 3u, 1, 2, 3);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignNonIterTest) {
  SCOPED_TRACE("AssignTest");

  this->theVector.push_back(Constructable(1));
  this->theVector.assign(2, 7);
  this->assertValuesInOrder(this->theVector, 2u, 7, 7);
}

TYPED_TEST(SmallVectorTest, AssignSmallVector) {
  SCOPED_TRACE("AssignSmallVector");

  SmallVector<Constructable, 3> otherVector = {7, 7};
  this->theVector.push_back(Constructable(1));
  this->theVector.assign(otherVector);
  this->assertValuesInOrder(this->theVector, 2u, 7, 7);
}

// Move-assign test
TYPED_TEST(SmallVectorTest, MoveAssignTest) {
  SCOPED_TRACE("MoveAssignTest");

  // Set up our vector with a single element, but enough capacity for 4.
  this->theVector.reserve(4);
  this->theVector.push_back(Constructable(1));

  // Set up the other vector with 2 elements.
  this->otherVector.push_back(Constructable(2));
  this->otherVector.push_back(Constructable(3));

  // Move-assign from the other vector.
  this->theVector = std::move(this->otherVector);

  // Make sure we have the right result.
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);

  // Make sure the # of constructor/destructor calls line up. There
  // are two live objects after clearing the other vector.
  this->otherVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls() - 2,
      Constructable::getNumDestructorCalls());

  // There shouldn't be any live objects any more.
  this->theVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls(),
      Constructable::getNumDestructorCalls());
}

// Erase a single element
TYPED_TEST(SmallVectorTest, EraseTest) {
  SCOPED_TRACE("EraseTest");

  this->makeSequence(this->theVector, 1, 3);
  const auto& theConstVector = this->theVector;
  this->theVector.erase(theConstVector.begin());
  this->assertValuesInOrder(this->theVector, 2u, 2, 3);
}

// Erase a range of elements
TYPED_TEST(SmallVectorTest, EraseRangeTest) {
  SCOPED_TRACE("EraseRangeTest");

  this->makeSequence(this->theVector, 1, 3);
  const auto& theConstVector = this->theVector;
  this->theVector.erase(theConstVector.begin(), theConstVector.begin() + 2);
  this->assertValuesInOrder(this->theVector, 1u, 3);
}

// Insert a single element.
TYPED_TEST(SmallVectorTest, InsertTest) {
  SCOPED_TRACE("InsertTest");

  this->makeSequence(this->theVector, 1, 3);
  typename TypeParam::iterator I =
      this->theVector.insert(this->theVector.begin() + 1, Constructable(77));
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);
}

// Insert a copy of a single element.
TYPED_TEST(SmallVectorTest, InsertCopy) {
  SCOPED_TRACE("InsertTest");

  this->makeSequence(this->theVector, 1, 3);
  Constructable C(77);
  typename TypeParam::iterator I =
      this->theVector.insert(this->theVector.begin() + 1, C);
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 4u, 1, 77, 2, 3);
}

// Insert repeated elements.
TYPED_TEST(SmallVectorTest, InsertRepeatedTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 1, 4);
  Constructable::reset();
  auto I =
      this->theVector.insert(this->theVector.begin() + 1, 2, Constructable(16));
  // Move construct the top element into newly allocated space, and optionally
  // reallocate the whole buffer, move constructing into it.
  // FIXME: This is inefficient, we shouldn't move things into newly allocated
  // space, then move them up/around, there should only be 2 or 4 move
  // constructions here.
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 2 ||
      Constructable::getNumMoveConstructorCalls() == 6);
  // Move assign the next two to shift them up and make a gap.
  EXPECT_EQ(1, Constructable::getNumMoveAssignmentCalls());
  // Copy construct the two new elements from the parameter.
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // All without any copy construction.
  EXPECT_EQ(0, Constructable::getNumCopyConstructorCalls());
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 16, 16, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedNonIterTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 1, 4);
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.begin() + 1, 2, 7);
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 7, 7, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedAtEndTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 1, 4);
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.end(), 2, Constructable(16));
  // Just copy construct them into newly allocated space
  EXPECT_EQ(2, Constructable::getNumCopyConstructorCalls());
  // Move everything across if reallocation is needed.
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 0 ||
      Constructable::getNumMoveConstructorCalls() == 4);
  // Without ever moving or copying anything else.
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());

  EXPECT_EQ(this->theVector.begin() + 4, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 2, 3, 4, 16, 16);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedEmptyTest) {
  SCOPED_TRACE("InsertRepeatedTest");

  this->makeSequence(this->theVector, 10, 15);

  // Empty insert.
  EXPECT_EQ(
      this->theVector.end(),
      this->theVector.insert(this->theVector.end(), 0, Constructable(42)));
  EXPECT_EQ(
      this->theVector.begin() + 1,
      this->theVector.insert(
          this->theVector.begin() + 1, 0, Constructable(42)));
}

// Insert range.
TYPED_TEST(SmallVectorTest, InsertRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] = {
      Constructable(77), Constructable(77), Constructable(77)};

  this->makeSequence(this->theVector, 1, 3);
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.begin() + 1, Arr, Arr + 3);
  // Move construct the top 3 elements into newly allocated space.
  // Possibly move the whole sequence into new space first.
  // FIXME: This is inefficient, we shouldn't move things into newly allocated
  // space, then move them up/around, there should only be 2 or 3 move
  // constructions here.
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 2 ||
      Constructable::getNumMoveConstructorCalls() == 5);
  // Copy assign the lower 2 new elements into existing space.
  EXPECT_EQ(2, Constructable::getNumCopyAssignmentCalls());
  // Copy construct the third element into newly allocated space.
  EXPECT_EQ(1, Constructable::getNumCopyConstructorCalls());
  EXPECT_EQ(this->theVector.begin() + 1, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 77, 77, 77, 2, 3);
}

TYPED_TEST(SmallVectorTest, InsertRangeAtEndTest) {
  SCOPED_TRACE("InsertRangeTest");

  Constructable Arr[3] = {
      Constructable(77), Constructable(77), Constructable(77)};

  this->makeSequence(this->theVector, 1, 3);

  // Insert at end.
  Constructable::reset();
  auto I = this->theVector.insert(this->theVector.end(), Arr, Arr + 3);
  // Copy construct the 3 elements into new space at the top.
  EXPECT_EQ(3, Constructable::getNumCopyConstructorCalls());
  // Don't copy/move anything else.
  EXPECT_EQ(0, Constructable::getNumCopyAssignmentCalls());
  // Reallocation might occur, causing all elements to be moved into the new
  // buffer.
  EXPECT_TRUE(
      Constructable::getNumMoveConstructorCalls() == 0 ||
      Constructable::getNumMoveConstructorCalls() == 3);
  EXPECT_EQ(0, Constructable::getNumMoveAssignmentCalls());
  EXPECT_EQ(this->theVector.begin() + 3, I);
  this->assertValuesInOrder(this->theVector, 6u, 1, 2, 3, 77, 77, 77);
}

TYPED_TEST(SmallVectorTest, InsertEmptyRangeTest) {
  SCOPED_TRACE("InsertRangeTest");

  this->makeSequence(this->theVector, 1, 3);

  // Empty insert.
  EXPECT_EQ(
      this->theVector.end(),
      this->theVector.insert(
          this->theVector.end(),
          this->theVector.begin(),
          this->theVector.begin()));
  EXPECT_EQ(
      this->theVector.begin() + 1,
      this->theVector.insert(
          this->theVector.begin() + 1,
          this->theVector.begin(),
          this->theVector.begin()));
}

// Comparison tests.
TYPED_TEST(SmallVectorTest, ComparisonTest) {
  SCOPED_TRACE("ComparisonTest");

  this->makeSequence(this->theVector, 1, 3);
  this->makeSequence(this->otherVector, 1, 3);

  EXPECT_TRUE(this->theVector == this->otherVector);
  EXPECT_FALSE(this->theVector != this->otherVector);

  this->otherVector.clear();
  this->makeSequence(this->otherVector, 2, 4);

  EXPECT_FALSE(this->theVector == this->otherVector);
  EXPECT_TRUE(this->theVector != this->otherVector);
}

// Constant vector tests.
TYPED_TEST(SmallVectorTest, ConstVectorTest) {
  const TypeParam constVector;

  EXPECT_EQ(0u, constVector.size());
  EXPECT_TRUE(constVector.empty());
  EXPECT_TRUE(constVector.begin() == constVector.end());
}

// Direct array access.
TYPED_TEST(SmallVectorTest, DirectVectorTest) {
  EXPECT_EQ(0u, this->theVector.size());
  this->theVector.reserve(4);
  EXPECT_LE(4u, this->theVector.capacity());
  EXPECT_EQ(0, Constructable::getNumConstructorCalls());
  this->theVector.push_back(1);
  this->theVector.push_back(2);
  this->theVector.push_back(3);
  this->theVector.push_back(4);
  EXPECT_EQ(4u, this->theVector.size());
  EXPECT_EQ(8, Constructable::getNumConstructorCalls());
  EXPECT_EQ(1, this->theVector[0].getValue());
  EXPECT_EQ(2, this->theVector[1].getValue());
  EXPECT_EQ(3, this->theVector[2].getValue());
  EXPECT_EQ(4, this->theVector[3].getValue());
}

TYPED_TEST(SmallVectorTest, IteratorTest) {
  std::list<int> L;
  this->theVector.insert(this->theVector.end(), L.begin(), L.end());
}

template <typename InvalidType>
class DualSmallVectorsTest;

template <typename VectorT1, typename VectorT2>
class DualSmallVectorsTest<std::pair<VectorT1, VectorT2>>
    : public SmallVectorTestBase {
 protected:
  VectorT1 theVector;
  VectorT2 otherVector;

  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }
};

typedef ::testing::Types<
    // Small mode -> Small mode.
    std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 4>>,
    // Small mode -> Big mode.
    std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 2>>,
    // Big mode -> Small mode.
    std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 4>>,
    // Big mode -> Big mode.
    std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 2>>>
    DualSmallVectorTestTypes;

TYPED_TEST_SUITE(DualSmallVectorsTest, DualSmallVectorTestTypes, );

TYPED_TEST(DualSmallVectorsTest, MoveAssignment) {
  SCOPED_TRACE("MoveAssignTest-DualVectorTypes");

  // Set up our vector with four elements.
  for (int I = 0; I < 4; ++I)
    this->otherVector.push_back(Constructable(I));

  const Constructable* OrigDataPtr = this->otherVector.data();

  // Move-assign from the other vector.
  this->theVector = std::move(
      static_cast<SmallVectorImpl<Constructable>&>(this->otherVector));

  // Make sure we have the right result.
  this->assertValuesInOrder(this->theVector, 4u, 0, 1, 2, 3);

  // Make sure the # of constructor/destructor calls line up. There
  // are two live objects after clearing the other vector.
  this->otherVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls() - 4,
      Constructable::getNumDestructorCalls());

  // If the source vector (otherVector) was in small-mode, assert that we just
  // moved the data pointer over.
  EXPECT_TRUE(
      this->NumBuiltinElts(this->otherVector) == 4 ||
      this->theVector.data() == OrigDataPtr);

  // There shouldn't be any live objects any more.
  this->theVector.clear();
  EXPECT_EQ(
      Constructable::getNumConstructorCalls(),
      Constructable::getNumDestructorCalls());

  // We shouldn't have copied anything in this whole process.
  EXPECT_EQ(Constructable::getNumCopyConstructorCalls(), 0);
}

struct notassignable {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  int& x;
  notassignable(int& x) : x(x) {}
};

TEST(SmallVectorCustomTest, NoAssignTest) {
  int x = 0;
  SmallVector<notassignable, 2> vec;
  vec.push_back(notassignable(x));
  x = 42;
  EXPECT_EQ(x, vec.pop_back_val().x);
}

struct MovedFrom {
  bool hasValue;
  MovedFrom() : hasValue(true) {}
  MovedFrom(MovedFrom&& m) noexcept : hasValue(m.hasValue) {
    m.hasValue = false;
  }
  MovedFrom& operator=(MovedFrom&& m) noexcept {
    hasValue = m.hasValue;
    m.hasValue = false;
    return *this;
  }
};

TEST(SmallVectorTest, MidInsert) {
  SmallVector<MovedFrom, 3> v;
  v.push_back(MovedFrom());
  v.insert(v.begin(), MovedFrom());
  for (MovedFrom& m : v)
    EXPECT_TRUE(m.hasValue);
}

enum EmplaceableArgState {
  EAS_Defaulted,
  EAS_Arg,
  EAS_LValue,
  EAS_RValue,
  EAS_Failure
};
template <int I>
struct EmplaceableArg {
  EmplaceableArgState State;
  EmplaceableArg() : State(EAS_Defaulted) {}
  EmplaceableArg(EmplaceableArg&& X) noexcept
      : State(X.State == EAS_Arg ? EAS_RValue : EAS_Failure) {}
  EmplaceableArg(EmplaceableArg& X)
      : State(X.State == EAS_Arg ? EAS_LValue : EAS_Failure) {}

  explicit EmplaceableArg(bool) : State(EAS_Arg) {}

  EmplaceableArg& operator=(EmplaceableArg&&) = delete;
  EmplaceableArg& operator=(const EmplaceableArg&) = delete;
};

enum EmplaceableState { ES_Emplaced, ES_Moved };
struct Emplaceable {
  EmplaceableArg<0> A0;
  EmplaceableArg<1> A1;
  EmplaceableArg<2> A2;
  EmplaceableArg<3> A3;
  EmplaceableState State;

  Emplaceable() : State(ES_Emplaced) {}

  template <class A0Ty>
  explicit Emplaceable(A0Ty&& A0)
      : A0(std::forward<A0Ty>(A0)), State(ES_Emplaced) {}

  template <class A0Ty, class A1Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        State(ES_Emplaced) {}

  template <class A0Ty, class A1Ty, class A2Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1, A2Ty&& A2)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        A2(std::forward<A2Ty>(A2)),
        State(ES_Emplaced) {}

  template <class A0Ty, class A1Ty, class A2Ty, class A3Ty>
  Emplaceable(A0Ty&& A0, A1Ty&& A1, A2Ty&& A2, A3Ty&& A3)
      : A0(std::forward<A0Ty>(A0)),
        A1(std::forward<A1Ty>(A1)),
        A2(std::forward<A2Ty>(A2)),
        A3(std::forward<A3Ty>(A3)),
        State(ES_Emplaced) {}

  Emplaceable(Emplaceable&&) noexcept : State(ES_Moved) {}
  Emplaceable& operator=(Emplaceable&&) noexcept {
    State = ES_Moved;
    return *this;
  }

  Emplaceable(const Emplaceable&) = delete;
  Emplaceable& operator=(const Emplaceable&) = delete;
};

TEST(SmallVectorTest, EmplaceBack) {
  EmplaceableArg<0> A0(true);
  EmplaceableArg<1> A1(true);
  EmplaceableArg<2> A2(true);
  EmplaceableArg<3> A3(true);
  {
    SmallVector<Emplaceable, 3> V;
    Emplaceable& back = V.emplace_back();
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_Defaulted);
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    SmallVector<Emplaceable, 3> V;
    Emplaceable& back = V.emplace_back(std::move(A0));
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    SmallVector<Emplaceable, 3> V;
    Emplaceable& back = V.emplace_back(A0);
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_LValue);
    EXPECT_TRUE(back.A1.State == EAS_Defaulted);
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    SmallVector<Emplaceable, 3> V;
    Emplaceable& back = V.emplace_back(A0, A1);
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_LValue);
    EXPECT_TRUE(back.A1.State == EAS_LValue);
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    SmallVector<Emplaceable, 3> V;
    Emplaceable& back = V.emplace_back(std::move(A0), std::move(A1));
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    EXPECT_TRUE(back.A1.State == EAS_RValue);
    EXPECT_TRUE(back.A2.State == EAS_Defaulted);
    EXPECT_TRUE(back.A3.State == EAS_Defaulted);
  }
  {
    SmallVector<Emplaceable, 3> V;
    // NOLINTNEXTLINE(bugprone-use-after-move)
    Emplaceable& back = V.emplace_back(std::move(A0), A1, std::move(A2), A3);
    EXPECT_TRUE(&back == &V.back());
    EXPECT_TRUE(V.size() == 1);
    EXPECT_TRUE(back.State == ES_Emplaced);
    EXPECT_TRUE(back.A0.State == EAS_RValue);
    EXPECT_TRUE(back.A1.State == EAS_LValue);
    EXPECT_TRUE(back.A2.State == EAS_RValue);
    EXPECT_TRUE(back.A3.State == EAS_LValue);
  }
  {
    SmallVector<int, 1> V;
    V.emplace_back();
    V.emplace_back(42);
    EXPECT_EQ(2U, V.size());
    EXPECT_EQ(0, V[0]);
    EXPECT_EQ(42, V[1]);
  }
}

TEST(SmallVectorTest, DefaultInlinedElements) {
  SmallVector<int> V;
  EXPECT_TRUE(V.empty());
  V.push_back(7);
  EXPECT_EQ(V[0], 7);

  // Check that at least a couple layers of nested SmallVector<T>'s are allowed
  // by the default inline elements policy. This pattern happens in practice
  // with some frequency, and it seems fairly harmless even though each layer of
  // SmallVector's will grow the total sizeof by a vector header beyond the
  // "preferred" maximum sizeof.
  SmallVector<SmallVector<SmallVector<int>>> NestedV;
  NestedV.emplace_back().emplace_back().emplace_back(42);
  EXPECT_EQ(NestedV[0][0][0], 42);
}

TEST(SmallVectorTest, InitializerList) {
  SmallVector<int, 2> V1 = {};
  EXPECT_TRUE(V1.empty());
  V1 = {0, 0};
  EXPECT_TRUE(makeArrayRef(V1).equals({0, 0}));
  V1 = {-1, -1};
  EXPECT_TRUE(makeArrayRef(V1).equals({-1, -1}));

  SmallVector<int, 2> V2 = {1, 2, 3, 4};
  EXPECT_TRUE(makeArrayRef(V2).equals({1, 2, 3, 4}));
  V2.assign({4});
  EXPECT_TRUE(makeArrayRef(V2).equals({4}));
  V2.append({3, 2});
  EXPECT_TRUE(makeArrayRef(V2).equals({4, 3, 2}));
  V2.insert(V2.begin() + 1, 5);
  EXPECT_TRUE(makeArrayRef(V2).equals({4, 5, 3, 2}));
}

template <class VectorT>
class SmallVectorReferenceInvalidationTest : public SmallVectorTestBase {
 protected:
  const char* AssertionMessage =
      "Attempting to reference an element of the vector in an operation \" "
      "\"that invalidates it";

  VectorT V;

  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }

  template <class T>
  static bool isValueType() {
    return std::is_same<T, typename VectorT::value_type>::value;
  }

  void SetUp() override {
    SmallVectorTestBase::SetUp();

    // Fill up the small size so that insertions move the elements.
    for (int I = 0, E = NumBuiltinElts(V); I != E; ++I)
      V.emplace_back(I + 1);
  }
};

// Test one type that's trivially copyable (int) and one that isn't
// (Constructable) since reference invalidation may be fixed differently for
// each.
using SmallVectorReferenceInvalidationTestTypes =
    ::testing::Types<SmallVector<int, 3>, SmallVector<Constructable, 3>>;

TYPED_TEST_SUITE(
    SmallVectorReferenceInvalidationTest,
    SmallVectorReferenceInvalidationTestTypes, );

TYPED_TEST(SmallVectorReferenceInvalidationTest, PushBack) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  int N = this->NumBuiltinElts(V);

  // Push back a reference to last element when growing from small storage.
  V.push_back(V.back());
  EXPECT_EQ(N, V.back());

  // Check that the old value is still there (not moved away).
  EXPECT_EQ(N, V[V.size() - 2]);

  // Fill storage again.
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // Push back a reference to last element when growing from large storage.
  V.push_back(V.back());
  EXPECT_EQ(int(V.size()) - 1, V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, PushBackMoved) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  int N = this->NumBuiltinElts(V);

  // Push back a reference to last element when growing from small storage.
  V.push_back(std::move(V.back()));
  EXPECT_EQ(N, V.back());
  if (this->template isValueType<Constructable>()) {
    // Check that the value was moved (not copied).
    EXPECT_EQ(0, V[V.size() - 2]);
  }

  // Fill storage again.
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // Push back a reference to last element when growing from large storage.
  V.push_back(std::move(V.back()));

  // Check the values.
  EXPECT_EQ(int(V.size()) - 1, V.back());
  if (this->template isValueType<Constructable>()) {
    // Check the value got moved out.
    EXPECT_EQ(0, V[V.size() - 2]);
  }
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Resize) {
  auto& V = this->V;
  (void)V;
  int N = this->NumBuiltinElts(V);
  V.resize(N + 1, V.back());
  EXPECT_EQ(N, V.back());

  // Resize to add enough elements that V will grow again. If reference
  // invalidation breaks in the future, sanitizers should be able to catch a
  // use-after-free here.
  V.resize(V.capacity() + 1, V.front());
  EXPECT_EQ(1, V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Append) {
  auto& V = this->V;
  (void)V;
  V.append(1, V.back());
  int N = this->NumBuiltinElts(V);
  EXPECT_EQ(N, V[N - 1]);

  // Append enough more elements that V will grow again. This tests growing
  // when already in large mode.
  //
  // If reference invalidation breaks in the future, sanitizers should be able
  // to catch a use-after-free here.
  V.append(V.capacity() - V.size() + 1, V.front());
  EXPECT_EQ(1, V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, AppendRange) {
  auto& V = this->V;
  (void)V;
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(V.append(V.begin(), V.begin() + 1), this->AssertionMessage);

  ASSERT_EQ(3u, this->NumBuiltinElts(V));
  ASSERT_EQ(3u, V.size());
  V.pop_back();
  ASSERT_EQ(2u, V.size());

  // Confirm this checks for growth when there's more than one element
  // appended.
  EXPECT_DEATH(V.append(V.begin(), V.end()), this->AssertionMessage);
#endif
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Assign) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  (void)V;
  int N = this->NumBuiltinElts(V);
  ASSERT_EQ(unsigned(N), V.size());
  ASSERT_EQ(unsigned(N), V.capacity());

  // Check assign that shrinks in small mode.
  V.assign(1, V.back());
  EXPECT_EQ(1u, V.size());
  EXPECT_EQ(N, V[0]);

  // Check assign that grows within small mode.
  ASSERT_LT(V.size(), V.capacity());
  V.assign(V.capacity(), V.back());
  for (int I = 0, E = V.size(); I != E; ++I) {
    EXPECT_EQ(N, V[I]);

    // Reset to [1, 2, ...].
    V[I] = I + 1;
  }

  // Check assign that grows to large mode.
  ASSERT_EQ(2, V[1]);
  V.assign(V.capacity() + 1, V[1]);
  for (int I = 0, E = V.size(); I != E; ++I) {
    EXPECT_EQ(2, V[I]);

    // Reset to [1, 2, ...].
    V[I] = I + 1;
  }

  // Check assign that shrinks in large mode.
  V.assign(1, V[1]);
  EXPECT_EQ(2, V[0]);
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, AssignRange) {
  auto& V = this->V;
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(V.assign(V.begin(), V.end()), this->AssertionMessage);
  EXPECT_DEATH(V.assign(V.begin(), V.end() - 1), this->AssertionMessage);
#endif
  V.assign(V.begin(), V.begin());
  EXPECT_TRUE(V.empty());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, Insert) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  (void)V;

  // Insert a reference to the back (not at end() or else insert delegates to
  // push_back()), growing out of small mode. Confirm the value was copied out
  // (moving out Constructable sets it to 0).
  V.insert(V.begin(), V.back());
  EXPECT_EQ(int(V.size() - 1), V.front());
  EXPECT_EQ(int(V.size() - 1), V.back());

  // Fill up the vector again.
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // Grow again from large storage to large storage.
  V.insert(V.begin(), V.back());
  EXPECT_EQ(int(V.size() - 1), V.front());
  EXPECT_EQ(int(V.size() - 1), V.back());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertMoved) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  (void)V;

  // Insert a reference to the back (not at end() or else insert delegates to
  // push_back()), growing out of small mode. Confirm the value was copied out
  // (moving out Constructable sets it to 0).
  V.insert(V.begin(), std::move(V.back()));
  EXPECT_EQ(int(V.size() - 1), V.front());
  if (this->template isValueType<Constructable>()) {
    // Check the value got moved out.
    EXPECT_EQ(0, V.back());
  }

  // Fill up the vector again.
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // Grow again from large storage to large storage.
  V.insert(V.begin(), std::move(V.back()));
  EXPECT_EQ(int(V.size() - 1), V.front());
  if (this->template isValueType<Constructable>()) {
    // Check the value got moved out.
    EXPECT_EQ(0, V.back());
  }
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertN) {
  auto& V = this->V;
  (void)V;

  // Cover NumToInsert <= this->end() - I.
  V.insert(V.begin() + 1, 1, V.back());
  int N = this->NumBuiltinElts(V);
  EXPECT_EQ(N, V[1]);

  // Cover NumToInsert > this->end() - I, inserting enough elements that V will
  // also grow again; V.capacity() will be more elements than necessary but
  // it's a simple way to cover both conditions.
  //
  // If reference invalidation breaks in the future, sanitizers should be able
  // to catch a use-after-free here.
  V.insert(V.begin(), V.capacity(), V.front());
  EXPECT_EQ(1, V.front());
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, InsertRange) {
  auto& V = this->V;
  (void)V;
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(
      V.insert(V.begin(), V.begin(), V.begin() + 1), this->AssertionMessage);

  ASSERT_EQ(3u, this->NumBuiltinElts(V));
  ASSERT_EQ(3u, V.size());
  V.pop_back();
  ASSERT_EQ(2u, V.size());

  // Confirm this checks for growth when there's more than one element
  // inserted.
  EXPECT_DEATH(V.insert(V.begin(), V.begin(), V.end()), this->AssertionMessage);
#endif
}

TYPED_TEST(SmallVectorReferenceInvalidationTest, EmplaceBack) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  int N = this->NumBuiltinElts(V);

  // Push back a reference to last element when growing from small storage.
  V.emplace_back(V.back());
  EXPECT_EQ(N, V.back());

  // Check that the old value is still there (not moved away).
  EXPECT_EQ(N, V[V.size() - 2]);

  // Fill storage again.
  V.back() = V.size();
  while (V.size() < V.capacity())
    V.push_back(V.size() + 1);

  // Push back a reference to last element when growing from large storage.
  V.emplace_back(V.back());
  EXPECT_EQ(int(V.size()) - 1, V.back());
}

template <class VectorT>
class SmallVectorInternalReferenceInvalidationTest
    : public SmallVectorTestBase {
 protected:
  const char* AssertionMessage =
      "Attempting to reference an element of the vector in an operation \" "
      "\"that invalidates it";

  VectorT V;

  template <typename T, unsigned N>
  static unsigned NumBuiltinElts(const SmallVector<T, N>&) {
    return N;
  }

  void SetUp() override {
    SmallVectorTestBase::SetUp();

    // Fill up the small size so that insertions move the elements.
    for (int I = 0, E = NumBuiltinElts(V); I != E; ++I)
      V.emplace_back(I + 1, I + 1);
  }
};

// Test pairs of the same types from SmallVectorReferenceInvalidationTestTypes.
using SmallVectorInternalReferenceInvalidationTestTypes = ::testing::Types<
    SmallVector<std::pair<int, int>, 3>,
    SmallVector<std::pair<Constructable, Constructable>, 3>>;

TYPED_TEST_SUITE(
    SmallVectorInternalReferenceInvalidationTest,
    SmallVectorInternalReferenceInvalidationTestTypes, );

TYPED_TEST(SmallVectorInternalReferenceInvalidationTest, EmplaceBack) {
  // Note: setup adds [1, 2, ...] to V until it's at capacity in small mode.
  auto& V = this->V;
  int N = this->NumBuiltinElts(V);

  // Push back a reference to last element when growing from small storage.
  V.emplace_back(V.back().first, V.back().second);
  EXPECT_EQ(N, V.back().first);
  EXPECT_EQ(N, V.back().second);

  // Check that the old value is still there (not moved away).
  EXPECT_EQ(N, V[V.size() - 2].first);
  EXPECT_EQ(N, V[V.size() - 2].second);

  // Fill storage again.
  V.back().first = V.back().second = V.size();
  while (V.size() < V.capacity())
    V.emplace_back(V.size() + 1, V.size() + 1);

  // Push back a reference to last element when growing from large storage.
  V.emplace_back(V.back().first, V.back().second);
  EXPECT_EQ(int(V.size()) - 1, V.back().first);
  EXPECT_EQ(int(V.size()) - 1, V.back().second);
}

} // end namespace
// NOLINTEND(*arrays, bugprone-forwarding-reference-overload)
