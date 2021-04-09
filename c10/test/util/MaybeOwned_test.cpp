#include <gtest/gtest.h>

#include <c10/util/MaybeOwned.h>

#include <memory>
#include <string>

template<typename T>
using MaybeOwned = c10::MaybeOwned<T>;

template <typename T>
static void assertBorrow(const MaybeOwned<T>& mo, const T& borrowedFrom) {
  EXPECT_EQ(*mo, borrowedFrom);
  EXPECT_EQ(&*mo, &borrowedFrom);
}

template <typename T>
static void assertOwn(const MaybeOwned<T>& mo, const T& original) {
  EXPECT_EQ(*mo, original);
  EXPECT_NE(&*mo, &original);
}


TEST(MaybeOwnedTest, SimpleDereferencingInt) {
  int x = 123;
  auto borrowed = MaybeOwned<int>::borrowed(x);
  auto owned = MaybeOwned<int>::owned(c10::in_place, x);

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
}

TEST(MaybeOwnedTest, SimpleDereferencingString) {
  std::string x = "hello";
  std::string y = x;
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::move(y));

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
  assertOwn(owned2, x);

  EXPECT_EQ(borrowed->size(), x.size());
  EXPECT_EQ(owned->size(), x.size());
  EXPECT_EQ(owned2->size(), x.size());
}

TEST(MaybeOwnedTest, DefaultCtorInt) {
  int x = 123;
  MaybeOwned<int> borrowed, owned;
  borrowed = MaybeOwned<int>::borrowed(x);
  owned = MaybeOwned<int>::owned(c10::in_place, x);

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
}

TEST(MaybeOwnedTest, CopyConstructor) {
  std::string x = "hello";

  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  auto copiedBorrowed(borrowed);
  auto copiedOwned(owned);
  auto copiedOwned2(owned2);

  for (auto *mo : {&borrowed, &copiedBorrowed}) {
    assertBorrow(*mo, x);
  }

  for (auto *mo : {&owned, &owned2, &copiedOwned, &copiedOwned2}) {
    assertOwn(*mo, x);
  }
}

TEST(MaybeOwnedTest, MoveDereferencingInt) {
  int x = 123;
  auto borrowed = MaybeOwned<int>::borrowed(x);
  auto owned = MaybeOwned<int>::owned(c10::in_place, x);

  // Moving from it gets the underlying int.
  EXPECT_EQ(*std::move(borrowed), x);
  EXPECT_EQ(*std::move(owned), x);

  // Borrowed is unaffected.
  EXPECT_EQ(*borrowed, x);
  EXPECT_EQ(&*borrowed, &x);

  // Owned is also unaffected because move is just copy for ints.
  EXPECT_EQ(*owned, x);
  EXPECT_NE(&*owned, &x);
}

// We use shared_ptr instead of string for this test because it has a
// specified moved-from state (null), unlike string.
TEST(MaybeOwnedTest, MoveDereferencingSharedPtr) {
  using MO = MaybeOwned<std::shared_ptr<int>>;
  std::shared_ptr<int> pBorrow = std::make_shared<int>(123);
  auto borrowed = MO::borrowed(pBorrow);
  auto owned = MO::owned(c10::in_place, std::make_shared<int>(456));

  EXPECT_EQ(**std::move(borrowed), 123);
  EXPECT_EQ(**std::move(owned), 456);

  // Borrowed is unaffected.
  EXPECT_EQ(**borrowed, 123);
  EXPECT_EQ(&*borrowed, &pBorrow);

  // Owned is a null shared_ptr<int>.
  EXPECT_EQ(*owned, nullptr);
}

TEST(MaybeOwnedTest, MoveConstructor) {
  std::string x = "hello";
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  auto movedBorrowed(std::move(borrowed));
  auto movedOwned(std::move(owned));
  auto movedOwned2(std::move(owned2));

  assertBorrow(movedBorrowed, x);
  assertOwn(movedOwned, x);
  assertOwn(movedOwned, x);
}

TEST(MaybeOwnedTest, CopyAssignmentIntoOwned) {
  std::string x = "hello";
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  auto copiedBorrowed = MaybeOwned<std::string>::owned(c10::in_place, "");
  auto copiedOwned = MaybeOwned<std::string>::owned(c10::in_place, "");
  auto copiedOwned2 = MaybeOwned<std::string>::owned(c10::in_place, "");

  copiedBorrowed = borrowed;
  copiedOwned = owned;
  copiedOwned2 = owned2;

  assertBorrow(borrowed, x);
  assertBorrow(copiedBorrowed, x);
  for (auto *mo : {&copiedOwned, &copiedOwned2, &owned, &owned2}) {
    assertOwn(*mo, x);
  }
}

TEST(MaybeOwnedTest, CopyAssignmentIntoBorrowed) {
  std::string x = "hello";
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  std::string y = "goodbye";
  auto copiedBorrowed = MaybeOwned<std::string>::borrowed(y);
  auto copiedOwned = MaybeOwned<std::string>::borrowed(y);
  auto copiedOwned2 = MaybeOwned<std::string>::borrowed(y);

  copiedBorrowed = borrowed;
  copiedOwned = owned;
  copiedOwned2 = owned2;

  assertBorrow(borrowed, x);
  assertBorrow(copiedBorrowed, x);
  for (auto *mo : {&copiedOwned, &copiedOwned2, &owned, &owned2}) {
    assertOwn(*mo, x);
  }
}


TEST(MaybeOwnedTest, MoveAssignmentIntoOwned) {
  std::string x = "hello";
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  auto movedBorrowed = MaybeOwned<std::string>::owned(c10::in_place, "");
  auto movedOwned = MaybeOwned<std::string>::owned(c10::in_place, "");
  auto movedOwned2 = MaybeOwned<std::string>::owned(c10::in_place, "");

  movedBorrowed = std::move(borrowed);
  movedOwned = std::move(owned);
  movedOwned2 = std::move(owned2);
}


TEST(MaybeOwnedTest, MoveAssignmentIntoBorrowed) {
  std::string x = "hello";
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  std::string y = "goodbye";
  auto movedBorrowed = MaybeOwned<std::string>::borrowed(y);
  auto movedOwned = MaybeOwned<std::string>::borrowed(y);
  auto movedOwned2 = MaybeOwned<std::string>::borrowed(y);

  movedBorrowed = std::move(borrowed);
  movedOwned = std::move(owned);
  movedOwned2 = std::move(owned2);

  assertBorrow(movedBorrowed, x);
  assertOwn(movedOwned, x);
  assertOwn(movedOwned2, x);
}

TEST(MaybeOwnedTest, SelfAssignment) {
  std::string x = "hello";

  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::string(x));

  borrowed = borrowed;
  owned = owned;
  owned2 = owned2;

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
}
