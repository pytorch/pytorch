#include <gtest/gtest.h>

#include <c10/util/MaybeOwned.h>

#include <memory>
#include <string>

template<typename T>
using MaybeOwned = c10::MaybeOwned<T>;

TEST(MaybeOwnedTest, SimpleDereferencingInt) {
  int x = 123;
  auto borrowed = MaybeOwned<int>::borrowed(x);
  auto owned = MaybeOwned<int>::owned(c10::in_place, x);
  EXPECT_EQ(*borrowed, x);
  EXPECT_EQ(*owned, x);
  EXPECT_EQ(&*borrowed, &x);
  EXPECT_NE(&*owned, &x);
}

TEST(MaybeOwnedTest, SimpleDereferencingString) {
  std::string x = "hello";
  std::string y = x;
  auto borrowed = MaybeOwned<std::string>::borrowed(x);
  auto owned = MaybeOwned<std::string>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<std::string>::owned(std::move(y));
  EXPECT_EQ(*borrowed, x);
  EXPECT_EQ(*owned, x);
  EXPECT_EQ(*owned2, x);
  EXPECT_EQ(&*borrowed, &x);
  EXPECT_NE(&*owned, &x);
  EXPECT_NE(&*owned2, &x);

  EXPECT_EQ(borrowed->size(), x.size());
  EXPECT_EQ(owned->size(), x.size());
  EXPECT_EQ(owned2->size(), x.size());
}

TEST(MaybeOwnedTest, DefaultCtorInt) {
  int x = 123;
  MaybeOwned<int> borrowed, owned;
  borrowed = MaybeOwned<int>::borrowed(x);
  owned = MaybeOwned<int>::owned(c10::in_place, x);
  EXPECT_EQ(*borrowed, x);
  EXPECT_EQ(*owned, x);
  EXPECT_EQ(&*borrowed, &x);
  EXPECT_NE(&*owned, &x);
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

  for (auto *mo : {&movedBorrowed, &movedOwned, &movedOwned2}) {
    EXPECT_EQ(**mo, x);
    EXPECT_EQ((*mo)->size(), x.size());
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

  for (auto *mo : {&movedBorrowed, &movedOwned, &movedOwned2}) {
    EXPECT_EQ(**mo, x);
    EXPECT_EQ((*mo)->size(), x.size());
  }
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

  for (auto *mo : {&movedBorrowed, &movedOwned, &movedOwned2}) {
    EXPECT_EQ(**mo, x);
    EXPECT_EQ((*mo)->size(), x.size());
  }
}
