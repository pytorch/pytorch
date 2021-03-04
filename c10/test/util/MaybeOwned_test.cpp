#include <gtest/gtest.h>

#include <c10/util/MaybeOwned.h>

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
