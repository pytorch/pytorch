#include <gtest/gtest.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/util/MaybeOwned.h>

#include <string>

template<typename T>
using MaybeOwned = c10::MaybeOwned<T>;

struct MyString : public c10::intrusive_ptr_target, public std::string {
  using std::string::string;
};

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


TEST(MaybeOwnedTest, SimpleDereferencingString) {
  auto x = c10::make_intrusive<MyString>("hello");
  auto y = x;
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(std::move(y));

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
  assertOwn(owned2, x);

  EXPECT_EQ((*borrowed)->size(), x->size());
  EXPECT_EQ((*owned)->size(), x->size());
  EXPECT_EQ((*owned2)->size(), x->size());
}

TEST(MaybeOwnedTest, DefaultCtor) {
  auto x = c10::make_intrusive<MyString>("123");
  MaybeOwned<c10::intrusive_ptr<MyString>> borrowed, owned;
  borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
}

TEST(MaybeOwnedTest, CopyConstructor) {
  auto x = c10::make_intrusive<MyString>("hello");

  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

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

TEST(MaybeOwnedTest, MoveConstructor) {
  auto x = c10::make_intrusive<MyString>("hello");
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  auto movedBorrowed(std::move(borrowed));
  auto movedOwned(std::move(owned));
  auto movedOwned2(std::move(owned2));

  assertBorrow(movedBorrowed, x);
  assertOwn(movedOwned, x);
  assertOwn(movedOwned, x);
}

TEST(MaybeOwnedTest, CopyAssignmentIntoOwned) {
  auto x = c10::make_intrusive<MyString>("hello");
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  auto copiedBorrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);
  auto copiedOwned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);
  auto copiedOwned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);

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
  auto x = c10::make_intrusive<MyString>("hello");
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  auto y = c10::make_intrusive<MyString>("goodbye");
  auto copiedBorrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);
  auto copiedOwned = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);
  auto copiedOwned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);

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
  auto x = c10::make_intrusive<MyString>("hello");
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  auto movedBorrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);
  auto movedOwned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);
  auto movedOwned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place);

  movedBorrowed = std::move(borrowed);
  movedOwned = std::move(owned);
  movedOwned2 = std::move(owned2);
}


TEST(MaybeOwnedTest, MoveAssignmentIntoBorrowed) {
  auto x = c10::make_intrusive<MyString>("hello");
  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  auto y = c10::make_intrusive<MyString>("goodbye");
  auto movedBorrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);
  auto movedOwned = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);
  auto movedOwned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(y);

  movedBorrowed = std::move(borrowed);
  movedOwned = std::move(owned);
  movedOwned2 = std::move(owned2);

  assertBorrow(movedBorrowed, x);
  assertOwn(movedOwned, x);
  assertOwn(movedOwned2, x);
}

TEST(MaybeOwnedTest, SelfAssignment) {
  auto x = c10::make_intrusive<MyString>("hello");

  auto borrowed = MaybeOwned<c10::intrusive_ptr<MyString>>::borrowed(x);
  auto owned = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::in_place, x);
  auto owned2 = MaybeOwned<c10::intrusive_ptr<MyString>>::owned(c10::intrusive_ptr<MyString>(x));

  borrowed = borrowed;
  owned = owned;
  owned2 = owned2;

  assertBorrow(borrowed, x);
  assertOwn(owned, x);
}
